"""`mack run`: one short instruction applied to every file in a folder.

A batch job is one task (so the dashboard, `mack task list` and the menu-bar app
show it like any other) whose items live in SQLite, one row per file. Each item
gets a fresh context holding only the grounding preamble, the instruction, the
output schema and that file: no plan, no digest, nothing from other items, so
order and resumption never matter.

Durability rules:
  * an item's output is written to the store before it is marked done
  * an item that was running when the process died goes back to pending; after
    two attempts it is failed
  * the JSONL output is rebuilt from the store on resume, so it never repeats lines
  * a file whose hash changed since enqueue is skipped, never processed stale
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from .config import Config
from .governor import load_json
from .schema import PREAMBLE, TEXT_SCHEMA, Schema, describe, extract_json, ground, parse_schema, retention, validate
from .tasks import PACES, Abandon, Stop, Task, TaskRunner, TaskStore, pace_rate
from .textsource import read_text, sha256
from .textutil import clip_middle
from . import system

DEFAULT_GLOB = "**/*.md,**/*.txt"
FINAL = ("done", "skipped", "failed", "flagged")
GUESS_OUT_TOKENS = 150
MAX_ENGINE_FAILURES = 3


# ---------------------------------------------------------------- store


class BatchStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS items (
        job_id TEXT NOT NULL, idx INTEGER NOT NULL, path TEXT NOT NULL, content_hash TEXT NOT NULL,
        status TEXT NOT NULL, reason TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        output TEXT, flags TEXT, error TEXT, started_at REAL, finished_at REAL,
        tokens_in INTEGER, tokens_out INTEGER, text TEXT,
        PRIMARY KEY (job_id, idx)
    );
    CREATE INDEX IF NOT EXISTS items_status ON items (job_id, status);
    """

    def __init__(self, cfg: Config):
        self.path = cfg.state_dir / "batch.db"
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(self.SCHEMA)

    def _x(self, sql: str, args=()):
        with self._lock:
            return self.db.execute(sql, args)

    def add(self, job_id: str, rows: list[dict]) -> None:
        with self._lock:
            self.db.execute("BEGIN")
            self.db.executemany(
                "INSERT INTO items (job_id, idx, path, content_hash, status, reason, text) VALUES (?,?,?,?,?,?,?)",
                [(job_id, r["idx"], r["path"], r["content_hash"], r["status"], r.get("reason"), r.get("text")) for r in rows],
            )
            self.db.execute("COMMIT")

    def items(self, job_id: str, status: Optional[str] = None) -> list[sqlite3.Row]:
        if status:
            return self._x("SELECT * FROM items WHERE job_id=? AND status=? ORDER BY idx", (job_id, status)).fetchall()
        return self._x("SELECT * FROM items WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()

    def item(self, job_id: str, idx: int) -> sqlite3.Row:
        return self._x("SELECT * FROM items WHERE job_id=? AND idx=?", (job_id, idx)).fetchone()

    def next_pending(self, job_id: str) -> Optional[sqlite3.Row]:
        return self._x("SELECT * FROM items WHERE job_id=? AND status='pending' ORDER BY idx LIMIT 1", (job_id,)).fetchone()

    def update(self, job_id: str, idx: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE items SET {cols} WHERE job_id=? AND idx=?", (*fields.values(), job_id, idx))

    def counts(self, job_id: str) -> dict:
        rows = self._x("SELECT status, COUNT(*) n FROM items WHERE job_id=? GROUP BY status", (job_id,)).fetchall()
        c = {r["status"]: r["n"] for r in rows}
        return {
            "total": sum(c.values()),
            "done": c.get("done", 0),
            "skipped": c.get("skipped", 0),
            "failed": c.get("failed", 0),
            "flagged": c.get("flagged", 0),
            "pending": c.get("pending", 0) + c.get("running", 0),
        }

    def recover(self, job_id: str) -> int:
        """Items left running by a dead process go back to pending (their attempt counts)."""
        return self._x("UPDATE items SET status='pending' WHERE job_id=? AND status='running'", (job_id,)).rowcount

    def recent(self, job_id: str, n: int = 20) -> list[sqlite3.Row]:
        return self._x(
            "SELECT * FROM items WHERE job_id=? AND finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT ?", (job_id, n)
        ).fetchall()

    def stats(self, job_id: str, window_s: float = 300) -> dict:
        r = self._x(
            "SELECT COUNT(*) n, AVG(tokens_out) avg_out, AVG(tokens_in) avg_in FROM items WHERE job_id=? AND status IN ('done','flagged')",
            (job_id,),
        ).fetchone()
        since = time.time() - window_s
        w = self._x(
            "SELECT SUM(tokens_out) t, MIN(started_at) a, MAX(finished_at) b FROM items WHERE job_id=? AND finished_at>=?",
            (job_id, since),
        ).fetchone()
        span = (w["b"] - w["a"]) if w["a"] and w["b"] else 0
        return {
            "finished": r["n"],
            "avg_tokens_out": r["avg_out"],
            "avg_tokens_in": r["avg_in"],
            "recent_tps": (w["t"] or 0) / span if span > 0 else None,
        }


# ---------------------------------------------------------------- jobs


def job_schema(task: Task) -> Schema:
    b = task.batch
    return parse_schema(b.get("schema") or TEXT_SCHEMA, default_grounded=b.get("default_grounded", False))


def expand(paths: list[str], glob: str, limit: Optional[int]) -> list[Path]:
    patterns = [g.strip() for g in (glob or DEFAULT_GLOB).split(",") if g.strip()]
    seen, out = set(), []

    def add(p: Path):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file():
            add(p)
        elif p.is_dir():
            matches = set()
            for pat in patterns:
                for m in p.glob(pat):
                    rel = m.relative_to(p)
                    if m.is_file() and not any(part.startswith(".") for part in rel.parts):
                        matches.add(m)
            for m in sorted(matches):
                add(m)
        else:
            raise FileNotFoundError(f"no such file or folder: {raw}")
    return out[:limit] if limit else out


def create(
    cfg: Config,
    store: TaskStore,
    bstore: BatchStore,
    instruction: str,
    paths: list[str],
    *,
    pace: str = "auto",
    schema: Optional[dict] = None,
    schema_path: Optional[str] = None,
    recipe: Optional[str] = None,
    glob: Optional[str] = None,
    out: Optional[str] = None,
    limit: Optional[int] = None,
    ocr: bool = False,
    default_grounded: bool = False,
    checks: Optional[list] = None,
    options: Optional[dict] = None,
    boost_when_idle: bool = False,
    max_output_tokens: Optional[int] = None,
    dry_run: bool = False,
) -> tuple[Optional[Task], dict]:
    """Expand, read and hash every file, then create the job (unless dry_run). Returns (task, plan)."""
    if pace not in PACES:
        raise ValueError(f"pace must be one of {', '.join(PACES)}")
    if not instruction.strip():
        raise ValueError("an instruction (or a recipe) is required")
    parsed = parse_schema(schema or TEXT_SCHEMA, default_grounded)  # fails early on a bad schema
    files = expand(paths, glob or DEFAULT_GLOB, limit)
    max_bytes = cfg.batch.max_file_kb * 1024
    rows = []
    for i, f in enumerate(files):
        row = {"idx": i, "path": str(f), "content_hash": sha256(f), "status": "pending"}
        text, why = read_text(f, max_bytes, ocr)
        if why:
            row.update(status="skipped", reason=why)
        elif f.suffix.lower() not in (".md", ".txt", ".markdown") or ocr:
            row["text"] = text  # cache recognized text so the run does not redo OCR
        row["chars"] = len(text) if text else 0
        rows.append(row)
    plan = plan_line(cfg, instruction, parsed, rows, pace)
    if dry_run:
        plan["first"] = [r["path"] + (f"  (skip: {r['reason']})" if r.get("reason") else "") for r in rows[:5]]
        return None, plan
    if not rows:
        raise ValueError("no files matched")
    task = store.create_batch(
        instruction,
        pace,
        out,
        {
            "instruction": instruction,
            "recipe": recipe,
            "schema": schema or TEXT_SCHEMA,
            "schema_path": schema_path,
            "roots": [str(Path(p).expanduser()) for p in paths],
            "glob": glob or DEFAULT_GLOB,
            "ocr": ocr,
            "default_grounded": default_grounded,
            "checks": checks or [],
            "options": options or {},
            "boost_when_idle": boost_when_idle,
            "max_output_tokens": max_output_tokens,
            "counts": {},
            "pace_note": "",
        },
    )
    bstore.add(task.id, rows)
    task.batch["counts"] = bstore.counts(task.id)
    store.save(task)
    return task, plan


def slowdown_at(cfg: Config, rate: Optional[float]) -> Optional[float]:
    """Measured slowdown of memory-heavy CPU work at a token rate, from calibration."""
    cal = load_json(cfg.state_dir / "calibration.json", cfg.model.main)
    if not cal:
        return None
    pts = sorted((c["tps"], c["slow_triad"]) for c in cal.get("rate_cells", []) if c["mode"] == "ahead1")
    if not pts:
        return None
    if rate is None:
        return pts[-1][1]
    prev = (0.0, 0.0)
    for tps, slow in pts:
        if rate <= tps:
            return prev[1] + (slow - prev[1]) * (rate - prev[0]) / (tps - prev[0])
        prev = (tps, slow)
    return pts[-1][1]


def fmt_duration(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def estimate_seconds(cfg: Config, pace: str, items: int, avg_in: float, avg_out: float, watts: Optional[float] = -1) -> float:
    """Seconds for `items` files at a pace: prompt reading is limited by the pace's power budget,
    generation by its token rate and power budget (see TaskRunner._generate)."""
    from .power import energy

    e = energy(cfg)
    rate = effective_rate(cfg, pace)
    if watts == -1:
        watts = getattr(cfg.pace, f"{pace}_watts", None) if pace != "full" else None
    read = max(avg_in / 300.0, e["prefill_j_per_token"] * avg_in / watts if watts else 0.0)
    gen = avg_out * max(1.0 / rate if rate else 1.0 / 30.0, e["decode_j_per_token"] / watts if watts else 0.0)
    return items * (read + gen + (cfg.batch.item_rest_s if rate else 0.0))


def plan_line(cfg: Config, instruction: str, schema: Schema, rows: list[dict], pace: str) -> dict:
    todo = [r for r in rows if r["status"] == "pending"]
    overhead = (len(PREAMBLE) + len(instruction) + len(describe(schema))) / 4
    avg_in = overhead + (sum(r["chars"] for r in todo) / len(todo) / 4 if todo else 0)
    skipped = len(rows) - len(todo)
    tail = f" · {skipped} skipped" if skipped else ""
    if pace == "auto":
        fast = estimate_seconds(cfg, "full", len(todo), avg_in, GUESS_OUT_TOKENS)
        slow = estimate_seconds(cfg, "gentle", len(todo), avg_in, GUESS_OUT_TOKENS, watts=cfg.background.busy_watts)
        text = (f"{len(todo)} items · about {fmt_duration(fast)} if you leave the Mac alone, up to {fmt_duration(slow)} "
                f"while you're busy on it · auto: full speed when you're away, gentle when your apps are busy{tail}")
        return {"items": len(todo), "skipped": skipped, "seconds": fast, "seconds_busy": slow, "rate": None,
                "slowdown": None, "text": text}
    secs = estimate_seconds(cfg, pace, len(todo), avg_in, GUESS_OUT_TOKENS)
    rate = effective_rate(cfg, pace)
    slow = slowdown_at(cfg, rate)
    if pace == "full":
        impact = "uncapped: up to 45% slower memory-heavy work while it runs"
        if system.power().on_battery:
            impact += " (on battery it runs at steady instead)"
    elif slow is not None:
        impact = f"slows memory-heavy apps by about {slow:.0%}"
    else:
        impact = "not calibrated on this Mac yet (run `mack calibrate`)"
    text = (
        f"{len(todo)} items · about {fmt_duration(secs)} at {pace}"
        + (f" ({rate:.1f} tok/s)" if rate else "")
        + f" · {impact}{tail}"
    )
    return {"items": len(todo), "skipped": skipped, "seconds": secs, "rate": rate, "slowdown": slow, "text": text}


def effective_rate(cfg: Config, pace: str) -> Optional[float]:
    override = getattr(cfg.pace, f"{pace}_tps", None) if pace != "full" else None
    return override or pace_rate(cfg, pace)


def output_path(store: TaskStore, task: Task) -> Path:
    return Path(task.batch.get("output_path") or (store.output_root / f"{task.id}.jsonl"))


def line_for(row) -> dict:
    return {
        "path": row["path"],
        "content_hash": row["content_hash"],
        "status": row["status"],
        "output": json.loads(row["output"]) if row["output"] else None,
        "flags": json.loads(row["flags"]) if row["flags"] else [],
        "reason": row["reason"] or row["error"],
        "tokens_in": row["tokens_in"],
        "tokens_out": row["tokens_out"],
        "duration_s": round(row["finished_at"] - row["started_at"], 2) if row["finished_at"] and row["started_at"] else None,
    }


def rebuild_output(store: TaskStore, bstore: BatchStore, task: Task) -> Path:
    path = output_path(store, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row in bstore.items(task.id):
            if row["status"] in FINAL:
                f.write(json.dumps(line_for(row)) + "\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- runner


class EngineFailure(RuntimeError):
    pass


class BatchRunner(TaskRunner):
    """Runs a batch job item by item. Same pacing, checks and control files as tasks."""

    def __init__(self, engine, cfg: Config, store: TaskStore, task: Task, bstore: BatchStore, governor=None, clock=None):
        super().__init__(engine, cfg, store, task, clock=clock, governor=governor)
        self.bstore = bstore
        self.schema = job_schema(task)
        self.base_rate = effective_rate(cfg, task.pace)
        self.out_path = output_path(store, task)

    def steps(self) -> Iterator[float]:
        t, b = self.task, self.task.batch
        previous_cap = self.governor.set_job_cap(self.base_rate) if self.governor else None
        failures_in_row = 0
        current = None
        try:
            self._check()
            t.status, t.error, t.stage = "running", "", "processing"
            t.started = t.started or time.time()
            self.bstore.recover(t.id)
            rebuild_output(self.store, self.bstore, t)
            self._refresh()
            while True:
                row = self.bstore.next_pending(t.id)
                if row is None:
                    break
                yield from self._held()
                self._check()
                current = row["idx"]
                t.current = current
                b["current_path"] = row["path"]
                try:
                    yield from self._process(row)
                    failures_in_row = 0
                except Abandon:
                    self._put_back(current)
                    current = None
                    yield from self._free_memory()
                    continue
                except EngineFailure as e:
                    failures_in_row += 1
                    self._finish(row, "failed", error=f"engine: {e}")
                    if failures_in_row >= MAX_ENGINE_FAILURES:
                        self.store.set_control(t.id, "pause")
                        t.error = f"paused after {MAX_ENGINE_FAILURES} engine failures in a row; last: {e}"
                        raise Stop("paused")
                current = None
                self._refresh()
                if self.rate and self.cfg.batch.item_rest_s:
                    yield self.cfg.batch.item_rest_s
            t.status, t.stage, t.finished, t.current = "done", "finished", time.time(), -1
            b["current_path"] = ""
            t.note(f"finished: {self._summary_counts()}")
        except Stop as e:
            reason = str(e)
            t.status = "cancelled" if reason == "cancelled" else "paused"
            if current is not None:
                self._put_back(current)
            t.note(reason)
        except Exception as e:
            t.status, t.error = "failed", f"{type(e).__name__}: {e}"
            t.note(t.error)
        finally:
            if self.governor:
                self.governor.restore_job_cap(previous_cap)
            self._refresh()
            self.store.write_live(t.id, "")

    def _put_back(self, idx: int) -> None:
        """An interruption that is not the item's fault: back to pending, attempt not counted."""
        row = self.bstore.item(self.task.id, idx)
        if row and row["status"] == "running":
            self.bstore.update(self.task.id, idx, status="pending", attempts=max(0, row["attempts"] - 1))

    def _refresh(self) -> None:
        self.task.batch["counts"] = self.bstore.counts(self.task.id)
        self._save()

    def _summary_counts(self) -> str:
        c = self.task.batch["counts"]
        return f"{c['done']} done, {c['flagged']} flagged, {c['failed']} failed, {c['skipped']} skipped"

    def _finish(self, row, status: str, *, output=None, flags=None, reason=None, error=None, tokens=(None, None)) -> None:
        self.bstore.update(
            self.task.id, row["idx"],
            status=status, output=json.dumps(output) if output is not None else None,
            flags=json.dumps(flags) if flags else None, reason=reason, error=error,
            finished_at=time.time(), tokens_in=tokens[0], tokens_out=tokens[1],
        )
        final = self.bstore.item(self.task.id, row["idx"])
        with open(self.out_path, "a") as f:
            f.write(json.dumps(line_for(final)) + "\n")

    def _process(self, row) -> Iterator[float]:
        t = self.task
        path = Path(row["path"])
        if row["attempts"] >= 2:
            self._finish(row, "failed", error="gave up after 2 attempts")
            return
        if not path.exists():
            self._finish(row, "skipped", reason="missing: the file is gone")
            return
        if sha256(path) != row["content_hash"]:
            self._finish(row, "skipped", reason="changed_since_enqueue")
            return
        text = row["text"]
        if text is None:
            text, why = read_text(path, self.cfg.batch.max_file_kb * 1024, t.batch.get("ocr", False))
            if why:
                self._finish(row, "skipped", reason=why)
                return
        self.bstore.update(t.id, row["idx"], status="running", attempts=row["attempts"] + 1, started_at=time.time())

        shown, truncated = self._fit(text)
        user = f"{t.batch['instruction'].strip()}\n\n{describe(self.schema)}\n\nFile: {path.name}\n<<<FILE\n{shown}\nFILE>>>"
        tokens_in = tokens_out = 0
        out, error = None, None
        for attempt in range(2):
            prompt = user if attempt == 0 else user + f"\n\nYour previous reply was rejected: {error}. Reply again with only the corrected JSON object."
            try:
                reply = yield from self._generate(PREAMBLE, prompt, t.batch.get("max_output_tokens") or self.cfg.batch.max_output_tokens, temp=0.0,
                                                  live_prefix=f"{path.name}\n\n")
            except (Stop, Abandon):
                raise  # pauses and memory releases are not engine failures
            except Exception as e:
                raise EngineFailure(f"{type(e).__name__}: {e}") from e
            tokens_in += self.last_tokens_in
            tokens_out += self.last_tokens_out
            try:
                obj = extract_json(reply)
            except ValueError as e:
                error, kind = str(e), "invalid_json"
                continue
            try:
                out = validate(self.schema, obj)
                break
            except ValueError as e:
                error, kind = str(e), "schema"
        if out is None:
            self._finish(row, "failed", reason=f"{kind}: {error}", tokens=(tokens_in, tokens_out))
            return
        flags = ground(self.schema, out, shown)
        for check in t.batch.get("checks", []):
            if check.get("kind") == "retention" and out.get(check["field"]):
                kept = retention(shown, out[check["field"]])
                if kept < check.get("min", 0.85):
                    flags.append(check["field"])
                    out[f"{check['field']}_retention"] = round(kept, 3)
        if truncated:
            out["_truncated"] = True
        self._finish(row, "flagged" if flags else "done", output=out, flags=flags, tokens=(tokens_in, tokens_out))

    def _fit(self, text: str) -> tuple[str, bool]:
        limit = self.cfg.batch.max_input_tokens
        n = self.engine.count(text)
        if n <= limit:
            return text, False
        return clip_middle(text, int(len(text) * limit / n * 0.95), "\n[… middle of the file left out …]\n"), True


# ---------------------------------------------------------------- review


def write_review(store: TaskStore, bstore: BatchStore, task: Task, path: Optional[Path] = None) -> Path:
    path = path or store.output_root / f"{task.id}.review.md"
    rows = bstore.items(task.id)
    c = bstore.counts(task.id)
    lines = [
        f"# Review: {task.batch.get('recipe') or 'job'} {task.id}",
        "",
        f"Instruction: {task.batch['instruction'].strip()}",
        "",
        f"{c['total']} items: {c['done']} done, {c['flagged']} flagged, {c['failed']} failed, {c['skipped']} skipped, {c['pending']} not yet run.",
        "",
        "Flagged items had a value the file did not support; that value was removed. Nothing is written to your files until `mack apply`.",
        "",
    ]
    attention = [r for r in rows if r["status"] in ("flagged", "failed")]
    if attention:
        lines += ["## Needs attention", ""]
        for r in attention:
            lines += [f"### {r['path']}", "", f"Status: **{r['status']}**"]
            if r["flags"]:
                lines.append(f"Flags: {', '.join(json.loads(r['flags']))}")
            if r["reason"] or r["error"]:
                lines.append(f"Reason: {r['reason'] or r['error']}")
            if r["output"]:
                lines += ["", "```json", json.dumps(json.loads(r["output"]), indent=2, ensure_ascii=False), "```"]
            lines += ["", "Start of the file:", "", "> " + _head(r).replace("\n", "\n> "), ""]
    done = [r for r in rows if r["status"] == "done"]
    if done:
        lines += ["## Done", "", "| file | output |", "|---|---|"]
        for r in done:
            out = json.dumps(json.loads(r["output"]), ensure_ascii=False).replace("|", "\\|")
            lines.append(f"| {Path(r['path']).name} | {out} |")
        lines.append("")
    skipped = [r for r in rows if r["status"] == "skipped"]
    if skipped:
        lines += ["## Skipped", ""] + [f"- {r['path']}: {r['reason']}" for r in skipped] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def _head(row) -> str:
    if row["text"]:
        return row["text"][:300]
    try:
        return Path(row["path"]).read_text(errors="replace")[:300]
    except OSError:
        return "(file not readable)"
