"""Background tasks: long jobs done slowly and methodically.

A task is something big — a many-page report, a small codebase — that Mack works
through on its own while the user does other things. It is built for low impact,
not speed:

  * **Paced.** Tokens are generated at a fixed low rate taken from this machine's
    calibration (how much each token rate slows the user's CPU work), with the
    synchronous decode loop, which spends less energy per token than the pipelined
    one (odi2027 S7: 0.131 vs 0.184 J/token). Prompt prefill runs in chunks with
    rests between them, so even the bursts are spread out. Work pauses on a low
    battery.
  * **Methodical.** Plan first, then one unit at a time (a report section or a
    source file), each written in a fresh context holding the brief, the plan
    and a digest of finished units — never an ever-growing transcript. Sections
    get a self-review and at most one revision; code gets a syntax check and a
    repair attempt. Generated code is never executed.
  * **Durable.** Everything is saved after every unit under state/tasks/<id>/,
    so a task survives quitting Mack or rebooting and resumes where it left off.
  * **Polite.** The runner is a generator that yields after every token and
    chunk. The engine thread serves interactive requests between yields, so a
    background task never blocks them for more than a moment.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from mlx_lm.sample_utils import make_sampler

from .config import ROOT, Config
from .engine import Decision, GenStats, strip_thinking
from .governor import load_json, rate_for_target, ruler_cells
from .pins import extract_pins, select_pins
from .verify import verify_summary
from . import system

WORDS_PER_PAGE = 500
MAX_SECTION_WORDS = 1000
MAX_FILES = 14

# pace -> (CPU-slowdown target on the triad ruler, rate if uncalibrated, prefill duty)
PACES = {
    "auto": (None, None, 1.0),  # adapts to what you are doing; see pacing.py
    "gentle": (0.05, 3.0, 0.25),
    "balanced": (0.10, 5.0, 0.40),
    "steady": (0.20, 10.0, 0.60),
    "full": (None, None, 1.0),
}
MIN_TASK_TPS = 2.0
BATTERY_PAUSE_BELOW = 25  # percent, while on battery power

CODE_HINTS = re.compile(
    r"\b(code ?base|app|application|script|library|package|api|cli|program|website|web site|server|"
    r"implement|python|javascript|typescript|rust|swift|java|golang|react|flask|fastapi|django|game)\b",
    re.I,
)
SAFE_PATH = re.compile(r"^[A-Za-z0-9_.\-/]+$")


# ---------------------------------------------------------------- data


@dataclass
class Unit:
    index: int
    kind: str  # section | file
    title: str  # section heading or file path
    spec: str  # what it should contain
    target_words: int = 0
    interfaces: str = ""  # files: what it must define
    status: str = "pending"  # pending | writing | done | failed
    words: int = 0
    tokens: int = 0
    digest: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str
    prompt: str
    kind: str  # report | code
    pages: int
    pace: str
    out_dir: str
    title: str = ""
    stack: str = ""
    status: str = "queued"  # queued | running | paused | done | failed | cancelled
    stage: str = "waiting"  # waiting | planning | writing | reviewing | assembling | finished
    current: int = -1  # unit being worked on
    units: list[Unit] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)  # exact quotes worth keeping across units
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    tokens: int = 0
    generating_s: float = 0.0
    error: str = ""
    log: list[str] = field(default_factory=list)
    batch: dict = field(default_factory=dict)  # kind == "batch": job settings and item counts (items live in batch.db)

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if self.kind == "batch":
            c = self.batch.get("counts") or {}
            total = c.get("total", 0)
            return round((total - c.get("pending", total)) / total, 3) if total else 0.0
        if not self.units:
            return 0.0
        done = sum(1 for u in self.units if u.status == "done")
        return round(0.05 + 0.9 * done / len(self.units), 3)

    def summary(self) -> dict:
        unit = self.units[self.current] if 0 <= self.current < len(self.units) else None
        if self.kind == "batch":
            c = self.batch.get("counts") or {}
            return {
                **self._common(),
                "units_done": c.get("total", 0) - c.get("pending", 0),
                "units_total": c.get("total", 0),
                "current": self.batch.get("current_path", ""),
                "counts": c,
                "recipe": self.batch.get("recipe"),
                "pace_note": self.batch.get("pace_note", ""),
                "output_path": self.batch.get("output_path", ""),
            }
        return {**self._common(),
            "units_done": sum(1 for u in self.units if u.status == "done"),
            "units_total": len(self.units),
            "current": unit.title if unit else "",
        }

    def _common(self) -> dict:
        return {
            "id": self.id,
            "title": self.title or self.prompt[:80],
            "prompt": self.prompt,
            "kind": self.kind,
            "pages": self.pages,
            "pace": self.pace,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "out_dir": self.out_dir,
            "pace_note": self.batch.get("pace_note", ""),
            "tokens": self.tokens,
            "tps": round(self.tokens / self.generating_s, 2) if self.generating_s else 0.0,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
        }

    def note(self, text: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {text}")
        del self.log[:-200]

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["units"] = [Unit(**u) for u in d.get("units", [])]
        return cls(**d)


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "task"


def guess_kind(prompt: str) -> str:
    return "code" if CODE_HINTS.search(prompt) else "report"


# ---------------------------------------------------------------- store


class TaskStore:
    """Tasks live as files, so the CLI, the server and the menu-bar app can all see
    them without talking to each other. Only the runner writes task.json; everyone
    else asks for pause/resume/cancel through a separate control file."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.state_dir / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.output_root = cfg.output_dir

    def path(self, task_id: str) -> Path:
        return self.dir / task_id

    def create(self, prompt: str, kind: str = "auto", pages: int = 5, pace: str = "auto", out_dir: Optional[str] = None) -> Task:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("the task needs a description")
        if pace not in PACES:
            raise ValueError(f"pace must be one of {', '.join(PACES)}")
        kind = guess_kind(prompt) if kind in ("auto", "", None) else kind
        if kind not in ("report", "code"):
            raise ValueError("kind must be report, code or auto")
        task_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        out = Path(out_dir).expanduser() if out_dir else self.output_root / f"{task_id[:15]}-{slugify(prompt)}"
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"{out} already exists and is not empty; Mack only writes into new folders")
        task = Task(id=task_id, prompt=prompt, kind=kind, pages=max(1, min(int(pages), 60)), pace=pace, out_dir=str(out))
        self.path(task_id).mkdir(parents=True)
        self.save(task)
        return task

    def create_batch(self, instruction: str, pace: str, out_path: Optional[str], batch: dict) -> Task:
        task_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        out = Path(out_path).expanduser() if out_path else self.output_root / f"{task_id}.jsonl"
        if out.exists() and out.stat().st_size:
            raise ValueError(f"{out} already exists; choose another --out")
        out.parent.mkdir(parents=True, exist_ok=True)
        batch = {**batch, "output_path": str(out)}
        title = (batch.get("recipe") + ": " if batch.get("recipe") else "") + instruction.strip().splitlines()[0][:70]
        task = Task(id=task_id, prompt=instruction.strip(), kind="batch", pages=0, pace=pace,
                    out_dir=str(out.parent), title=title, batch=batch)
        self.path(task_id).mkdir(parents=True)
        self.save(task)
        return task

    def save(self, task: Task) -> None:
        p = self.path(task.id)
        tmp = p / "task.json.tmp"
        tmp.write_text(json.dumps(asdict(task), indent=1))
        tmp.replace(p / "task.json")

    def load(self, task_id: str) -> Task:
        return Task.from_dict(json.loads((self.path(task_id) / "task.json").read_text()))

    def all(self) -> list[Task]:
        tasks = []
        for p in sorted(self.dir.glob("*/task.json")):
            try:
                tasks.append(Task.from_dict(json.loads(p.read_text())))
            except Exception:
                continue
        # ids only have one-second resolution; creation time decides the order
        return sorted(tasks, key=lambda t: t.created)

    def next_runnable(self, cfg: Optional[Config] = None) -> Optional[Task]:
        for t in self.all():  # oldest first
            control = self.control(t.id)
            if control == "battery" and battery_ok(cfg):
                (self.path(t.id) / "control").unlink(missing_ok=True)  # power is back: carry on by itself
                t.status = "queued"
                t.note("battery recovered: resuming")
                self.save(t)
                control = ""
            if t.status in ("queued", "running") and control == "":
                return t
        return None

    def pause_for_battery(self, task_id: str) -> None:
        (self.path(task_id) / "control").write_text("battery")

    # control: written by anyone, read by the runner
    def control(self, task_id: str) -> str:
        f = self.path(task_id) / "control"
        return f.read_text().strip() if f.exists() else ""

    def set_control(self, task_id: str, action: str) -> None:
        if action not in ("pause", "resume", "cancel"):
            raise ValueError("action must be pause, resume or cancel")
        task = self.load(task_id)
        f = self.path(task_id) / "control"
        if action == "resume":
            f.unlink(missing_ok=True)
            if task.status == "paused":
                task.status = "queued"
                self.save(task)
        else:
            f.write_text(action)
            if task.status in ("queued", "paused") and action == "cancel":
                task.status = "cancelled"
                self.save(task)
            elif task.status == "queued" and action == "pause":
                task.status = "paused"
                self.save(task)

    def live(self, task_id: str) -> str:
        f = self.path(task_id) / "live.txt"
        return f.read_text() if f.exists() else ""

    def write_live(self, task_id: str, text: str) -> None:
        (self.path(task_id) / "live.txt").write_text(text)


# ---------------------------------------------------------------- planning


def parse_report_plan(text: str) -> tuple[str, list[Unit]]:
    title, units = "", []
    for line in strip_thinking(text).splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip().strip('"*')
        elif line.upper().startswith("SECTION:"):
            parts = [p.strip() for p in line.split(":", 1)[1].split("|")]
            if not parts or not parts[0]:
                continue
            words = int(m.group(0)) if len(parts) > 1 and (m := re.search(r"\d+", parts[1])) else 600
            spec = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 and not re.search(r"\d", parts[1]) else "")
            units.append(
                Unit(index=len(units), kind="section", title=parts[0].strip("#* "), spec=spec,
                     target_words=max(250, min(words, MAX_SECTION_WORDS)))
            )
    return title, units


def safe_rel_path(path: str) -> Optional[str]:
    """A plain relative path inside the project, or None. Checked before any
    normalization, so "../x" and "/etc/x" are rejected rather than rewritten."""
    path = path.strip().strip("`'\"")
    if path.startswith("./"):
        path = path[2:]
    if not path or path.startswith(("/", "~")) or ".." in path.split("/") or not SAFE_PATH.match(path):
        return None
    return path


def parse_code_plan(text: str) -> tuple[str, str, list[Unit]]:
    project, stack, units, seen = "", "", [], set()
    for line in strip_thinking(text).splitlines():
        line = line.strip().lstrip("-*• ").strip()
        upper = line.upper()
        if upper.startswith("PROJECT:"):
            project = line.split(":", 1)[1].strip().strip('"*')
        elif upper.startswith("STACK:"):
            stack = line.split(":", 1)[1].strip()
        elif upper.startswith("FILE:"):
            parts = [p.strip() for p in line.split(":", 1)[1].split("|")]
            path = safe_rel_path(parts[0]) if parts else None
            if not path or path in seen:
                continue
            seen.add(path)
            units.append(
                Unit(index=len(units), kind="file", title=path, spec=parts[1] if len(parts) > 1 else "",
                     interfaces=parts[2] if len(parts) > 2 else "")
            )
    return project, stack, units[:MAX_FILES]


def report_plan_prompt(task: Task) -> str:
    words = task.pages * WORDS_PER_PAGE
    lo = max(2, round(words / MAX_SECTION_WORDS))
    hi = max(lo + 1, round(words / 450))
    return f"""Task: {task.prompt}

Target length: about {words} words (about {task.pages} pages).

Plan this report. Reply with the outline only, in exactly this format:

TITLE: <title of the report>
SECTION: <section heading> | <number of words> | <what the section covers, in one sentence>
SECTION: <section heading> | <number of words> | <what the section covers, in one sentence>

Use between {lo} and {hi} sections, each 400 to {MAX_SECTION_WORDS} words, adding up to about {words} words. Give every section a distinct job so nothing is covered twice. Start with an introduction and end with a conclusion."""


def code_plan_prompt(task: Task) -> str:
    return f"""Task: {task.prompt}

Plan a small, complete, working codebase for this. Reply with the plan only, in exactly this format:

PROJECT: <short project name>
STACK: <language and key libraries, one line>
FILE: <relative/path.ext> | <what the file contains, one sentence> | <the functions or classes it defines, comma-separated>
FILE: <relative/path.ext> | <what the file contains, one sentence> | <the functions or classes it defines, comma-separated>

Rules: at most {MAX_FILES} files. List each file after the files it depends on. Include tests if they fit, and make README.md the last file. Prefer the standard library over extra dependencies."""


# ---------------------------------------------------------------- digests


def code_interfaces(path: str, source: str) -> list[str]:
    """Public signatures of a source file, so later files can call it correctly."""
    out = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                out.append(f"def {node.name}({ast.unparse(node.args)})" + _doc(node))
            elif isinstance(node, ast.ClassDef):
                methods = [
                    f"{m.name}({ast.unparse(m.args)})"
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not m.name.startswith("_") or m.name == "__init__")
                ]
                out.append(f"class {node.name}: " + ", ".join(methods) + _doc(node))
        return out
    for line in source.splitlines():
        if re.match(r"^\s*(export\s+)?(default\s+)?(async\s+)?(function|class|interface|type|const|struct|fn|pub fn|func)\b", line):
            out.append(line.strip().rstrip("{").strip())
    return out[:20]


def _doc(node) -> str:
    doc = ast.get_docstring(node)
    return f"  # {doc.splitlines()[0][:80]}" if doc else ""


def extract_code(text: str) -> str:
    text = strip_thinking(text)
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).rstrip() + "\n"
    if "```" in text:  # opened but never closed (cut off)
        return text.split("```", 1)[1].split("\n", 1)[-1].rstrip() + "\n"
    return text.strip() + "\n"


def syntax_error(path: str, source: str) -> Optional[str]:
    """Static checks only. Generated code is never run."""
    try:
        if path.endswith(".py"):
            ast.parse(source)
        elif path.endswith(".json"):
            json.loads(source)
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    except ValueError as e:
        return str(e)
    return None


def missing_names(path: str, source: str, interfaces: str) -> list[str]:
    if not path.endswith(".py") or not interfaces:
        return []
    wanted = [re.split(r"[\s(]", w.strip())[0] for w in interfaces.split(",") if w.strip()]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return [w for w in wanted if re.match(r"^[A-Za-z_]\w*$", w) and w not in defined]


def repeating_tail(text: str) -> Optional[int]:
    """If generation has started looping, the index where the loop begins.

    A real loop repeats the same stretch back to back; a transcript or a note that
    says the same sentence twice does not. So the last 200 characters must already
    have appeared at least twice before (three times in all), or the last four
    substantial lines must be identical."""
    if len(text) < 600:
        return None
    tail = text[-200:]
    seen, start = [], 0
    while (i := text.find(tail, start)) != -1 and i < len(text) - 200:
        seen.append(i)
        start = i + 1
    if len(seen) >= 2:
        return seen[0] + 200
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 20]
    if len(lines) >= 4 and len(set(lines[-4:])) == 1:
        return text.rfind(lines[-1], 0, text.rfind(lines[-1]))
    return None


def battery_low(cfg: Optional[Config] = None) -> bool:
    below = cfg.batch.battery_pause_below if cfg else BATTERY_PAUSE_BELOW
    p = system.power()
    return p.on_battery and p.percent is not None and p.percent < below


def battery_ok(cfg: Optional[Config] = None) -> bool:
    """Safe to resume: on mains power, or charged back above the resume level."""
    resume = cfg.batch.battery_resume_at if cfg else BATTERY_PAUSE_BELOW + 5
    p = system.power()
    return not p.on_battery or p.percent is None or p.percent >= resume


def pace_rate(cfg: Config, pace: str) -> Optional[float]:
    target, fallback, _ = PACES[pace]
    if target is None:
        return None
    cal = load_json(cfg.state_dir / "calibration.json", cfg.model.main)
    if not cal:
        return fallback
    cap, _ = rate_for_target(ruler_cells(cal, "triad"), target, MIN_TASK_TPS)
    return cap  # None would mean even full speed is within the target


class _Sync:
    def decide(self) -> Decision:
        return Decision(mode="sync")


# ---------------------------------------------------------------- runner


class Stop(Exception):
    """Raised inside the runner when the task is paused or cancelled."""


class Abandon(Exception):
    """Memory is critical: drop the piece in progress so the model can be unloaded."""


class TaskRunner:
    """Drives one task. `steps()` is a generator yielding how long to wait (seconds)
    before calling it again; 0 means continue whenever convenient."""

    def __init__(self, engine, cfg: Config, store: TaskStore, task: Task, clock=None, governor=None):
        from .pacing import Pacer

        self.engine, self.cfg, self.store, self.task = engine, cfg, store, task
        self.governor = governor
        self.out = Path(task.out_dir)
        self.clock = clock or time.monotonic  # injectable so tests can run hours of pacing instantly
        self.pacer = Pacer(cfg, task.pace, governor, self.clock, boost_when_idle=bool(task.batch.get("boost_when_idle")))
        self.rate = self.pacer.state.rate
        self._next_battery_check = -1e18
        self.last_tokens_in = self.last_tokens_out = 0

    def pace(self):
        """The pace right now (rate, chunk size, burst duty), recording any change in the task."""
        st = self.pacer.current()
        self.rate = st.rate
        if self.governor is not None:
            self.governor.job_cap = st.rate or None
        if st.note != self.task.batch.get("pace_note", ""):
            self.task.batch["pace_note"] = st.note
            if st.note:
                self.task.note(st.note)
        return st

    def current_rate(self) -> Optional[float]:
        return self.pace().rate

    def _held(self) -> Iterator[float]:
        """Wait while the pacer says hold, or while there is no room to load the model.
        Use as `state = yield from self._held()`."""
        while True:
            st = self.pace()
            if st.rate == 0.0:
                self._check()
                if st.unload and hasattr(self.engine, "unload"):
                    raise Abandon()
                yield 1.0
                continue
            if not self._room_to_load():
                self._check()
                yield 10.0
                continue
            return st

    def _room_to_load(self) -> bool:
        """Loading ~3 GB of weights onto a Mac that is short of memory pushes your apps into
        swap. Only load when macOS will still have `load_headroom_gb` available afterwards."""
        if getattr(self.engine, "loaded", True):
            return True
        need = self.engine.weights_gb + self.cfg.background.load_headroom_gb
        have = system.available_memory_gb()
        ok = have >= need and system.memory_pressure() < system.MEMORY_CRITICAL
        note = "" if ok else f"waiting for memory to load the model ({have:.1f} GB available, needs {need:.1f})"
        if note != self.task.batch.get("pace_note", "") and (note or self.task.batch.get("pace_note", "").startswith("waiting for memory")):
            self.task.batch["pace_note"] = note
            if note:
                self.task.note(note)
            self._save()
        return ok

    def _free_memory(self) -> Iterator[float]:
        """After Abandon: unload the model and wait until macOS memory pressure eases.

        Waits at least a minute, doubling on each repeat (up to 10 minutes), so that a
        model whose own loading tips the Mac into critical pressure does not thrash
        between loading and unloading."""
        self.engine.unload()
        self._memory_events = getattr(self, "_memory_events", 0) + 1
        wait = min(60.0 * 2 ** (self._memory_events - 1), 600.0)
        self.task.note(f"freed the model's memory: macOS reported critical memory pressure; retrying in {wait:.0f} s or later")
        self._save()
        until = self.clock() + wait
        while self.clock() < until or self.pace().unload:
            self._check()
            yield 5.0

    # ---- control ---------------------------------------------------------

    def _check(self) -> None:
        action = self.store.control(self.task.id)
        if action == "cancel":
            raise Stop("cancelled")
        if action == "pause":
            raise Stop("paused")
        now = self.clock()
        if now >= self._next_battery_check:
            self._next_battery_check = now + 60
            if battery_low(self.cfg):
                self.store.pause_for_battery(self.task.id)
                self.task.note(f"paused: on battery below {self.cfg.batch.battery_pause_below}%; resumes at {self.cfg.batch.battery_resume_at}% or on power")
                raise Stop("paused")

    def _save(self) -> None:
        self.store.save(self.task)

    # ---- generation --------------------------------------------------------

    def _generate(self, system: str, user: str, max_tokens: int, temp: float, live_prefix: str = "") -> Iterator[float]:
        """Paced generation in a fresh context. Use as `text = yield from self._generate(...)`."""
        yield from self._held()  # before touching the engine: it may need loading
        fmt = self.engine.fmt
        prompt = self.engine.encode(fmt.system(system) + fmt.user(user) + fmt.generation_prompt())
        ctx = self.engine.new_context()
        from .power import energy

        joules = energy(self.cfg)
        e_pre, e_dec = joules["prefill_j_per_token"], joules["decode_j_per_token"]
        clock = self.clock
        self.last_tokens_in, self.last_tokens_out = len(prompt), 0
        # Prompt reading and generation spend one power budget (`watts`): measured on
        # this Mac, reading draws ~18 W flat out but barely slows memory-heavy apps,
        # while generating draws ~7 W and is what slows them, which the token rate caps.
        # Chunks are short (128 tokens) while you are at the Mac, so no single
        # full-power burst lasts long, and never busier than the pace's duty.
        i, end = 0, len(prompt) - 1
        while i < end:
            self._check()
            st = yield from self._held()
            chunk = prompt[i : min(i + st.chunk, end)]
            t0 = clock()
            self.engine.prefill(ctx, chunk)
            busy = clock() - t0
            i += len(chunk)
            budget = max(busy / st.duty, e_pre * len(chunk) / st.watts if st.watts else 0.0)
            yield max(0.0, budget - busy)
        stats = GenStats()
        detok = self.engine.detok()
        sampler = make_sampler(temp=temp, top_p=0.9 if temp > 0 else 0.0)
        text = ""
        next_t = clock()
        t_gen = clock()
        stopped = None
        looped = False
        gen = self.engine.generate(ctx, prompt[-1:], max_tokens, policy=_Sync(), sampler=sampler, stats=stats)
        try:
            for n, tok in enumerate(gen):
                text += detok.add(tok)
                self.task.tokens += 1
                self.last_tokens_out += 1
                if n % 16 == 0:
                    self.store.write_live(self.task.id, live_prefix + text)
                    if (cut := repeating_tail(text)) is not None:
                        text = text[:cut]
                        looped = True
                        self.task.note("stopped a repeating loop in the output")
                        break
                    try:
                        self._check()
                    except Stop as e:
                        stopped = e
                        break
                st = yield from self._held()
                gap = max(1.0 / st.rate if st.rate else 0.0, e_dec / st.watts if st.watts else 0.0)
                if gap:
                    next_t = max(next_t + gap, clock() - 1.0)
                    yield max(0.0, next_t - clock())
                else:
                    next_t = clock()
                    yield 0.0
        finally:
            gen.close()
            self.task.generating_s += clock() - t_gen
        text += detok.flush()
        self.store.write_live(self.task.id, live_prefix + text)
        if stopped:
            raise stopped
        self.finish = "stop" if looped else (stats.finish or "stop")
        return strip_thinking(text).strip()

    # ---- task --------------------------------------------------------------

    def steps(self) -> Iterator[float]:
        t = self.task
        try:
            self._check()
            if not t.started:
                t.started = time.time()
            t.status, t.error = "running", ""
            self._save()
            if not t.units:
                yield from self._plan()
            self.out.mkdir(parents=True, exist_ok=True)
            for unit in t.units:
                if unit.status == "done":
                    continue
                t.current = unit.index
                while True:
                    unit.status = "writing"
                    t.stage = "writing"
                    self._save()
                    try:
                        if unit.kind == "section":
                            yield from self._write_section(unit)
                        else:
                            yield from self._write_file(unit)
                        break
                    except Abandon:
                        unit.status = "pending"
                        yield from self._free_memory()
                unit.status = "done"
                self._save()
            t.stage, t.current = "assembling", -1
            self._save()
            yield from self._assemble()
            t.status, t.stage, t.finished = "done", "finished", time.time()
            t.note(f"finished: {t.tokens} tokens in {t.generating_s / 60:.0f} minutes of generation")
            self.store.write_live(t.id, "")
        except Stop as e:
            t.status = "cancelled" if str(e) == "cancelled" else "paused"
            for u in t.units:
                if u.status == "writing":
                    u.status = "pending"  # restarts cleanly on resume
            t.note(str(e))
        except Exception as e:
            t.status, t.error = "failed", f"{type(e).__name__}: {e}"
            t.note(t.error)
        finally:
            self._save()

    def _plan(self) -> Iterator[float]:
        t = self.task
        t.stage = "planning"
        self._save()
        system = "You are Mack, working methodically through a long task in the background. You plan carefully before writing."
        for attempt in range(3):
            prompt = report_plan_prompt(t) if t.kind == "report" else code_plan_prompt(t)
            text = yield from self._generate(system, prompt, 900, temp=0.0 if attempt else 0.3, live_prefix="Planning…\n\n")
            if t.kind == "report":
                title, units = parse_report_plan(text)
                ok = len(units) >= 2
            else:
                title, t.stack, units = parse_code_plan(text)
                ok = len(units) >= 1
            if ok:
                t.title = title or t.prompt[:80]
                t.units = units
                (Path(self.store.path(t.id)) / "plan.txt").write_text(text)
                t.note(f"planned {len(units)} {'sections' if t.kind == 'report' else 'files'}")
                return
            t.note(f"plan attempt {attempt + 1} could not be parsed; retrying")
        raise RuntimeError("could not produce a usable plan after 3 attempts")

    def _outline(self, current: Unit) -> str:
        lines = []
        for u in self.task.units:
            mark = "✓" if u.status == "done" else ("→" if u is current else " ")
            extra = f" — {u.spec}" if u.spec else ""
            lines.append(f"{mark} {u.index + 1}. {u.title}{extra}")
        return "\n".join(lines)

    def _digest(self, budget_words: int = 900) -> str:
        """What finished units established, newest first in detail, oldest trimmed."""
        done = [u for u in self.task.units if u.status == "done" and u.digest]
        parts, used = [], 0
        for u in reversed(done):
            keep = u.digest if used < budget_words // 2 else u.digest[:2]
            block = f"{u.title}:\n" + "\n".join(f"- {d}" for d in keep)
            used += len(block.split())
            if used > budget_words:
                break
            parts.append(block)
        facts = "\n".join(f'- "{q}"' for q in self.task.facts)
        text = "\n\n".join(reversed(parts))
        return (text + (f"\n\nExact facts to stay consistent with:\n{facts}" if facts else "")).strip()

    # ---- report sections --------------------------------------------------

    def _write_section(self, unit: Unit) -> Iterator[float]:
        t = self.task
        system = (
            "You are Mack, writing one section of a long report in the background. Write clear, specific, "
            "well-organized prose. Never write meta commentary about the task or the outline."
        )
        digest = self._digest()
        last = unit.index == len(t.units) - 1
        brief = f"""Report: {t.title}
Original request: {t.prompt}

Outline (✓ written, → this section):
{self._outline(unit)}

{"What the earlier sections already established:" + chr(10) + digest if digest else "This is the first section."}

Write section {unit.index + 1}, "{unit.title}": {unit.spec or "as planned"}. About {unit.target_words} words.
- Begin with the line "## {unit.title}".
- Cover only this section's job. Do not repeat what earlier sections established, and do not start other sections.
- {"This is the final section, so it may close the whole report." if last else "Do not write a conclusion for the whole report."}
- Use ### subheadings, lists or a table only where they genuinely help."""
        max_tokens = int(unit.target_words * 1.7) + 200
        text = yield from self._generate(system, brief, max_tokens, temp=0.6, live_prefix="")
        if self.finish == "length":
            more = yield from self._generate(
                system,
                brief + f"\n\nThe section so far (it was cut off):\n\n{text[-2500:]}\n\nContinue exactly where it stops and finish the section. Do not repeat anything.",
                max(400, max_tokens // 2), temp=0.6, live_prefix=text,
            )
            text = text + ("" if text.endswith((" ", "\n")) else " ") + more
            unit.notes.append("continued after hitting the length limit")
        if not text.lstrip().startswith("#"):
            text = f"## {unit.title}\n\n{text}"

        t.stage = "reviewing"
        self._save()
        review = yield from self._generate(
            "You are a strict, concise editor.",
            f"""Outline of the report:
{self._outline(unit)}

{"Earlier sections established:" + chr(10) + digest if digest else ""}

Draft of section "{unit.title}" ({unit.spec}):

{text}

Reply with exactly OK unless the draft has a serious problem. Only these count as serious:
- it stops in the middle of a sentence or is visibly unfinished
- it states something that contradicts what the earlier sections established
- most of it covers a different section's topic instead of its own
Style, depth and minor overlap are not serious problems. If there is a serious problem, list each one on its own line starting with "- ", in one sentence each.""",
            220, temp=0.0, live_prefix=text + "\n\n— reviewing —\n",
        )
        problems = [ln.strip()[2:].strip()[:200] for ln in review.splitlines() if ln.strip().startswith("- ")]
        if problems and not review.strip().upper().startswith("OK"):
            unit.notes.append("revised: " + "; ".join(problems[:3]))
            t.stage = "writing"
            self._save()
            revised = yield from self._generate(
                system,
                brief + f"\n\nYour draft:\n\n{text}\n\nAn editor found these problems:\n" + "\n".join(f"- {p}" for p in problems[:6])
                + "\n\nRewrite the section fixing them. Keep everything that was good. Begin with the heading line.",
                max_tokens, temp=0.5, live_prefix="— revising —\n",
            )
            if len(revised.split()) > 0.5 * len(text.split()):  # a revision that lost half the section is worse
                text = revised if revised.lstrip().startswith("#") else f"## {unit.title}\n\n{revised}"
        t.stage = "writing"

        sections = self.out / "sections"
        sections.mkdir(parents=True, exist_ok=True)
        (sections / f"{unit.index + 1:02d}-{slugify(unit.title)}.md").write_text(text.strip() + "\n")
        unit.words = len(text.split())
        yield from self._summarize_unit(unit, text)

    def _summarize_unit(self, unit: Unit, text: str) -> Iterator[float]:
        raw = yield from self._generate(
            "You summarize precisely.",
            f"Summarize what this section establishes as 3 to 6 short factual bullets starting with \"- \". "
            f"Copy any numbers, names and definitions exactly.\n\n{text}",
            220, temp=0.0, live_prefix=text + "\n\n— noting what this section established —\n",
        )
        unit.digest = [v.line for v in verify_summary(raw, text) if v.kept][:6]
        pins = extract_pins([type("T", (), {"idx": unit.index, "role": "user", "text": text})()])
        kept = select_pins(pins, 80, self.engine.count)
        self.task.facts = (self.task.facts + [p.quote for p in kept])[-12:]

    # ---- code files ---------------------------------------------------------

    def _write_file(self, unit: Unit) -> Iterator[float]:
        t = self.task
        digest = self._digest()
        system = (
            "You are Mack, building a codebase one file at a time in the background. Write complete, working, "
            "readable code with no placeholders or TODOs. Output only the file contents in one fenced code block."
        )
        brief = f"""Project: {t.title}
Stack: {t.stack or "choose sensibly"}
Original request: {t.prompt}

Files (✓ written, → this file):
{self._outline(unit)}

{"Interfaces of files already written (use these exact names and signatures):" + chr(10) + digest if digest else "No files have been written yet."}

Write the complete contents of `{unit.title}`: {unit.spec or "as planned"}.{(" It must define: " + unit.interfaces + ".") if unit.interfaces else ""}"""
        text = yield from self._generate(system, brief, 2600, temp=0.2, live_prefix=f"# {unit.title}\n")
        code = extract_code(text)
        error = syntax_error(unit.title, code)
        missing = missing_names(unit.title, code, unit.interfaces)
        if error or missing or self.finish == "length":
            why = f"syntax error at {error}" if error else (f"it does not define {', '.join(missing)}" if missing else "it was cut off")
            unit.notes.append(f"repaired: {why}")
            t.stage = "reviewing"
            self._save()
            fixed = yield from self._generate(
                system,
                brief + f"\n\nYour previous version:\n```\n{code[-6000:]}\n```\nProblem: {why}. Write the corrected, complete file.",
                2600, temp=0.1, live_prefix=f"# {unit.title} (repairing)\n",
            )
            fixed_code = extract_code(fixed)
            if not syntax_error(unit.title, fixed_code) or error:
                code = fixed_code
            t.stage = "writing"
            if (err := syntax_error(unit.title, code)):
                unit.notes.append(f"still has a syntax error at {err}")
        target = (self.out / unit.title).resolve()
        if self.out.resolve() not in target.parents:
            raise RuntimeError(f"refusing to write outside the output folder: {unit.title}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)
        unit.words = len(code.split())
        unit.digest = code_interfaces(unit.title, code) or [f"(no public definitions; {len(code.splitlines())} lines)"]

    # ---- assembly ----------------------------------------------------------

    def _assemble(self) -> Iterator[float]:
        t = self.task
        if t.kind == "report":
            parts = [f"# {t.title}", "", "## Contents", ""]
            parts += [f"{u.index + 1}. {u.title}" for u in t.units]
            parts.append("")
            for f in sorted((self.out / "sections").glob("*.md")):
                parts += [f.read_text().strip(), ""]
            (self.out / "report.md").write_text("\n".join(parts))
        notes = [f"# How Mack made this", "", f"Request: {t.prompt}", "", f"Model: {self.cfg.model.main}. Pace: {t.pace}.", ""]
        for u in t.units:
            notes.append(f"- {u.title}: {u.words} words" + (f" ({'; '.join(u.notes)})" if u.notes else ""))
        if t.kind == "code":
            notes += ["", "Generated code was syntax-checked but never run. Read it and run the tests yourself."]
        (self.out / "MACK_NOTES.md").write_text("\n".join(notes) + "\n")
        yield 0.0


class WorkerLock:
    """Only one process may run background jobs. An exclusive flock on state/worker.lock;
    the kernel drops it when the process exits, even on SIGKILL, so it cannot go stale."""

    def __init__(self, cfg: Config):
        self.path = cfg.state_dir / "worker.lock"
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        import fcntl

        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class TaskScheduler:
    """Picks the oldest runnable task and advances it one small step per tick."""

    def __init__(self, engine, cfg: Config, store: TaskStore, governor=None, clock=None):
        self.engine, self.cfg, self.store = engine, cfg, store
        self.governor, self.clock = governor, clock
        self.runner: Optional[TaskRunner] = None
        self._steps: Optional[Iterator[float]] = None
        self._bstore = None
        self._last_beat = 0.0
        self.lock = WorkerLock(cfg)

    def tick(self) -> Optional[float]:
        """Seconds to wait before the next tick, or None when there is nothing to do."""
        if not self.lock.acquire():
            return None  # another process is the worker; it will run the queue
        self._heartbeat()
        if self._steps is None:
            task = self.store.next_runnable(self.cfg)
            if task is None:
                return None
            if task.kind == "batch":
                from .batch import BatchRunner, BatchStore

                self._bstore = self._bstore or BatchStore(self.cfg)
                self.runner = BatchRunner(self.engine, self.cfg, self.store, task, self._bstore, self.governor, self.clock)
            else:
                self.runner = TaskRunner(self.engine, self.cfg, self.store, task, clock=self.clock, governor=self.governor)
            self._steps = self.runner.steps()
        try:
            return next(self._steps)
        except StopIteration:
            self._steps, self.runner = None, None
            return 0.0

    def active(self) -> Optional[dict]:
        return self.runner.task.summary() if self.runner else None

    def _heartbeat(self) -> None:
        """Lets `mack run` know a worker is alive and will pick its job up."""
        now = time.time()
        if now - self._last_beat > 3:
            self._last_beat = now
            (self.cfg.state_dir / "worker.json").write_text(json.dumps({"pid": os.getpid(), "time": now}))


def worker_alive(cfg: Config) -> bool:
    try:
        beat = json.loads((cfg.state_dir / "worker.json").read_text())
        os.kill(beat["pid"], 0)
        return time.time() - beat["time"] < 20
    except Exception:
        return False
