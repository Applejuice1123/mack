"""`mack apply` and `mack undo`: the only code that ever writes into the user's files.

Order of operations, always:
  1. Plan every change in memory.
  2. Refuse the whole apply, touching nothing, if any target changed since the job
     read it (hash differs from the one recorded at enqueue).
  3. Write the undo log — original bytes of every file whose contents change, the
     old and new name of every rename, every file created — and fsync it.
  4. Only then make the changes.

`undo` replays the log backwards. It refuses if any file was edited after apply,
unless --force, which restores only the files that still match.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .recipes import Recipe, find
from .tasks import Task, TaskStore
from .textsource import sha256


class NothingToApply(ValueError):
    pass


class Refused(RuntimeError):
    def __init__(self, message: str, paths: list[str]):
        super().__init__(message)
        self.paths = paths


@dataclass
class Change:
    op: str  # modify | rename | create
    path: Path
    new_path: Optional[Path] = None
    data: Optional[bytes] = None  # new contents (modify, create)
    expect_hash: Optional[str] = None  # the file must still be what the job read
    summary: str = ""


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- frontmatter


def merge_frontmatter(text: str, values: dict, overwrite: bool = False) -> tuple[str, list[str]]:
    """Add keys to a note's YAML frontmatter without rewriting what is already there.
    Existing keys are kept unless overwrite. Returns (new text, keys written)."""
    def lines_for(k, v) -> str:
        # Block style for the key, flow style for a list of scalars: "tags: [a, b]".
        flow = None if isinstance(v, list) else False
        return yaml.safe_dump({k: v}, default_flow_style=flow, allow_unicode=True, sort_keys=False, width=10**6)

    m = re.match(r"\A---[ \t]*\n(.*?)(^---[ \t]*$\n?)", text, re.S | re.M)
    if m:
        block = m.group(1)
        try:
            existing = yaml.safe_load(block) or {}
        except yaml.YAMLError:
            raise ValueError("existing frontmatter is not valid YAML")
        if not isinstance(existing, dict):
            raise ValueError("existing frontmatter is not a mapping")
        written, add = [], ""
        for k, v in values.items():
            if v in (None, [], ""):
                continue
            if k in existing:
                if not overwrite:
                    continue
                block = _drop_key(block, k)
            add += lines_for(k, v)
            written.append(k)
        if not written:
            return text, []
        if block and not block.endswith("\n"):
            block += "\n"
        return text[: m.start(1)] + block + add + text[m.start(2):], written
    written = [k for k, v in values.items() if v not in (None, [], "")]
    if not written:
        return text, []
    front = "".join(lines_for(k, values[k]) for k in written)
    return f"---\n{front}---\n" + text, written


def _drop_key(block: str, key: str) -> str:
    out, skipping = [], False
    for line in block.splitlines(keepends=True):
        if re.match(rf"^{re.escape(key)}\s*:", line):
            skipping = True
            continue
        if skipping and (line.startswith((" ", "\t", "- ")) or not line.strip()):
            continue
        skipping = False
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------- planning


SAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def rename_target(path: Path, out: dict, pattern: str, taken: set) -> Path:
    vals = {
        "date": out.get("date") or "undated",
        "source": out.get("source") or "",
        "title": out.get("title") or "",
        "kind": out.get("kind") or "",
    }
    stem = pattern.format(**vals)
    stem = SAFE_NAME.sub("-", re.sub(r"\s+", " ", stem)).strip(" .-") or "untitled"
    stem = stem[:120]
    candidate, n = path.with_name(stem + path.suffix.lower()), 2
    while (candidate.exists() and candidate != path) or candidate in taken:
        candidate = path.with_name(f"{stem}-{n}{path.suffix.lower()}")
        n += 1
    taken.add(candidate)
    return candidate


def render_sidecar(template: str, path: Path, out: dict) -> str:
    if template == "json":
        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        return json.dumps(clean, indent=2, ensure_ascii=False) + "\n"
    if template == "summary":
        lines = [f"# Summary of {path.name}", "", out.get("summary") or "_No summary._", ""]
        points = out.get("key_points") or []
        quotes = out.get("key_points_quotes") or []
        if points:
            lines += ["## Key points", ""]
            lines += [f"- {p}\n  > {quotes[i]}" if i < len(quotes) else f"- {p}" for i, p in enumerate(points)]
        return "\n".join(lines) + "\n"
    if template == "clean":
        lines = [out.get("cleaned") or "", ""]
        items = out.get("action_items") or []
        quotes = out.get("action_items_quotes") or []
        if items:
            lines += ["## Action items", ""]
            lines += [f"- [ ] {a}" + (f"  (“{quotes[i]}”)" if i < len(quotes) else "") for i, a in enumerate(items)]
        return "\n".join(lines) + "\n"
    return "\n".join(f"**{k}**: {v}" for k, v in out.items() if not k.startswith("_")) + "\n"


def plan(store: TaskStore, bstore, task: Task, only: str = "done", overwrite: bool = False,
         csv_path: Optional[str] = None, recipe: Optional[Recipe] = None) -> tuple[list[Change], list[str]]:
    """Every change apply would make, and the items it skipped (with why)."""
    name = task.batch.get("recipe")
    if not name and not csv_path:
        raise NothingToApply("this job has no recipe, so there is nothing to apply; its results are in the output and review files")
    recipe = recipe or (find(name, store.cfg.recipes_dir) if name else None)
    rule = recipe.apply if recipe else {"kind": "none"}
    if rule.get("kind") == "none" and not csv_path:
        raise NothingToApply(f"recipe {name!r} has nothing to apply")
    statuses = {"done"} if only == "done" else {"done", "flagged"}
    rows = [r for r in bstore.items(task.id) if r["status"] in statuses and r["output"]]
    changes, notes, taken = [], [], set()
    csv_rows = []
    for r in rows:
        path, out = Path(r["path"]), json.loads(r["output"])
        if csv_path:
            csv_rows.append((path, out))
            continue
        kind = rule["kind"]
        if kind == "frontmatter":
            try:
                text = path.read_text()
                fields = {k: out.get(k) for k in rule.get("fields", out.keys())}
                new, written = merge_frontmatter(text, fields, overwrite)
            except (ValueError, OSError) as e:
                notes.append(f"{path}: skipped ({e})")
                continue
            if written:
                changes.append(Change("modify", path, data=new.encode(), expect_hash=r["content_hash"],
                                      summary=f"{path}: frontmatter + {', '.join(written)}"))
        elif kind == "rename":
            target = rename_target(path, out, rule.get("pattern", "{date} {source} {title}"), taken)
            if target != path:
                changes.append(Change("rename", path, new_path=target, expect_hash=r["content_hash"],
                                      summary=f"{path.name} → {target.name}"))
        elif kind == "sidecar":
            suffix = rule.get("suffix", ".mack.md")
            base = path.with_name(path.stem + suffix)
            target, n = base, 2
            while target.exists() or target in taken:
                target = base.with_name(f"{path.stem}-{n}{suffix}")
                n += 1
            taken.add(target)
            changes.append(Change("create", target, data=render_sidecar(rule.get("template", ""), path, out).encode(),
                                  summary=f"{path.name} → new file {target.name}"))
    if csv_path:
        target = Path(csv_path).expanduser()
        fields = sorted({k for _, o in csv_rows for k in o if not k.startswith("_")})
        buf = io.StringIO()
        w = csv.writer(buf)
        exists = target.exists()
        if not exists:
            w.writerow(["path", *fields])
        for path, o in csv_rows:
            w.writerow([str(path), *["; ".join(o[k]) if isinstance(o.get(k), list) else ("" if o.get(k) is None else o.get(k)) for k in fields]])
        if exists:
            old = target.read_bytes()
            changes.append(Change("modify", target, data=old + buf.getvalue().encode(), expect_hash=_digest(old),
                                  summary=f"{target}: +{len(csv_rows)} rows"))
        else:
            changes.append(Change("create", target, data=buf.getvalue().encode(), summary=f"new file {target} ({len(csv_rows)} rows)"))
    skipped = len(bstore.items(task.id)) - len(rows)
    if skipped:
        notes.append(f"{skipped} items not applied (status other than {' or '.join(sorted(statuses))})")
    return changes, notes


# ---------------------------------------------------------------- apply / undo


def undo_log_path(store: TaskStore, task: Task) -> Path:
    return store.output_root / f"{task.id}.undo.jsonl"


def apply(store: TaskStore, bstore, task: Task, only: str = "done", overwrite: bool = False,
          csv_path: Optional[str] = None, dry_run: bool = False) -> tuple[list[Change], list[str]]:
    changes, notes = plan(store, bstore, task, only, overwrite, csv_path)
    stale = [str(c.path) for c in changes if c.expect_hash and (not c.path.exists() or sha256(c.path) != c.expect_hash)]
    if stale:
        raise Refused("these files changed after the job read them; nothing was applied", stale)
    if dry_run or not changes:
        return changes, notes
    log = undo_log_path(store, task)
    if log.exists():
        raise Refused(f"this job was already applied; run `mack undo {task.id}` first", [str(log)])
    entries = []
    for c in changes:
        if c.op == "modify":
            entries.append({"op": "modify", "path": str(c.path), "original_b64": base64.b64encode(c.path.read_bytes()).decode(),
                            "after_hash": _digest(c.data)})
        elif c.op == "rename":
            entries.append({"op": "rename", "path": str(c.path), "new_path": str(c.new_path), "after_hash": sha256(c.path)})
        else:
            entries.append({"op": "create", "path": str(c.path), "after_hash": _digest(c.data)})
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if os.environ.get("MACK_TEST_CRASH_AFTER_UNDO_LOG"):
        os._exit(9)  # tests: prove nothing is written before the log is safely on disk
    for c in changes:
        if c.op == "modify":
            tmp = c.path.with_name(c.path.name + ".mack-tmp")
            tmp.write_bytes(c.data)
            os.replace(tmp, c.path)
        elif c.op == "rename":
            os.rename(c.path, c.new_path)
        else:
            c.path.parent.mkdir(parents=True, exist_ok=True)
            with open(c.path, "xb") as f:  # never overwrite an existing file
                f.write(c.data)
    task.batch["applied"] = {"undo_log": str(log), "changes": len(changes)}
    store.save(task)
    return changes, notes


def undo(store: TaskStore, task: Task, force: bool = False) -> tuple[list[str], list[str]]:
    """Returns (restored, left alone because they changed after apply)."""
    log = undo_log_path(store, task)
    if not log.exists():
        raise NothingToApply(f"job {task.id} has nothing to undo")
    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]

    def current(e) -> Optional[str]:
        p = Path(e["new_path"] if e["op"] == "rename" else e["path"])
        return sha256(p) if p.exists() else None

    changed = [e.get("new_path") or e["path"] for e in entries if current(e) != e["after_hash"]]
    if changed and not force:
        raise Refused("these files were changed after apply; nothing was undone (use --force to restore the others)", changed)
    restored, left = [], []
    for e in reversed(entries):
        if current(e) != e["after_hash"]:
            left.append(e.get("new_path") or e["path"])
            continue
        if e["op"] == "modify":
            p = Path(e["path"])
            tmp = p.with_name(p.name + ".mack-tmp")
            tmp.write_bytes(base64.b64decode(e["original_b64"]))
            os.replace(tmp, p)
        elif e["op"] == "rename":
            if Path(e["path"]).exists():
                left.append(e["new_path"])
                continue
            os.rename(e["new_path"], e["path"])
        else:
            Path(e["path"]).unlink()  # a file Mack created and nobody has touched since
        restored.append(e["path"])
    log.rename(log.with_suffix(".undone.jsonl"))
    task.batch["applied"] = None
    store.save(task)
    return restored, left
