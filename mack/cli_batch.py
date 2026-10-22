"""CLI for `mack run` and friends. Exit codes: 0 ok, 1 user error, 2 engine/model error, 3 refused."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import Config, load_config

OK, USER_ERROR, ENGINE_ERROR, REFUSED = 0, 1, 2, 3


def _err(msg: str, code: int) -> int:
    print(f"mack: {msg}", file=sys.stderr)
    return code


def _stores(cfg: Config):
    from .batch import BatchStore
    from .tasks import TaskStore

    return TaskStore(cfg), BatchStore(cfg)


def resolve_job(store, ident: str | None):
    """A job id, a unique prefix of one, or 'last' (also the default)."""
    jobs = [t for t in store.all() if t.kind == "batch"]
    if not jobs:
        raise ValueError("there are no batch jobs yet; start one with `mack run`")
    if not ident or ident == "last":
        return jobs[-1]
    matches = [t for t in jobs if t.id == ident or t.id.startswith(ident)]
    if len(matches) != 1:
        raise ValueError(f"no single job matches {ident!r}" + (f" ({len(matches)} do)" if matches else ""))
    return matches[0]


def progress_line(task, bstore) -> str:
    from .batch import estimate_seconds, fmt_duration

    c = bstore.counts(task.id)
    st = bstore.stats(task.id)
    tps = st["recent_tps"]
    finished = c["total"] - c["pending"]
    avg_in = st["avg_tokens_in"] or 600
    avg_out = st["avg_tokens_out"] or 150
    left = estimate_seconds(load_config(), task.pace, c["pending"], avg_in, avg_out) if c["pending"] else 0
    parts = [f"{finished}/{c['total']}", f"{tps:.1f} tok/s" if tps else "– tok/s", f"~{fmt_duration(left)} left" if c["pending"] else "finished"]
    if c["flagged"]:
        parts.append(f"{c['flagged']} flagged")
    if c["failed"]:
        parts.append(f"{c['failed']} failed")
    note = task.batch.get("pace_note")
    return " · ".join(parts) + (f" · {note}" if note else "")


# ---------------------------------------------------------------- run


def cmd_run(args) -> int:
    from .batch import create
    from .recipes import RecipeError, find, parse_options
    from .schema import SchemaError, load_schema
    from .tasks import worker_alive

    cfg = load_config(args.config)
    store, bstore = _stores(cfg)
    items = list(args.items)
    try:
        recipe = find(args.recipe, cfg.recipes_dir) if args.recipe else None
        if recipe:
            extra = ""
            if items and not Path(items[0]).expanduser().exists() and items[0] != "-" and len(items) > 1:
                extra = items.pop(0)
            instruction = recipe.instruction + (f"\n{extra}" if extra else "")
        else:
            if len(items) < 2:
                return _err('usage: mack run [options] "INSTRUCTION" PATH [PATH...]   (or --recipe NAME PATH...)', USER_ERROR)
            instruction = items.pop(0)
        paths = []
        for p in items:
            paths += [line.strip() for line in sys.stdin if line.strip()] if p == "-" else [p]
        if not paths:
            return _err("give at least one file or folder (or - to read paths from stdin)", USER_ERROR)
        user_schema = None
        if args.schema:
            load_schema(args.schema)  # clear error before anything else
            user_schema = json.loads(Path(args.schema).expanduser().read_text())
        schema = recipe.resolved_schema(parse_options(args.option), user_schema) if recipe else user_schema
        task, plan = create(
            cfg, store, bstore, instruction, paths,
            pace=args.pace, schema=schema, schema_path=args.schema, recipe=recipe.name if recipe else None,
            glob=args.glob or (recipe.glob if recipe else None), out=args.out, limit=args.limit,
            ocr=args.ocr or (recipe.ocr if recipe else False),
            default_grounded=recipe.default_grounded if recipe else False,
            checks=recipe.checks if recipe else None, options=parse_options(args.option),
            boost_when_idle=args.boost_when_idle,
            max_output_tokens=recipe.max_output_tokens if recipe else None,
            dry_run=args.dry_run,
        )
    except (RecipeError, SchemaError, ValueError, FileNotFoundError) as e:
        return _err(str(e), USER_ERROR)

    print(plan["text"])
    if args.dry_run:
        for line in plan["first"]:
            print(f"  {line}")
        if plan["items"] > 5:
            print(f"  … and {plan['items'] + plan['skipped'] - 5} more")
        return OK
    print(f"job {task.id} · output {task.batch['output_path']}")
    if worker_alive(cfg):
        if not args.wait:
            print("Mack's worker is running and will pick this up. Follow it with: mack status " + task.id)
            return OK
        return _attach(cfg, store, bstore, task.id)
    if not args.wait:
        print("Mack's worker is not running. Start it with:  mack work   (or rerun with --wait to run here)")
        return OK
    return _run_here(cfg, store, bstore, task.id)


def _attach(cfg, store, bstore, job_id) -> int:
    try:
        while True:
            task = store.load(job_id)
            print("\r\033[K" + progress_line(task, bstore), end="", flush=True)
            if task.status not in ("queued", "running"):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n(detached; the job keeps running in the worker)")
        return OK
    print()
    return _final(store.load(job_id))


def _run_here(cfg, store, bstore, job_id) -> int:
    from .batch import BatchRunner
    from .engine import LazyEngine, ModelMissing
    from .governor import Governor, set_thread_user_initiated

    set_thread_user_initiated()
    engine = LazyEngine(cfg, load_draft=False)
    try:
        engine.fmt  # load now, so a missing model is reported before anything starts
    except ModelMissing as e:
        return _err(str(e), ENGINE_ERROR)
    governor = Governor(cfg)
    governor.start()
    runner = BatchRunner(engine, cfg, store, store.load(job_id), bstore, governor=governor)
    last = 0.0
    try:
        for wait in runner.steps():
            now = time.monotonic()
            if now - last > 1.0:
                last = now
                print("\r\033[K" + progress_line(runner.task, bstore), end="", flush=True)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\nstopped; progress is saved. Resume with: mack work   (or mack run --wait again is not needed)")
        return OK
    print("\r\033[K" + progress_line(runner.task, bstore))
    return _final(store.load(job_id))


def _final(task) -> int:
    if task.status == "done":
        print(f"done. Review with: mack review {task.id}")
        return OK
    print(f"job is {task.status}" + (f": {task.error}" if task.error else ""))
    return ENGINE_ERROR if task.error else OK


# ---------------------------------------------------------------- status / review


def cmd_status(args) -> int:
    from .batch import fmt_duration

    cfg = load_config(args.config)
    store, bstore = _stores(cfg)
    if not args.job:
        jobs = [t for t in store.all() if t.kind == "batch"]
        if not jobs:
            print("no batch jobs yet")
            return OK
        print(f"{'job':22s} {'status':9s} {'done/total':>11s} {'flagged':>7s}  {'pace':8s} started")
        for t in reversed(jobs):
            c = t.batch.get("counts") or {}
            started = time.strftime("%b %d %H:%M", time.localtime(t.started)) if t.started else "-"
            print(f"{t.id:22s} {t.status:9s} {c.get('total', 0) - c.get('pending', 0):>5}/{c.get('total', 0):<5} {c.get('flagged', 0):>7}  {t.pace:8s} {started}")
        return OK
    try:
        t = resolve_job(store, args.job)
    except ValueError as e:
        return _err(str(e), USER_ERROR)
    c, st = bstore.counts(t.id), bstore.stats(t.id)
    print(f"{t.title}\n{t.id} · {t.status} · pace {t.pace} · {t.batch.get('output_path')}")
    print(progress_line(t, bstore))
    print(f"counts: {c['done']} done, {c['flagged']} flagged, {c['failed']} failed, {c['skipped']} skipped, {c['pending']} to go")
    if t.batch.get("current_path") and t.status == "running":
        print(f"now: {t.batch['current_path']}")
    if st["avg_tokens_out"]:
        print(f"average per item: {st['avg_tokens_in']:.0f} tokens in, {st['avg_tokens_out']:.0f} out")
    if st["recent_tps"]:
        print(f"last 5 minutes: {st['recent_tps']:.2f} tok/s")
    errors = [r for r in bstore.recent(t.id, 200) if r["status"] == "failed"][:3]
    for r in errors:
        print(f"error: {Path(r['path']).name}: {r['reason'] or r['error']}")
    if t.error:
        print(f"job error: {t.error}")
    return OK


def cmd_review(args) -> int:
    from .batch import write_review

    cfg = load_config(args.config)
    store, bstore = _stores(cfg)
    try:
        t = resolve_job(store, args.job)
    except ValueError as e:
        return _err(str(e), USER_ERROR)
    path = write_review(store, bstore, t)
    print(path)
    if not args.no_open:
        subprocess.run(["open", str(path)], check=False)
    return OK


# ---------------------------------------------------------------- apply / undo


def cmd_apply(args) -> int:
    from .apply import NothingToApply, Refused, apply

    cfg = load_config(args.config)
    store, bstore = _stores(cfg)
    try:
        t = resolve_job(store, args.job)
        changes, notes = apply(store, bstore, t, only=args.only, overwrite=args.overwrite, csv_path=args.csv, dry_run=args.dry_run)
    except NothingToApply as e:
        print(str(e))
        return OK
    except Refused as e:
        print(f"mack: refused: {e}", file=sys.stderr)
        for p in e.paths:
            print(f"  {p}", file=sys.stderr)
        return REFUSED
    except ValueError as e:
        return _err(str(e), USER_ERROR)
    for c in changes:
        print(("would change " if args.dry_run else "") + c.summary)
    for n in notes:
        print(n)
    if not changes:
        print("nothing to change")
    elif not args.dry_run:
        print(f"applied {len(changes)} changes. Undo with: mack undo {t.id}")
    return OK


def cmd_undo(args) -> int:
    from .apply import NothingToApply, Refused, undo

    cfg = load_config(args.config)
    store, _ = _stores(cfg)
    try:
        t = resolve_job(store, args.job)
        restored, left = undo(store, t, force=args.force)
    except NothingToApply as e:
        print(str(e))
        return OK
    except Refused as e:
        print(f"mack: refused: {e}", file=sys.stderr)
        for p in e.paths:
            print(f"  {p}", file=sys.stderr)
        return REFUSED
    except ValueError as e:
        return _err(str(e), USER_ERROR)
    print(f"restored {len(restored)} files")
    for p in left:
        print(f"left as is (changed after apply): {p}")
    return OK


# ---------------------------------------------------------------- recipes / model / bench


def cmd_recipes(args) -> int:
    from .recipes import RecipeError, all_recipes, find

    cfg = load_config(args.config)
    try:
        if args.action == "show":
            if not args.name:
                return _err("usage: mack recipes show NAME", USER_ERROR)
            r = find(args.name, cfg.recipes_dir)
            print(f"# {r.path}{' (built-in)' if r.builtin else ''}")
            print(r.path.read_text())
            return OK
        for r in all_recipes(cfg.recipes_dir):
            print(f"{r.name:10s} {'built-in' if r.builtin else 'yours   '}  {r.description}")
        print(f"\nCopy one to {cfg.recipes_dir}/ to customize it; yours replaces the built-in of the same name.")
    except RecipeError as e:
        return _err(str(e), USER_ERROR)
    return OK


def cmd_model(args) -> int:
    cfg = load_config(args.config)
    repo = args.model or cfg.model.main
    sizes = {"mlx-community/Qwen3.5-4B-4bit": 3.1, "mlx-community/Qwen3.5-9B-4bit": 6.0, "mlx-community/Qwen3.5-0.8B-4bit": 0.7}
    need = sizes.get(repo, 4.0) + 1.0
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < need:
        return _err(f"{repo} needs about {need:.1f} GB free and there is {free:.1f} GB. Free some space first.", REFUSED)
    from huggingface_hub import snapshot_download

    print(f"downloading {repo} (about {need - 1:.1f} GB)…")
    try:
        path = snapshot_download(repo)
    except Exception as e:
        return _err(f"download failed: {e}", ENGINE_ERROR)
    print(f"ready: {path}")
    return OK


def cmd_bench(args) -> int:
    from . import bench

    cfg = load_config(args.config)
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < 2.0:
        return _err(f"only {free:.1f} GB of disk free; the benchmark needs 2 GB. Free some space first.", REFUSED)
    try:
        result = bench.run(cfg, seconds=args.seconds, paces=args.paces.split(","))
    except Exception as e:
        return _err(f"benchmark failed: {type(e).__name__}: {e}", ENGINE_ERROR)
    return OK if result["all_within_contract"] else REFUSED


def register(sub) -> None:
    c = sub.add_parser("run", help="apply one instruction (or a recipe) to every file in a folder, slowly, in the background")
    c.add_argument("items", nargs="*", help='"INSTRUCTION" PATH [PATH...]  — with --recipe, just PATH...')
    c.add_argument("--pace", default="auto", choices=["auto", "gentle", "balanced", "steady", "full"],
                   help="auto (default): full speed while you're away, gentle while you're busy")
    c.add_argument("--schema", help="JSON Schema file for each item's output")
    c.add_argument("--recipe", help="built-in or user recipe (see `mack recipes`)")
    c.add_argument("--option", action="append", default=[], help="recipe option, e.g. tags=work,home,idea")
    c.add_argument("--out", help="output .jsonl file")
    c.add_argument("--glob", help="patterns for folders, comma-separated (default **/*.md,**/*.txt)")
    c.add_argument("--limit", type=int, help="only the first N matching files")
    c.add_argument("--ocr", action="store_true", help="read text from images and PDFs")
    c.add_argument("--boost-when-idle", action="store_true", help="run at steady while the Mac is idle")
    c.add_argument("--dry-run", action="store_true", help="show the plan and the first files, create nothing")
    c.add_argument("--wait", action="store_true", help="stay attached (runs here if no worker is running)")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("status", help="list batch jobs, or show one in detail")
    c.add_argument("job", nargs="?")
    c.set_defaults(fn=cmd_status)

    c = sub.add_parser("review", help="write the review file for a job and open it")
    c.add_argument("job", nargs="?")
    c.add_argument("--no-open", action="store_true")
    c.set_defaults(fn=cmd_review)

    c = sub.add_parser("apply", help="apply a job's results to your files (with an undo log)")
    c.add_argument("job", nargs="?")
    c.add_argument("--only", default="done", choices=["done", "done+flagged"])
    c.add_argument("--overwrite", action="store_true", help="frontmatter: replace keys that already exist")
    c.add_argument("--csv", help="extract: append one row per item to this CSV instead")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_apply)

    c = sub.add_parser("undo", help="reverse a job's apply")
    c.add_argument("job", nargs="?")
    c.add_argument("--force", action="store_true", help="restore the files that still match, leave edited ones")
    c.set_defaults(fn=cmd_undo)

    c = sub.add_parser("recipes", help="list recipes, or show one")
    c.add_argument("action", nargs="?", choices=["list", "show"], default="list")
    c.add_argument("name", nargs="?")
    c.set_defaults(fn=cmd_recipes)

    c = sub.add_parser("model", help="download the model")
    c.add_argument("action", choices=["download"])
    c.add_argument("--model", help="Hugging Face repo (default: model.main in mack.toml)")
    c.set_defaults(fn=cmd_model)

    c = sub.add_parser("bench", help="measure how much each pace slows memory-heavy and compute-heavy work")
    c.add_argument("--seconds", type=float, default=60)
    c.add_argument("--paces", default="gentle,balanced,steady,full,auto")
    c.set_defaults(fn=cmd_bench)
