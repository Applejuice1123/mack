"""Command line: mack chat | serve | phase0 | calibrate | doctor | lora ..."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def cmd_chat(args) -> int:
    from .app import Mack

    cfg = load_config(args.config)
    print(f"loading {cfg.model.main}…", file=sys.stderr)
    mack = Mack(cfg)
    session = mack.session(args.session)
    print(f"session {session.id}. /status, /memory, /new, /quit", file=sys.stderr)
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/status":
            print(json.dumps(mack.status(session), indent=2))
            continue
        if text == "/memory":
            print(json.dumps(session.memory_view(), indent=2))
            continue
        if text == "/new":
            session = mack.new_session()
            print(f"session {session.id}")
            continue
        print("mack> ", end="", flush=True)
        try:
            for event in session.chat(text):
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "status":
                    if event["phase"] == "compact_start":
                        print("[compacting context…] ", end="", flush=True, file=sys.stderr)
                    elif event["phase"] == "compact_done":
                        print(
                            f"[{event['tokens_before']} → {event['tokens_after']} tokens in {event['seconds']}s]",
                            file=sys.stderr,
                        )
                    elif event["phase"] == "recall":
                        print(f"[recalled messages {event['turns']}]", file=sys.stderr)
                elif event["type"] == "done":
                    print(
                        f"\n[{event['tps']} tok/s · {event['context_tokens']}/{event['budget_tokens']} ctx · "
                        f"{'/'.join(k for k, v in event['modes'].items() if v)}]",
                        file=sys.stderr,
                    )
        except KeyboardInterrupt:
            print("\n[interrupted]", file=sys.stderr)
        mack.save(session)
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    print(f"Mack.ai on http://{args.host}:{args.port}", file=sys.stderr)
    serve(load_config(args.config), args.host, args.port)
    return 0


def cmd_phase0(args) -> int:
    from . import phase0

    result = phase0.run(load_config(args.config), max_context=args.max_context)
    print(f"wrote state/phase0.json and reports/phase0.md (context ceiling {result['memory']['context_ceiling_tokens']:,} tokens)")
    return 0


def cmd_calibrate(args) -> int:
    from . import calibrate

    cfg = load_config(args.config)
    if args.energy:
        from . import power
        from .engine import make_engine

        if not power.available():
            print("mack: powermetrics is not allowed without a password; add a sudoers rule for /usr/bin/powermetrics", file=sys.stderr)
            return 3
        power.measure_energy(cfg, make_engine(cfg, load_draft=False))
        print("wrote state/energy.json")
        return 0
    calibrate.run(cfg, quick=args.quick)
    print("wrote state/calibration.json")
    return 0


def cmd_doctor(args) -> int:
    from .engine import adapter_shipped
    from .governor import ensure_foreground, load_json, rate_for_target, ruler_cells

    cfg = load_config(args.config)
    state = cfg.state_dir
    phase0 = load_json(state / "phase0.json", cfg.model.main)
    cal = load_json(state / "calibration.json", cfg.model.main)
    print(f"model            {cfg.model.main}")
    print(f"draft            {cfg.model.draft}")
    print(f"priority         {ensure_foreground()}")
    print(f"adapter shipped  {adapter_shipped(cfg.adapter_dir)}")
    if phase0:
        m, s = phase0["memory"], phase0["speculative"]
        print(f"phase0           ceiling {m['context_ceiling_tokens']:,} tok, cache {m['cache_kib_per_token']} KiB/tok, speculative x{s.get('speedup')}")
    else:
        print("phase0           not run for this model  ->  mack phase0")
    if cal:
        cap, note = rate_for_target(ruler_cells(cal, cfg.governor.ruler), cfg.governor.cpu_slowdown_target, cfg.governor.min_tps)
        print(
            f"calibration      rate cap {cap and round(cap, 1)} tok/s at target {cfg.governor.cpu_slowdown_target:.0%} "
            f"on the {cfg.governor.ruler} ruler ({note}); sync loop at duty {cal['sync_threshold']}"
        )
    else:
        print("calibration      not run for this model  ->  mack calibrate")
    return 0


def cmd_task(args) -> int:
    from .tasks import TaskStore

    store = TaskStore(load_config(args.config))
    if args.action == "add":
        if not args.text:
            print("usage: mack task add \"describe the task\" [--kind report|code] [--pages N] [--pace ...]")
            return 2
        task = store.create(" ".join(args.text), args.kind, args.pages, args.pace, args.out)
        print(f"queued {task.id} ({task.kind}, {task.pages} pages, {task.pace} pace)")
        print(f"output: {task.out_dir}")
        print("Mack works on it while its engine runs (the menu-bar app, `mack serve`, or `mack work`).")
    elif args.action == "list":
        for t in reversed(store.all()):
            s = t.summary()
            print(f"{s['id']}  {s['status']:9s} {s['progress']:4.0%}  {s['units_done']}/{s['units_total']}  {s['title'][:60]}")
    elif args.action in ("show", "follow"):
        task_id = args.text[0] if args.text else store.all()[-1].id
        while True:
            t = store.load(task_id)
            s = t.summary()
            print(f"{s['title']}\n{s['status']} · {s['stage']} · {s['progress']:.0%} · {s['units_done']}/{s['units_total']} units · {s['tps']} tok/s · {s['out_dir']}")
            for u in t.units:
                print(f"  [{'x' if u.status == 'done' else ('>' if u.status == 'writing' else ' ')}] {u.title}" + (f"  ({u.words} words)" if u.words else ""))
            for line in t.log[-5:]:
                print("  " + line)
            if args.action == "show" or t.status not in ("queued", "running"):
                break
            import time as _t
            _t.sleep(10)
            print()
    elif args.action in ("pause", "resume", "cancel"):
        task_id = args.text[0] if args.text else ""
        store.set_control(task_id, args.action)
        print(f"{args.action}: {task_id}")
    return 0


def cmd_work(args) -> int:
    """Run queued tasks in the foreground, without the HTTP server."""
    import time

    from .engine import LazyEngine, ModelMissing
    from .governor import Governor, set_thread_user_initiated
    from .tasks import TaskScheduler, TaskStore

    cfg = load_config(args.config)
    set_thread_user_initiated()
    engine = LazyEngine(cfg, load_draft=False)  # loads when there is work, unloads when there isn't
    governor = Governor(cfg)
    governor.start()
    scheduler = TaskScheduler(engine, cfg, TaskStore(cfg), governor=governor)
    if not scheduler.lock.acquire():
        print("mack: another Mack worker is already running (the menu-bar app, `mack serve` or `mack work`); "
              "it will process the queue", file=sys.stderr)
        return 1
    print("working through queued tasks; Ctrl-C to stop (progress is saved after every unit)", file=sys.stderr)
    last = None
    last_work = time.monotonic()
    try:
        while True:
            try:
                wait = scheduler.tick()
            except ModelMissing as e:
                print(f"mack: {e}", file=sys.stderr)
                return 2
            if wait is not None:
                last_work = time.monotonic()
            elif engine.loaded and time.monotonic() - last_work > cfg.background.unload_after_idle_s:
                engine.unload()
                print("nothing to do: freed the model's memory", file=sys.stderr)
            active = scheduler.active()
            if active and (active["current"], active["stage"]) != last:
                last = (active["current"], active["stage"])
                print(f"[{active['progress']:.0%}] {active['stage']}: {active['current'] or active['title']}", file=sys.stderr)
            if wait is None:
                if args.once:
                    return 0
                time.sleep(2.0)
            elif wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\nstopped; run `mack work` again to resume", file=sys.stderr)
    return 0


def cmd_lora(args) -> int:
    from .lora import dataset, evals, train

    cfg = load_config(args.config)
    if args.action == "data":
        counts = dataset.build(cfg, seed=args.seed)
        print(f"wrote {counts}")
    elif args.action == "train":
        return train.run(cfg, iters=args.iters, num_layers=args.layers)
    elif args.action == "baseline":
        evals.baseline(cfg, limit=args.limit)
    elif args.action == "eval":
        report = evals.gate(cfg, limit=args.limit)
        print(json.dumps(report["decision"], indent=2))
    return 0


def main(argv=None) -> int:
    from . import __version__
    from . import cli_batch

    p = argparse.ArgumentParser(prog="mack", description="Run a local AI over thousands of your files without your Mac noticing.")
    p.add_argument("--config", help="path to mack.toml")
    p.add_argument("--version", action="version", version=f"mack {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    cli_batch.register(sub)

    c = sub.add_parser("chat", help="chat in the terminal")
    c.add_argument("--session", help="resume a saved session id")
    c.set_defaults(fn=cmd_chat)

    c = sub.add_parser("serve", help="run the web UI and API")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8082)
    c.set_defaults(fn=cmd_serve)

    c = sub.add_parser("phase0", help="measure memory slopes, loops and speculative decoding on this model")
    c.add_argument("--max-context", type=int, default=0)
    c.set_defaults(fn=cmd_phase0)

    c = sub.add_parser("calibrate", help="measure how decode slows CPU work; sets the rate cap")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--energy", action="store_true", help="measure joules per prompt and generated token (powermetrics)")
    c.set_defaults(fn=cmd_calibrate)

    c = sub.add_parser("doctor", help="show what is measured and configured")
    c.set_defaults(fn=cmd_doctor)

    c = sub.add_parser("task", help="background tasks: add | list | show | follow | pause | resume | cancel")
    c.add_argument("action", choices=["add", "list", "show", "follow", "pause", "resume", "cancel"])
    c.add_argument("text", nargs="*", help="the task description (add) or a task id")
    c.add_argument("--kind", default="auto", choices=["auto", "report", "code"])
    c.add_argument("--pages", type=int, default=5, help="target length for reports")
    c.add_argument("--pace", default="auto", choices=["auto", "gentle", "balanced", "steady", "full"])
    c.add_argument("--out", help="output folder (must be new or empty)")
    c.set_defaults(fn=cmd_task)

    c = sub.add_parser("work", help="run queued tasks in the foreground (no server)")
    c.add_argument("--once", action="store_true", help="exit when the queue is empty")
    c.set_defaults(fn=cmd_work)

    c = sub.add_parser("lora", help="behaviour fine-tune: data | baseline | train | eval")
    c.add_argument("action", choices=["data", "baseline", "train", "eval"])
    c.add_argument("--iters", type=int, default=300)
    c.add_argument("--layers", type=int, default=4, help="how many final layers get LoRA (memory grows with this)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--limit", type=int, default=0, help="eval only the first N cases per category")
    c.set_defaults(fn=cmd_lora)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
