"""Live stress test: Mack working through a big queue while you use the Mac.

    .venv/bin/python scripts/stress_live.py [--minutes 30] [--notes 150] [--no-kill]

Generates a workload (notes of mixed length, an inbox of PDFs/scans/screenshots,
messy transcripts), starts `mack work` as its own process exactly as in daily use,
queues three jobs at the auto pace, and records every 2 seconds:

  * the level Mack chose (away / active / busy / hold) and progress
  * other apps' CPU duty (everything except the worker), memory pressure, swap
  * whole-Mac package power and thermal pressure (powermetrics, if allowed)

Halfway through it SIGKILLs the worker and restarts it, to check that nothing is
lost or duplicated under load. At the end it writes a JSON + Markdown report into
the run folder (output/stress-<time>/). It cannot measure how much *you* were
slowed down; that is your call, so note it in the report afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from mack import power, system  # noqa: E402
from mack.config import load_config  # noqa: E402
from tests.fixtures import make_fixtures as fx  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")


def level_of(note: str, status: str) -> str:
    if status not in ("running",):
        return status
    n = (note or "").lower()
    if n.startswith("paused"):
        return "hold"
    if "away" in n:
        return "away"
    if "busy" in n:
        return "busy"
    if "using the mac" in n:
        return "active"
    return "running"


class OtherCPU:
    """CPU duty of every process except the worker, in performance-core units."""

    def __init__(self, n_perf: int):
        self.n_perf, self.prev, self.t = n_perf, {}, time.monotonic()

    def sample(self, exclude: set[int]) -> float:
        now, snap = time.monotonic(), {}
        for p in psutil.process_iter():
            if p.pid in exclude:
                continue
            try:
                t = p.cpu_times()
                snap[p.pid] = t.user + t.system
            except (psutil.Error, OSError):
                pass
        dt = now - self.t
        used = sum(max(0.0, snap[k] - self.prev[k]) for k in snap if k in self.prev)
        self.prev, self.t = snap, now
        return min(4.0, used / dt / self.n_perf) if dt > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--notes", type=int, default=150)
    ap.add_argument("--no-kill", action="store_true")
    a = ap.parse_args()

    base = load_config()
    run = ROOT / "output" / time.strftime("stress-%Y%m%d-%H%M%S")
    (run / "state").mkdir(parents=True)
    (run / "mack.toml").write_text(
        (ROOT / "mack.toml").read_text().replace('state = "state"', f'state = "{run}/state"\noutput = "{run}/results"'))
    for f in ("calibration.json", "energy.json"):
        if (base.state_dir / f).exists():
            (run / "state" / f).write_text((base.state_dir / f).read_text())
    env = dict(os.environ, MACK_CONFIG=str(run / "mack.toml"))
    env.pop("MACK_FAKE_POWER", None)
    env.pop("MACK_STUB_ENGINE", None)

    print(f"workload → {run}")
    fx.vault(run / "notes", a.notes, seed=99)
    fx.inbox(run / "inbox")
    r = random.Random(3)
    (run / "transcripts").mkdir()
    for i in range(6):
        lines = [r.choice(["um, ", "so, like, ", "uh ", "you know, ", ""]) +
                 r.choice(fx.SENTENCES).format(t=r.choice(fx.TOPICS), p=r.choice(fx.PEOPLE)) for _ in range(14)]
        lines.append(f"okay so {r.choice(fx.PEOPLE)} will send the notes by friday")
        (run / "transcripts" / f"call{i}.txt").write_text("\n".join(lines) + "\n")

    log = open(run / "worker.log", "a")
    start_worker = lambda: subprocess.Popen([PY, "-m", "mack.cli", "work"], env=env, cwd=str(ROOT), stdout=log, stderr=log)
    worker = start_worker()
    jobs = []
    for recipe, folder in (("rename", "inbox"), ("clean", "transcripts"), ("tag", "notes")):
        out = subprocess.run([PY, "-m", "mack.cli", "run", "--recipe", recipe, str(run / folder)],
                             env=env, cwd=str(ROOT), capture_output=True, text=True)
        jid = next(l.split()[1] for l in out.stdout.splitlines() if l.startswith("job "))
        jobs.append((recipe, jid))
        print(f"queued {recipe}: {jid}  ({out.stdout.splitlines()[0]})")

    meter = power.PowerMeter(2000) if power.available() else None
    other = OtherCPU(4)
    swap0 = psutil.swap_memory().used
    rows, kills = [], []
    t0 = time.monotonic()
    killed = a.no_kill
    tasks_dir = run / "state" / "tasks"
    print(f"recording for up to {a.minutes:.0f} minutes; use your Mac as you normally would. Ctrl-C ends early.")
    try:
        while time.monotonic() - t0 < a.minutes * 60:
            time.sleep(2)
            now = time.monotonic()
            tasks = {}
            for recipe, jid in jobs:
                try:
                    tasks[recipe] = json.loads((tasks_dir / jid / "task.json").read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            active = next((t for t in tasks.values() if t["status"] == "running"), None)
            done = sum((t["batch"].get("counts") or {}).get("total", 0) - (t["batch"].get("counts") or {}).get("pending", 0)
                       for t in tasks.values())
            row = {
                "t": round(now - t0, 1),
                "level": level_of(active["batch"].get("pace_note", ""), "running") if active else "idle",
                "job": next((k for k, t in tasks.items() if t is active), None),
                "items_finished": done,
                "other_cpu": round(other.sample({worker.pid}), 3),
                "memory_pressure": system.memory_pressure(),
                "swap_growth_gb": round((psutil.swap_memory().used - swap0) / 1e9, 3),
                "idle_s": round(system.input_idle_seconds(), 1),
            }
            if meter:
                row["watts"] = meter.mean(now - 2.5, now)
                row["thermal"] = meter.worst_thermal(now - 2.5, now)
            rows.append(row)
            print(f"\r{row['t']:6.0f}s  {row['level']:7s} {row['job'] or '':7s} finished {done:4d}  "
                  f"other CPU {row['other_cpu']:.2f}  mem {row['memory_pressure']}  "
                  f"{(row.get('watts') or 0):5.1f} W  {row.get('thermal') or ''}   ", end="", flush=True)
            if not killed and now - t0 > a.minutes * 30:
                worker.send_signal(signal.SIGKILL)
                worker.wait()
                kills.append({"t": row["t"], "finished_before": done})
                print(f"\n-- SIGKILLed the worker at {row['t']:.0f}s; restarting in 5 s")
                time.sleep(5)
                worker = start_worker()
                killed = True
            if all(t["status"] in ("done", "failed", "cancelled") for t in tasks.values()) and len(tasks) == len(jobs):
                print("\nall jobs finished")
                break
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        worker.send_signal(signal.SIGINT)
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
        if meter:
            meter.close()

    # ---- analysis ----
    db = sqlite3.connect(str(run / "state" / "batch.db"))
    db.row_factory = sqlite3.Row
    per_job = {}
    for recipe, jid in jobs:
        items = db.execute("SELECT * FROM items WHERE job_id=?", (jid,)).fetchall()
        c = Counter(i["status"] for i in items)
        lines = (run / "results" / f"{jid}.jsonl")
        paths = [json.loads(l)["path"] for l in lines.read_text().splitlines()] if lines.exists() else []
        per_job[recipe] = {
            "counts": dict(c), "max_attempts": max((i["attempts"] for i in items), default=0),
            "jsonl_lines": len(paths), "duplicate_lines": len(paths) - len(set(paths)),
            "tokens_in": sum(i["tokens_in"] or 0 for i in items), "tokens_out": sum(i["tokens_out"] or 0 for i in items),
            "failed_reasons": Counter((i["reason"] or i["error"] or "")[:60] for i in items if i["status"] == "failed"),
        }
    truth = json.loads((run / "inbox-answers.json").read_text())
    rename_job = dict(jobs)["rename"]
    right = wrong = undated = 0
    for i in db.execute("SELECT * FROM items WHERE job_id=?", (rename_job,)):
        name = Path(i["path"]).name
        if name in truth and i["output"]:
            o = json.loads(i["output"])
            if o.get("date") == truth[name]["date"]:
                right += 1
            elif o.get("date") is None or i["status"] == "flagged":
                undated += 1
            else:
                wrong += 1
    levels = Counter(r["level"] for r in rows)
    by_level = defaultdict(list)
    for prev, cur in zip(rows, rows[1:]):
        by_level[cur["level"]].append(cur["items_finished"] - prev["items_finished"])
    watts = [r["watts"] for r in rows if r.get("watts")]
    duration = rows[-1]["t"] if rows else 0
    summary = {
        "minutes": round(duration / 60, 1),
        "time_share_by_level": {k: round(v / len(rows), 3) for k, v in levels.items()} if rows else {},
        "items_per_minute_by_level": {k: round(sum(v) / (len(v) * 2 / 60), 2) for k, v in by_level.items() if v},
        "items_finished": rows[-1]["items_finished"] if rows else 0,
        "other_cpu_mean": round(sum(r["other_cpu"] for r in rows) / len(rows), 3) if rows else None,
        "other_cpu_p90": round(sorted(r["other_cpu"] for r in rows)[int(0.9 * (len(rows) - 1))], 3) if rows else None,
        "input_active_share": round(sum(1 for r in rows if r["idle_s"] < 60) / len(rows), 3) if rows else None,
        "watts_mean": round(sum(watts) / len(watts), 2) if watts else None,
        "watts_p95": round(sorted(watts)[int(0.95 * (len(watts) - 1))], 2) if watts else None,
        "thermal_levels": dict(Counter(r.get("thermal") for r in rows if r.get("thermal"))),
        "memory_pressure_levels": dict(Counter(r["memory_pressure"] for r in rows)),
        "swap_growth_gb_max": max((r["swap_growth_gb"] for r in rows), default=0),
        "kills": kills,
        "jobs": per_job,
        "rename_accuracy": {"invoice_dates_right": right, "undated_or_flagged": undated, "wrong_unflagged": wrong},
        "worker_log_errors": [l.strip() for l in (run / "worker.log").read_text().splitlines()
                              if "Traceback" in l or "Error" in l][:10],
    }
    (run / "telemetry.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "report.json").write_text(json.dumps(summary, indent=2, default=dict))
    print(json.dumps(summary, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
