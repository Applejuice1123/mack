"""E2E-1, the Tuesday-morning run: durability and politeness on the real model.

    .venv/bin/python scripts/e2e_tuesday.py [--n 300] [--pace gentle] [--kill-at 40] [--no-sleep]

1. Writes a deterministic vault of N notes.
2. Measures the memory-heavy CPU probe idle for 60 s, then starts `mack work` as a
   separate process and queues `mack run --recipe tag --pace <pace>` on the vault.
3. Logs the probe's throughput every 10 s for the whole run.
4. After --kill-at items, SIGKILLs the worker, checks the interrupted item, restarts
   the worker and checks it resumes at that item.
5. After --sleep-at items, asks you to put the Mac to sleep for 5 minutes (skip with
   --no-sleep) and checks the job resumes after wake with no duplicate output.
6. Reports wall time, tok/s, probe slowdown (mean and 95th percentile), flagged
   count, output line count and uniqueness, max attempts; pass/fail per the plan.

At gentle pace a 300-note vault takes many hours. Use --n 30 for a quick rehearsal.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mack.config import load_config  # noqa: E402
from mack.probe import CpuProbe, build  # noqa: E402
from tests.fixtures.make_fixtures import vault  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")


def counts(db: Path, job: str) -> dict:
    con = sqlite3.connect(str(db))
    rows = dict(con.execute("SELECT status, COUNT(*) FROM items WHERE job_id=? GROUP BY status", (job,)).fetchall())
    con.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pace", default="gentle")
    ap.add_argument("--kill-at", type=int, default=40)
    ap.add_argument("--sleep-at", type=int, default=80)
    ap.add_argument("--no-sleep", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    work = Path(tempfile.mkdtemp(prefix="mack-e2e1-"))
    vault(work / "vault", a.n)
    (work / "mack.toml").write_text(
        (ROOT / "mack.toml").read_text().replace('state = "state"', f'state = "{work}/state"\noutput = "{work}/output"')
    )
    env = dict(os.environ, MACK_CONFIG=str(work / "mack.toml"))
    (work / "state").mkdir()
    if (cfg.state_dir / "calibration.json").exists():
        (work / "state" / "calibration.json").write_text((cfg.state_dir / "calibration.json").read_text())

    probe = CpuProbe(build(cfg.state_dir))
    print("idle probe baseline (60 s)…")
    idle = statistics.median(probe.warm("t") for _ in range(int(60 / 0.6)))
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append((time.time(), probe.warm("t")))
            stop.wait(10)

    run = subprocess.run([PY, "-m", "mack.cli", "run", "--recipe", "tag", "--pace", a.pace, str(work / "vault")],
                         env=env, capture_output=True, text=True, cwd=str(ROOT))
    print(run.stdout)
    job = next(l.split()[1] for l in run.stdout.splitlines() if l.startswith("job "))
    db = work / "state" / "batch.db"
    worker = subprocess.Popen([PY, "-m", "mack.cli", "work"], env=env, cwd=str(ROOT), stderr=subprocess.DEVNULL)
    t0 = time.time()
    threading.Thread(target=sampler, daemon=True).start()
    killed_idx = None
    slept = a.no_sleep
    try:
        while True:
            c = counts(db, job)
            finished = sum(v for k, v in c.items() if k in ("done", "flagged", "failed", "skipped"))
            print(f"\r{finished}/{a.n}  {c}", end="", flush=True)
            if killed_idx is None and finished >= a.kill_at:
                con = sqlite3.connect(str(db))
                row = con.execute("SELECT idx FROM items WHERE job_id=? AND status='running'", (job,)).fetchone()
                con.close()
                worker.send_signal(signal.SIGKILL)
                worker.wait()
                killed_idx = row[0] if row else -1
                print(f"\nSIGKILLed the worker during item {killed_idx}; restarting")
                status = subprocess.run([PY, "-m", "mack.cli", "status", job], env=env, capture_output=True, text=True, cwd=str(ROOT))
                print(status.stdout)
                worker = subprocess.Popen([PY, "-m", "mack.cli", "work"], env=env, cwd=str(ROOT), stderr=subprocess.DEVNULL)
            if not slept and finished >= a.sleep_at:
                input("\nPut the Mac to sleep now for 5 minutes (Apple menu → Sleep), wake it, then press Return here… ")
                slept = True
            if finished >= a.n:
                break
            time.sleep(5)
    finally:
        stop.set()
        worker.terminate()
        probe.close()
    wall = time.time() - t0
    out_file = work / "output" / f"{job}.jsonl"
    lines = out_file.read_text().splitlines() if out_file.exists() else []
    con = sqlite3.connect(str(db))
    attempts = con.execute("SELECT MAX(attempts) FROM items WHERE job_id=?", (job,)).fetchone()[0]
    killed_attempts = con.execute("SELECT attempts FROM items WHERE job_id=? AND idx=?", (job, killed_idx)).fetchone()
    tokens = con.execute("SELECT SUM(tokens_out) FROM items WHERE job_id=?", (job,)).fetchone()[0] or 0
    con.close()
    slow = [1 - v / idle for _, v in samples]
    paths = [json.loads(l)["path"] for l in lines]
    c = counts(db, job)
    result = {
        "n": a.n, "pace": a.pace, "wall_s": round(wall), "tokens_out": tokens, "avg_tps": round(tokens / wall, 2),
        "probe_slowdown_mean": round(statistics.mean(slow), 4) if slow else None,
        "probe_slowdown_p95": round(sorted(slow)[int(0.95 * (len(slow) - 1))], 4) if slow else None,
        "flagged": c.get("flagged", 0), "output_lines": len(lines), "unique_paths": len(set(paths)),
        "max_attempts": attempts, "killed_item": killed_idx, "killed_item_attempts": killed_attempts[0] if killed_attempts else None,
    }
    limit = getattr(cfg.pace, f"{a.pace}_max_slowdown", 1.0)
    result["pass"] = (len(lines) == a.n == len(set(paths)) and (result["probe_slowdown_mean"] or 0) <= limit and attempts <= 2)
    print("\n" + json.dumps(result, indent=2))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "e2e1_tuesday.json").write_text(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
