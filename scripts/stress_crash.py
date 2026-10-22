"""Crash and concurrency stress (stub engine, no model).

    .venv/bin/python scripts/stress_crash.py

1. Kill storm: a worker is SIGKILLed every few seconds and restarted until the
   job finishes. Expect every file exactly once in the output, no file tried more
   than twice, nothing failed except files killed twice in a row.
2. Two workers at once on the same queue. Expect each file processed once.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def setup(name: str, n: int):
    run = ROOT / "output" / time.strftime(f"{name}-%Y%m%d-%H%M%S")
    (run / "vault").mkdir(parents=True)
    for i in range(n):
        (run / "vault" / f"n{i:03d}.md").write_text(f"Note {i} about topic {i} and the weekly meeting.\n" * 3)
    (run / "mack.toml").write_text(
        f'[paths]\nstate = "{run}/state"\noutput = "{run}/results"\n[batch]\nitem_rest_s = 0.0\n')
    env = dict(os.environ, MACK_CONFIG=str(run / "mack.toml"), MACK_STUB_ENGINE="1", MACK_FAKE_POWER="ac",
               MACK_STUB_DELAY="0.05", PYTHONPATH=str(ROOT))
    out = subprocess.run([PY, "-m", "mack.cli", "run", "--pace", "full", "Echo the first words.", str(run / "vault")],
                         env=env, cwd=str(ROOT), capture_output=True, text=True)
    job = next(l.split()[1] for l in out.stdout.splitlines() if l.startswith("job "))
    return run, env, job


def audit(run: Path, job: str, n: int) -> dict:
    db = sqlite3.connect(str(run / "state" / "batch.db"))
    db.row_factory = sqlite3.Row
    items = db.execute("SELECT status, attempts, error, reason FROM items WHERE job_id=?", (job,)).fetchall()
    lines = [json.loads(l) for l in (run / "results" / f"{job}.jsonl").read_text().splitlines()]
    paths = Counter(l["path"] for l in lines)
    task = json.loads((run / "state" / "tasks" / job / "task.json").read_text())
    return {
        "task_status": task["status"],
        "statuses": dict(Counter(i["status"] for i in items)),
        "max_attempts": max(i["attempts"] for i in items),
        "output_lines": len(lines),
        "unique_files_in_output": len(paths),
        "duplicated_files": [p for p, c in paths.items() if c > 1][:5],
        "failed_reasons": dict(Counter((i["error"] or i["reason"] or "")[:40] for i in items if i["status"] == "failed")),
        "expected": n,
    }


def kill_storm(n=60) -> dict:
    run, env, job = setup("crash-killstorm", n)
    r = random.Random(7)
    kills = 0
    t0 = time.time()
    while time.time() - t0 < 600:
        w = subprocess.Popen([PY, "-m", "mack.cli", "work", "--once"], env=env, cwd=str(ROOT),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            w.wait(timeout=r.uniform(1.5, 6.0))
            break  # finished the queue on its own
        except subprocess.TimeoutExpired:
            w.send_signal(signal.SIGKILL)
            w.wait()
            kills += 1
    res = audit(run, job, n)
    res["kills"] = kills
    res["passed"] = (res["task_status"] == "done" and res["unique_files_in_output"] == n
                     and not res["duplicated_files"] and res["max_attempts"] <= 2)
    return res


def two_workers(n=40) -> dict:
    run, env, job = setup("crash-twoworkers", n)
    ws = [subprocess.Popen([PY, "-m", "mack.cli", "work", "--once"], env=env, cwd=str(ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [w.communicate(timeout=900)[0] for w in ws]
    res = audit(run, job, n)
    res["worker_messages"] = [o.strip().splitlines()[-1] if o.strip() else "" for o in outs]
    res["passed"] = res["unique_files_in_output"] == n and not res["duplicated_files"] and res["task_status"] == "done"
    return res


if __name__ == "__main__":
    result = {"kill_storm": kill_storm(), "two_workers": two_workers()}
    print(json.dumps(result, indent=2))
    sys.exit(0 if all(v["passed"] for v in result.values()) else 1)
