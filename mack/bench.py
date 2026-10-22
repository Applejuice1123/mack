"""`mack bench`: does each pace keep its promise on this Mac, in slowdown and in power?

For each pace a real batch job runs on synthetic notes while:
  * the CPU probe from the calibration (STREAM triad = memory-heavy work, an FMA
    loop = compute-heavy work; foreground QoS; warm readings) stands in for your apps
    and measures how much slower they get, and
  * powermetrics (if allowed through sudo) records package power and thermal pressure.

Idle references are re-taken before every pace. The auto pace is measured twice:
with the probe running (your apps are busy, so auto should back off) and without
it (nothing else running, so auto should go fast). Results are compared with the
contract in mack.toml [pace] and written to state/bench.json and reports/bench.md.
"""

from __future__ import annotations

import copy
import json
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path

from .config import ROOT, Config
from .governor import now_iso
from .probe import CpuProbe, build

NOTE = (
    "Meeting notes, week {i}. We reviewed the roadmap for the storage service and agreed to move the "
    "migration to the second quarter. Priya will draft the rollout plan; Tomas owns the load test. "
    "Open questions: whether to keep the legacy export, and how to handle customers on the old API. "
    "Next check-in is Thursday. Action: send the summary to the wider team by Friday."
)


def _probe_window(probe: CpuProbe, seconds: float) -> dict:
    tri, comp = [], []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        tri.append(probe.warm("t"))
        comp.append(probe.warm("c"))
    return {"triad": statistics.median(tri), "compute": statistics.median(comp)}


def run(cfg: Config, seconds: float = 60, paces=("gentle", "balanced", "steady", "full", "auto"), log=print) -> dict:
    from . import power
    from .batch import BatchRunner, BatchStore, create
    from .engine import make_engine
    from .governor import Governor
    from .tasks import TaskStore

    tmp = Path(tempfile.mkdtemp(prefix="mack-bench-"))
    meter = power.PowerMeter() if power.available() else None
    if meter is None:
        log("power: unavailable (add a sudoers rule for /usr/bin/powermetrics to measure watts)")
    try:
        bcfg = copy.deepcopy(cfg)
        bcfg.paths.state = str(tmp / "state")
        bcfg.paths.output = str(tmp / "out")
        cal = cfg.state_dir / "calibration.json"
        if cal.exists():
            shutil.copy(cal, bcfg.state_dir / "calibration.json")
        notes = tmp / "notes"
        notes.mkdir()
        for i in range(600):
            (notes / f"note{i:03d}.md").write_text(NOTE.format(i=i))
        engine = make_engine(cfg, load_draft=False)
        governor = Governor(bcfg)
        governor.start()  # auto needs to see other apps' CPU use
        store, bstore = TaskStore(bcfg), BatchStore(bcfg)
        plan = [(p, True) for p in paces if p != "auto"]
        if "auto" in paces:
            plan += [("auto", True), ("auto", False)]
        rows = []
        with CpuProbe(build(cfg.state_dir)) as probe:
            for pace, with_probe in plan:
                label = pace if pace != "auto" else ("auto, your apps busy" if with_probe else "auto, nothing else running")
                log(f"{label}: idle reference…")
                # The baseline runs exactly what the measurement runs minus Mack: with the
                # probe for probe rows, so "added power" is Mack's alone.
                t_idle = time.monotonic()
                governor.sample_once()
                idle = _probe_window(probe, min(20, seconds)) if with_probe else (time.sleep(10) or None)
                idle_w = meter.mean(t_idle, time.monotonic()) if meter else None
                others = governor.sample_once() - (0.25 if with_probe else 0.0)  # the probe keeps one core busy
                if others > 0.15:
                    log(f"  warning: other apps were using {others:.0%} of the performance cores; results will be noisy")
                task, _ = create(bcfg, store, bstore, "Summarize this note in one sentence.", [str(notes)], pace=pace)
                runner = BatchRunner(engine, bcfg, store, task, bstore, governor=governor)
                done, result = threading.Event(), {}

                def sample(with_probe=with_probe, runner=runner):
                    time.sleep(5)  # let the job reach its steady pace
                    t0, tok0 = time.monotonic(), runner.task.tokens
                    if with_probe:
                        result.update(_probe_window(probe, seconds))
                    else:
                        time.sleep(seconds)
                    t1 = time.monotonic()
                    result.update(tps=(runner.task.tokens - tok0) / (t1 - t0), t0=t0, t1=t1)
                    done.set()

                th = threading.Thread(target=sample, daemon=True)
                th.start()
                log(f"{label}: running a batch job for {seconds + 5:.0f} s…")
                for wait in runner.steps():
                    if done.is_set():
                        store.set_control(task.id, "cancel")
                    if wait > 0:
                        time.sleep(min(wait, 0.5))
                th.join()
                watts = meter.mean(result["t0"], result["t1"]) if meter else None
                total = sum(runner.pacer.levels.values()) or 1.0
                row = {
                    "pace": label,
                    "target_tps": runner.base_rate if pace != "auto" else None,
                    "measured_tps": round(result["tps"], 2),
                    "memory_slowdown": round(1 - result["triad"] / idle["triad"], 4) if with_probe else None,
                    "compute_slowdown": round(1 - result["compute"] / idle["compute"], 4) if with_probe else None,
                    "added_watts": round(watts - idle_w, 2) if watts is not None and idle_w is not None else None,
                    "package_watts": round(watts, 2) if watts is not None else None,
                    "thermal": meter.worst_thermal(result["t0"], result["t1"]) if meter else None,
                    "levels": {k: round(v / total, 2) for k, v in runner.pacer.levels.items() if v},
                    "other_apps_duty": round(max(0.0, others), 3),
                    "limit": getattr(cfg.pace, f"{pace}_max_slowdown", None) if with_probe else None,
                }
                if row["added_watts"] is not None and row["measured_tps"]:
                    row["joules_per_token"] = round(row["added_watts"] / row["measured_tps"], 3)
                row["within_contract"] = row["limit"] is None or row["memory_slowdown"] <= row["limit"]
                rows.append(row)
                log(
                    f"  {label}: {row['measured_tps']} tok/s"
                    + (f", memory-heavy {row['memory_slowdown']:+.1%}, compute-heavy {row['compute_slowdown']:+.1%}" if with_probe else "")
                    + (f", +{row['added_watts']} W" if row["added_watts"] is not None else "")
                    + (f", thermal {row['thermal']}" if row["thermal"] else "")
                    + (f", levels {row['levels']}" if pace == "auto" else "")
                    + ("" if row["within_contract"] else "  ← over the contract")
                )
    finally:
        if meter:
            meter.close()
        shutil.rmtree(tmp, ignore_errors=True)
    result = {"model": cfg.model.main, "created": now_iso(), "seconds": seconds, "rows": rows,
              "all_within_contract": all(r["within_contract"] for r in rows)}
    (cfg.state_dir / "bench.json").write_text(json.dumps(result, indent=2))
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    lines = [
        "| pace | tok/s (target) | tok/s (measured) | memory-heavy slowdown | compute-heavy slowdown | added power | thermal | contract |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        limit = f"≤ {r['limit']:.0%} {'✓' if r['within_contract'] else '✗'}" if r["limit"] else "—"
        watts = f"+{r['added_watts']} W" if r["added_watts"] is not None else "—"
        lines.append(f"| {r['pace']} | {r['target_tps'] or '—'} | {r['measured_tps']} | {pct(r['memory_slowdown'])} | "
                     f"{pct(r['compute_slowdown'])} | {watts} | {r['thermal'] or '—'} | {limit} |")
    report = ROOT / "reports" / "bench.md"
    report.parent.mkdir(exist_ok=True)
    noisy = [r["pace"] for r in rows if r["other_apps_duty"] > 0.15]
    warning = (f"\n**Noisy:** other apps were busy during {', '.join(noisy)}; rerun with the Mac otherwise idle.\n"
               if noisy else "")
    report.write_text(f"# mack bench: {cfg.model.main}\n\nMeasured {result['created']}, {seconds:.0f} s per pace.\n\n"
                      + "\n".join(lines) + "\n" + warning)
    print("\n" + "\n".join(lines))
    return result
