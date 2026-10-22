"""`mack calibrate`: measure what decode does to the CPU on this machine and model.

The contention research measured how CPU load slows decode. Mack needs the other
direction, how decode slows CPU work, so this measures it directly:

  Part 1, rate cells. Decode at several capped token rates while a CPU probe
  (STREAM triad for the memory fabric, a cache-resident FMA loop for core
  availability) runs beside it. CPU slowdown = 1 - probe rate / idle probe rate,
  with the idle reference re-taken next to every cell. The triad is the worst
  case (work that is purely memory-bandwidth bound); real applications are hit
  less than it predicts (contention_harness F5, F6). The compute ruler is the
  other end: work that never leaves the cache.

  Part 2, loop cells. Under a synthetic memory-bandwidth load at several duties,
  time the pipelined loop against the synchronous one, and find the CPU duty past
  which pipelining stops being faster.

Close other apps first; anything else running lands in the numbers.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Callable, Optional

from .config import ROOT, Config
from .engine import Decision, Engine
from .governor import Governor, now_iso, rate_for_target, ruler_cells, write_json
from .probe import CpuProbe, StreamLoad, build

PROMPT = (
    "Write a very long, detailed technical essay about how computer memory hierarchies work, "
    "covering caches, DRAM, bandwidth, and virtual memory. Do not stop early."
)
RATE_FRACTIONS = (0.75, 0.5, 0.3, 0.15)
LOAD_DUTIES = (0.0, 0.3, 0.6, 1.0)
AHEAD_MARGIN = 1.03  # pipelined must beat synchronous by 3% to count as faster


class Fixed:
    def __init__(self, mode: str, rate_cap: Optional[float] = None):
        self._d = Decision(mode=mode, rate_cap=rate_cap)

    def decide(self) -> Decision:
        return self._d


def decode_until(engine: Engine, stop: Callable[[], bool], policy) -> list[float]:
    """Generate continuously until stop() is true; return a timestamp per token."""
    prompt = engine.render_chat([{"role": "user", "content": PROMPT}])
    stamps: list[float] = []
    while not stop():
        ctx = engine.new_context()
        for _ in engine.generate(ctx, prompt, 2000, policy=policy, cancel=stop):
            stamps.append(time.perf_counter())
    return stamps


def window_tps(stamps: list[float], t0: float, t1: float) -> float:
    n = sum(1 for s in stamps if t0 <= s <= t1)
    return n / (t1 - t0) if t1 > t0 else 0.0


def timed_decode(engine: Engine, policy, seconds: float, skip: float = 1.0) -> float:
    start = time.perf_counter()
    stamps = decode_until(engine, lambda: time.perf_counter() - start > seconds, policy)
    return window_tps(stamps, start + skip, start + seconds)


def rate_cell(engine: Engine, probe: CpuProbe, mode: str, cap: Optional[float], n: int) -> dict:
    idle_t, idle_c = probe.median("t", 2), probe.median("c", 2)
    result: dict = {}
    done = threading.Event()

    def sample():
        time.sleep(1.5)  # let decode reach steady state
        t0 = time.perf_counter()
        tri = [probe.warm("t") for _ in range(n)]
        comp = [probe.warm("c") for _ in range(n)]
        result.update(t0=t0, t1=time.perf_counter(), triad=statistics.median(tri), compute=statistics.median(comp))
        done.set()

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    stamps = decode_until(engine, done.is_set, Fixed(mode, cap))
    th.join()
    slow_t = 1.0 - result["triad"] / idle_t
    slow_c = 1.0 - result["compute"] / idle_c
    return {
        "mode": mode,
        "rate_cap": cap,
        "tps": round(window_tps(stamps, result["t0"], result["t1"]), 2),
        "idle_triad_gbps": round(idle_t, 2),
        "triad_gbps": round(result["triad"], 2),
        "slow_triad": round(slow_t, 4),
        "slow_compute": round(slow_c, 4),
    }


def merge(cells: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Median across rounds for cells that share `keys`."""
    groups: dict = {}
    for c in cells:
        groups.setdefault(tuple(c[k] for k in keys), []).append(c)
    out = []
    for group in groups.values():
        merged = dict(group[0])
        for field in group[0]:
            if isinstance(group[0][field], float):
                merged[field] = round(statistics.median(c[field] for c in group), 4)
        merged["n"] = len(group)
        out.append(merged)
    return out


def sync_threshold(loop_cells: list[dict]) -> tuple[float, str]:
    """Lowest CPU duty at which the pipelined loop is no longer meaningfully faster."""
    cells = sorted(loop_cells, key=lambda c: c["gov_duty"])
    prev = None
    for c in cells:
        ratio = c["tps_ahead1"] / c["tps_sync"] if c["tps_sync"] else 99.0
        if ratio < AHEAD_MARGIN:
            if prev is None:
                return c["gov_duty"], "pipelined loop is not faster even with no load"
            p_ratio = prev["tps_ahead1"] / prev["tps_sync"]
            frac = (p_ratio - AHEAD_MARGIN) / (p_ratio - ratio) if p_ratio != ratio else 0.0
            return round(prev["gov_duty"] + frac * (c["gov_duty"] - prev["gov_duty"]), 3), "interpolated crossover"
        prev = c
    return 1.01, "pipelined loop stayed faster at every measured load; synchronous loop never selected"


def run(cfg: Config, rounds: int = 3, quick: bool = False, log=print) -> dict:
    engine = Engine(cfg, load_draft=False)
    gov = Governor(cfg)
    binary = build(cfg.state_dir)
    n = 2 if quick else 4
    rounds = 1 if quick else rounds

    log("warming up…")
    timed_decode(engine, Fixed("ahead1"), 3.0)
    free_tps = timed_decode(engine, Fixed("ahead1"), 4.0)
    log(f"unloaded decode: {free_tps:.1f} tok/s")

    rate_cells: list[dict] = []
    with CpuProbe(binary) as probe:
        plan = [("ahead1", None), ("sync", None)] + [("ahead1", round(free_tps * f, 1)) for f in RATE_FRACTIONS]
        for r in range(rounds):
            for mode, cap in plan if r % 2 == 0 else reversed(plan):
                time.sleep(2.0)  # let the GPU go idle before the next idle reference
                cell = rate_cell(engine, probe, mode, cap, n)
                rate_cells.append(cell)
                log(
                    f"  rate {mode:6s} cap={cap or 'none':>6} -> {cell['tps']:6.1f} tok/s, "
                    f"CPU slowdown triad {cell['slow_triad']:+.1%} compute {cell['slow_compute']:+.1%}"
                )
        idle = {"triad_gbps": round(probe.median("t", 3), 2), "compute": round(probe.median("c", 3), 4)}

    loop_cells: list[dict] = []
    threads = max(1, gov.n_perf // 2)
    seconds = 4.0 if quick else 6.0
    for r in range(rounds):
        for duty in LOAD_DUTIES:
            load = StreamLoad(binary, threads, duty) if duty > 0 else None
            try:
                time.sleep(1.0)
                gov.sample_once()
                order = ("ahead1", "sync") if r % 2 == 0 else ("sync", "ahead1")
                tps = {m: timed_decode(engine, Fixed(m), seconds) for m in order}
                gov_duty = gov.sample_once()
                gbps = load.achieved_gbps() if load else 0.0
            finally:
                if load:
                    load.close()
            cell = {
                "load_duty": duty,
                "load_threads": threads,
                "load_gbps": round(gbps or 0.0, 1),
                "gov_duty": round(gov_duty, 3),
                "tps_ahead1": round(tps["ahead1"], 2),
                "tps_sync": round(tps["sync"], 2),
            }
            loop_cells.append(cell)
            log(
                f"  loop load duty {duty:.1f} (cpu duty {gov_duty:.2f}, {cell['load_gbps']} GB/s): "
                f"ahead1 {cell['tps_ahead1']:.1f} vs sync {cell['tps_sync']:.1f} tok/s"
            )

    rate_cells = merge(rate_cells, ("mode", "rate_cap"))
    loop_cells = merge(loop_cells, ("load_duty",))
    threshold, note = sync_threshold(loop_cells)
    result = {
        "model": cfg.model.main,
        "device": engine.device_name,
        "created": now_iso(),
        "free_tps": round(free_tps, 2),
        "idle": idle,
        "rate_cells": rate_cells,
        "loop_cells": loop_cells,
        "sync_threshold": threshold,
        "sync_threshold_note": note,
    }
    write_json(cfg.state_dir / "calibration.json", result)
    write_report(result, cfg)
    log(f"sync threshold: cpu duty {threshold} ({note})")
    return result


def write_report(r: dict, cfg: Config) -> None:
    g = cfg.governor
    lines = [
        f"# Calibration: how {r['model']} decode slows CPU work",
        "",
        f"Measured {r['created']} on {r['device']}. Regenerate with `mack calibrate` (close other apps first).",
        "",
        f"Unloaded decode: {r['free_tps']} tok/s. Idle CPU probe: {r['idle']['triad_gbps']} GB/s triad.",
        "",
        "## Rate cells: CPU slowdown at each token rate",
        "",
        "Triad = memory-bandwidth-bound work (worst case). Compute = cache-resident work.",
        "",
        "| loop | rate cap | achieved tok/s | triad GB/s | triad slowdown | compute slowdown | runs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(r["rate_cells"], key=lambda c: (c["mode"], -c["tps"])):
        lines.append(
            f"| {c['mode']} | {c['rate_cap'] or 'none'} | {c['tps']} | {c['triad_gbps']} | "
            f"{c['slow_triad']:.1%} | {c['slow_compute']:.1%} | {c['n']} |"
        )
    lines += ["", "## Rate cap this implies", "", "| target | ruler | cap |", "|---:|---|---|"]
    for ruler in ("triad", "compute"):
        for target in (0.05, 0.10, 0.20, 0.30):
            cap, note = rate_for_target(ruler_cells(r, ruler), target, g.min_tps)
            lines.append(f"| {target:.0%} | {ruler} | {f'{cap:.1f} tok/s' if cap else 'none'} ({note}) |")
    lines += [
        "",
        "## Loop cells: pipelined vs synchronous under a stream load",
        "",
        "| load duty | load GB/s | other CPU duty | ahead1 tok/s | sync tok/s | ahead1 / sync | runs |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(r["loop_cells"], key=lambda c: c["load_duty"]):
        ratio = c["tps_ahead1"] / c["tps_sync"] if c["tps_sync"] else 0.0
        lines.append(
            f"| {c['load_duty']} | {c['load_gbps']} | {c['gov_duty']} | {c['tps_ahead1']} | {c['tps_sync']} | "
            f"{ratio:.2f} | {c['n']} |"
        )
    lines += ["", f"Synchronous-loop threshold: other CPU duty {r['sync_threshold']} ({r['sync_threshold_note']}).", ""]
    path = ROOT / "reports" / "calibration.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
