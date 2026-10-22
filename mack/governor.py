"""Contention governor: decides, per token, how hard Mack may drive the machine.

Inputs
  * calibration.json  measured by `mack calibrate` on this machine and model:
      - how much each token rate slows CPU work (the direction the earlier
        research did not measure: decode -> CPU, not CPU -> decode)
      - the CPU duty past which the pipelined loop stops beating the synchronous one
  * live CPU duty of everything that is not Mack

Rules
  * Rate: the highest token rate whose measured CPU slowdown is within the target.
  * Loop: one token in flight normally; synchronous under heavy CPU load, where
    pipelining no longer buys speed and still costs energy (odi2027 N28).
  * Speculative decoding only when unthrottled, quiet, and Phase 0 measured a gain.
  * Never background priority: it puts the submitting thread on the E-cluster and
    makes decode worse (odi2027 N12, N22-N24).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

from .config import Config
from .engine import Decision, GenStats

PRIO_DARWIN_PROCESS = 4
QOS_CLASS_USER_INITIATED = 0x19
SPEC_MIN_SPEEDUP = 1.05


def perf_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip())
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def ensure_foreground() -> dict:
    """Clear Darwin background state on this process and report what was found."""
    report = {"was_background": False, "nice": os.nice(0)}
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.getpriority(PRIO_DARWIN_PROCESS, 0) == 1:
            report["was_background"] = True
            libc.setpriority(PRIO_DARWIN_PROCESS, 0, 0)
            report["cleared"] = libc.getpriority(PRIO_DARWIN_PROCESS, 0) == 0
    except Exception as e:  # not Darwin, or libc lookup failed
        report["error"] = str(e)
    return report


def set_thread_user_initiated() -> bool:
    """Pin the calling thread's QoS to user-initiated so it is never demoted to
    utility/background (which would move it to the efficiency cores)."""
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("pthread") or ctypes.util.find_library("c"))
        return lib.pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) == 0
    except Exception:
        return False


def load_json(path: Path, model: str) -> Optional[dict]:
    """A measurement file is only valid for the model it was taken on."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("model") == model else None


def ruler_cells(calibration: dict, ruler: str) -> list[dict]:
    """Pipelined-loop calibration points as {"tps", "slowdown"} for the chosen ruler."""
    if ruler not in ("triad", "compute"):
        raise ValueError(f"governor.ruler must be triad or compute, not {ruler!r}")
    return [
        {"tps": c["tps"], "slowdown": c[f"slow_{ruler}"]}
        for c in calibration.get("rate_cells", [])
        if c["mode"] == "ahead1"
    ]


def rate_for_target(cells: list[dict], target: float, floor: float = 0.0) -> tuple[Optional[float], str]:
    """Highest token rate whose CPU slowdown stays within `target`.

    `cells` are calibration points for the pipelined loop: {"tps", "slowdown"}.
    The curve is anchored at (0 tok/s, no slowdown) and never returns less than
    `floor`. Returns (rate cap or None for unthrottled, note).
    """
    pts = sorted((c["tps"], max(0.0, c["slowdown"])) for c in cells)
    if not pts:
        return None, "uncalibrated"
    # More tokens per second cannot cost less; flatten measurement noise.
    mono, worst = [(0.0, 0.0)], 0.0
    for tps, slow in pts:
        worst = max(worst, slow)
        mono.append((tps, worst))
    if mono[-1][1] <= target:
        return None, "full speed is within target"
    for (t0, s0), (t1, s1) in zip(mono, mono[1:]):
        if s0 <= target < s1:
            rate = t0 + (t1 - t0) * (target - s0) / (s1 - s0)
            if rate < floor:
                return floor, f"target needs {rate:.1f} tok/s; held at the {floor:g} tok/s floor"
            return rate, "interpolated"
    return max(floor, mono[1][0]), "fallback"


class Governor:
    def __init__(self, cfg: Config, spec_supported: bool = False):
        self.cfg = cfg
        g = cfg.governor
        state = cfg.state_dir
        self.calibration = load_json(state / "calibration.json", cfg.model.main)
        self.phase0 = load_json(state / "phase0.json", cfg.model.main)
        self.n_perf = perf_cores()
        self.sync_duty = g.sync_duty
        self.rate_cap: Optional[float] = None
        self.rate_note = "uncalibrated: no rate cap applied"
        if self.calibration:
            if self.calibration.get("sync_threshold") is not None:
                self.sync_duty = self.calibration["sync_threshold"]
            self.rate_cap, self.rate_note = rate_for_target(
                ruler_cells(self.calibration, g.ruler), g.cpu_slowdown_target, g.min_tps
            )
        spec = cfg.generation.speculative
        measured = (self.phase0 or {}).get("speculative", {}).get("speedup")
        self.spec_enabled = spec_supported and (
            spec == "on" or (spec == "auto" and measured is not None and measured >= SPEC_MIN_SPEEDUP)
        )
        self.duty = 0.0
        self.last_tps = 0.0
        # Cap of the background job currently running (tasks pace themselves at it).
        # Separate from chat's rate cap above, which a job never changes.
        self.job_cap: Optional[float] = None
        self._pid = os.getpid()
        self._last: tuple[float, dict[int, float]] = (0.0, {})
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.priority = ensure_foreground()

    # ---- live CPU duty -----------------------------------------------------

    def start(self) -> None:
        if self._thread:
            return
        self.sample_once()
        self._thread = threading.Thread(target=self._sample_loop, name="mack-governor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _cpu_seconds(self) -> dict[int, float]:
        """CPU seconds used so far by every process we are allowed to read, except Mack."""
        out = {}
        for p in psutil.process_iter():
            if p.pid == self._pid:
                continue
            try:
                t = p.cpu_times()
                out[p.pid] = t.user + t.system
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return out

    def sample_once(self) -> float:
        """CPU duty of the user's other processes since the last call, in units of
        performance cores (1.0 = all of them busy).

        Summed per process rather than read from the system-wide counter: the
        system total also contains the kernel's own work while Mack decodes and
        thermal management after sustained load, which read as 0.5 "other" duty
        with nothing else running.
        """
        now, snap = time.monotonic(), self._cpu_seconds()
        prev_t, prev = self._last
        self._last = (now, snap)
        dt = now - prev_t
        if not prev or dt <= 0:
            return 0.0
        used = sum(max(0.0, snap[pid] - prev[pid]) for pid in snap if pid in prev)
        return min(1.0, used / dt / self.n_perf)

    def _sample_loop(self) -> None:
        interval = self.cfg.governor.sample_interval_s
        alpha = min(1.0, interval / 2.0)  # ~2 s smoothing so the loop does not flap
        while not self._stop.wait(interval):
            self.duty += alpha * (self.sample_once() - self.duty)

    # ---- policy ------------------------------------------------------------

    def policy(self) -> "Governor":
        return self

    def decide(self) -> Decision:
        g = self.cfg.governor
        duty = self.duty
        capped = g.protect == "always" or (g.protect == "when_busy" and duty >= g.busy_duty)
        cap = self.rate_cap if capped else None
        if duty >= self.sync_duty:
            mode = "sync"
        elif self.spec_enabled and cap is None and duty < g.busy_duty:
            mode = "spec"
        else:
            mode = "ahead1"
        return Decision(mode=mode, rate_cap=cap)

    def set_job_cap(self, rate: Optional[float]) -> Optional[float]:
        """Install a job's cap; returns the previous one so it can be restored."""
        previous, self.job_cap = self.job_cap, rate
        return previous

    def restore_job_cap(self, previous: Optional[float]) -> None:
        self.job_cap = previous

    def observe(self, stats: GenStats) -> None:
        self.last_tps = stats.tps

    def status(self) -> dict:
        d = self.decide()
        return {
            "cpu_duty": round(self.duty, 3),
            "mode": d.mode,
            "rate_cap": round(d.rate_cap, 1) if d.rate_cap else None,
            "calibrated_rate_cap": round(self.rate_cap, 1) if self.rate_cap else None,
            "rate_note": self.rate_note,
            "cpu_slowdown_target": self.cfg.governor.cpu_slowdown_target,
            "ruler": self.cfg.governor.ruler,
            "sync_duty": round(self.sync_duty, 3),
            "calibrated": self.calibration is not None,
            "speculative": self.spec_enabled,
            "job_cap": round(self.job_cap, 2) if self.job_cap else None,
            "last_tps": round(self.last_tps, 1),
            "priority": self.priority,
        }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
