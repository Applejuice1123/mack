"""How fast a background job may run right now.

The default pace, `auto`, trades speed against interference by watching you:

    away     no keyboard or mouse for a minute and little else running
             → full speed (steady on battery): get the work done
    active   you are using the Mac, lightly → balanced (~10% on memory-heavy work)
    busy     your other apps are using real CPU → gentle (~5%)
    hold     heavy load, macOS reports serious heat, or memory is critical
             → no work at all until it clears

Fixed paces (gentle, balanced, steady, full) keep their rate, except that full
drops to steady on battery, a "fair" thermal state caps any pace at balanced,
and the hold conditions still pause them.

While you are present, prompt reading uses short chunks (`active_chunk`, 128 tokens)
so each full-power GPU burst lasts a fraction of a second; when you are away it
uses the normal 512-token chunks.

Signals, re-read at most once a second:
  * keyboard/mouse idle time (CoreGraphics)
  * other processes' CPU duty, from the governor's per-process sampler
  * macOS thermal state and memory pressure level
  * power source
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import system
from .config import Config

HOLD_THERMAL_S = 60.0  # a serious thermal state pauses for at least this long
DUTY = {"gentle": 0.25, "balanced": 0.40, "steady": 0.60, "full": 1.0}


@dataclass
class PaceState:
    rate: Optional[float]  # tokens/s; None = uncapped; 0.0 = hold
    level: str  # away | active | busy | hold | fixed
    note: str
    chunk: int
    unload: bool = False  # memory is critical: give the model's memory back
    duty: float = 1.0  # share of time prompt reading may keep the GPU busy
    watts: Optional[float] = None  # average extra power allowed; None = uncapped


class Pacer:
    def __init__(self, cfg: Config, pace: str, governor=None, clock=None, boost_when_idle: bool = False):
        from .batch import effective_rate  # rates come from this Mac's calibration

        self.cfg, self.pace, self.governor = cfg, pace, governor
        self.clock = clock or time.monotonic
        self.boost = boost_when_idle
        self.rates = {p: effective_rate(cfg, p) for p in ("gentle", "balanced", "steady")}
        self.rates["full"] = None
        b, pc = cfg.background, cfg.pace
        self.watts = {"gentle": pc.gentle_watts, "balanced": pc.balanced_watts, "steady": pc.steady_watts, "full": None}
        self.level_watts = {"away": None, "active": b.active_watts, "busy": b.busy_watts}
        self._checked = -1e18
        self._hold_until = -1e18
        self.state = PaceState(self.rates.get(pace), "fixed", "", cfg.context.prefill_step, duty=DUTY.get(pace, 1.0),
                               watts=self.watts.get(pace))
        self.levels: dict[str, float] = {}  # seconds spent per level, for reports

    def _duty(self) -> float:
        return self.governor.duty if self.governor else 0.0

    def current(self) -> PaceState:
        now = self.clock()
        if now - self._checked < 1.0:
            return self.state
        elapsed = min(now - self._checked, 5.0) if self._checked > -1e17 else 0.0
        self.levels[self.state.level] = self.levels.get(self.state.level, 0.0) + elapsed
        self._checked = now
        self.state = self._decide(now)
        return self.state

    def _decide(self, now: float) -> PaceState:
        b = self.cfg.background
        big, small = self.cfg.context.prefill_step, b.active_chunk
        if now < self._hold_until:  # "pause 60 seconds, then check again"
            return PaceState(0.0, "hold", "paused: the Mac is hot", small)
        thermal = system.thermal_state()
        if thermal >= system.THERMAL_SERIOUS:
            self._hold_until = now + HOLD_THERMAL_S
            return PaceState(0.0, "hold", "paused: the Mac is hot", small)
        memory = system.memory_pressure()
        power = system.power()
        duty = self._duty()
        idle = system.input_idle_seconds()
        if memory >= system.MEMORY_CRITICAL:
            return PaceState(0.0, "hold", "paused: memory is critically low", small, unload=True)
        # Hysteresis: once busy (or holding), stay until load is clearly lower, so a
        # bursty browser tab does not flip the pace every few seconds.
        prev = self.state.level
        hold_at = b.hold_duty - (0.1 if prev == "hold" else 0.0)
        busy_at = b.busy_duty - (0.1 if prev in ("busy", "hold") else 0.0)
        if duty >= hold_at:
            return PaceState(0.0, "hold", "paused: your Mac is under heavy load", small)

        away = idle >= b.away_after_s and duty < b.away_max_duty
        if self.pace == "auto":
            if away:
                level, pace = "away", ("steady" if power.on_battery else "full")
            elif duty >= busy_at:
                level, pace = "busy", "gentle"
            else:
                level, pace = "active", "balanced"
        else:
            level, pace = "fixed", self.pace
            if pace == "full" and power.on_battery:
                pace = "steady"
            elif self.boost and pace in ("gentle", "balanced") and idle >= 600 and duty < 0.2 and not power.on_battery:
                level, pace = "away", "steady"
        warm = thermal >= system.THERMAL_FAIR and pace in ("steady", "full")
        if warm:
            pace = "balanced"
        notes = []
        if level == "away":
            notes.append(f"you're away: running at {pace}")
        elif level == "active":
            notes.append("you're using the Mac: balanced")
        elif level == "busy":
            notes.append("your apps are busy: gentle")
        elif self.pace == "full" and power.on_battery:
            notes.append("on battery: full speed lowered to steady")
        if warm:
            notes.append("the Mac is warm: capped at balanced")
        chunk = big if pace == "full" else small
        if level in self.level_watts:
            watts = self.level_watts[level]
            if level == "away" and power.on_battery:
                watts = b.away_battery_watts
        else:
            watts = self.watts[pace]
        if warm and watts is not None:
            watts = min(watts, self.watts["balanced"])
        elif warm:
            watts = self.watts["balanced"]
        return PaceState(self.rates[pace], level, "; ".join(notes), chunk, duty=DUTY[pace], watts=watts)


def fixed(rate: Optional[float]) -> "Pacer":
    """A pacer pinned to one rate (tests, benchmarks)."""
    p = Pacer.__new__(Pacer)
    p.state = PaceState(rate, "fixed", "", 512, duty=1.0, watts=None)
    p.current = lambda: p.state
    p.levels = {}
    return p
