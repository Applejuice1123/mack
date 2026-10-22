"""Package power and thermal pressure from `powermetrics`.

powermetrics needs root. Mack only ever runs it as `sudo -n /usr/bin/powermetrics`
(non-interactive), which works when the user has added a sudoers rule for that one
binary; otherwise power is reported as unavailable and nothing else changes.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Optional

CMD = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power,gpu_power,thermal"]
_POWER = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW")
_THERMAL = re.compile(r"Current pressure level:\s*(\w+)")
THERMAL_LEVELS = ["Nominal", "Fair", "Serious", "Critical", "Trapping", "Sleeping"]


def available() -> bool:
    try:
        r = subprocess.run(CMD + ["-n", "1", "-i", "200"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(_POWER.search(r.stdout))
    except Exception:
        return False


class PowerMeter:
    """Streams samples in the background; ask for the mean over any time window."""

    def __init__(self, interval_ms: int = 1000):
        self.watts: list[tuple[float, float]] = []
        self.thermal: list[tuple[float, str]] = []
        self._p = subprocess.Popen(CMD + ["-i", str(interval_ms)], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._p.stdout:
            now = time.monotonic()
            if m := _POWER.search(line):
                self.watts.append((now, int(m.group(1)) / 1000.0))
            elif m := _THERMAL.search(line):
                self.thermal.append((now, m.group(1)))

    def mean(self, t0: float, t1: float) -> Optional[float]:
        vals = [w for t, w in self.watts if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else None

    def worst_thermal(self, t0: float, t1: float) -> Optional[str]:
        levels = [lv for t, lv in self.thermal if t0 <= t <= t1]
        return max(levels, key=lambda lv: THERMAL_LEVELS.index(lv) if lv in THERMAL_LEVELS else 0) if levels else None

    def close(self) -> None:
        self._p.terminate()
        try:
            self._p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Measured 2026-10-22 on this Mac (Qwen3.5-4B-4bit, `mack calibrate --energy` refreshes):
# continuous prompt reading +17.8 W at 310 tok/s, continuous generation +7.0 W at ~34 tok/s.
DEFAULT_ENERGY = {"prefill_j_per_token": 0.057, "decode_j_per_token": 0.205}


def energy(cfg) -> dict:
    """Joules per prompt token and per generated token for the configured model."""
    import json

    path = cfg.state_dir / "energy.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("model") == cfg.model.main:
            return data
    return dict(DEFAULT_ENERGY)


def measure_energy(cfg, engine, seconds: float = 12.0, log=print) -> dict:
    """Added package power during continuous prompt reading and continuous generation,
    divided by their token rates. Needs powermetrics through sudo."""
    import json

    from .calibrate import Fixed
    from .governor import now_iso

    toks = engine.encode("The harbour town kept its ledgers in three languages and argued about salt. " * 200)[:2048]

    def prefill_once():
        ctx = engine.new_context()
        for i in range(0, 2048, 128):
            engine.prefill(ctx, toks[i : i + 128])
        return 2048

    def decode_once():
        ctx = engine.new_context()
        return sum(1 for _ in engine.generate(ctx, toks[:64], 96, policy=Fixed("sync"), sampler=engine.greedy))

    with PowerMeter(500) as meter:
        def window(fn):
            t0 = time.monotonic()
            n = 0
            while time.monotonic() - t0 < seconds:
                n += fn() if fn else (time.sleep(0.5) or 0)
            t1 = time.monotonic()
            return meter.mean(t0 + 1, t1), n / (t1 - t0)

        prefill_once(), decode_once()  # warm up
        idle_w, _ = window(None)
        pre_w, pre_tps = window(prefill_once)
        idle2_w, _ = window(None)
        dec_w, dec_tps = window(decode_once)
    idle = (idle_w + idle2_w) / 2
    result = {
        "model": cfg.model.main, "created": now_iso(), "idle_watts": round(idle, 2),
        "prefill_watts": round(pre_w - idle, 2), "prefill_tps": round(pre_tps, 1),
        "decode_watts": round(dec_w - idle, 2), "decode_tps": round(dec_tps, 1),
        "prefill_j_per_token": round((pre_w - idle) / pre_tps, 4),
        "decode_j_per_token": round((dec_w - idle) / dec_tps, 4),
    }
    (cfg.state_dir / "energy.json").write_text(json.dumps(result, indent=2))
    log(f"prompt reading +{result['prefill_watts']} W at {result['prefill_tps']} tok/s = {result['prefill_j_per_token']} J/token; "
        f"generation +{result['decode_watts']} W at {result['decode_tps']} tok/s = {result['decode_j_per_token']} J/token")
    return result
