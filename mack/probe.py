"""Driver for the native CPU probe and load generator (native/mack_probe.c)."""

from __future__ import annotations

import statistics
import subprocess
from pathlib import Path
from typing import Optional

SRC = Path(__file__).resolve().parent / "native" / "mack_probe.c"


def build(state_dir: Path) -> Path:
    out = state_dir / "bin" / "mack_probe"
    if out.exists() and out.stat().st_mtime >= SRC.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["clang", "-O2", "-Wall", str(SRC), "-o", str(out), "-lpthread"], check=True)
    return out


class CpuProbe:
    """Persistent probe process. `warm(kind)` is the reading to use."""

    def __init__(self, binary: Path, burst_ms: int = 300):
        self.burst_ms = burst_ms
        self._p = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        ready = self._p.stdout.readline()
        if not ready.startswith("READY"):
            raise RuntimeError(f"probe failed to start: {ready!r}")

    def burst(self, kind: str) -> float:
        self._p.stdin.write(f"{kind} {self.burst_ms}\n")
        self._p.stdin.flush()
        return float(self._p.stdout.readline().split()[1])

    def warm(self, kind: str) -> float:
        """Two back-to-back bursts, second one returned. The first only ramps the
        core's clock; using it measures DVFS state rather than contention (N10)."""
        self.burst(kind)
        return self.burst(kind)

    def median(self, kind: str, n: int = 3) -> float:
        return statistics.median(self.warm(kind) for _ in range(n))

    def close(self) -> None:
        try:
            self._p.stdin.write("q\n")
            self._p.stdin.flush()
            self._p.wait(timeout=2)
        except Exception:
            self._p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class StreamLoad:
    """Synthetic memory-bandwidth load at a controlled duty, for calibration only."""

    def __init__(self, binary: Path, threads: int, duty: float):
        self._p = subprocess.Popen(
            [str(binary), "load", str(threads), f"{duty:.3f}"], stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def achieved_gbps(self) -> Optional[float]:
        line = self._p.stdout.readline()
        return float(line.split()[1]) if line.startswith("LOAD") else None

    def close(self) -> None:
        self._p.kill()
        self._p.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
