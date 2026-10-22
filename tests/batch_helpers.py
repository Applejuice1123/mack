"""Shared helpers for batch tests: a fake clock, a stub-backed environment, and a runner loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mack import system
from mack.batch import BatchRunner, BatchStore, create
from mack.config import Config
from mack.stub import StubEngine, file_text
from mack.tasks import TaskStore


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += max(0.0, dt)


class Killed(BaseException):
    """Simulates the process dying: escapes every `except Exception`."""


@pytest.fixture(autouse=True)
def quiet_mac(monkeypatch):
    """Tests must not depend on whether this laptop is on battery or idle."""
    monkeypatch.setattr(system, "power", lambda: system.Power(False, 100))
    monkeypatch.setattr(system, "thermal_state", lambda: system.THERMAL_NOMINAL)
    monkeypatch.setattr(system, "input_idle_seconds", lambda: 0.0)
    monkeypatch.setattr(system, "memory_pressure", lambda: system.MEMORY_NORMAL)


def make_env(tmp_path: Path, responder=None, clock=None, **cfgkw):
    cfg = Config()
    cfg.paths.state = str(tmp_path / "state")
    cfg.batch.item_rest_s = 0.0
    for k, v in cfgkw.items():
        setattr(cfg.batch, k, v)
    store = TaskStore(cfg)
    store.output_root = tmp_path / "output"
    bstore = BatchStore(cfg)
    engine = StubEngine(cfg, responder=responder, clock=clock)
    return cfg, store, bstore, engine


def write_notes(folder: Path, n: int, text=lambda i: f"Note {i} talks about topic {i}.") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"note{i:03d}.md"
        p.write_text(text(i))
        paths.append(p)
    return paths


def run(cfg, store, bstore, engine, task, clock=None, governor=None):
    runner = BatchRunner(engine, cfg, store, store.load(task.id), bstore, governor=governor, clock=clock)
    for wait in runner.steps():
        if clock is not None:
            clock.advance(wait)
    return runner


def jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def first_word(prompt: str) -> str:
    return json.dumps({"text": file_text(prompt).split()[0]})


__all__ = ["FakeClock", "Killed", "quiet_mac", "make_env", "write_notes", "run", "jsonl", "first_word", "create", "file_text"]
