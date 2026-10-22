"""Configuration: dataclass defaults, overridden by mack.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
# Where config, state, outputs and your recipes live: the source checkout when run from
# one, otherwise ~/MACK (an installed package must not write into site-packages).
# MACK_HOME overrides both.
BASE = Path(os.environ["MACK_HOME"]).expanduser() if os.environ.get("MACK_HOME") else (
    ROOT if (ROOT / "mack.toml").exists() else Path.home() / "MACK"
)


@dataclass
class ModelCfg:
    main: str = "mlx-community/Qwen3.5-4B-4bit"
    draft: Optional[str] = "mlx-community/Qwen3.5-0.8B-4bit"
    adapter: Optional[str] = "adapters/mack"


@dataclass
class GenerationCfg:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    enable_thinking: bool = False
    speculative: str = "auto"  # auto | on | off
    num_draft_tokens: int = 3


@dataclass
class ContextCfg:
    budget_tokens: int = 8192
    high_water: float = 0.85
    recent_fraction: float = 0.25
    pin_fraction: float = 0.10
    summary_fraction: float = 0.08
    prefill_step: int = 512
    # There is no kv_bits option on purpose: the KV cache stays fp16. Quantizing
    # it raised peak memory during prefill on this stack (contention_harness F28).
    verify_with_model: bool = False


@dataclass
class RecallCfg:
    enabled: bool = True
    max_tokens: int = 600
    top_k: int = 3


@dataclass
class GovernorCfg:
    cpu_slowdown_target: float = 0.20
    min_tps: float = 4.0  # the rate cap never goes below this
    protect: str = "when_busy"  # when_busy | always | never
    busy_duty: float = 0.25
    sync_duty: float = 0.60
    sample_interval_s: float = 1.0
    ruler: str = "triad"  # triad | compute


@dataclass
class BatchCfg:
    max_file_kb: int = 200  # larger files are skipped at enqueue
    max_input_tokens: int = 6000  # longer files are shown to the model head + tail
    max_output_tokens: int = 512
    item_rest_s: float = 0.5  # rest between items at a capped pace
    battery_pause_below: int = 25  # percent, on battery power
    battery_resume_at: int = 30


@dataclass
class BackgroundCfg:
    # The auto pace (mack/pacing.py).
    away_after_s: float = 60.0  # no keyboard/mouse this long, and little else running → full speed
    away_max_duty: float = 0.15  # "little else running": other apps below this CPU duty
    busy_duty: float = 0.35  # other apps at or above this duty (~1.4 busy cores) → gentle
    hold_duty: float = 0.90  # at or above this (most performance cores busy) → pause
    active_chunk: int = 128  # prompt-reading chunk while you are present (shorter GPU bursts)
    # Average extra power Mack may draw at each auto level (heat on a fanless Mac).
    active_watts: float = 3.0  # about what one busy browser tab draws
    busy_watts: float = 1.5
    away_battery_watts: float = 6.0  # away on battery; away on power is uncapped
    unload_after_idle_s: float = 300.0  # free the model's memory after this long with nothing to do
    load_headroom_gb: float = 2.0  # only load the model if this much memory stays available afterwards


@dataclass
class PaceCfg:
    # Optional overrides of the calibrated rates (tok/s). Phase 4: if `mack bench`
    # shows a pace breaking its contract, lower its rate here and say why in FINDINGS.md.
    gentle_tps: Optional[float] = None
    balanced_tps: Optional[float] = None
    steady_tps: Optional[float] = None
    # Average extra power for the fixed paces (watts; full is uncapped).
    gentle_watts: float = 1.5
    balanced_watts: float = 3.0
    steady_watts: float = 6.0
    # The published contract: max slowdown of memory-heavy CPU work, per pace.
    gentle_max_slowdown: float = 0.15
    balanced_max_slowdown: float = 0.25
    steady_max_slowdown: float = 0.30


@dataclass
class PathsCfg:
    state: str = "state"
    output: str = "output"  # batch outputs, review files and undo logs
    recipes: str = "recipes"  # your recipes; they shadow the built-ins


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    context: ContextCfg = field(default_factory=ContextCfg)
    recall: RecallCfg = field(default_factory=RecallCfg)
    governor: GovernorCfg = field(default_factory=GovernorCfg)
    batch: BatchCfg = field(default_factory=BatchCfg)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    pace: PaceCfg = field(default_factory=PaceCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    @property
    def state_dir(self) -> Path:
        p = Path(self.paths.state).expanduser()
        p = p if p.is_absolute() else BASE / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir(self) -> Path:
        p = Path(self.paths.output).expanduser()
        return p if p.is_absolute() else BASE / p

    @property
    def recipes_dir(self) -> Path:
        p = Path(self.paths.recipes).expanduser()
        return p if p.is_absolute() else BASE / p

    @property
    def adapter_dir(self) -> Optional[Path]:
        if not self.model.adapter:
            return None
        p = Path(self.model.adapter)
        return p if p.is_absolute() else BASE / p


def _apply(obj, data: dict, where: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key {where}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg = Config()
    path = Path(path or os.environ.get("MACK_CONFIG") or BASE / "mack.toml")
    if path.exists():
        with open(path, "rb") as f:
            _apply(cfg, tomllib.load(f), "")
    return cfg
