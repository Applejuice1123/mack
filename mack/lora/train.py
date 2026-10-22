"""`mack lora train`: train the behaviour adapter with mlx-lm's trainer.

Prompts are rendered with Mack's runtime format and masked out of the loss, so
the adapter only learns the replies. Sized for a 16 GB machine: batch 1, gradient
checkpointing, LoRA on the last layers only, and logits computed only at reply
positions (the vocabulary is 248K wide, so full-sequence logits alone are
~0.7 GB per copy for a 700-token example).

Training is the one thing in Mack that can push the machine into swap, and swap
lands on the disk. The first attempt on the 4B did exactly that: 4 GB of swap in
five minutes with 3 GB of disk free. So there is a preflight check and a watchdog
that aborts the run rather than let it fill the disk.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import psutil
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from ..config import Config
from ..prompts import detect_format
from . import dataset

LORA = {"rank": 8, "scale": 16.0, "dropout": 0.05}
# Backprop has to keep every recurrent layer's per-token state between the first
# adapted layer and the output, so memory grows with this number. Measured on the
# 0.8B at ~600-token examples: 8 layers peaked at 4.8 GB.
NUM_LAYERS = 4
MAX_SEQ = 1536
MAX_REPLY = 128  # reply positions the loss looks at per example
MIN_FREE_DISK_GB = 6.0
MIN_AVAILABLE_RAM_GB = 8.0
MAX_SWAP_GROWTH_GB = 0.75


def reply_loss(model, batch, lengths):
    """Cross-entropy on the reply tokens, projecting only those positions to the vocabulary."""
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "model"):
        return default_loss(model, batch, lengths)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    hidden = lm.model(inputs)
    rows = mx.arange(inputs.shape[0])[:, None]
    pos = lengths[:, 0:1] - 1 + mx.arange(MAX_REPLY)[None, :]  # target index of each reply token
    valid = pos < lengths[:, 1:] - 1  # targets are one shorter than the sequence
    pos = mx.minimum(pos, inputs.shape[1] - 1)
    h = hidden[rows, pos]
    logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    ce = nn.losses.cross_entropy(logits, targets[rows, pos]) * valid
    ntoks = valid.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def preflight(min_free_disk_gb: float, min_available_ram_gb: float) -> None:
    free_disk = shutil.disk_usage(Path.home()).free / 1e9
    available = psutil.virtual_memory().available / 1e9
    problems = []
    if free_disk < min_free_disk_gb:
        problems.append(f"{free_disk:.1f} GB of disk free, need {min_free_disk_gb:g} GB (swap grows onto the disk)")
    if available < min_available_ram_gb:
        problems.append(f"{available:.1f} GB of RAM available, need {min_available_ram_gb:g} GB (close other apps)")
    if problems:
        raise SystemExit("not safe to train right now: " + "; ".join(problems))


def start_watchdog(max_swap_growth_gb: float = MAX_SWAP_GROWTH_GB) -> None:
    """Kill this process if training starts paging. Checked every second on its own thread,
    because the damage happens inside a single training step, not between reports."""
    swap0 = psutil.swap_memory().used
    disk0 = shutil.disk_usage(Path.home()).free

    def watch():
        while True:
            threading.Event().wait(1.0)
            swap = (psutil.swap_memory().used - swap0) / 1e9
            disk = (disk0 - shutil.disk_usage(Path.home()).free) / 1e9
            if swap > max_swap_growth_gb or disk > max_swap_growth_gb:
                print(
                    f"\nABORTED: swap grew {swap:.2f} GB and free disk fell {disk:.2f} GB since training started. "
                    "The model does not fit in memory for training right now.",
                    flush=True,
                )
                os._exit(3)

    threading.Thread(target=watch, name="mack-train-watchdog", daemon=True).start()


class Rendered:
    """Items are (tokens, prompt_length): loss is taken on the reply tokens only."""

    def __init__(self, cases: list[dict], tokenizer, fmt):
        self.items = []
        for case in cases:
            prompt = tokenizer.encode(dataset.render_prompt(fmt, case), add_special_tokens=False)
            reply = tokenizer.encode(case["reply"] + fmt.close, add_special_tokens=False)
            if len(prompt) + len(reply) <= MAX_SEQ:
                self.items.append((prompt + reply, len(prompt)))

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)


def run(
    cfg: Config,
    iters: int = 300,
    learning_rate: float = 1e-4,
    num_layers: int = NUM_LAYERS,
    min_free_disk_gb: float = MIN_FREE_DISK_GB,
    min_available_ram_gb: float = MIN_AVAILABLE_RAM_GB,
) -> int:
    adapter_dir = cfg.adapter_dir
    if adapter_dir is None:
        raise SystemExit("model.adapter is not set in mack.toml")
    preflight(min_free_disk_gb, min_available_ram_gb)
    start_watchdog()
    if not (dataset.DATA_DIR / "train.jsonl").exists():
        dataset.build(cfg)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # A new adapter has not passed the gate yet, whatever the previous one did.
    (adapter_dir / "SHIP.json").unlink(missing_ok=True)

    model, tokenizer = load(cfg.model.main)
    fmt = detect_format(tokenizer, cfg.generation.enable_thinking)
    train_set = Rendered(dataset.read("train"), tokenizer, fmt)
    valid_set = Rendered(dataset.read("valid"), tokenizer, fmt)
    print(f"{len(train_set)} train / {len(valid_set)} valid examples, {iters} iterations")

    model.freeze()
    linear_to_lora_layers(model, num_layers, LORA)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": LORA,
                "model": cfg.model.main,
                "iters": iters,
                "learning_rate": learning_rate,
            },
            indent=2,
        )
    )
    mx.random.seed(0)
    model.train()
    train(
        model,
        optim.AdamW(learning_rate=learning_rate),
        train_set,
        valid_set,
        TrainingArgs(
            batch_size=1,
            iters=iters,
            val_batches=min(20, len(valid_set)),
            steps_per_report=20,
            steps_per_eval=100,
            steps_per_save=iters,  # one adapter file, no intermediate checkpoints (disk is tight)
            max_seq_length=MAX_SEQ,
            adapter_file=str(adapter_dir / "adapters.safetensors"),
            grad_checkpoint=True,
        ),
        loss=reply_loss,
    )
    for checkpoint in adapter_dir.glob("0*_adapters.safetensors"):
        checkpoint.unlink()  # the trainer's numbered copy of the final weights
    print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"adapter written to {adapter_dir}. It is not used until `mack lora eval` passes the gate.")
    return 0
