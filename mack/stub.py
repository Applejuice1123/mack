"""A stand-in for the MLX engine, for tests and for trying the CLI without a model.

It has the same interface the task and batch runners use. Replies come from a
`responder(prompt_text) -> str`. Set it directly in Python, or from the
environment for subprocess tests:

    MACK_STUB_ENGINE=1                         use this instead of the real engine
    MACK_STUB_RESPONDER=package.module:func    reply function (default: echo_first_words)

Time can be simulated: give it a clock with `advance(seconds)` and it advances the
clock by what the work would have taken at `tps` / `prefill_tps`, instead of
sleeping.
"""

from __future__ import annotations

import importlib
import os
import re
from types import SimpleNamespace
from typing import Callable, Optional

from .prompts import ChatFormat

FMT = ChatFormat(
    system_open="<|im_start|>system\n",
    user_open="<|im_start|>user\n",
    assistant_open="<|im_start|>assistant\n",
    close="<|im_end|>",
    sep="\n",
)
FILE_OPEN, FILE_CLOSE = "<<<FILE\n", "\nFILE>>>"


def file_text(prompt: str) -> str:
    """The file content inside a batch item prompt."""
    if FILE_OPEN in prompt:
        return prompt.split(FILE_OPEN, 1)[1].split(FILE_CLOSE, 1)[0]
    return prompt


def echo_first_words(prompt: str) -> str:
    import json

    words = file_text(prompt).split()[:12]
    return json.dumps({"text": " ".join(words)})


class _Detok:
    def __init__(self, engine):
        self.engine = engine

    def add(self, token):
        return self.engine.vocab[token]

    def flush(self):
        return ""


class StubEngine:
    adapter_loaded = False
    hybrid = True
    spec_supported = False
    draft = None
    device_name = "stub"
    wired_limit_bytes = 11_450_000_000

    def __init__(self, cfg=None, responder: Optional[Callable[[str], str]] = None, clock=None,
                 tps: float = 40.0, prefill_tps: float = 300.0, **_):
        self.cfg = cfg
        self.fmt = FMT
        self.vocab: list[str] = []
        self.ids: dict[str, int] = {}
        self.eos_id = self._id("<|im_end|>")
        self.eos_ids = {self.eos_id}
        self.responder = responder or _env_responder() or echo_first_words
        self.clock, self.tps, self.prefill_tps = clock, tps, prefill_tps
        self.calls: list[dict] = []  # what each generate() saw, for assertions

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.vocab)
            self.vocab.append(piece)
        return self.ids[piece]

    def encode(self, text: str) -> list[int]:
        return [self._id(p) for p in re.findall(r"\s*\S+\s*|\s+", text)]

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, tokens) -> str:
        return "".join(self.vocab[t] for t in tokens)

    def detok(self):
        return _Detok(self)

    def new_context(self):
        return SimpleNamespace(cache=[], tokens=[])

    def prefill(self, ctx, tokens, progress=None):
        ctx.tokens.extend(tokens)
        if self.clock is not None:
            self.clock.advance(len(tokens) / self.prefill_tps)

    def cache_bytes(self, ctx) -> int:
        return len(ctx.tokens) * 100

    def memory(self) -> dict:
        return {"active_gb": 0.0, "peak_gb": 0.0, "wired_limit_gb": 11.45}

    def render_chat(self, messages, add_generation_prompt=True):
        text = "".join(self.fmt.system(m["content"]) if m["role"] == "system" else self.fmt.user(m["content"]) for m in messages)
        return self.encode(text + self.fmt.generation_prompt())

    def generate(self, ctx, pending, max_tokens, policy=None, sampler=None, cancel=None, stats=None):
        prompt = self.decode(ctx.tokens + list(pending))
        ctx.tokens.extend(pending)
        if self.clock is not None:
            self.clock.advance(len(pending) / self.prefill_tps)
        reply = self.responder(prompt)
        out = self.encode(reply)
        truncated = len(out) > max_tokens
        out = out[:max_tokens]
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "emitted": len(out)})
        n = 0
        try:
            for tok in out:
                if cancel and cancel():
                    break
                if self.clock is not None:
                    self.clock.advance(1.0 / self.tps)
                ctx.tokens.append(tok)
                n += 1
                if stats is not None:
                    stats.n_tokens = n
                yield tok
        finally:
            ctx.tokens.append(self.eos_id)
            if stats is not None:
                stats.n_tokens = n
                stats.finish = "length" if truncated and n == len(out) else "stop"
                stats.decode_s = 0.01

    def complete(self, messages, max_tokens=256, policy=None) -> str:
        ctx = self.new_context()
        return self.decode(list(self.generate(ctx, self.render_chat(messages), max_tokens)))


def _env_responder() -> Optional[Callable[[str], str]]:
    spec = os.environ.get("MACK_STUB_RESPONDER")
    if not spec:
        return None
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def stub_enabled() -> bool:
    return os.environ.get("MACK_STUB_ENGINE", "") not in ("", "0")
