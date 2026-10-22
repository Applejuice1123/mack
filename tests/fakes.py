"""A stand-in engine so session logic can be tested without weights or a GPU."""

from __future__ import annotations

import re
from types import SimpleNamespace

from mack.prompts import ChatFormat

FMT = ChatFormat(
    system_open="<|im_start|>system\n",
    user_open="<|im_start|>user\n",
    assistant_open="<|im_start|>assistant\n",
    close="<|im_end|>",
    sep="\n",
)


class FakeDetok:
    def __init__(self, engine):
        self.engine = engine

    def add(self, token):
        return self.engine.vocab[token]

    def flush(self):
        return ""


class FakeEngine:
    """Tokens are whitespace-delimited pieces. Replies and summaries are scripted."""

    def __init__(self, replies=None, summary="- The user likes tea."):
        self.fmt = FMT
        self.vocab: list[str] = []
        self.ids: dict[str, int] = {}
        self.eos_id = self._id("<|im_end|>")
        self.replies = list(replies or [])
        self.summary = summary
        self.summary_prompts: list[str] = []
        self.prefills = 0

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.vocab)
            self.vocab.append(piece)
        return self.ids[piece]

    def encode(self, text: str) -> list[int]:
        return [self._id(p) for p in re.findall(r"\s*\S+\s*", text)]

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, tokens) -> str:
        return "".join(self.vocab[t] for t in tokens)

    def detok(self):
        return FakeDetok(self)

    def new_context(self):
        return SimpleNamespace(cache=[], tokens=[])

    def prefill(self, ctx, tokens, progress=None):
        self.prefills += 1
        ctx.tokens.extend(tokens)

    def cache_bytes(self, ctx) -> int:
        return len(ctx.tokens) * 100

    def generate(self, ctx, pending, max_tokens, policy=None, sampler=None, cancel=None, stats=None):
        ctx.tokens.extend(pending)
        reply = self.replies.pop(0) if self.replies else "Okay."
        out = self.encode(reply)[:max_tokens]
        for i, tok in enumerate(out):
            if cancel and cancel():
                break
            ctx.tokens.append(tok)
            if stats is not None:
                stats.n_tokens = i + 1
            yield tok
        ctx.tokens.append(self.eos_id)
        if stats is not None:
            stats.finish = "stop"
            stats.decode_s = 0.01

    def complete(self, messages, max_tokens=256, policy=None) -> str:
        self.summary_prompts.append(messages[-1]["content"])
        return self.summary(messages) if callable(self.summary) else self.summary
