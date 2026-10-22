"""Explainer mode: text captured from a region of the screen goes in, a short
explanation comes out. One shot, no conversation and no memory: every request
gets a fresh context, so nothing here touches compaction or the archive.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Iterator, Optional

from .textutil import clip_middle

MAX_INPUT_CHARS = 12_000  # about 3,000 tokens of captured text
DEFAULT_MAX_TOKENS = 450

SYSTEM = """You are Mack, a helper that runs locally on this Mac. The user dragged a box over part of their screen and the text inside it was captured. Your job is to help them understand it.

How to answer:
- Start with the explanation itself. No greeting, no "Sure", no restating the request, no closing question or offer.
- First say in one or two sentences what this is and what it means. Then give the points the reader needs, as short bullets if there are several.
- Put jargon, symbols and abbreviations into plain words. For code, say what it does. For an error, say what went wrong and the likely fix. For math, walk through the steps. For a dense paragraph, say what it is claiming.
- The text was read off the screen, so it may contain stray characters, menu labels or broken line wraps. Work around them silently.
- Use the text plus general knowledge. Do not invent details the text does not support. If it is too fragmentary to explain, say what is missing.
- Keep it short. It is read in a small panel beside the user's work."""

STYLES = {
    "explain": "Explain this.",
    "simpler": "Explain this as simply as you can, for someone new to the subject. Short sentences, everyday words, one analogy if it helps.",
    "detail": "Explain this in more depth: the reasoning behind it, how the parts relate, and anything a careful reader should watch for.",
    "summary": "Summarize this in at most three short bullets.",
}


def clean(text: str) -> str:
    """Tidy captured text: trim lines, drop blank runs, rejoin words split by a line-end hyphen."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.replace("\r", "\n").split("\n")]
    out = "\n".join(ln for ln in lines if ln)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", out)


def build_messages(text: str, style: str = "explain", question: Optional[str] = None) -> tuple[str, str, bool]:
    """Returns (system, user, clipped)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
    text = clean(text)
    clipped = len(text) > MAX_INPUT_CHARS
    text = clip_middle(text, MAX_INPUT_CHARS, "\n[… the middle of the selection was left out …]\n")
    ask = f"Answer this question about it: {question.strip()}" if question and question.strip() else STYLES[style]
    return SYSTEM, f'Text from the screen:\n"""\n{text}\n"""\n\n{ask}', clipped


def explain(
    engine,
    text: str,
    style: str = "explain",
    question: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    policy=None,
    cancel: Optional[Callable[[], bool]] = None,
) -> Iterator[dict]:
    """Yields {"type": "token", "text"} then {"type": "done", ...}."""
    from .engine import GenStats, strip_thinking

    if not clean(text):
        yield {"type": "error", "message": "There was no text in that selection."}
        return
    system, user, clipped = build_messages(text, style, question)
    fmt = engine.fmt
    prompt = engine.encode(fmt.system(system) + fmt.user(user) + fmt.generation_prompt())
    ctx = engine.new_context()
    stats = GenStats()
    detok = engine.detok()
    pieces: list[str] = []
    t0 = time.perf_counter()
    for tok in engine.generate(ctx, prompt, max_tokens, policy=policy, cancel=cancel, stats=stats):
        seg = detok.add(tok)
        if seg:
            pieces.append(seg)
            yield {"type": "token", "text": seg}
    seg = detok.flush()
    if seg:
        pieces.append(seg)
        yield {"type": "token", "text": seg}
    yield {
        "type": "done",
        "reply": strip_thinking("".join(pieces)).strip(),
        "input_tokens": len(prompt),
        "clipped": clipped,
        "seconds": round(time.perf_counter() - t0, 2),
        **stats.as_dict(),
    }
