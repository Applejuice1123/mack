"""Exact-quote pins: sentences that must survive compaction word for word.

Numbers, names and decisions are the things a summary gets wrong and a user notices,
so they are never paraphrased. A pin is a verbatim sentence from the conversation
with the turn it came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .textutil import entities, numbers, specials, split_sentences

_DECISION = re.compile(
    r"\b(?:decided?|decision|agreed?|let'?s (?:go with|use|do|stick)|we(?:'ll| will| should| are going to)"
    r"|i(?:'ll| will| am going to|'m going to)|going with|go with|chose|choose|chosen|settled on|plan is"
    r"|the plan|must|never|always|do not|don'?t|make sure|from now on|instead of|rather than|deadline"
    r"|due (?:on|by)|by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next))\b",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(?:my name is|call me|i am|i'm|i live|i work|i study|i prefer|i like|i hate|i need|i want|i have"
    r"|we are|we're|we have|(?:my|our) (?:\w+ ){0,2}(?:is|are|was|were)"
    r"|remember that|note that|important|keep in mind|for reference)\b",
    re.I,
)

MAX_QUOTE_CHARS = 320
USER_WEIGHT = 2.0
ASSISTANT_WEIGHT = 0.6


@dataclass
class Pin:
    turn: int
    role: str
    quote: str
    kind: str
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def score_sentence(sentence: str, role: str) -> tuple[str, float]:
    kinds = []
    score = 0.0
    if _DECISION.search(sentence):
        kinds.append("decision")
        score += 3.0
    if role == "user" and _PERSONAL.search(sentence):
        kinds.append("fact")
        score += 2.5
    nums = numbers(sentence)
    if nums:
        kinds.append("number")
        score += 2.0 + min(len(nums), 3) * 0.25
    if specials(sentence):
        kinds.append("reference")
        score += 2.0
    names = entities(sentence)
    if names:
        kinds.append("name")
        score += 1.0 + min(len(names), 3) * 0.25
    if not kinds:
        return "", 0.0
    # What the user stated is the ground truth of the conversation. Mack's own
    # sentences can be regenerated, and its numbers are often estimates.
    score *= USER_WEIGHT if role == "user" else ASSISTANT_WEIGHT
    if sentence.rstrip().endswith("?"):
        score *= 0.4  # questions rarely carry the fact itself
    return kinds[0], score


def extract_pins(turns: Iterable) -> list[Pin]:
    """Candidate pins from turns (objects with .idx, .role, .text)."""
    pins: list[Pin] = []
    for turn in turns:
        for sent in split_sentences(turn.text):
            if len(sent) > MAX_QUOTE_CHARS or len(sent) < 8:
                continue
            kind, score = score_sentence(sent, turn.role)
            if score > 0:
                pins.append(Pin(turn=turn.idx, role=turn.role, quote=sent, kind=kind, score=score))
    return pins


def select_pins(pins: list[Pin], budget_tokens: int, count: Callable[[str], int]) -> list[Pin]:
    """Highest-scoring pins that fit the budget, returned in conversation order.
    Ties go to the more recent turn."""
    seen: set[str] = set()
    chosen: list[Pin] = []
    used = 0
    for pin in sorted(pins, key=lambda p: (-p.score, -p.turn)):
        key = pin.quote.lower()
        if key in seen:
            continue
        cost = count(pin.quote) + 8
        if used + cost > budget_tokens:
            continue
        seen.add(key)
        chosen.append(pin)
        used += cost
    return sorted(chosen, key=lambda p: p.turn)
