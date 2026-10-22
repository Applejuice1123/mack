"""Summary verification: a summary line is kept only if the original supports it.

The check is deliberately biased toward dropping. A dropped line costs nothing,
because the original turn is still in the archive and can be recalled; a kept
line that is wrong becomes a "memory" the model will repeat with confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .textutil import content_words, entities, numbers, specials, split_sentences

MIN_COVERAGE = 0.6
_META = re.compile(
    r"^(?:here (?:is|are)|summary|the (?:conversation|excerpt|user and)|in summary|overall|bullet points?)\b.*:$",
    re.I,
)


@dataclass
class Verdict:
    line: str
    kept: bool
    reason: str = ""


def check_line(line: str, source: str, source_lower: Optional[str] = None) -> Verdict:
    source_lower = source_lower if source_lower is not None else source.lower()
    if _META.match(line.strip()):
        return Verdict(line, False, "meta")
    missing_nums = numbers(line) - numbers(source)
    if missing_nums:
        return Verdict(line, False, f"number not in original: {sorted(missing_nums)[0]}")
    for item in specials(line) | entities(line):
        if item.lower() not in source_lower:
            return Verdict(line, False, f"name not in original: {item}")
    words = content_words(line)
    if not words:
        return Verdict(line, False, "no content")
    covered = len(words & content_words(source)) / len(words)
    if covered < MIN_COVERAGE:
        return Verdict(line, False, f"only {covered:.0%} of its words appear in the original")
    return Verdict(line, True)


def verify_summary(summary: str, source: str, engine=None, threshold: float = 0.5) -> list[Verdict]:
    """Check each line of a model-written summary against the text it summarizes.

    With `engine`, lines that pass the lexical check also need the model to judge
    them supported (one extra prefill per line, so off by default).
    """
    source_lower = source.lower()
    verdicts = []
    for line in split_sentences(summary):
        v = check_line(line, source, source_lower)
        if v.kept and engine is not None:
            p = engine.yes_probability(
                [
                    {
                        "role": "user",
                        "content": f"Text:\n{source}\n\nClaim: {line}\n\n"
                        "Is the claim fully supported by the text? Answer Yes or No.",
                    }
                ]
            )
            if p < threshold:
                v = Verdict(line, False, f"model judged unsupported (p={p:.2f})")
        verdicts.append(v)
    return verdicts
