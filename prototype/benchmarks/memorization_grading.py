"""
Memorization / context-retention scoring for gemma_chat benchmark outputs.

Produces percentages and structured metrics independent of strict PASS/FAIL.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FactTrace:
    needle: str
    found: bool
    char_index: int = -1
    pct_from_start: float | None = None
    in_answer_tail: bool = False


@dataclass
class MemorizationReport:
    task_id: str
    output_chars: int
    est_tokens: float
    fact_recall_pct: float
    format_match_tier_pct: float
    length_adequacy_pct: float
    tail_placement_pct: float
    memorization_index: float
    facts: list[FactTrace] = field(default_factory=list)
    format_pattern_index: int | None = None
    min_output_chars: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["facts"] = [asdict(f) for f in self.facts]
        return d


def _est_tokens(chars: int) -> float:
    return round(chars / 3.5, 1)


def _fact_traces(text: str, needles: list[str], tail_frac: float = 0.32) -> list[FactTrace]:
    if not text:
        return [
            FactTrace(needle=n, found=False, char_index=-1, pct_from_start=None, in_answer_tail=False)
            for n in needles
        ]
    tlow = text.lower()
    nlen = len(text)
    cutoff = (1.0 - tail_frac) * nlen
    out: list[FactTrace] = []
    for n in needles:
        idx = tlow.find(n.lower())
        if idx < 0:
            out.append(FactTrace(needle=n, found=False))
            continue
        pct = 100.0 * idx / nlen if nlen else 0.0
        out.append(
            FactTrace(
                needle=n,
                found=True,
                char_index=idx,
                pct_from_start=round(pct, 2),
                in_answer_tail=idx >= cutoff,
            )
        )
    return out


def _format_tier_pct(match_any: list[str] | None, text: str) -> tuple[float, int | None]:
    """First matching pattern = strictest (index 0 best). Returns (score 0-100, index)."""
    if not match_any or not text:
        return 100.0, None
    for i, rx in enumerate(match_any):
        try:
            if re.search(rx, text, re.DOTALL):
                n = len(match_any)
                if n <= 1:
                    return 100.0, i
                return round(100.0 - i * (80.0 / max(1, n - 1)), 1), i
        except re.error:
            continue
    return 0.0, None


def _length_adequacy_pct(text: str, min_chars: int | None) -> float:
    if not min_chars:
        return 100.0 if text else 0.0
    if not text:
        return 0.0
    return min(100.0, 100.0 * len(text) / float(min_chars))


def _tail_placement_pct(facts: list[FactTrace], tail_frac: float = 0.32) -> float:
    """Reward facts appearing in the last `tail_frac` of the output (recall after long context)."""
    found = [f for f in facts if f.found]
    if not found:
        return 0.0
    scores: list[float] = []
    for f in found:
        if f.in_answer_tail:
            scores.append(100.0)
        elif f.char_index >= 0 and f.pct_from_start is not None:
            scores.append(f.pct_from_start)
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 1)


def analyze_memorization(
    text: str,
    task: dict[str, Any],
    task_id: str,
    tail_frac: float = 0.32,
) -> MemorizationReport:
    """
    Compute recall %, format tier %, length %, tail placement %, and a composite memorization_index.

    memorization_index (0-100) weights:
      35% fact recall (must_contain hits)
      25% format tier (strictest regex match)
      20% length adequacy vs min_output_chars
      20% tail placement (facts in final segment or late in text)
    """
    needles = list(task.get("must_contain", []))
    min_c = task.get("min_output_chars")
    if min_c is not None:
        min_c = int(min_c)

    facts = _fact_traces(text, needles, tail_frac=tail_frac)
    n_need = len(needles)
    if n_need == 0:
        fact_recall_pct = 100.0
    else:
        fact_recall_pct = round(100.0 * sum(1 for f in facts if f.found) / n_need, 1)

    ma = task.get("match_any_regex")
    fmt_pct, fmt_idx = _format_tier_pct(ma, text)

    len_pct = _length_adequacy_pct(text, min_c)

    if n_need == 0:
        tail_pct = 100.0
    else:
        tail_pct = _tail_placement_pct(facts, tail_frac=tail_frac)

    notes: list[str] = []
    if not text:
        notes.append("empty_output")
    elif min_c and len(text) < min_c:
        notes.append(f"below_min_chars({len(text)}<{min_c})")

    mi = (
        0.35 * fact_recall_pct
        + 0.25 * fmt_pct
        + 0.20 * len_pct
        + 0.20 * tail_pct
    )
    mi = round(min(100.0, max(0.0, mi)), 1)

    return MemorizationReport(
        task_id=task_id,
        output_chars=len(text),
        est_tokens=_est_tokens(len(text)),
        fact_recall_pct=fact_recall_pct,
        format_match_tier_pct=fmt_pct,
        length_adequacy_pct=round(len_pct, 1),
        tail_placement_pct=tail_pct,
        memorization_index=mi,
        facts=facts,
        format_pattern_index=fmt_idx,
        min_output_chars=min_c,
        notes=notes,
    )


def aggregate_mode(reports: list[MemorizationReport]) -> dict[str, Any]:
    if not reports:
        return {
            "mean_memorization_index": 0.0,
            "mean_fact_recall_pct": 0.0,
            "mean_tail_placement_pct": 0.0,
            "mean_length_adequacy_pct": 0.0,
            "mean_format_tier_pct": 0.0,
            "total_output_est_tokens": 0.0,
        }
    return {
        "mean_memorization_index": round(
            sum(r.memorization_index for r in reports) / len(reports), 2
        ),
        "mean_fact_recall_pct": round(
            sum(r.fact_recall_pct for r in reports) / len(reports), 2
        ),
        "mean_tail_placement_pct": round(
            sum(r.tail_placement_pct for r in reports) / len(reports), 2
        ),
        "mean_length_adequacy_pct": round(
            sum(r.length_adequacy_pct for r in reports) / len(reports), 2
        ),
        "mean_format_tier_pct": round(
            sum(r.format_match_tier_pct for r in reports) / len(reports), 2
        ),
        "total_output_est_tokens": round(sum(r.est_tokens for r in reports), 1),
        "total_output_chars": sum(r.output_chars for r in reports),
    }


def print_mode_memorization_table(mode_name: str, reports: list[MemorizationReport]) -> None:
    agg = aggregate_mode(reports)
    print(f"  --- Memorization metrics ({mode_name}) ---")
    print(
        f"  Mean memorization index: {agg['mean_memorization_index']:.1f}/100 "
        f"(composite: facts + format + length + tail placement)"
    )
    print(
        f"  Mean fact recall: {agg['mean_fact_recall_pct']:.1f}%  |  "
        f"Mean tail placement score: {agg['mean_tail_placement_pct']:.1f}%  |  "
        f"Mean length adequacy: {agg['mean_length_adequacy_pct']:.1f}%"
    )
    print(
        f"  Mean format tier: {agg['mean_format_tier_pct']:.1f}%  |  "
        f"Σ est. output tokens: {agg['total_output_est_tokens']:.0f}"
    )
    for r in reports:
        fi = ", ".join(
            f"{x.needle[:20]}…@{x.pct_from_start}%"
            if x.found and x.pct_from_start is not None and len(x.needle) > 20
            else (
                f"{x.needle}@{x.pct_from_start}%"
                if x.found and x.pct_from_start is not None
                else f"{x.needle}:missing"
            )
            for x in r.facts
        )
        print(
            f"    {r.task_id}: idx={r.memorization_index:.0f} "
            f"facts={r.fact_recall_pct:.0f}% tail={r.tail_placement_pct:.0f}% "
            f"| {fi or '—'}"
        )


def print_comparison_memorization(
    on_reports: list[MemorizationReport], off_reports: list[MemorizationReport]
) -> None:
    ao = aggregate_mode(on_reports)
    af = aggregate_mode(off_reports)
    print("  --- Comparison: memorization (ON − OFF) ---")
    d = ao["mean_memorization_index"] - af["mean_memorization_index"]
    print(
        f"  Δ mean memorization index: {d:+.2f}  "
        f"(ON {ao['mean_memorization_index']:.1f} vs OFF {af['mean_memorization_index']:.1f})"
    )
    d2 = ao["mean_fact_recall_pct"] - af["mean_fact_recall_pct"]
    print(f"  Δ mean fact recall %: {d2:+.2f}")
    d3 = ao["mean_tail_placement_pct"] - af["mean_tail_placement_pct"]
    print(f"  Δ mean tail placement %: {d3:+.2f}")
