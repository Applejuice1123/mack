"""`mack lora eval`: the gate that decides whether the adapter ships.

The adapter exists to cut made-up answers about things Mack can no longer see.
It ships only if, on held-out cases, it

  * fabricates less than the base model on unanswerable questions, and
  * does not refuse more on answerable ones (within a small tolerance), and
  * does not get worse at answering from memory or at ordinary questions.

The decision is written to <adapter>/SHIP.json. The engine loads the adapter
only when that file says ship.
"""

from __future__ import annotations

import gc
import json
import re
import unicodedata

import mlx.core as mx

from ..config import ROOT, Config
from ..engine import Engine, strip_thinking
from ..governor import now_iso
from . import dataset

TOLERANCE = 0.02
GENERAL_TOLERANCE = 0.05
MAX_REPLY_TOKENS = 96

_ABSTAIN = re.compile(
    r"can(?:'|no)?t (?:see|find|recall|access|view)|cannot (?:see|find|recall|access|view)|no longer (?:visible|see|have)"
    r"|don'?t (?:have|see|recall|know)|do not (?:have|see|recall|know)|not (?:visible|in (?:what|our|the))|isn'?t (?:visible|in (?:what|our|the))"
    r"|(?:tell|give|share|remind) (?:me|it)(?: \w+){0,3} again|repeat (?:it|that)|haven'?t (?:told|mentioned|given)"
    r"|didn'?t (?:tell|mention|give)|not sure what|won'?t guess|rather not guess|no (?:record|mention|information)",
    re.I,
)
_OTHER_IDENTITY = re.compile(r"\b(?:i am|i'm|this is)\s+(?:qwen|chatgpt|claude|gemini|gpt)\b|alibaba|openai", re.I)


def norm(text: str) -> str:
    # NFKC folds typographic variants, e.g. the subscript in H₂O.
    return re.sub(r"[\s,$]", "", unicodedata.normalize("NFKC", text).lower())


def has_gold(reply: str, gold) -> bool:
    golds = gold if isinstance(gold, list) else [gold]
    return any(norm(g).rstrip(".") in norm(reply) for g in golds)


def score(case: dict, reply: str) -> str:
    """One of: correct, abstained, fabricated, refused, wrong."""
    abstained = bool(_ABSTAIN.search(reply))
    kind = case["kind"]
    if kind == "identity":
        return "correct" if has_gold(reply, case["gold"]) and not _OTHER_IDENTITY.search(reply) else "wrong"
    if kind == "general":
        if has_gold(reply, case["gold"]):
            return "correct"
        return "refused" if abstained else "wrong"
    if case["answerable"]:
        if has_gold(reply, case["gold"]):
            return "correct"
        return "refused" if abstained else "wrong"
    # Unanswerable: anything other than saying so is a made-up answer.
    return "abstained" if abstained else "fabricated"


def rate(rows: list[dict], kinds: tuple[str, ...], outcome: str) -> float:
    sel = [r for r in rows if r["kind"] in kinds]
    return sum(r["outcome"] == outcome for r in sel) / len(sel) if sel else 0.0


def summarize(rows: list[dict]) -> dict:
    unanswerable = ("missing", "missing_near", "fresh_missing")
    answerable = ("pinned", "recalled", "visible")
    return {
        "n": len(rows),
        "fabrication_rate": round(rate(rows, unanswerable, "fabricated"), 4),
        "unnecessary_refusal_rate": round(rate(rows, answerable, "refused"), 4),
        "answerable_accuracy": round(rate(rows, answerable, "correct"), 4),
        "identity_accuracy": round(rate(rows, ("identity",), "correct"), 4),
        "general_accuracy": round(rate(rows, ("general",), "correct"), 4),
    }


def run_cases(engine: Engine, cases: list[dict], log=print) -> list[dict]:
    rows = []
    for i, case in enumerate(cases):
        ctx = engine.new_context()
        prompt = engine.encode(dataset.render_prompt(engine.fmt, case))
        out = list(engine.generate(ctx, prompt, MAX_REPLY_TOKENS, sampler=engine.greedy))
        reply = strip_thinking(engine.decode(out)).strip()
        rows.append({"kind": case["kind"], "gold": case["gold"], "user": case["user"][-200:], "reply": reply, "outcome": score(case, reply)})
        del ctx
        if (i + 1) % 20 == 0:
            mx.clear_cache()
            log(f"  {i + 1}/{len(cases)}")
    return rows


def decide(base: dict, adapter: dict) -> dict:
    reasons = []
    if adapter["fabrication_rate"] >= base["fabrication_rate"] and base["fabrication_rate"] > 0:
        reasons.append("does not reduce made-up answers")
    if adapter["unnecessary_refusal_rate"] > base["unnecessary_refusal_rate"] + TOLERANCE:
        reasons.append("refuses answerable questions more often")
    if adapter["answerable_accuracy"] < base["answerable_accuracy"] - TOLERANCE:
        reasons.append("answers from memory less accurately")
    if adapter["general_accuracy"] < base["general_accuracy"] - GENERAL_TOLERANCE:
        reasons.append("ordinary answers got worse")
    if adapter["identity_accuracy"] < base["identity_accuracy"]:
        reasons.append("identity got worse")
    return {"ship": not reasons, "reasons": reasons or ["fewer made-up answers without more refusals"]}


def load_cases(limit: int = 0) -> list[dict]:
    cases = dataset.read("eval")
    if limit:
        by_kind: dict = {}
        for c in cases:
            by_kind.setdefault(c["kind"], []).append(c)
        cases = [c for group in by_kind.values() for c in group[:limit]]
    return cases


def baseline(cfg: Config, limit: int = 0, log=print) -> dict:
    """Score the base model alone: how often it makes things up before any fine-tune."""
    cases = load_cases(limit)
    log(f"evaluating base model on {len(cases)} held-out cases…")
    rows = run_cases(Engine(cfg, use_adapter=False, load_draft=False), cases, log)
    report = {"model": cfg.model.main, "created": now_iso(), "cases": len(cases), "base": summarize(rows)}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "lora_baseline.json").write_text(json.dumps({**report, "rows": rows}, indent=2))
    log(f"base     {report['base']}")
    return report


def gate(cfg: Config, limit: int = 0, log=print) -> dict:
    adapter_dir = cfg.adapter_dir
    if adapter_dir is None or not (adapter_dir / "adapters.safetensors").exists():
        raise SystemExit("no trained adapter found; run `mack lora train` first")
    cases = load_cases(limit)

    results = {}
    for name, use_adapter in (("base", False), ("adapter", True)):
        log(f"evaluating {name} on {len(cases)} held-out cases…")
        engine = Engine(cfg, use_adapter=use_adapter, load_draft=False)
        results[name] = run_cases(engine, cases, log)
        del engine
        gc.collect()
        mx.clear_cache()

    base, adapter = summarize(results["base"]), summarize(results["adapter"])
    decision = decide(base, adapter)
    report = {
        "model": cfg.model.main,
        "created": now_iso(),
        "cases": len(cases),
        "base": base,
        "adapter": adapter,
        "decision": decision,
        "ship": decision["ship"],
    }
    (adapter_dir / "SHIP.json").write_text(json.dumps(report, indent=2))
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "lora_eval.json").write_text(json.dumps({**report, "rows": results}, indent=2))
    log(f"base     {base}")
    log(f"adapter  {adapter}")
    log(f"ship: {decision['ship']} ({'; '.join(decision['reasons'])})")
    return report
