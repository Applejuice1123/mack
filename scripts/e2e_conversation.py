"""End-to-end check on the real model: a short conversation in a deliberately small
context, so compaction, pins, verified summary, recall and abstention all happen.

    .venv/bin/python scripts/e2e_conversation.py [--budget 1500]

Uses a throwaway state directory: no calibration is loaded, so decode runs unthrottled.
Prints the transcript and a PASS/FAIL line per expectation.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mack.app import Mack  # noqa: E402
from mack.config import load_config  # noqa: E402

SCRIPT = [
    "Hi! My name is Alex. I'm planning a trip to Lisbon from March 14 to March 21 with my sister Maria. Our total budget is 2,400 euros.",
    "We decided to stay in the Alfama district instead of Baixa. The apartment door code is 4471. What are two things we should see there?",
    "Explain briefly how the Lisbon tram system works.",
    "What kind of food should we try? Keep it to a short list.",
    "Give me a short packing list for mild spring weather.",
    "What is fado music? Two sentences.",
    "What was the door code for the apartment again?",
    "And what is our total budget, and who am I travelling with?",
    "What was the name of the restaurant I told you my cousin recommended?",
]
ABSTAIN = ("can't see", "cannot see", "don't have", "do not have", "no longer", "don't see", "not see")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1500)
    args = ap.parse_args()

    cfg = load_config()
    cfg.paths.state = tempfile.mkdtemp(prefix="mack-e2e-")
    cfg.context.budget_tokens = args.budget
    cfg.generation.max_tokens = 220
    mack = Mack(cfg)
    session = mack.new_session()
    print(f"model {cfg.model.main}, adapter={mack.engine.adapter_loaded}, budget {session.budget}, system {len(session.ctx.tokens)} tok")

    replies = []
    for msg in SCRIPT:
        print(f"\n>>> {msg}")
        for e in session.chat(msg):
            if e["type"] == "status":
                print("   [status]", {k: v for k, v in e.items() if k != "type"})
            elif e["type"] == "done":
                replies.append(e["reply"])
                print("<<<", e["reply"].replace("\n", " ")[:400])
                print(f"   [{e['tps']} tok/s, {e['n_tokens']} tok, ctx {e['context_tokens']}/{e['budget_tokens']}, ttft {e['ttft_s']}s, recalled {e['recalled']}]")

    view = session.memory_view()
    print("\nPINS:")
    for p in view["pins"]:
        print(f"  #{p['turn'] + 1} {p['role']}: {p['quote']}")
    print("SUMMARY:", json.dumps(view["summary"], indent=1))
    print("COMPACTIONS:", [(c["tokens_before"], c["tokens_after"], c["summary_dropped"], c["seconds"]) for c in view["compactions"]])

    checks = {
        "compaction happened": bool(view["compactions"]),
        "no reply opens with a recap heading": not any(r.lstrip("*# ").lower().startswith(("recalled", "memory")) for r in replies),
        "door code recovered exactly": "4471" in replies[6],
        "budget recovered exactly": "2,400" in replies[7] or "2400" in replies[7],
        "sister recovered": "Maria" in replies[7],
        "unknown restaurant: abstains": any(a in replies[8].lower() for a in ABSTAIN),
        "every pin is a verbatim quote": all(p["quote"] in session.turns[p["turn"]].text for p in view["pins"]),
        "originals archived": "4471" in session.archive.turn_text(session.id, 2),
    }
    print()
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print("memory:", mack.engine.memory())
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
