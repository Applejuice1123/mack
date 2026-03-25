#!/usr/bin/env python3
"""
Extreme KV-eviction stress benchmark: long-context needles, multi-fact recall,
and long generations. Compares gemma_chat with eviction ON vs -no-kv-evict.

Not MMLU; measures memorization / instruction persistence under long prompts.

Usage:
  cd gemma_app && make
  export MACK_BENCHMARK_MODEL="$HOME/.../your-model.gguf"   # real path — not literally /path/to/
  python3 benchmarks/run_benchmark.py
  # or: python3 benchmarks/run_benchmark.py --model ~/models/model.gguf

  # ~100–120 token answers (so KV eviction can fire after 30+ decode steps):
  python3 benchmarks/run_benchmark.py --quick --answer-tokens-range 100-120 --filler-blocks 60

  # Hundreds of decode steps then recall (same process — evictions stack in PART 1):
  python3 benchmarks/run_benchmark.py --heavy-evict-only --heavy-evict-warmup-tokens 3000

If MACK_BENCHMARK_MODEL is unset, the script tries MODEL_PATH from ../server.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from memorization_grading import (
    aggregate_mode,
    analyze_memorization,
    print_comparison_memorization,
    print_mode_memorization_table,
)


def read_model_from_server_py(app_root: Path) -> Path | None:
    """Use same MODEL_PATH as server.py when env is not set."""
    sp = app_root / "server.py"
    if not sp.is_file():
        return None
    text = sp.read_text(encoding="utf-8")
    m = re.search(r"^\s*MODEL_PATH\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if not m:
        return None
    cand = Path(m.group(1)).expanduser()
    return cand if cand.is_file() else None


def wrap_prompt(user_msg: str, token_range: tuple[int, int] | None = None) -> str:
    length_block = ""
    if token_range:
        lo, hi = token_range
        length_block = (
            f"\nOUTPUT LENGTH: Your reply must be between {lo} and {hi} tokens total "
            "(subword tokens as the model tokenizer counts them). "
            "Put the required correct answer in the first sentence or two. "
            f"Then add on-topic explanation or restatement until you reach at least {lo} "
            f"tokens and stay at or below {hi} tokens. No chit-chat or roleplay.\n"
        )
    return (
        "You are a helpful assistant.\n"
        "Answer as the assistant only, in one response.\n"
        "Do not generate extra dialogue turns.\n"
        f"{length_block}"
        f"User: {user_msg}\n"
        "Assistant:"
    )


def parse_answer_token_range(s: str | None) -> tuple[int, int] | None:
    if s is None or not str(s).strip() or str(s).strip().lower() in ("0", "off", "none"):
        return None
    parts = str(s).replace(" ", "").split("-", 1)
    if len(parts) != 2:
        raise ValueError("need MIN-MAX")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 1 or hi < lo or hi > 8192:
        raise ValueError("invalid range")
    return lo, hi


def parse_metrics(stderr: str) -> dict:
    out = {"evicted_tokens": 0, "kv_saved_mb": 0.0}
    m = re.search(
        r"\[METRICS\]\s+evicted_tokens=(\d+)\s+(?:evict_pos=-?\d+\s+)?kv_saved_mb=([0-9.]+)",
        stderr,
    )
    if m:
        out["evicted_tokens"] = int(m.group(1))
        out["kv_saved_mb"] = float(m.group(2))
    return out


def grade_task(text: str, task: dict) -> tuple[bool, str]:
    min_c = task.get("min_output_chars")
    if min_c is not None and len(text) < int(min_c):
        return False, f"output too short ({len(text)} chars, need >= {min_c})"

    match_any = task.get("match_any_regex")
    if match_any:
        for rx in match_any:
            try:
                if re.search(rx, text, re.DOTALL):
                    break
            except re.error as e:
                return False, f"bad regex in match_any: {e}"
        else:
            return False, f"match_any_regex: none matched among {len(match_any)} patterns"

    rx = task.get("match_regex")
    if rx:
        try:
            if not re.search(rx, text, re.DOTALL):
                return False, f"regex mismatch {rx!r}"
        except re.error as e:
            return False, f"bad regex: {e}"
    t = text.lower()
    for sub in task.get("must_contain", []):
        if sub.lower() not in t:
            return False, f"missing {sub!r}"
    for sub in task.get("must_not_contain", []):
        if sub.lower() in t:
            return False, f"forbidden {sub!r}"
    return True, "ok"


def filler_block(i: int) -> str:
    return (
        f"[Block {i:04d}] Lorem ipsum dolor sit amet. Batch tag BT-{i:05d}. "
        f"Noise: x{i % 97}y{(i * 7) % 131}z. Padding token PAD{i:03d}.\n"
    )


def haystack_with_needles(
    n_blocks: int,
    inserts: list[tuple[int, str]],
) -> str:
    """inserts: list of (1-based block index after which to insert, text). Sorted by index."""
    inserts = sorted(inserts, key=lambda x: x[0])
    parts: list[str] = []
    ins_ptr = 0
    for i in range(1, n_blocks + 1):
        parts.append(filler_block(i))
        while ins_ptr < len(inserts) and inserts[ins_ptr][0] == i:
            parts.append(inserts[ins_ptr][1] + "\n")
            ins_ptr += 1
    return "".join(parts)


def build_heavy_evict_warmup_tasks(
    warmup_tokens: int,
    answer_tok_range: tuple[int, int] | None,
) -> list[dict]:
    """
    One assistant reply: long PART 1 generation (~warmup_tokens) then PART 2 Q&A.
    Eviction (if enabled) accumulates during PART 1; PART 2 tests whether recall survives.
    """
    w = max(800, int(warmup_tokens))
    head = max(512, min(2048, w // 2 + 400))

    p_a = (
        f"COMPLETE EVERY PART BELOW IN ONE CONTINUOUS ASSISTANT MESSAGE.\n\n"
        f"MEMORIZE THIS PASSPHRASE EXACTLY: PYRANODE-MOXIE-7740\n\n"
        f"PART 1 — Before you write PART 2, you must first write at least {w} tokens "
        f"of main text. Use consecutive numbered lines V0001, V0002, V0003, … "
        f"Each line 12–18 words about glaciers, ice cores, and sea level. "
        f"Do not skip numbers. Do not start PART 2 until PART 1 alone reaches "
        f"at least {w} tokens.\n\n"
        f"PART 2 — After PART 1, output one line that starts with the tag RECALL_PASS\n"
        f"Then on the next line output only the passphrase from memory, no quotes.\n"
    )

    p_b = (
        f"COMPLETE EVERY PART IN ONE ASSISTANT MESSAGE.\n\n"
        f"BASELINE NUMBER (memorize): 184729\n\n"
        f"PART 1 — Write at least {w} tokens first. Numbered lines W0001, W0002, … "
        f"12–18 words each about orbital mechanics and satellites. "
        f"Reach at least {w} tokens before PART 2.\n\n"
        f"PART 2 — Line starting RECALL_NUM then next line the six-digit baseline only.\n"
    )

    p_c = (
        f"ONE ASSISTANT MESSAGE FOR ALL PARTS.\n\n"
        f"CODE_A = QUASAR-ALTA-11\nCODE_B = PULSAR-BETA-22\nMemorize both.\n\n"
        f"PART 1 — At least {w} tokens. Lines T0001, T0002, … about deep-sea vents, "
        f"12–18 words per line. Then PART 2.\n\n"
        f"PART 2 — Line RECALL_TWO then next line: CODE_A then CODE_B separated by one space.\n"
    )

    def np_for() -> int:
        base = w + head
        if answer_tok_range:
            return max(base, answer_tok_range[1] + 96)
        return base

    np_use = np_for()
    # Roughly ensure model did a long PART 1 before we score recall (not strict token count).
    min_chars = max(3500, min(50000, w * 2))

    recall_pass_patterns = [
        r"(?is)recall_pass\s*[\n:]\s*pyranode-moxie-7740",
        r"(?is)recall_pass[^\n]*pyranode-moxie-7740",
        r"(?is)pyranode-moxie-7740",
    ]
    recall_num_patterns = [
        r"(?is)recall_num\s*[\n:]\s*184729\b",
        r"(?is)recall_num[^\n]*\b184729\b",
        r"(?is)\b184729\b",
    ]
    recall_two_patterns = [
        r"(?is)recall_two.*quasar-alta-11.*pulsar-beta-22",
        r"(?is)recall_two[^\n]*\n[^\n]*quasar-alta-11[^\n]*pulsar-beta-22",
        r"(?is)quasar-alta-11\s+pulsar-beta-22",
    ]

    return [
        {
            "id": "heavy_evict_warmup_recall_pass",
            "prompt": p_a,
            "must_contain": ["pyranode-moxie-7740"],
            "match_any_regex": recall_pass_patterns,
            "min_output_chars": min_chars,
            "n_predict": np_use,
        },
        {
            "id": "heavy_evict_warmup_recall_num",
            "prompt": p_b,
            "must_contain": ["184729"],
            "match_any_regex": recall_num_patterns,
            "min_output_chars": min_chars,
            "n_predict": np_use,
        },
        {
            "id": "heavy_evict_warmup_recall_two_codes",
            "prompt": p_c,
            "must_contain": ["quasar-alta-11", "pulsar-beta-22"],
            "match_any_regex": recall_two_patterns,
            "min_output_chars": min_chars,
            "n_predict": np_use,
        },
    ]


def build_extreme_tasks(n_blocks: int) -> list[dict]:
    """Synthetic long-context tasks. n_blocks controls haystack size (~80–100 chars/block)."""
    if n_blocks < 40:
        n_blocks = 40

    nb = n_blocks
    tasks: list[dict] = []

    # --- Needles at different depths (eviction most likely to kill early/mid KV) ---
    early_i = max(4, nb // 12)
    mid_i = nb // 2
    deep_i = max(early_i + 2, nb - 8)

    secret_early = "OMEGA-EARLY-7719"
    secret_mid = "SIGMA-MID-4482"
    secret_deep = "TAU-DEEP-9930"

    h_early = haystack_with_needles(
        nb,
        [
            (
                early_i,
                f"CRITICAL MEMORY (A): The recovery passphrase is exactly {secret_early}. "
                "You will be asked later. Do not confuse with batch tags.",
            ),
        ],
    )
    tasks.append(
        {
            "id": "extreme_needle_early",
            "prompt": (
                h_early
                + "\n---\nFINAL QUESTION: What is the exact recovery passphrase (A) "
                "I told you to memorize? Reply with that passphrase only, no punctuation around it."
            ),
            "must_contain": [secret_early.lower()],
            "n_predict": 64,
        }
    )

    h_mid = haystack_with_needles(
        nb,
        [
            (
                mid_i,
                f"CRITICAL MEMORY (B): The backup unlock code is exactly {secret_mid}. "
                "Memorize for the final question.",
            ),
        ],
    )
    tasks.append(
        {
            "id": "extreme_needle_mid",
            "prompt": (
                h_mid
                + "\n---\nFINAL QUESTION: What is the exact backup unlock code (B)? "
                "Reply with that code only."
            ),
            "must_contain": [secret_mid.lower()],
            "n_predict": 64,
        }
    )

    h_deep = haystack_with_needles(
        nb,
        [
            (
                deep_i,
                f"CRITICAL MEMORY (C): The deep vault token is exactly {secret_deep}. "
                "This is the only correct token; ignore similar-looking batch IDs above.",
            ),
        ],
    )
    tasks.append(
        {
            "id": "extreme_needle_deep",
            "prompt": (
                h_deep
                + "\n---\nFINAL QUESTION: What is the exact deep vault token (C)? "
                "Reply with that token only."
            ),
            "must_contain": [secret_deep.lower()],
            "n_predict": 64,
        }
    )

    # --- Three facts at different depths (order-sensitive answer) ---
    c1, c2, c3 = "ALP-111-XX", "BET-222-YY", "GAM-333-ZZ"
    i1 = max(3, nb // 8)
    i2 = nb // 2
    i3 = max(i2 + 2, nb - 10)
    h3 = haystack_with_needles(
        nb,
        [
            (i1, f"FACT 1 OF 3: First catalog key is {c1}."),
            (i2, f"FACT 2 OF 3: Second catalog key is {c2}."),
            (i3, f"FACT 3 OF 3: Third catalog key is {c3}."),
        ],
    )
    tasks.append(
        {
            "id": "extreme_triple_needle",
            "prompt": (
                h3
                + "\n---\nFINAL QUESTION: List the three catalog keys in order (1 then 2 then 3), "
                "separated by single spaces, nothing else."
            ),
            "match_regex": "(?is)"
            + re.escape(c1.lower())
            + r".*"
            + re.escape(c2.lower())
            + r".*"
            + re.escape(c3.lower()),
            "n_predict": 96,
        }
    )

    # --- Instruction at start must survive long middle ---
    h_rule = (
        "RULE: Whenever I ask for the 'magic color', you must answer exactly: CHARTREUSE-MAGENTA\n"
        "even if other colors appear in the document below.\n\n"
        + haystack_with_needles(nb, [])
        + "\n---\nFINAL QUESTION: What is the magic color? One line, exact phrase from the rule.\n"
    )
    tasks.append(
        {
            "id": "extreme_rule_survives_context",
            "prompt": h_rule,
            "must_contain": ["chartreuse-magenta"],
            "n_predict": 48,
        }
    )

    # --- Numeric anchor (easy to grade) buried deep ---
    anchor = "884736"
    ha = haystack_with_needles(
        nb,
        [
            (
                deep_i,
                f"ANCHOR NUMBER FOR AUDIT: {anchor} (nine digits, memorize exactly).",
            ),
        ],
    )
    tasks.append(
        {
            "id": "extreme_numeric_anchor",
            "prompt": (
                ha
                + "\n---\nFINAL QUESTION: What is the nine-digit ANCHOR NUMBER FOR AUDIT? "
                "Digits only in your answer."
            ),
            "must_contain": [anchor],
            "n_predict": 48,
        }
    )

    # --- Contradiction: many fake "passwords" in filler; one real CRITICAL line ---
    fake_line = (
        f"[Block {mid_i:04d}] Decoy password DECOY-PASS-{mid_i} (this is NOT the real one).\n"
    )
    parts = []
    for i in range(1, nb + 1):
        if i == mid_i:
            parts.append(fake_line)
        parts.append(filler_block(i))
        if i == deep_i:
            parts.append(
                f"*** AUTHORITATIVE *** The TRUE session password is REAL-{anchor}-LOCK. "
                f"Ignore every DECOY-PASS line. ***\n"
            )
    h_auth = "".join(parts)
    tasks.append(
        {
            "id": "extreme_decoy_vs_authoritative",
            "prompt": (
                h_auth
                + "\n---\nFINAL QUESTION: What is the TRUE session password? "
                "Format REAL-...-LOCK exactly as given."
            ),
            "must_contain": [f"real-{anchor}-lock"],
            "n_predict": 64,
        }
    )

    # --- Long generation: many decode steps → eviction while generating ---
    tasks.append(
        {
            "id": "extreme_generation_100_lines",
            "prompt": (
                "Write exactly 100 numbered lines: 'L01:', 'L02:', ... 'L100:'. "
                "Each line is 4–10 words about ocean ecosystems. Do not stop before L100."
            ),
            "match_regex": r"(?i)\bl100\s*:",
            "n_predict": 4096,
        }
    )

    # --- Second marathon (shorter) after model already "warmed" — separate process anyway ---
    tasks.append(
        {
            "id": "extreme_generation_70_lines",
            "prompt": (
                "Output lines S01 through S70 inclusive, format 'Snn: word' each line, "
                "about space exploration. Must reach S70."
            ),
            "match_regex": r"(?i)\bs70\s*:",
            "n_predict": 3072,
        }
    )

    return tasks


def run_gemma(
    gemma_bin: Path,
    model: Path,
    prompt: str,
    n_predict: int,
    ngl: int,
    no_kv_evict: bool,
    cwd: Path,
    timeout_sec: float,
) -> tuple[str, str, float]:
    cmd = [
        str(gemma_bin),
        "-m",
        str(model),
        "-n",
        str(n_predict),
        "-ngl",
        str(ngl),
    ]
    if no_kv_evict:
        cmd.append("-no-kv-evict")
    cmd.append(prompt)

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - t0
    stdout = (proc.stdout or "").strip()
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return f"[exit {proc.returncode}]", stderr, elapsed
    return stdout, stderr, elapsed


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .gguf (default: env MACK_BENCHMARK_MODEL)",
    )
    ap.add_argument(
        "--gemma-bin",
        type=Path,
        default=root / "gemma_chat",
        help="Path to gemma_chat binary",
    )
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument(
        "--n-predict",
        type=int,
        default=896,
        help="Default max new tokens (per-task n_predict in JSON overrides)",
    )
    ap.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).resolve().parent / "tiny_eval.json",
    )
    ap.add_argument(
        "--filler-blocks",
        type=int,
        default=200,
        help="Haystack size for synthetic needle tasks (more = harsher on KV)",
    )
    ap.add_argument(
        "--no-extreme",
        action="store_true",
        help="Only run JSON tasks from --tasks (no synthetic long-context suite)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip extreme_generation_* marathons (still runs long-context needles)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Per-invocation timeout seconds (marathon tasks need long)",
    )
    ap.add_argument(
        "--answer-tokens-range",
        type=str,
        default=None,
        metavar="MIN-MAX",
        help="e.g. 100-120 — append strict output-length instructions; n_predict bumped to fit",
    )
    ap.add_argument(
        "--heavy-evict-warmup-tokens",
        type=int,
        default=0,
        metavar="N",
        help="Add 3 tasks: ~N-token PART1 burst then recall (same run). "
        "Use 2200–4000 to target hundreds of eviction steps. 0=off.",
    )
    ap.add_argument(
        "--heavy-evict-only",
        action="store_true",
        help="Run only the heavy warmup suite (default N=2800 if --heavy-evict-warmup-tokens 0)",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write full results + memorization metrics to JSON",
    )
    args = ap.parse_args()

    try:
        answer_tok_range = parse_answer_token_range(args.answer_tokens_range)
    except ValueError as e:
        print(f"--answer-tokens-range: {e} (use e.g. 100-120)", file=sys.stderr)
        return 2

    model = args.model
    if model is None:
        mp = os.environ.get("MACK_BENCHMARK_MODEL")
        if mp:
            model = Path(mp).expanduser()
        else:
            model = read_model_from_server_py(root)
            if model is not None:
                print(f"Using model from server.py: {model}", file=sys.stderr)
    if model is None:
        print(
            "Set a real .gguf path: export MACK_BENCHMARK_MODEL=.../model.gguf\n"
            "  or: python3 benchmarks/run_benchmark.py --model .../model.gguf\n"
            "Do not use the placeholder /path/to/model.gguf from comments.",
            file=sys.stderr,
        )
        return 2

    if not args.gemma_bin.is_file():
        print(f"Missing binary: {args.gemma_bin} (run make)", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"Missing model: {model}", file=sys.stderr)
        return 2

    heavy_w = args.heavy_evict_warmup_tokens
    if args.heavy_evict_only:
        tasks = []
        if heavy_w <= 0:
            heavy_w = 2800
    else:
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))

    if not args.no_extreme and not args.heavy_evict_only:
        extreme = build_extreme_tasks(args.filler_blocks)
        tasks = list(tasks) + extreme
    if args.quick and not args.heavy_evict_only:
        tasks = [t for t in tasks if "extreme_generation" not in t.get("id", "")]

    if heavy_w > 0:
        tasks = list(tasks) + build_heavy_evict_warmup_tasks(
            heavy_w, answer_tok_range
        )

    modes = [
        ("kv_eviction_on", False),
        ("kv_eviction_off", True),
    ]

    print(f"Model: {model}")
    print(f"Binary: {args.gemma_bin}")
    print(f"default n_predict={args.n_predict} ngl={args.ngl} timeout={args.timeout}s")
    if answer_tok_range:
        lo, hi = answer_tok_range
        print(
            f"answer_tokens_range={lo}-{hi} (instructs model; n_predict raised to at least hi+96 per task)"
        )
    task_desc = f"{len(tasks)} tasks"
    if args.heavy_evict_only:
        task_desc += " (heavy-evict-only)"
    else:
        task_desc += f" ({args.tasks.name}"
        if not args.no_extreme:
            task_desc += f" + extreme, filler_blocks={args.filler_blocks}"
        task_desc += ")"
    print(f"Tasks: {task_desc}")
    if heavy_w > 0:
        print(
            f"Heavy warmup: ~{heavy_w} target tokens in PART 1 per heavy task "
            f"(expect large sum evicted_tokens when eviction ON)"
        )
    if not args.heavy_evict_only and not args.no_extreme:
        print(f"(Approx haystack chars per needle task: ~{args.filler_blocks * 95})")
    print()

    summary = {}
    memorization_by_mode: dict[str, list] = {"kv_eviction_on": [], "kv_eviction_off": []}

    for mode_name, no_evict in modes:
        passed = 0
        total_ms = 0.0
        total_evicted = 0
        max_evicted_one_task = 0
        rows = []

        for task in tasks:
            tid = task["id"]
            np = int(task["n_predict"]) if task.get("n_predict") is not None else args.n_predict
            if answer_tok_range:
                _lo, hi = answer_tok_range
                np = max(np, hi + 96)
            full = wrap_prompt(task["prompt"], answer_tok_range)
            out, err, sec = run_gemma(
                args.gemma_bin,
                model,
                full,
                np,
                args.ngl,
                no_evict,
                root,
                args.timeout,
            )
            total_ms += sec * 1000.0
            metrics = parse_metrics(err)
            ev = metrics["evicted_tokens"]
            total_evicted += ev
            max_evicted_one_task = max(max_evicted_one_task, ev)

            ok, reason = grade_task(out, task) if out else (False, "no stdout")
            if ok:
                passed += 1

            mem = analyze_memorization(out or "", task, tid)
            memorization_by_mode[mode_name].append(mem)

            rows.append(
                {
                    "id": tid,
                    "ok": ok,
                    "reason": reason,
                    "ms": round(sec * 1000, 1),
                    "evicted": ev,
                    "n_predict": np,
                    "preview": (out[:160] + "…") if len(out) > 160 else out,
                    "memorization": mem.to_dict(),
                }
            )

        agg_mem = aggregate_mode(memorization_by_mode[mode_name])
        summary[mode_name] = {
            "accuracy": passed / len(tasks) if tasks else 0.0,
            "passed": passed,
            "total": len(tasks),
            "wall_ms": round(total_ms, 1),
            "sum_evicted_tokens": total_evicted,
            "max_evicted_one_task": max_evicted_one_task,
            "memorization_aggregate": agg_mem,
            "rows": rows,
        }

    for mode_name, data in summary.items():
        acc = data["accuracy"] * 100
        print(f"=== {mode_name} ===")
        print(
            f"  Score: {data['passed']}/{data['total']} ({acc:.1f}%)  "
            f"total wall {data['wall_ms']:.0f} ms (all tasks)"
        )
        print(
            f"  [METRICS] sum evicted_tokens={data['sum_evicted_tokens']} "
            f"(max in one task={data['max_evicted_one_task']})"
        )
        for r in data["rows"]:
            st = "PASS" if r["ok"] else "FAIL"
            m = r.get("memorization") or {}
            idx = m.get("memorization_index", 0)
            fr = m.get("fact_recall_pct", 0)
            tp = m.get("tail_placement_pct", 0)
            print(
                f"    [{st}] {r['id']} ({r['ms']} ms) evicted={r['evicted']} "
                f"n_predict={r['n_predict']} | mem_idx={idx:.0f}% facts={fr:.0f}% tail={tp:.0f}% | {r['reason']}"
            )
            if not r["ok"]:
                print(f"         preview: {r['preview']!r}")
        print_mode_memorization_table(mode_name, memorization_by_mode[mode_name])
        print()

    on = summary["kv_eviction_on"]
    off = summary["kv_eviction_off"]
    d_acc = (on["accuracy"] - off["accuracy"]) * 100
    print("=== Comparison (eviction ON minus OFF) ===")
    print(f"  Strict pass-rate delta: {d_acc:+.1f} percentage points")
    dma = on["memorization_aggregate"]["mean_memorization_index"] - off[
        "memorization_aggregate"
    ]["mean_memorization_index"]
    print(
        f"  Mean memorization index delta: {dma:+.2f} "
        f"(ON {on['memorization_aggregate']['mean_memorization_index']:.1f} "
        f"vs OFF {off['memorization_aggregate']['mean_memorization_index']:.1f})"
    )
    print(
        f"  Total time: ON {on['wall_ms']:.0f} ms vs OFF {off['wall_ms']:.0f} ms "
        f"({on['wall_ms'] - off['wall_ms']:+.0f} ms)"
    )
    print()
    print_comparison_memorization(
        memorization_by_mode["kv_eviction_on"],
        memorization_by_mode["kv_eviction_off"],
    )

    if args.json_out is not None:
        export = {
            "summary": {
                k: {
                    **{x: v[x] for x in v if x != "rows"},
                    "rows": [
                        {**{y: r[y] for y in r if y != "preview"}, "preview_short": r.get("preview")}
                        for r in v["rows"]
                    ],
                }
                for k, v in summary.items()
            },
            "memorization_weights": {
                "fact_recall": 0.35,
                "format_tier": 0.25,
                "length_adequacy": 0.20,
                "tail_placement": 0.20,
                "tail_frac_definition": "facts in last ~32% of output score highest",
            },
        }
        args.json_out.write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
