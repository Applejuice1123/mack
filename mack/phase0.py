"""`mack phase0`: re-measure, on the current model, the things the design relies on.

The earlier memory results (peak-memory slopes, tiled prefill) were taken on pure
attention models. Qwen3.5 is a hybrid: only every fourth layer keeps a growing KV
cache and the rest carry a fixed-size recurrent state. Nothing below is assumed
to transfer; it is measured here and written to state/phase0.json, which the
governor and session read.

  1. architecture  which layers grow, whether the cache can be trimmed
  2. memory        cache bytes/token, peak during prefill vs context and chunk size,
                   and the context ceiling that follows from the wired limit
  3. decode        pipelined vs synchronous loop, tokens/s
  4. speculative   snapshot-rewind speculative decoding: acceptance and real speedup
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import psutil

from .calibrate import Fixed
from .config import ROOT, Config
from .engine import GB, Engine, GenStats
from .governor import now_iso, write_json

FILLER = (
    "The harbour town kept its ledgers in three languages, and every clerk learned to read the tide "
    "tables before the tax rolls. Merchants argued about the price of salt, the width of the new quay, "
    "and whether the lighthouse keeper had miscounted the ships again. "
)
SPEC_PROMPTS = (
    "Explain how a hash table handles collisions, step by step.",
    "Write a Python function that merges two sorted lists, with comments.",
    "Summarize the causes of the French Revolution in two paragraphs.",
)
SAFETY = 0.80  # fraction of the wired limit Mack plans to stay under


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx_, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx_) ** 2 for x in xs)
    slope = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return slope, my - slope * mx_


def measure_prefill(engine: Engine, tokens: list[int], step: int) -> dict:
    engine.cfg.context.prefill_step = step
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    ctx = engine.new_context()
    t0 = time.perf_counter()
    engine.prefill(ctx, tokens)
    dt = time.perf_counter() - t0
    row = {
        "tokens": len(tokens),
        "step": step,
        "cache_mb": round(engine.cache_bytes(ctx) / 1e6, 1),
        "peak_over_base_mb": round((mx.get_peak_memory() - base) / 1e6, 1),
        "peak_gb": round(mx.get_peak_memory() / GB, 3),
        "prefill_tps": round(len(tokens) / dt, 1),
    }
    del ctx
    mx.clear_cache()
    return row


def decode_tps(engine: Engine, mode: str, prompt: list[int], n: int = 160) -> tuple[float, GenStats]:
    ctx = engine.new_context()
    stats = GenStats()
    t_first = None
    count = 0
    for _ in engine.generate(ctx, prompt, n, policy=Fixed(mode), sampler=engine.greedy, stats=stats):
        count += 1
        if t_first is None:
            t_first = time.perf_counter()  # exclude prefill and the first token
    dt = time.perf_counter() - t_first if t_first else 0.0
    return ((count - 1) / dt if dt > 0 and count > 1 else 0.0), stats


def run(cfg: Config, max_context: int = 0, log=print) -> dict:
    step_default = cfg.context.prefill_step
    cfg.generation.speculative = "on"  # load the draft so it can be measured
    mx.reset_peak_memory()
    engine = Engine(cfg, use_adapter=False)
    weights_gb = mx.get_active_memory() / GB
    wired_gb = engine.wired_limit_bytes / GB
    probe_cache = engine.new_context().cache
    layer_types = [type(c).__name__ for c in probe_cache]
    growing = sum(1 for t in layer_types if t == "KVCache")
    log(f"model {cfg.model.main}: {len(layer_types)} layers, {growing} with a growing KV cache, hybrid={engine.hybrid}")
    log(f"weights (main + draft) {weights_gb:.2f} GB; wired limit {wired_gb:.2f} GB")

    # ---- memory ----
    max_context = max_context or max(cfg.context.budget_tokens, 8192)
    filler = engine.encode(FILLER)
    sizes = [n for n in (1024, 2048, 4096, 8192, 16384) if n <= max_context]
    rows = []
    for n in sizes:
        toks = (filler * (n // len(filler) + 1))[:n]
        for step in (128, 512, 2048):
            row = measure_prefill(engine, toks, step)
            rows.append(row)
            log(
                f"  prefill {n:6d} tok step {step:4d}: cache {row['cache_mb']:7.1f} MB, "
                f"peak +{row['peak_over_base_mb']:7.1f} MB, {row['prefill_tps']:7.1f} tok/s"
            )
    cfg.context.prefill_step = step_default
    at_default = [r for r in rows if r["step"] == step_default] or [r for r in rows if r["step"] == 512]
    cache_slope, cache_base = linear_fit([r["tokens"] for r in at_default], [r["cache_mb"] for r in at_default])
    # Chunked prefill adds a transient on top of the cache it is building. It does
    # not grow with context at a fixed chunk size, so it is taken as a bound (the
    # worst row) rather than fitted.
    transient_mb = max(r["peak_over_base_mb"] - r["cache_mb"] for r in at_default)
    headroom_mb = (wired_gb * SAFETY - weights_gb) * 1000
    available_gb = psutil.virtual_memory().available / GB
    ceiling = int((headroom_mb - cache_base - transient_mb) / cache_slope) if cache_slope > 0 else 0
    by_step = {}
    for step in (128, 512, 2048):
        big = [r for r in rows if r["step"] == step and r["tokens"] == sizes[-1]]
        by_step[step] = big[0]["peak_over_base_mb"] if big else None

    # ---- decode loops ----
    short = engine.render_chat([{"role": "user", "content": SPEC_PROMPTS[0]}])
    decode_tps(engine, "ahead1", short, 32)  # warm up
    loops = {m: round(decode_tps(engine, m, short)[0], 2) for m in ("ahead1", "sync")}
    log(f"decode: ahead1 {loops['ahead1']} tok/s, sync {loops['sync']} tok/s")

    # ---- speculative ----
    spec = {"supported": engine.spec_supported}
    if engine.spec_supported:
        base_tps, spec_tps, drafted, accepted = [], [], 0, 0
        for p in SPEC_PROMPTS:
            prompt = engine.render_chat([{"role": "user", "content": p}])
            base_tps.append(decode_tps(engine, "ahead1", prompt)[0])
            tps, st = decode_tps(engine, "spec", prompt)
            spec_tps.append(tps)
            drafted += st.drafted
            accepted += st.accepted
        spec.update(
            ahead1_tps=round(sum(base_tps) / len(base_tps), 2),
            spec_tps=round(sum(spec_tps) / len(spec_tps), 2),
            acceptance=round(accepted / drafted, 3) if drafted else 0.0,
            num_draft_tokens=cfg.generation.num_draft_tokens,
        )
        spec["speedup"] = round(spec["spec_tps"] / spec["ahead1_tps"], 3) if spec["ahead1_tps"] else 0.0
        log(
            f"speculative: {spec['spec_tps']} vs {spec['ahead1_tps']} tok/s "
            f"(x{spec['speedup']}, acceptance {spec['acceptance']:.0%})"
        )

    result = {
        "model": cfg.model.main,
        "draft": cfg.model.draft,
        "device": engine.device_name,
        "created": now_iso(),
        "architecture": {
            "layers": len(layer_types),
            "kv_layers": growing,
            "recurrent_layers": len(layer_types) - growing,
            "hybrid": engine.hybrid,
            "trimmable": not engine.hybrid,
        },
        "memory": {
            "weights_gb": round(weights_gb, 3),
            "wired_limit_gb": round(wired_gb, 2),
            "available_gb_at_run": round(available_gb, 2),
            "cache_kib_per_token": round(cache_slope * 1e6 / 1024, 2),
            "cache_base_mb": round(cache_base, 1),
            "prefill_transient_mb": round(transient_mb, 1),
            "prefill_step": at_default[0]["step"],
            "peak_mb_by_step_at_max": by_step,
            "context_ceiling_tokens": ceiling,
            "rows": rows,
        },
        "decode": loops,
        "speculative": spec,
    }
    write_json(cfg.state_dir / "phase0.json", result)
    write_report(result, ROOT / "reports" / "phase0.md")
    return result


def write_report(r: dict, path: Path) -> None:
    a, m, s = r["architecture"], r["memory"], r["speculative"]
    lines = [
        f"# Phase 0 measurements: {r['model']}",
        "",
        f"Measured {r['created']} on {r['device']}. Regenerate with `mack phase0`.",
        "",
        "## Architecture",
        f"- {a['layers']} layers: {a['kv_layers']} keep a growing KV cache, {a['recurrent_layers']} carry fixed-size recurrent state.",
        f"- Cache trimmable: **{a['trimmable']}**. "
        + ("Token-level eviction is not available; context is reduced by rebuild." if not a["trimmable"] else ""),
        "",
        "## Memory",
        f"- Weights (main + draft): {m['weights_gb']} GB. Wired limit: {m['wired_limit_gb']} GB.",
        f"- Cache growth: {m['cache_kib_per_token']} KiB/token (base {m['cache_base_mb']} MB).",
        f"- Prefill transient above the cache at chunk {m['prefill_step']}: {m['prefill_transient_mb']} MB, bounded (see table: it does not grow with context).",
        f"- Peak at the largest context by chunk size (MB over resting): {m['peak_mb_by_step_at_max']}.",
        f"- Context ceiling at {int(SAFETY * 100)}% of the wired limit: **{m['context_ceiling_tokens']:,} tokens** "
        f"(weights + cache + transient). Available memory when measured: {m['available_gb_at_run']} GB.",
        "",
        "| context | chunk | cache MB | peak over resting MB | prefill tok/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {x['tokens']} | {x['step']} | {x['cache_mb']} | {x['peak_over_base_mb']} | {x['prefill_tps']} |"
        for x in m["rows"]
    ]
    lines += ["", "## Decode", f"- ahead1 (one token in flight): {r['decode']['ahead1']} tok/s", f"- sync: {r['decode']['sync']} tok/s", ""]
    lines.append("## Speculative decoding (snapshot-rewind)")
    if s.get("supported"):
        lines.append(
            f"- {s['spec_tps']} tok/s vs {s['ahead1_tps']} tok/s plain: **x{s['speedup']}**, "
            f"acceptance {s['acceptance']:.0%} with {s['num_draft_tokens']} draft tokens."
        )
        lines.append("- `speculative = \"auto\"` enables it only at x1.05 or better.")
    else:
        lines.append("- Not supported for this model/draft pair.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
