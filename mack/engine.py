"""MLX inference engine: prefill, decode loops, and snapshot-rewind speculative decoding.

Three decode modes, chosen per token by a policy (see governor.py):

  ahead1  one token in flight: step n+1 is submitted before token n is read.
          Closes ~95% of the host-side idle gap (odi2027 N25) and is the default.
  sync    submit, wait, read. Slower when the machine is quiet, but past ~0.6 CPU
          duty the pipelined loop stops being faster and only costs more energy.
  spec    draft K tokens with the small model, verify them in one pass of the
          main model. mlx-lm's own speculative path needs a trimmable cache and
          refuses Qwen3.5's recurrent (Gated DeltaNet) layers, so this one
          snapshots both caches before a round and rewinds on a rejection.

Invariant kept across all three: `ctx.tokens` is exactly the token sequence the
main cache has consumed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .config import Config
from .prompts import detect_format

GB = 1e9  # decimal, to match the 11.45 GB wired limit quoted in the research


@dataclass
class Decision:
    mode: str = "ahead1"  # ahead1 | sync | spec
    rate_cap: Optional[float] = None  # tokens/s, None = unthrottled


class FreeRun:
    """Policy used when no governor is attached: always pipelined, never capped."""

    def __init__(self, mode: str = "ahead1"):
        self._d = Decision(mode=mode)

    def decide(self) -> Decision:
        return self._d


@dataclass
class LiveContext:
    cache: list
    tokens: list[int] = field(default_factory=list)
    draft_cache: Optional[list] = None
    draft_n: int = 0  # draft cache has consumed (tokens + pending)[:draft_n]


@dataclass
class GenStats:
    n_tokens: int = 0
    decode_s: float = 0.0  # whole call, including prefill of the pending prompt
    first_token_s: float = 0.0
    finish: str = ""
    modes: dict = field(default_factory=lambda: {"ahead1": 0, "sync": 0, "spec": 0})
    drafted: int = 0
    accepted: int = 0

    @property
    def tps(self) -> float:
        """Decode rate after the first token, so prompt prefill does not count."""
        dt = self.decode_s - self.first_token_s
        return (self.n_tokens - 1) / dt if dt > 0 and self.n_tokens > 1 else 0.0

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "decode_s": round(self.decode_s, 3),
            "first_token_s": round(self.first_token_s, 3),
            "tps": round(self.tps, 2),
            "finish": self.finish,
            "modes": dict(self.modes),
            "drafted": self.drafted,
            "accepted": self.accepted,
        }


def _snapshot(cache: list) -> list:
    return [
        (list(c.cache), c.left_padding, c.lengths) if isinstance(c, ArraysCache) else c.offset
        for c in cache
    ]


def _restore(cache: list, snap: list) -> None:
    for c, s in zip(cache, snap):
        if isinstance(c, ArraysCache):
            c.cache, c.left_padding, c.lengths = list(s[0]), s[1], s[2]
        else:
            c.offset = s


def _rewindable(cache: list) -> bool:
    return all(type(c) in (ArraysCache, KVCache) for c in cache)


def adapter_shipped(adapter_dir: Optional[Path]) -> bool:
    """An adapter is only loaded if its eval gate passed (lora/evals.py writes SHIP.json)."""
    if adapter_dir is None:
        return False
    ship = adapter_dir / "SHIP.json"
    if not ship.exists() or not (adapter_dir / "adapters.safetensors").exists():
        return False
    return bool(json.loads(ship.read_text()).get("ship"))


class Engine:
    def __init__(self, cfg: Config, use_adapter: Optional[bool] = None, load_draft: bool = True):
        self.cfg = cfg
        adapter_dir = cfg.adapter_dir
        if use_adapter is None:
            use_adapter = adapter_shipped(adapter_dir)
        self.adapter_loaded = bool(use_adapter)
        self.model, self.tokenizer = load(
            cfg.model.main, adapter_path=str(adapter_dir) if use_adapter else None
        )
        self.draft = None
        if load_draft and cfg.model.draft and cfg.generation.speculative != "off":
            self.draft, _ = load(cfg.model.draft)
        g = cfg.generation
        self.fmt = detect_format(self.tokenizer, g.enable_thinking)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.fmt.close)
        self.eos_ids = set(self.tokenizer.eos_token_ids) | {self.eos_id}
        self.sampler = make_sampler(temp=g.temperature, top_p=g.top_p, top_k=g.top_k)
        self.greedy = make_sampler(temp=0.0)
        info = mx.device_info()
        self.wired_limit_bytes = int(info["max_recommended_working_set_size"])
        self.device_name = info.get("device_name", "")
        # Freed buffers otherwise sit in MLX's pool and count against the wired limit.
        mx.set_cache_limit(int(1 * GB))
        probe = make_prompt_cache(self.model)
        self.hybrid = any(isinstance(c, ArraysCache) for c in probe)
        self.spec_supported = self.draft is not None and _rewindable(probe)

    # ---- tokens -----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def detok(self) -> "StreamDetok":
        return StreamDetok(self.tokenizer)

    # ---- context ----------------------------------------------------------

    def new_context(self) -> LiveContext:
        return LiveContext(cache=make_prompt_cache(self.model))

    def prefill(
        self,
        ctx: LiveContext,
        tokens: list[int],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Feed tokens through the main model in chunks, keeping peak memory flat."""
        step = self.cfg.context.prefill_step
        total = len(tokens)
        for i in range(0, total, step):
            chunk = tokens[i : i + step]
            self.model(mx.array(chunk)[None], cache=ctx.cache)
            mx.eval([c.state for c in ctx.cache])
            ctx.tokens.extend(chunk)
            mx.clear_cache()
            if progress:
                progress(min(i + step, total), total)

    def cache_bytes(self, ctx: LiveContext) -> int:
        n = sum(c.nbytes for c in ctx.cache)
        if ctx.draft_cache is not None:
            n += sum(c.nbytes for c in ctx.draft_cache)
        return n

    def memory(self) -> dict:
        return {
            "active_gb": round(mx.get_active_memory() / GB, 3),
            "peak_gb": round(mx.get_peak_memory() / GB, 3),
            "wired_limit_gb": round(self.wired_limit_bytes / GB, 2),
        }

    # ---- decode -----------------------------------------------------------

    def _step(self, cache: list, inp: mx.array, sampler) -> mx.array:
        logits = self.model(inp[None], cache=cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs)

    def generate(
        self,
        ctx: LiveContext,
        pending: list[int],
        max_tokens: int,
        policy=None,
        sampler=None,
        cancel: Optional[Callable[[], bool]] = None,
        stats: Optional[GenStats] = None,
    ) -> Iterator[int]:
        """Yield generated token ids (EOS excluded). `pending` are prompt tokens the
        cache has not consumed yet. On return the cache has consumed the prompt, the
        reply and a closing EOS, so the next turn can be appended directly."""
        assert pending, "generate() needs at least one pending token"
        policy = policy or FreeRun()
        sampler = sampler or self.sampler
        stats = stats if stats is not None else GenStats()
        P = list(pending)  # valid tokens not yet fed to the main cache
        inflight: Optional[mx.array] = None  # sampled-but-unread token (its input is fed)
        finish = "length"
        t0 = time.perf_counter()
        last_emit = t0
        try:
            while stats.n_tokens < max_tokens:
                if cancel and cancel():
                    finish = "cancel"
                    break
                d = policy.decide()
                mode = d.mode
                room = max_tokens - stats.n_tokens
                if mode == "spec" and not (self.spec_supported and d.rate_cap is None and room >= 2):
                    mode = "ahead1"

                if mode == "spec" and inflight is None:
                    k_max = min(self.cfg.generation.num_draft_tokens, room - 1)
                    emitted, done = self._spec_round(ctx, P, sampler, stats, k_max)
                    for tok in emitted:
                        stats.n_tokens += 1
                        stats.modes["spec"] += 1
                        if stats.n_tokens == 1:
                            stats.first_token_s = time.perf_counter() - t0
                        yield tok
                    if done:
                        finish = "stop"
                        break
                    continue

                if inflight is None:
                    if len(P) > 1:
                        self.prefill(ctx, P[:-1])
                    x = P[-1]
                    P.clear()
                    inflight = self._step(ctx.cache, mx.array([x]), sampler)
                    mx.async_eval(inflight)
                    ctx.tokens.append(x)
                y = inflight
                nxt = None
                if mode == "ahead1" and stats.n_tokens + 1 < max_tokens:
                    nxt = self._step(ctx.cache, y, sampler)
                    mx.async_eval(nxt)
                tok = y.item()
                if nxt is not None:
                    ctx.tokens.append(tok)
                else:
                    P.append(tok)
                inflight = nxt
                if tok in self.eos_ids:
                    finish = "stop"
                    break
                stats.n_tokens += 1
                stats.modes["ahead1" if nxt is not None else "sync"] += 1
                if stats.n_tokens == 1:
                    stats.first_token_s = time.perf_counter() - t0
                yield tok
                if stats.n_tokens % 256 == 0:
                    mx.clear_cache()
                if d.rate_cap:
                    wait = last_emit + 1.0 / d.rate_cap - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                last_emit = time.perf_counter()
        finally:
            stats.decode_s = time.perf_counter() - t0
            stats.finish = finish
            # Leave the context well-formed: everything emitted is fed, and the turn is closed.
            if finish != "stop":
                P.append(self.eos_id)
            if P:
                self.prefill(ctx, P)
            mx.eval([c.state for c in ctx.cache])

    def _spec_round(self, ctx: LiveContext, P: list[int], sampler, stats: GenStats, K: int):
        """One draft-and-verify round of at most K draft tokens (emits at most K+1).
        Mutates ctx and P; returns (emitted, finished)."""
        if ctx.draft_cache is None:
            ctx.draft_cache = make_prompt_cache(self.draft)
            ctx.draft_n = 0
        # Repeated rejections grow the pending lists; commit them before they get expensive.
        if len(P) > 2 * K + 1:
            self.prefill(ctx, P[:-1])
            del P[:-1]
        seq = ctx.tokens + P
        dp = seq[ctx.draft_n :]
        if len(dp) > 2 * K + 1:
            step = self.cfg.context.prefill_step
            for i in range(0, len(dp) - 1, step):
                chunk = dp[i : min(i + step, len(dp) - 1)]
                self.draft(mx.array(chunk)[None], cache=ctx.draft_cache)
                mx.eval([c.state for c in ctx.draft_cache])
            ctx.draft_n += len(dp) - 1
            dp = dp[-1:]

        snap_d, snap_m = _snapshot(ctx.draft_cache), _snapshot(ctx.cache)

        inp = mx.array(dp)
        lazy = []
        for _ in range(K):
            logits = self.draft(inp[None], cache=ctx.draft_cache)[:, -1, :]
            inp = mx.argmax(logits, axis=-1)
            lazy.append(inp)
        mx.eval(lazy)
        drafts = [a.item() for a in lazy]
        for i, tok in enumerate(drafts):
            if tok in self.eos_ids:
                drafts = drafts[: i + 1]
                break
        k = len(drafts)

        logits = self.model(mx.array(P + drafts)[None], cache=ctx.cache)[0, -(k + 1) :, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = sampler(logprobs).tolist()

        n = 0
        while n < k and t[n] == drafts[n]:
            n += 1
        stats.drafted += k
        stats.accepted += n

        # Draft cache consumed dp + drafts[:K-1] (all K steps ran even if EOS truncated the list).
        if n >= K - 1 and k == K:
            ctx.draft_n += len(dp) + K - 1
        else:
            _restore(ctx.draft_cache, snap_d)

        if n == k:
            ctx.tokens.extend(P + drafts)
            P.clear()
            if drafts[-1] in self.eos_ids:
                return drafts[:-1], True
            P.append(t[k])
            emitted = drafts + [t[k]]
        else:
            _restore(ctx.cache, snap_m)
            P.extend(drafts[:n] + [t[n]])
            emitted = drafts[:n] + [t[n]]
        if emitted[-1] in self.eos_ids:
            return emitted[:-1], True
        return emitted, False

    # ---- one-shot helpers (summaries, verification, evals) -----------------

    def render_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.cfg.generation.enable_thinking,
        )

    def complete(self, messages: list[dict], max_tokens: int = 256, policy=None) -> str:
        """Greedy one-shot completion on a throwaway cache."""
        ctx = self.new_context()
        out = list(
            self.generate(ctx, self.render_chat(messages), max_tokens, policy=policy, sampler=self.greedy)
        )
        del ctx
        mx.clear_cache()
        return strip_thinking(self.decode(out)).strip()

    def yes_probability(self, messages: list[dict]) -> float:
        """P(Yes) / (P(Yes) + P(No)) for the next token after the prompt."""
        ctx = self.new_context()
        toks = self.render_chat(messages)
        logits = self.model(mx.array(toks)[None], cache=ctx.cache)[0, -1, :]
        yes = [self.encode(w)[0] for w in ("Yes", "yes", " Yes")]
        no = [self.encode(w)[0] for w in ("No", "no", " No")]
        probs = mx.softmax(logits, axis=-1)
        p_yes = sum(probs[i].item() for i in set(yes))
        p_no = sum(probs[i].item() for i in set(no))
        return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.5


class StreamDetok:
    """Incremental detokenizer: add(token) returns the newly completed text."""

    def __init__(self, tokenizer):
        self._d = tokenizer.detokenizer
        self._d.reset()

    def add(self, token: int) -> str:
        self._d.add_token(token)
        return self._d.last_segment

    def flush(self) -> str:
        self._d.finalize()
        return self._d.last_segment


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


class ModelMissing(RuntimeError):
    pass


def make_engine(cfg: Config, **kwargs):
    """The real MLX engine, or the stub when MACK_STUB_ENGINE is set (tests, demos)."""
    from .stub import StubEngine, stub_enabled

    if stub_enabled():
        return StubEngine(cfg)
    try:
        from huggingface_hub import snapshot_download

        if not Path(cfg.model.main).exists():
            snapshot_download(cfg.model.main, local_files_only=True)
    except Exception as e:
        raise ModelMissing(
            f"No model found for {cfg.model.main}. Run `mack model download` (about 3.1 GB; needs 4 GB free)."
        ) from e
    return Engine(cfg, **kwargs)


class LazyEngine:
    """Loads the model on first use and can give its memory back.

    Background work wraps the engine in this so the ~3 GB of weights are only
    resident while there is something to do: `unload()` after a quiet spell or
    when macOS reports critical memory pressure, and the next call reloads.
    """

    def __init__(self, cfg: Config, **kwargs):
        self._cfg, self._kwargs, self._engine = cfg, kwargs, None
        self.loads = 0

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def weights_gb(self) -> float:
        """Size of the model's weights, from the files on disk (what loading will take)."""
        if not hasattr(self, "_weights_gb"):
            try:
                from huggingface_hub import snapshot_download

                path = Path(self._cfg.model.main)
                if not path.exists():
                    path = Path(snapshot_download(self._cfg.model.main, local_files_only=True))
                self._weights_gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9 or 3.0
            except Exception:
                self._weights_gb = 3.0
        return self._weights_gb

    def _get(self):
        if self._engine is None:
            self._engine = make_engine(self._cfg, **self._kwargs)
            self.loads += 1
        return self._engine

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def unload(self) -> None:
        if self._engine is None:
            return
        self._engine = None
        import gc

        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
