"""Live engine tests against the small draft model. Skipped if weights are not cached.

Run with: pytest tests/test_engine_live.py -m live
"""

import itertools

import mlx.core as mx
import pytest

from mack.config import Config
from mack.engine import Decision, Engine, GenStats

pytestmark = pytest.mark.live

SMALL = "mlx-community/Qwen3.5-0.8B-4bit"


class Cycle:
    def __init__(self, modes):
        self._it = itertools.cycle(modes)

    def decide(self):
        return Decision(mode=next(self._it))


@pytest.fixture(scope="module")
def engine():
    cfg = Config()
    cfg.model.main = SMALL
    cfg.model.draft = SMALL
    cfg.model.adapter = None
    cfg.generation.speculative = "on"
    try:
        return Engine(cfg)
    except Exception as e:  # weights not downloaded
        pytest.skip(f"model unavailable: {e}")


def run(engine, modes, max_tokens=48):
    prompt = engine.render_chat(
        [{"role": "user", "content": "List the first eight prime numbers, then say done."}]
    )
    ctx = engine.new_context()
    stats = GenStats()
    out = list(engine.generate(ctx, prompt, max_tokens, policy=Cycle(modes), sampler=engine.greedy, stats=stats))
    return prompt, ctx, out, stats


def next_token(engine, tokens):
    ctx = engine.new_context()
    engine.prefill(ctx, tokens)
    logits = engine.model(mx.array([engine.eos_id])[None], cache=ctx.cache)
    return mx.argmax(logits[0, -1]).item()


def consistent(engine, ctx):
    """The live cache behaves like a fresh cache prefilled with the same tokens."""
    live = engine.model(mx.array([engine.eos_id])[None], cache=ctx.cache)[0, -1]
    return mx.argmax(live).item() == next_token(engine, ctx.tokens)


def test_modes_agree_and_cache_matches_tokens(engine):
    prompt, ctx_a, ahead, _ = run(engine, ["ahead1"])
    assert consistent(engine, ctx_a)
    _, _, sync, _ = run(engine, ["sync"])
    assert sync == ahead
    # Batched verification can break exact logit ties differently from single-token
    # steps, so speculative output is checked for state consistency, not identity.
    for modes in (["spec"], ["ahead1", "sync", "spec", "spec", "sync", "ahead1"]):
        _, ctx, out, stats = run(engine, modes)
        assert ctx.tokens == prompt + out + [engine.eos_id], modes
        assert stats.accepted > 0
        assert consistent(engine, ctx), modes


def test_length_limit_closes_turn(engine):
    for modes in (["ahead1"], ["sync"], ["spec"]):
        prompt, ctx, out, stats = run(engine, modes, max_tokens=7)
        assert len(out) == 7 and stats.finish == "length"
        assert ctx.tokens == prompt + out + [engine.eos_id]


def test_spec_state_survives_rejections(engine):
    """Force rejections by sampling the main model at high temperature; the cache must
    still equal the emitted tokens."""
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(7)
    prompt = engine.render_chat([{"role": "user", "content": "Write a short poem about rain."}])
    ctx = engine.new_context()
    stats = GenStats()
    out = list(
        engine.generate(ctx, prompt, 60, policy=Cycle(["spec"]), sampler=make_sampler(temp=1.3), stats=stats)
    )
    assert stats.accepted < stats.drafted  # some rejections happened
    assert ctx.tokens == prompt + out + [engine.eos_id]
    assert consistent(engine, ctx)
