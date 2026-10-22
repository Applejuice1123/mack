# Contributing

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # stub suite, no model needed
app/build.sh                           # optional: the helper app (OCR, PDFs, menu bar)
```

## Tests: stub vs model

Almost everything runs against `mack/stub.py:StubEngine`, which has the same interface as the
real engine, returns replies from a function you give it, and can simulate time with a fake
clock. That keeps the suite fast and deterministic.

- In-process: `StubEngine(cfg, responder=lambda prompt: '{"text": "..."}', clock=FakeClock())`
  (see `tests/batch_helpers.py`).
- Subprocess (CLI tests): `MACK_STUB_ENGINE=1`, `MACK_STUB_RESPONDER=tests.stub_responders:name`,
  `MACK_FAKE_POWER=ac` (otherwise a laptop on battery runs `full` at `steady`, by design).

Tests that need the real model are marked `@pytest.mark.model` and skipped unless
`MACK_RUN_MODEL=1`. Run them one at a time; each records its numbers in `reports/model_results.json`.

Every behaviour change comes with a test. The three guarantees each have a test file:
`test_grounding.py`, `test_pace.py`, `test_batch_resume.py` + `test_cli_batch.py`.

## Ground rules

Read `CLAUDE.md`: it lists what is settled by measurement (fp16 KV cache, no background priority,
MLX only) and the rules `mack run` depends on (only `apply.py` writes user files; items are
independent; grounding flags rather than repairs).

## Good first recipe

Write a recipe for a job you actually have, such as receipts to a CSV, meeting notes to action
items, or bookmarks to tags, and try it with `--limit 5 --pace full --wait`. If it works on your
files with few flags, open a pull request adding it to `mack/recipes/` with three fixture files
and a test in `tests/test_recipes.py`. Format reference: `docs/recipes.md`.
