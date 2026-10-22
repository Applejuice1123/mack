# Mack 0.1.0: release notes (draft, not tagged)

## What was built

`mack run` applies one instruction, or one of five recipes (tag, summarize, rename, extract,
clean), to every file in a folder, one file per fresh model context, on Qwen3.5-4B-4bit via MLX.
Every output must match a JSON schema, and every field that claims to come from the file is checked
against it mechanically; anything unsupported is nulled and flagged for review. Jobs run at a chosen
pace, with prompt reading counted against it, pause on low battery and heat, and resume after kill,
quit or reboot without losing or repeating work. Results go to a JSONL file and a Markdown review
file; `mack apply` writes them into files only after an undo log is on disk, and `mack undo` reverses it.

## Measurements

Pace table: **not yet measured with a real job.** The rates and expected slowdowns come from this
machine's calibration (gentle 2.3 tok/s ≈ 5%, balanced 4.5 ≈ 10%, steady 9.7 ≈ 20%, full up to 45%
on memory-heavy work). `mack bench` is built to produce the real table.

Per-recipe correct / flagged / wrong-unflagged:

| test | correct | flagged | wrong, unflagged |
|---|---:|---:|---:|
| 30 invoices, vendor + date + amount grounded (`extract`-style schema), first build | 26 | 0 | 4 |
| same, after the ambiguous-date fix | 26 | 4 | **0** |
| tag, summarize, rename, clean, extract on the 40-file mixed set | not run | | |

## Decisions where the plan left a choice

- **One task with N items**, items in SQLite (`state/batch.db`): the dashboard and menu bar already show one task's progress.
- **Generate, validate, retry once** rather than grammar-constrained decoding: MLX has no constrained decoding here, and the retry is simple and testable.
- **Prefill weight 0.25**, plus the existing duty-cycle rests, as a config value.
- **A separate `job_cap` on the governor** so a job's pace is visible without touching chat's cap.
- **Battery pauses resume by themselves** at 30% or on power.
- **Ambiguous numeric dates are flagged**, not trusted (found by the model test).
- **License: not chosen**; MIT or Apache-2.0 is the user's call.
- **Git: not initialised**, at the user's request (code first). The plan's per-phase commits did not happen.

## Where the real model differed from the plan's assumptions

- The 4B read four day-first dates (03/09/2025) US-style. The first grounding check could not catch it because both readings appear in the text; now such dates are flagged.
- The plan pairs the pace rates with ~12% / ~22% / ~45% slowdown; calibration says ~5% / ~10% / ~20%. The contract (15 / 25 / 30%) is looser than needed.
- At gentle pace, prefill weighting makes a typical note cost about two minutes, so 2,400 notes take about three days, not one night. Steady (about 18 hours) or gentle with `--boost-when-idle` fits the plan's overnight scenario.

## What remains unverified

- `mack bench` (Phase 4 contract) and the YouTube-and-scrolling feel test.
- The Phase 5 recipe model test and E2E-1 (`scripts/e2e_tuesday.py`), E2E-2, E2E-3, E2E-5 (`tests/test_model.py`). All are written; none has been run.
- The fresh-eyes quick-start run from a clean clone.
- `pipx install` from a clean machine (`MACK_HOME`/`~/MACK` placement is implemented but untested outside the checkout).
- The Swift menu-bar app's batch display (it reads the unchanged `/api/tasks` fields), the 9B model, idle boost on a real idle Mac.
- Thermal pause on a real hot Mac (tested only with a faked thermal state).

## What a maintainer should do next

1. Plug in, close other apps, run `mack bench`, then `MACK_RUN_MODEL=1 pytest tests/test_model.py` one test at a time; record results in FINDINGS.md and replace the README pace table with measured numbers.
2. Run `scripts/e2e_tuesday.py --n 300` overnight, then the fresh-eyes quick start from a clean clone.
3. `git init`, pick the license, and tag `v0.1.0` once the Definition of done in mack/PLAN.md holds.
