# FINDINGS

Working notes for building `mack run` (PLAN.md). Phase 0 maps what existed; later
sections record decisions, measurements, and where reality differed from the plan.
Function names are `module.py:name`.

## Phase 0: the engine as found (2026-10-22, before any change)

Full suite on the starting tree: **46 passed** (8.3 s; the 3 live tests loaded the 0.8B model).

### 1. Engine entry points
- Model load: `engine.py:Engine.__init__` (mlx-lm `load`, optional draft model, chat format via `prompts.py:detect_format`).
- Prefill: `engine.py:Engine.prefill`, chunked by `config.context.prefill_step` (512), each chunk evaluated then `mx.clear_cache()`.
- Decode: one generator, `engine.py:Engine.generate`, choosing a loop per token from `policy.decide()`
  (`engine.py:Decision.mode`): `ahead1` (pipelined), `sync`, `spec` (`engine.py:Engine._spec_round`,
  snapshot-rewind speculative decoding). Callers pick by passing a policy (`calibrate.py:Fixed`, `tasks.py:_Sync`, the governor).
- Invariant: `LiveContext.tokens` equals what the cache consumed (tests/test_engine_live.py).

### 2. Governor
- `governor.py:Governor.decide` returns `Decision(mode, rate_cap)`; the cap comes from `rate_for_target(ruler_cells(calibration), target, min_tps)`.
- Load signal: `Governor.sample_once` sums other processes' CPU time per process (not the system counter, which counted kernel/thermal work as "other load").
- Efficiency-core probe fix: `native/mack_probe.c` runs at user-interactive QoS; readings use `probe.py:CpuProbe.warm` (second of two bursts).
- Pace presets did **not** go through the governor: `tasks.py:pace_rate` maps a pace to a CPU-slowdown target and calls `rate_for_target`; `TaskRunner` paces itself. Chat's cap was per-process and could not be set per job.

### 3. Task queue and durability
- Store: **one JSON file per task**, `state/tasks/<id>/task.json` (`tasks.py:TaskStore`), not SQLite. Control requests in a separate `control` file (`TaskStore.set_control`).
- Checkpoint: `TaskRunner._save` after every unit. Fresh context per unit: `TaskRunner._write_section` / `_write_file` build brief + outline (`_outline`) + digest (`_digest`).
- Resume: `TaskScheduler.tick` → `TaskStore.next_runnable` picks queued/running tasks; units left `writing` go back to `pending` in `TaskRunner.steps`.
- Runs through `cli.py:cmd_work` (foreground) or the server's `server.py:Worker._run` (interleaved with interactive requests).

### 4. Battery and rest
- Battery: `tasks.py:battery_low` (pmset), checked in `TaskRunner._check` once a minute; paused via the control file and needed a manual resume.
- Rests: `TaskRunner._generate` yielded `busy * (1/duty - 1)` after each prefill chunk; decode tokens paced at `1/rate`. Prefill did **not** count against the rate.

### 5. Self-review
- Sections: `TaskRunner._write_section` → review prompt → at most one revision. Code: `tasks.py:syntax_error` (ast/json only), `missing_names`, one repair.

### 6. Recall and compaction
- SQLite FTS5 table `passages(text, session, turn, part, role, ts)` in `archive.py:Archive`; compaction in `session.py:Session._compact`. Not needed by any recipe.

### 7. Interfaces
- CLI: argparse in `cli.py:main`. Dashboard: FastAPI `server.py:create_app` on 8082 reading `TaskStore` files. Swift app polls `GET /api/tasks` and reads `title, status, progress, units_done, units_total, current`.

### 8. Tests
- 46 tests; a fake engine existed only in `tests/fakes.py` (canned replies, no rate, no clock). Real-model tests: `tests/test_engine_live.py` (0.8B).

### Design for mack run
- **One task with N items** (plan's default). The dashboard, `mack task list` and the menu bar already understand one task's progress; a batch job is a `Task` with `kind="batch"` and its items in SQLite.
- **Per-piece context builder**: not reusable as is (it assumes a plan and a digest). Items use a new, smaller builder (`batch.py:BatchRunner._process`): preamble + instruction + schema + file. The paced generator `TaskRunner._generate` *is* reused (BatchRunner subclasses TaskRunner).
- **Constrained decoding**: none on MLX here. Built **generate → parse → validate → one retry with the error** (`batch.py:BatchRunner._process`). Logit masking against a JSON grammar would need a token-level automaton over the 248K vocabulary inside `Engine.generate`; not worth it given the retry works.
- **Compatibility**: task.json gains one optional field (`batch`), defaulting to `{}`; old task files load unchanged. Item rows are a new file, `state/batch.db`.

## Decisions made where the plan left a choice

| decision | choice | why |
|---|---|---|
| batch = one task or many | one task, items in `state/batch.db` (table `items`) | existing progress plumbing works unchanged; one row per item keeps per-item commits cheap (rewriting a 2,400-unit JSON per item would be quadratic) |
| item store | SQLite with WAL, separate from task.json | the plan expected SQLite; task.json stays the job's header so nothing else had to change |
| decoding route | generate, validate, retry once | see above |
| prefill weighting | 0.25, `mack.toml [batch] prefill_weight` | the plan's number; prefill also keeps the old duty-cycle rest, whichever is longer (`TaskRunner._generate`) |
| governor cap per job | new `Governor.job_cap`, set/restored by `BatchRunner.steps`; chat's `rate_cap` untouched | the plan wants both "set the governor's cap" and "chat's cap untouched"; a separate slot satisfies both |
| battery pause | control value `battery`, auto-resumes at 30% or on power (`TaskStore.next_runnable`) | the plan's test requires 30% to resume without the user |
| ambiguous dates | a date matching only a form like 03/09/2025 is kept but flagged (`schema.py:dates_in`, `ground`) | found by the Phase 2 model test, below |
| input size | files over 200 KB skipped; text over 6,000 tokens shown head + tail and marked `_truncated` | a 200 KB note is ~50K tokens: ~1.5 h of prefill per file at gentle |
| recipe format | YAML (PyYAML already installed with mlx-lm's dependencies) | as planned |
| PDFs | PDFKit text layer, OCR of the first 3 pages otherwise, both inside the Swift helper (`Mack --pdf-text`) | no PDF library was installed; reuses the explainer's Vision code |
| OCR scope | images/PDFs are read only by recipes with `ocr: true` (rename) or `--ocr`; otherwise skipped `unsupported_type` | keeps Phase 1's "skip PNG" behaviour for plain runs |
| `model`/`live` tests | both skipped unless `MACK_RUN_MODEL=1` | the plan's "suite runs without a model"; the 0.8B live tests used to run by default |
| test hooks | `MACK_STUB_ENGINE`, `MACK_STUB_RESPONDER`, `MACK_FAKE_POWER`, `MACK_TEST_CRASH_AFTER_UNDO_LOG` | subprocess CLI tests need them; this laptop is often on battery, which (correctly) demotes `full` to `steady` |
| E2E-4 | `mack bench` (time-boxed per pace) instead of exactly 20 items per pace | same probe, same table; time-boxing gives every pace an equal measurement window |
| git | not initialised | the user asked to build in code first and do git later |
| license | not chosen | reserved for the user (MIT or Apache-2.0) |

## Measurements

### Phase 2 model test: 30 invoices, vendor + date + amount all `x-grounded` (Qwen3.5-4B-4bit, full pace)
| run | right | flagged | wrong but unflagged | failed |
|---|---:|---:|---:|---:|
| 2026-10-22 00:40, first build | 26 | 0 | **4** | 0 |
| 2026-10-22 00:44, after the date fix | 26 | 4 | **0** | 0 |

All four misses were dates written day-first (`03/09/2025` = 3 September) read US-style. The first grounding
check accepted either reading, so it could not catch it. Now a date supported only by an ambiguous numeric form
is flagged for review (value kept, `_ambiguous: ["date"]`). The bar is ≤ 2 of 30 wrong-unflagged: met.

### Not yet run (each needs the real model for minutes to hours; see RELEASE_NOTES.md)
- Phase 4 `mack bench` pace table and the contract check.
- Phase 5 recipe model test (`tests/test_model.py -k recipes`).
- E2E-1 (`scripts/e2e_tuesday.py`), E2E-2/3/5 (`tests/test_model.py`), the YouTube-and-scrolling feel test, the fresh-eyes quick start.

## Where reality differed from the plan

- **Pace ↔ slowdown pairing.** The plan pairs gentle 2.3 / balanced 4.5 / steady 9.7 tok/s with ~12% / ~22% / ~45%.
  The calibration (reports/calibration.md) gives ~5% / ~10% / ~20% at those rates; 45% is uncapped and 12% is 5.4 tok/s.
  The Phase 4 contract (15 / 25 / 30%) is therefore loose for this machine, which is fine; `mack bench` will confirm.
- **Gentle cannot finish 2,400 notes overnight.** With prefill weighted at 0.25, a 650-token note costs
  (0.25 × 650 + ~100 out) / 2.3 ≈ 115 s at gentle: about 77 hours for 2,400 notes. Balanced is ~38 h, steady ~18 h.
  The plan's scenario (start Tuesday morning, review Wednesday) fits `steady`, or `gentle` with `--boost-when-idle`
  overnight. Not changed; the plan line prints the honest estimate.
- **Task store** was JSON files, not SQLite (above). **Stub engine** did not exist in the package; `mack/stub.py` added.
- **The CPU probe is not pinned to cores**; it runs at user-interactive QoS so macOS keeps it on the performance cores.
- **"mack model download"** did not exist; added (`cli_batch.py:cmd_model`, refuses without enough free disk).
- **The laptop's own battery** throttled the first CLI test run from 14 s to 104 s: `full` on battery runs at `steady`
  by design. Tests fake mains power.

## Auto pace and power (2026-10-22, after the user enabled powermetrics)

**What heats the Mac and what slows it are different things.** Measured beside the CPU probe, with
the owner's Chrome tab busy in the background:

| continuous | memory-heavy slowdown | compute-heavy slowdown | added package power |
|---|---:|---:|---:|
| prompt reading (prefill, 310 tok/s) | 12% | 4% | +17.8 W |
| generation (sync loop) | 38% | 2% | +7.0 W |

`mack calibrate --energy` then stored 0.051 J per prompt token and 0.257 J per generated token
(`state/energy.json`). The plan's fixed 0.25 prompt weighting was replaced by a **power budget**:
prompt reading and generation spend one budget of average extra watts (auto: 3 W active, 1.5 W busy,
uncapped away on power, 6 W away on battery; fixed paces 1.5 / 3 / 6 W), while the token-rate cap still
limits the memory-heavy slowdown. Prompt chunks are 128 tokens unless running at full.

**Thresholds.** The first `mack bench` (40 s per pace) was taken while a Chrome renderer used a full
core. With `busy_duty = 0.25` / `hold_duty = 0.6`, Chrome plus the probe tipped auto into "hold" 46–75%
of the time, and the full row too. One busy tab is not heavy load (it is barely slowed by Mack), so
the defaults became 0.35 / 0.90 with hysteresis (0.1 lower to leave a level). Those bench numbers are
not quoted anywhere: the power baseline also omitted the probe's own draw (fixed), and the Mac was not quiet.

**Real job while in use.** 12 fixture notes (100–2,000 words), tag recipe, auto pace, on battery,
owner using the Mac:

| version | wall time | finished | whole-Mac power | thermal |
|---|---:|---:|---:|---|
| fixed weighting, old thresholds | 11 min for 4 of 12 (stopped) | 3 done, 1 failed | — | — |
| power budget, new thresholds | **5 min 53 s** for 12 | 12 done | **2.46 W mean** | Nominal |

Levels used in the second run: balanced while active, gentle while Chrome was busy, steady during short
away spells (on battery). The failed item in the first run was a 200-character limit on a free-text
summary; over-long free text is now trimmed at a word boundary (grounded values never are).

Still to do: `mack bench` on a quiet Mac (it now warns when other apps are busy), so the fixed-pace
table has clean numbers.

## Demo run on generated files (2026-10-22, real 4B, on battery so "full" ran at steady)

31 generated files: 8 notes, an inbox of 20 (8 invoice PDFs, half image-only; 6 screenshots; 4 notes;
2 blank images), 3 messy transcripts.

- **rename** (20 files, 2 min 39 s): 7 of 8 invoice dates and 8 of 8 vendors exact; the 8th ("20/10/2025",
  read correctly by OCR) came back with no date rather than a wrong one. Both blank images skipped
  (`no_text`). Apply then undo restored all 18 names.
- **tag** (8 notes, 2 min 3 s) and **clean** (3 transcripts, 1 min 13 s): all done, none flagged; clean kept
  every sentence and quoted each action item from the transcript.
- **Bug found:** critical memory pressure while the model loaded raised `Abandon` between items, outside
  the handler, and failed the job. Fixed (the hold check is inside the handler) and the reload now backs
  off at least 60 s, doubling on repeats, so loading cannot thrash. Test: `test_memory_release_between_items…`.
- **Quality problems found and fixed:** rename titles repeated the sender ("Invoice #1000 from Northwind
  Traders" after "Northwind Traders") and plain notes were renamed "undated Note 1"; titles now drop the
  sender, date and "#", and files with neither a date nor a sender keep their names. Tag used all three
  slots on nearly every note; the instruction now asks for usually one tag. Rerun on fresh files: 7 of 8
  notes got one correct tag, 1 got two; renames like "2025-10-11 Northwind Traders Invoice for consulting
  services and support plan.pdf"; 7 unidentifiable files kept their names. One screenshot that was dated in
  the first run came back undated in the second (the model is more willing to say null), which is the safe
  direction.
