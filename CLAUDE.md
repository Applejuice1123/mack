# Mack — standing rules

The main product is **`mack run`**: one instruction applied to every file in a folder,
grounded, paced and reversible (`SPEC.md`). Chat, the screen explainer and long tasks
remain in the code as secondary features (`docs/explainer.md`). Decisions and
measurements are in `FINDINGS.md`; the build plan is `mack/PLAN.md`.

## Commands

```bash
.venv/bin/python -m pytest -q                 # stub suite: no model, no GPU, well under a minute
MACK_RUN_MODEL=1 .venv/bin/python -m pytest tests/test_model.py -k invoices -s   # one real-model test at a time
MACK_STUB_ENGINE=1 .venv/bin/python -m mack.cli run --wait "..." DIR            # try the CLI without a model
.venv/bin/python -m mack.cli bench            # pace contract table (real model, minutes)
.venv/bin/python scripts/e2e_tuesday.py --n 30   # E2E-1 rehearsal
.venv/bin/python -m mack.cli doctor           # what is measured and configured
.venv/bin/python -m mack.cli phase0           # re-measure memory, loops, speculative
.venv/bin/python -m mack.cli calibrate        # re-measure decode -> CPU slowdown
.venv/bin/python -m mack.cli serve            # UI on http://127.0.0.1:8082
app/build.sh run                              # build and open the menu-bar explainer
app/Mack.app/Contents/MacOS/Mack --selftest   # OCR round-trip, no permissions needed
```

## Rules that come from measurements

1. **Engine is MLX, not llama.cpp.** They are separate engines; do not add a
   llama.cpp path. Basis: odi2027 N27 (llama.cpp spent 58% more energy per token
   than a pipelined MLX loop on this machine).
2. **KV cache stays fp16.** There is no `kv_bits` option and the config loader
   rejects unknown keys. Quantizing the cache raised peak memory during prefill
   (contention_harness F28).
3. **Never run the engine at background priority.** No `taskpolicy -b`, no
   background or utility QoS, no `nice`. It moves the submitting thread to the
   E-cluster and slows decode (odi2027 N12, N22 to N24). `ensure_foreground()`
   and `set_thread_user_initiated()` in `governor.py` enforce this at startup.
4. **Eviction is compaction, never cache surgery.** Do not delete, trim or
   reorder entries in a live cache. Qwen3.5 is a hybrid: its recurrent layers
   cannot be trimmed, and dropping attention entries while the recurrent state
   still holds them is a state the model never trained on. Context shrinks only
   by `Session._compact()`, which rebuilds and re-prefills.
5. **Compaction is rare and big.** Rebuilding costs a full prefill. Do not lower
   `high_water` or add incremental trimming to "smooth" it.
6. **Pins are exact quotes.** A pin must be a verbatim substring of the turn it
   cites. Never paraphrase, merge or "clean up" a pin.
7. **Unverified summary lines are dropped, not repaired.** `verify.py` is biased
   toward dropping because the original is still in the archive. Never
   re-summarize a summary.
8. **Nothing is destroyed.** Every turn is written to the archive when it
   happens. Compaction changes what is in the live context, not what exists.
9. **The adapter loads only through the gate.** `Engine` loads
   `adapters/mack` only if `SHIP.json` says ship. Training deletes `SHIP.json`.
   Do not bypass this in app code; `use_adapter=True` is for evals only.
10. **Measurements are per model.** `state/phase0.json` and
    `state/calibration.json` are ignored unless their `model` field matches
    `mack.toml`. After changing the model, rerun `phase0` and `calibrate`
    before trusting the budget ceiling, rate cap or loop threshold.
11. **Use the warm probe reading.** `CpuProbe.warm()` takes two bursts and
    returns the second. A single burst after an idle gap measures the DVFS ramp,
    not contention (odi2027 N10).
12. **One thread owns MLX.** The server's worker thread builds the engine and
    runs every model call. HTTP handlers only pass messages.
13. **Never bypass the training preflight or watchdog.** LoRA training is the
    one operation that can push this machine into swap, and swap lands on a
    nearly full disk: the first 4B attempt consumed 4 GB of swap in five
    minutes. `mack lora train` refuses to start under 6 GB free disk or 8 GB
    available RAM and kills itself if swap grows. Run it from the CLI, not by
    calling `train.run` with the limits lowered.

14. **The explainer is stateless and local.** `explain()` builds a fresh context
    per request and never writes to the archive or a session. The app sends
    captured text only to 127.0.0.1 and deletes the screenshot right after OCR.
    Do not add history or a remote endpoint. The one exception is the opt-in
    debug record (`defaults write ai.mack.explainer Debug -bool YES`), which
    keeps only the last capture, its text and the reply in `state/debug/`. It
    must stay off by default; turn it off and delete the folder when done.
15. **Do not trust `CGPreflightScreenCaptureAccess` alone.** Without real access
    macOS still returns a capture, containing only the desktop picture. The app
    checks `canSeeOtherWindows()` as well, and `--diagnose` (run through
    `open -n Mack.app --args --diagnose <file>`) is the ground truth.
16. **Keep the signing requirement in `app/build.sh`.** The designated
    requirement is `identifier "ai.mack.explainer"`; with a plain ad-hoc
    signature every rebuild silently drops the Screen Recording grant.

## Rules for `mack run`

17. **Never write to a user's file outside `mack/apply.py`.** Runs only read. Apply
    checks every target's hash against the one recorded at enqueue, writes and
    fsyncs the undo log, and only then changes anything. Undo deletes only files
    Mack created that still match their recorded hash.
18. **Every item is independent.** Its context is the preamble, the instruction,
    the schema and that one file. Never add earlier items, a plan or a digest.
19. **Grounding is mechanical and biased to flag.** A value the file does not
    support is nulled and flagged, never repaired by the model. An unverifiable
    value (an ambiguous numeric date) is flagged even if it may be right. Do not
    loosen `schema.py:ground` to raise the "done" count.
20. **Prefill counts against the pace** (`batch.prefill_weight`). Do not add a
    path that prefills at full speed under a capped pace.
21. **Item rows are written before their status.** Output first, then `done`,
    then the JSONL line; the JSONL is rebuilt from the store on resume.
22. **Tests use the stub.** `mack/stub.py:StubEngine` with a fake clock; subprocess
    tests set `MACK_STUB_ENGINE=1`, `MACK_STUB_RESPONDER=tests.stub_responders:...`
    and `MACK_FAKE_POWER=ac` (this laptop is often on battery, which demotes `full`).
    Real-model tests are `@pytest.mark.model`, opt-in, run one at a time.

23. **`auto` is the default pace and must stay adaptive** (`mack/pacing.py`): full
    speed only when you have been away a minute and other apps are quiet; balanced
    while you use the Mac; gentle when your apps are busy; hold on heavy load,
    serious heat (60 s, then re-check) or critical memory. Memory "warn" is normal
    on this 16 GB Mac and must not slow anything down.
24. **Short GPU bursts while the user is present.** Prompt chunks are
    `background.active_chunk` (128) unless the pace is running at full.
25. **Give memory back.** Background engines are `LazyEngine`s: unloaded after
    `unload_after_idle_s` with nothing to do, and on critical memory pressure (the
    item in progress is put back, its attempt not counted). A memory release is
    never an engine failure.
26. **Power comes only from `sudo -n /usr/bin/powermetrics`** (`mack/power.py`),
    which the user allowed through sudoers. Never run anything else with sudo.

## Resource rules

- Free disk is often under 1 GB. Before writing more than 200 MB (models, training,
  big outputs) check `df -h ~` and stop below 2 GB free. `mack model download` and
  `mack bench` enforce this.
- No LoRA training unless the user asks; never in parallel with model tests.

## Invariants the tests hold

- `ctx.tokens` is exactly what the main cache has consumed, in every decode
  mode, including after cancel, length limit and speculative rejections
  (`tests/test_engine_live.py`).
- The live context always ends right after an end-of-turn token; turns are
  appended as deltas, never by re-rendering the history.
- Sync and pipelined loops produce identical greedy output. Speculative output
  may differ only at exact logit ties, so it is tested for state consistency.

## Working conventions

- Numbers in `README.md`, `FINDINGS.md` and `reports/` come from runs on this
  machine. Do not edit them by hand; rerun the command that produced them.
- Open choices that belong to the user: the license, and any change to the
  on-disk format of jobs, items or the undo log.
- Disk is tight (the 9B model needs about 7 GB free). Check `df -h ~` before
  downloading weights or starting a LoRA run.
- No new config keys without a consumer; no dead options.
