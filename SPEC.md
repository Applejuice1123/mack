# Mack: specification

`mack run` applies one instruction to every file in a folder, one file per fresh model
context, writes results to a review file, and stays under a chosen CPU-slowdown budget.
The earlier design (chat, screen explainer, long tasks, engine and governor internals) is in
[docs/explainer.md](docs/explainer.md). Decisions and measurements are in [FINDINGS.md](FINDINGS.md).

## Guarantees

1. **Grounded.** Each output is derived from its own file only. Values that cannot be traced to the file are nulled and the item is flagged.
2. **Polite.** A job at pace P never exceeds P's token rate, prefill included, pauses on low battery and heat, and never runs at background priority.
3. **Durable and reversible.** Progress is checkpointed per item and survives kill, sleep and reboot. Nothing is written to user files before `mack apply`; every apply has an undo log on disk first.

## Data model

A batch job is a task (`state/tasks/<id>/task.json`, `mack/tasks.py:Task`) with `kind = "batch"`
and a `batch` object:

| field | meaning |
|---|---|
| `instruction` | the instruction (a recipe's, plus any extra text given on the command line) |
| `recipe` | recipe name or null |
| `schema` | the resolved JSON Schema for each item's output (`{"text": string}` when none was given) |
| `schema_path` | the `--schema` file, if any |
| `roots`, `glob` | what was enqueued |
| `ocr` | read images and PDFs |
| `default_grounded` | string fields are `x-grounded` unless the schema says otherwise (`extract`) |
| `checks` | extra checks, e.g. `{kind: retention, field: cleaned, min: 0.85}` |
| `options` | recipe options as given (`--option`) |
| `boost_when_idle`, `max_output_tokens` | per-job settings |
| `output_path` | the JSONL file |
| `counts` | `total, done, skipped, failed, flagged, pending`, refreshed after every item |
| `current_path`, `pace_note` | for status displays |
| `applied` | `{undo_log, changes}` after `mack apply`, null after `mack undo` |

`status` is `queued | running | paused | done | failed | cancelled`. Pause, resume and cancel are
requests written to `state/tasks/<id>/control` (`pause`, `cancel`, or `battery` for an automatic
battery pause); only the runner writes `task.json`.

Items are rows in `state/batch.db`, table `items` (`mack/batch.py:BatchStore`):

| column | meaning |
|---|---|
| `job_id`, `idx` | primary key; `idx` is enqueue order |
| `path`, `content_hash` | absolute path; sha256 of the file's bytes at enqueue |
| `status` | `pending | running | done | skipped | failed | flagged` |
| `reason` | why skipped or failed: `too_large`, `binary`, `unsupported_type`, `no_text`, `ocr_unavailable`, `changed_since_enqueue`, `missing`, `invalid_json: …`, `schema: …` |
| `attempts` | incremented when processing starts; an item interrupted twice is failed |
| `output`, `flags` | the validated, grounded JSON; names of fields that were nulled or are unverifiable |
| `error` | engine errors |
| `started_at`, `finished_at`, `tokens_in`, `tokens_out` | per-item accounting |
| `text` | text recognized by OCR at enqueue, cached so it is not recognized twice |

## Item processing (`BatchRunner._process`)

1. Skip if the file is gone or its hash changed since enqueue (`changed_since_enqueue`).
2. Fresh context: the grounding preamble (`schema.py:PREAMBLE`, under 80 words) as the system
   message, then the instruction, the field list (`schema.py:describe`) and the file between
   `<<<FILE` and `FILE>>>`. Text over `max_input_tokens` is shown head + tail and marked `_truncated`.
3. Generate with the synchronous decode loop, greedy, non-thinking, capped at 512 tokens (recipes may raise it).
4. Parse the first JSON object; validate against the schema. On failure, one retry with the error appended; then `failed: invalid_json` or `failed: schema`.
5. Ground (`schema.py:ground`): `x-grounded` values must appear in the file (case- and whitespace-insensitive;
   numbers match their written forms; dates match any written form of the same date, but a date that only
   matches an ambiguous numeric form like 03/09/2025 is kept and flagged as unverifiable). `x-quote` values need
   a quote of at most 25 words found in the file. Failures are nulled and named in `flags`.
6. Recipe checks (e.g. `clean` must keep ≥ 85% of the original's distinct words).
7. Write the output row, then the status (`flagged` if any flags, else `done`), then append one JSONL line.

Three engine exceptions in a row pause the job with an error. On start or resume the runner puts `running`
items back to `pending` and rebuilds the JSONL from the store, so it never holds duplicate lines.

## Pacing (`mack/pacing.py:Pacer`, `TaskRunner._generate`)

The default pace is **auto**. Once a second the pacer reads keyboard/mouse idle time, other
processes' CPU duty (the governor's per-process sampler), macOS thermal state, memory pressure
and the power source, and picks a level:

| level | when | rate |
|---|---|---|
| away | no input for `away_after_s` (60 s) and other apps under `away_max_duty` (0.15) | full (steady on battery) |
| active | you are using the Mac, other apps light | balanced |
| busy | other apps at or above `busy_duty` (0.25) | gentle |
| hold | other apps at or above `hold_duty` (0.6), thermal serious (held 60 s, then re-checked), or memory critical | none |

A "fair" thermal state caps any level at balanced. Memory "warn" changes nothing (it is normal on a
16 GB Mac with a browser open). Fixed paces (gentle 2.3, balanced 4.5, steady 9.7 tok/s, full) keep
their rate, except full drops to steady on battery, fair heat caps them at balanced, and the hold
conditions pause them.

- Prompt tokens are prefilled in chunks of `active_chunk` (128) while the pace is below full, 512 at
  full; after each chunk the runner rests until `max(busy / duty, prefill_weight × tokens / rate)` has
  passed (`prefill_weight` 0.25; duty 0.25 / 0.4 / 0.6 / 1.0 for gentle / balanced / steady / full).
  Each generated token waits `1 / rate`.
- The runner yields every wait to the worker, which serves interactive requests during it.
- Memory critical: the item in progress is put back (attempt not counted), the model is unloaded
  (`engine.py:LazyEngine.unload`), and work resumes when pressure eases. The worker also unloads the
  model after `unload_after_idle_s` (300 s) with nothing to do.
- Below 25% on battery the job pauses (`control = battery`) and resumes at 30% or on power.
- The governor's `job_cap` mirrors the running job's current rate; chat's cap is separate.

## Recipes (`mack/recipes.py`)

YAML with keys `name, description, glob, instruction, schema, apply, options, ocr, default_grounded, checks,
max_output_tokens`. Unknown keys are rejected. A schema value `"$name"` is replaced by option `name`
(`--option name=a,b`). Built-ins in `mack/recipes/`; files in `~/MACK/recipes/` shadow them. Format reference:
[docs/recipes.md](docs/recipes.md).

Apply kinds (`mack/apply.py`): `frontmatter` (merge keys, keep existing ones unless `--overwrite`, existing
frontmatter bytes untouched), `rename` (`pattern` with `{date} {source} {title} {kind}`, `undated` when no date,
`-2`, `-3` on collision), `sidecar` (new file `<stem><suffix>`, never overwriting), `none`. Any job can instead
append rows to a CSV with `mack apply --csv FILE`.

## On-disk layout

```
~/MACK/                      (or the source checkout; MACK_HOME overrides)
  mack.toml                  configuration
  recipes/<name>.yaml        your recipes
  state/tasks/<id>/          task.json, control, live.txt
  state/batch.db             batch items
  state/worker.json          heartbeat: {pid, time}; `mack run` uses it to tell if a worker is alive
  output/<id>.jsonl          one line per finished item
  output/<id>.review.md      the review file
  output/<id>.undo.jsonl     undo log; renamed to .undone.jsonl after `mack undo`
```

## Undo log

One JSON object per line, written and fsynced before the first change:

```
{"op": "modify", "path": "/abs/note.md", "original_b64": "<original bytes>", "after_hash": "<sha256 after apply>"}
{"op": "rename", "path": "/abs/old.pdf", "new_path": "/abs/2026-03-03 Northwind Invoice.pdf", "after_hash": "<sha256>"}
{"op": "create", "path": "/abs/note.summary.md", "after_hash": "<sha256>"}
```

`mack apply` refuses, touching nothing, if any target's hash differs from `content_hash`. `mack undo` replays the
log in reverse; it refuses if any file's current hash differs from `after_hash`, and `--force` restores only the
files that still match. It only deletes files Mack created and nobody has changed since.

## Interfaces

- CLI (`mack/cli_batch.py`): `run, status, review, apply, undo, recipes, model download, bench`; exit codes 0 / 1 user / 2 engine / 3 refused.
- Worker: `mack work`, or the server (`mack serve`, port 8082), which interleaves jobs with chat and explain requests.
- Dashboard (`/`): batch jobs show counts and the latest items. The Swift menu-bar app reads `GET /api/tasks` unchanged.
- OCR: the Swift helper, `Mack --ocr <image>` and `Mack --pdf-text <pdf>` (PDFKit text layer, else Vision on the first 3 pages).

## Tests

`pytest -q` runs the stub suite (no model) in well under a minute. `MACK_RUN_MODEL=1` enables the `model` and `live`
tests; run them one at a time. `scripts/e2e_tuesday.py` is E2E-1; `mack bench` is the pace table.
