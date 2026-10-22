# Mack `mack run`: build and test directions for Claude Code

Oct 22, 2026 · @Srish

## What we are building

Build `mack run`: a resumable background worker that applies one short instruction to every file in a folder, one file per fresh model context, writes results to a review file, and stays under a user-chosen CPU-slowdown budget. This replaces the current "big task" mode (multi-section reports, multi-file codebases) as Mack's main product. Chat mode and the screen explainer stay in the code but are not touched in this work.

The product makes three guarantees, and every phase below exists to make one of them true and tested:

1. **Grounded.** Every output is derived only from the file it was given. The model never answers from memory. Outputs that cannot be traced to the input are dropped and flagged, not shown.
2. **Polite.** The user picks a pace (`gentle` 2.3 tok/s, `balanced` 4.5, `steady` 9.7, `full` uncapped) and Mack never exceeds the matching measured slowdown (\~12% / \~22% / \~45% of memory-heavy CPU work). It pauses below 25% battery and never runs at background priority.
3. **Durable and reversible.** Progress is checkpointed after every item and survives quit, sleep and reboot. Nothing is written into the user's files until `mack apply`, and every apply has a `mack undo`.

Reference hardware is a base M4 MacBook with 16 GB and Qwen3.5-4B-4bit on MLX. Everything must work on that machine with no cloud fallback. The 9B is optional and out of scope here.

Target user and scenario, for taste decisions: a solo developer with \~2,400 Obsidian notes who runs `mack run --pace gentle --recipe tag ~/Obsidian` on a Tuesday morning, keeps using Xcode and Zoom all day, and reviews a tags file the next morning.

## Ground rules for Claude Code

Work in `~/MACK`. Read this whole document first, then do the phases in order. Do not start Phase N+1 until Phase N's tests pass and its checklist is ticked.

**Working habits**

- Work on a git branch `mack-run`. Commit after every phase with a message naming the phase. Never force-push.
- Before touching a module, read it. The engine, governor, task queue, compaction and SQLite recall already exist; reuse them. Do not write a second decode loop, a second cache, a second queue or a second CPU probe. If something existing is in the way, note it in `FINDINGS.md` (Phase 0) and propose the smallest change.
- Add a test with every behavior. The existing 46 tests must keep passing. Run the full suite before every commit.
- Tests that need the real 4B model are marked `@pytest.mark.model` and skipped by default. Everything else runs on a stub engine that returns canned tokens at a configurable rate, so the suite finishes in under a minute without a model.
- Never run anything at background priority, never quantize the KV cache, never delete the user's files. These are settled by prior measurement; do not revisit them.
- Do not execute user code. Code jobs get a syntax check only, as today.

**Resource rules for this machine**

- Free disk is often under 1 GB. Before any step that writes more than 200 MB (model downloads, training, large outputs), check `df -h ~` and stop with a clear message if free space is under 2 GB.
- Do not start LoRA training in this work. The pipeline stays as is.
- Model-backed tests run one at a time, never in parallel, and the engine is unloaded afterward.

**Stopping conditions**

Stop and ask the user rather than guessing when: a choice changes the on-disk format of jobs or outputs; an existing test must be deleted or its assertion changed; a step needs more than 2 GB of disk; or the real model behaves differently from this document's assumptions (for example, the governor cannot hold a pace). Everything else, decide and record the decision in `FINDINGS.md`.

**Final deliverable**

A branch where `mack run` works end to end on a real folder at every pace, with tests, docs and a `RELEASE_NOTES.md` summarizing what was built, what was measured and what remains unverified (Phase 7 and Definition of done).

## Phase 0: map the existing engine

Goal: write `FINDINGS.md` at the repo root before changing any code, so every later phase builds on what is actually there rather than on this document's summary.

Read the code and record, with file paths and function names:

1. **Engine entry points.** How a prompt is turned into tokens: the class or function that loads the model, the three decode loops (pipelined, synchronous, speculative), and how a caller picks one. Note the 512-token prefill chunking and where it lives.
2. **The governor.** Where the live CPU governor sets a token-rate cap, how it measures CPU load, how the pace presets (2.3 / 4.5 / 9.7 / uncapped) map to it, and the fix for the efficiency-core probe problem. Note whether a caller can set a cap for one job and leave chat mode's cap alone.
3. **Task queue and durability.** The task model behind `mack task add|list|show|pause|resume|cancel` and `mack work`: where tasks are stored, how progress is checkpointed after each piece, how a fresh context is built per piece (brief + plan + digest), and how resume after reboot works.
4. **Battery and rest.** Where the 25% battery pause lives and where rests between prefill bursts are scheduled.
5. **Self-review loop.** How a section is reviewed and revised once, and how code gets its syntax check and single repair.
6. **Recall and compaction.** The SQLite index schema and the compaction routine, in case Phase 5 recipes want the index for similarity.
7. **Interfaces.** The CLI framework in use, the dashboard server on port 8082 and what it reads, and how the Swift menu-bar app learns task progress.
8. **Tests.** How the 46 tests are organized, whether a stub engine exists, and which tests touch the real model.

Then write a short section **Design for mack run** in `FINDINGS.md` that answers:

- Is a batch job best modeled as one task with N items, or as N tasks? Recommend one. The default assumption is one task with N items, each item checkpointed separately, because the dashboard and menu bar already understand one task's progress.
- Can the existing per-piece fresh-context builder be reused for per-item contexts, or does it assume a plan and digest that an item does not have?
- Does the engine already support constrained or grammar-guided decoding on MLX? If not, Phase 2 must add it; say which route (logit masking against a JSON grammar, or generate-then-validate with one retry) is cheaper given the code.
- Any existing config or on-disk format Phases 1–5 must stay compatible with.

**Checklist**

- [ ] `FINDINGS.md` written with paths and function names, not prose summaries
- [ ] Full test suite run once on a clean checkout; result recorded
- [ ] Stub engine confirmed present, or its absence recorded as the first Phase 1 task
- [ ] No code changed in this phase

## Phase 1: the batch job core

Goal: a `BatchJob` that takes a list of files and an instruction, runs them one at a time through the engine with a fresh context each, checkpoints after every item, and can be killed and resumed at any point without repeating or losing work. No CLI yet; drive it from tests and a small Python entry point.

**Data model** (adapt names to what Phase 0 found; keep these fields)

- `BatchJob`: `id`, `created_at`, `instruction`, `recipe` (nullable until Phase 5), `schema_path` (nullable until Phase 2), `pace`, `root` (folder), `glob`, `output_path`, `status` (`queued | running | paused | done | cancelled`), `counts` (`total, done, skipped, failed, flagged`).
- `BatchItem`: `job_id`, `index`, `path`, `content_hash` (sha256 of the file at enqueue time), `status` (`pending | done | skipped | failed | flagged`), `attempts`, `output` (JSON), `error`, `started_at`, `finished_at`, `tokens_in`, `tokens_out`.
- Store items in the same store the task queue uses today (SQLite is expected). One row per item. Status transitions are the only writes during a run.

**Enqueue**

- Expand the glob, skip anything over a size cap (default 200 KB of text; configurable), skip binaries and files Mack cannot read as text (PDF and image handling arrives in Phase 5 via OCR), and record every skip with a reason.
- Hash every file at enqueue. If a file's hash differs at run time, mark the item `skipped: changed_since_enqueue` rather than processing stale content.
- Print a plan line before starting: item count, estimated duration at the chosen pace (use a running average of tokens per item once 10 items are done; before that, a fixed guess of 150 output tokens per item), and the slowdown the pace implies.

**Run loop**

- Pull the next `pending` item, build a fresh context containing only: a short system preamble (the grounding rule from Phase 2), the instruction, and the file content. No plan, no digest of earlier items. Items must be independent so order and resumption never matter.
- Use the synchronous (lower-energy) decode loop. Rest between items as the existing task runner rests between prefill bursts.
- On any exception, record it on the item, set `failed`, and continue. Three consecutive engine failures stop the job with status `paused` and an error message, so one bad model state does not burn the whole night.
- Write the item's output to the store and only then mark it `done`. Commit per item.

**Output file**

- JSON Lines at `output_path` (default `~/MACK/output/<job_id>.jsonl`), one line per finished item: `{path, content_hash, status, output, tokens_in, tokens_out, duration_s}`. Append-only; on resume, rebuild the file from the store so it never has duplicate lines.

**Resumability**

- `resume(job_id)` picks up at the first `pending` item. An item that was `running` when the process died goes back to `pending` with `attempts` incremented; after 2 attempts it is `failed`.
- Reuse the existing survive-quit-and-reboot path. If `mack work` is what runs queued tasks today, batch jobs must run through it too.

**Tests (stub engine, no model)**

- Enqueue 50 small text files; all 50 run; output has 50 lines; counts match.
- Kill the worker (raise inside the engine stub) after item 23; resume; items 24–50 run; output has exactly 50 lines, item 23 has `attempts: 2`.
- Modify a file between enqueue and run; it is `skipped: changed_since_enqueue`.
- Include one 5 MB file and one PNG; both skipped with reasons; job completes.
- Engine raises on three consecutive items; job ends `paused` with 3 `failed` items and the rest `pending`.
- Two items with identical content produce independent outputs (no cross-item leakage): instruction "echo the first word" with a stub that returns the first word of its prompt.

**Checklist**

- [ ] `BatchJob` and `BatchItem` stored in the existing store; schema recorded in `FINDINGS.md`
- [ ] Plan line prints count, estimate and slowdown
- [ ] All six tests pass on the stub; the original 46 still pass
- [ ] Dashboard at 8082 shows a batch job's `done/total` without changes to the Swift app

## Phase 2: schema-constrained outputs and grounding checks

Goal: every item's output is valid JSON matching a user-supplied schema, and every field that claims to come from the file is checked against the file. This is what makes a 4B model trustworthy at this job: the model is never asked a question it could answer from memory, and anything it invents is caught mechanically.

**Schema**

- Accept a JSON Schema file via `schema_path`. Support `object` with `string`, `number`, `boolean`, `array` of strings, and `enum` (a fixed list of allowed strings). Reject other types with a clear message; this is a batch tool, not a general validator.
- Two optional extensions, read from each property: `"x-grounded": true` means the value must appear verbatim in the input file (case-insensitive, whitespace-normalized); `"x-quote": true` means the model must also return a `<field>_quote` string of at most 25 words copied from the file that supports the value, and that quote must be found in the file.
- Decoding route per Phase 0's finding. Preferred: constrain decoding so only schema-valid JSON can be produced. Acceptable fallback: generate, parse, validate, and on failure retry once with the validation error appended to the prompt. Record which route was built.
- Always run the model in non-thinking mode for batch items and cap output at 512 tokens by default. A 4B model in thinking mode produces far more tokens than its answers need, and at 2.3 tok/s that is the whole budget.

**Grounding rule, in the system preamble of every item**

The preamble says, in plain words: answer only from the text provided; if the text does not contain the answer, return `null` for that field; never use outside knowledge; for enum fields pick exactly one option or `null`. Keep it under 80 words so it costs little prefill.

**Validation after generation**

1. Parse JSON. Fail → one retry → `failed: invalid_json`.
2. Validate against the schema. Fail → one retry → `failed: schema`.
3. For every `x-grounded` field, check the value is in the file. Not found → set that field to `null` and add the field name to `flags`.
4. For every `x-quote` field, check the quote is in the file. Not found → set the field and its quote to `null`, add to `flags`.
5. Any item with a non-empty `flags` list gets status `flagged`, not `done`, so the review step surfaces it.

**Tests (stub engine returning canned JSON)**

- Valid output passes all five steps and is `done`.
- Stub returns prose instead of JSON; after one retry the item is `failed: invalid_json`.
- Stub returns an enum value not in the list; `failed: schema` after retry.
- Stub returns `"vendor": "Acme"` for a file that never mentions Acme, with `x-grounded`; result has `vendor: null` and `flags: ["vendor"]`, status `flagged`.
- Stub returns a correct value with a fabricated quote under `x-quote`; both are nulled and flagged.
- Grounding check is case- and whitespace-insensitive: `"ACME  Corp"` matches `acme corp` in the file.
- Output token cap is passed to the engine and respected by the stub.

**Model test (marked `model`, run once by hand)**

- 30 short invoice-like text files generated by a script with known vendor, date and amount. Schema with all three `x-grounded`. Record how many of 30 are exactly right, how many flagged, how many wrong-but-unflagged. The last number is the one that matters; anything above 2 of 30 means the grounding rule or preamble needs work before Phase 5.

**Checklist**

- [ ] Decoding route chosen and recorded; constrained if feasible
- [ ] Grounding preamble under 80 words, stored in one place, used by every item
- [ ] Seven stub tests pass
- [ ] Model test run once; the three counts recorded in `FINDINGS.md`

## Phase 3: the CLI

Goal: the five commands below, added to the existing `mack` CLI in the same framework. `mack task ...` and `mack work` keep working. A batch job is visible to `mack task list` like any other task.

**`mack run`**

```
mack run [--pace gentle|balanced|steady|full] [--schema FILE] [--recipe NAME]
         [--out FILE] [--glob PATTERN] [--limit N] [--dry-run] [--wait]
         "INSTRUCTION" PATH [PATH...]
```

- Either `--recipe` (Phase 5) or an `INSTRUCTION` plus optional `--schema` is required. With no schema, output is a single `"text"` string field, still grounded by the preamble and capped at 512 tokens.
- `PATH` may be a file, a folder (recursed with `--glob`, default `**/*.md,**/*.txt`), or `-` for a newline-separated list on stdin.
- Default pace is `balanced`. Default `--out` is `~/MACK/output/<job_id>.jsonl`.
- Prints the plan line from Phase 1, then returns immediately with the job id, leaving the job queued for `mack work` (which must be running or started; print the exact command if it is not). With `--wait`, stays attached and shows a one-line progress bar: `412/2400 · 4.4 tok/s · ~6h12m left · 3 flagged`.
- `--dry-run` prints the plan line and the first 5 items that would be enqueued, then exits.
- `--limit N` enqueues only the first N matches. Use it in docs for the two-minute first run.

**`mack status [JOB_ID]`**

- With no id, lists batch jobs newest first: id, status, done/total, flagged, pace, started. With an id, shows counts, the current item's path, average tokens per item, measured tok/s over the last 5 minutes, estimated finish, and the last 3 errors.

**`mack review JOB_ID`**

- Writes `~/MACK/output/<job_id>.review.md`: a Markdown file with one section per flagged or failed item first (path, the output, the flags, and the first 300 characters of the file so the user can judge), then a table of all `done` items (path and output, one row each). Opens it with `open` unless `--no-open`.
- The review file is the product's trust surface. Keep it readable in a plain editor; no HTML.

**`mack apply JOB_ID [--only done|done+flagged] [--dry-run]`**

- Applies outputs to the user's files according to the recipe's `apply` rule (Phase 5). Jobs with no recipe have nothing to apply and say so.
- Default `--only done`; flagged items are applied only when explicitly asked.
- Before writing anything, write `~/MACK/output/<job_id>.undo.jsonl` containing, per file touched, the original path, the new path (if renamed), and the original bytes of any file whose contents change. Only after the undo log is flushed to disk does the first write happen. Refuse if a target file's hash no longer matches `content_hash` (changed since the job ran) and report the path.
- `--dry-run` prints every change as `path: old → new` without writing.

**`mack undo JOB_ID`**

- Reverses the undo log in reverse order. Refuses with a clear message if any file was changed after apply (hash mismatch), listing the files, and offers `--force` to restore only the files that still match.

**Exit codes and messages**

- 0 success, 1 user error (bad args, missing schema), 2 engine or model error, 3 refused (disk, battery, hash mismatch). Error messages name the fix: "No model found. Run `mack model download` (3.1 GB, needs 4 GB free)."

**Tests (stub engine, `subprocess` against the CLI)**

- `mack run --dry-run` on a folder of 12 files prints a plan with 12 items and exits 0 without creating a job.
- `mack run --limit 5 --wait` completes 5 items and the output file has 5 lines.
- `mack status` lists the job; `mack status <id>` shows 5/5.
- `mack review` writes a file with a flagged section when one stub output is flagged.
- `mack apply --dry-run` on a rename recipe prints old → new and changes nothing; `mack apply` renames; `mack undo` restores; file hashes before and after are identical.
- `mack apply` refuses when a file changed after the job ran, exit 3, and touches nothing.
- `mack undo` after a post-apply edit refuses, lists the file, and `--force` restores the rest.

**Checklist**

- [ ] Five commands wired into the existing CLI; `mack --help` lists them with one-line descriptions
- [ ] Seven CLI tests pass
- [ ] Undo log is written and flushed before any file is modified; a test proves it by killing the process between the two

## Phase 4: pacing, battery and the slowdown contract

Goal: a batch job at pace P never produces more than the token rate for P, across both prefill and decode, and the job's actual effect on the rest of the Mac matches the published numbers. This is the feature that distinguishes Mack from Ollama and LM Studio, so it gets its own measurement, not just a unit test.

**Wire the pace to the governor**

- Each batch job carries its pace; the worker sets the governor's cap from it when the job starts and restores the previous cap when it ends or pauses. Chat mode's live cap (default 9.7 tok/s while other apps are busy) is untouched.
- Prefill counts too. The current cap is on decode tokens per second. A 2,000-token note at `gentle` must not prefill at full speed and only decode slowly; spread prefill chunks (the existing 512-token chunks) with rests so that prompt tokens plus output tokens per wall-clock second stay under the cap, weighting a prefill token at 0.25 of a decode token (prefill is compute-bound and measured to disturb other work far less). Record the weighting in `FINDINGS.md` and make it a config value.
- `full` means uncapped, same as today, and the plan line says "up to 45% slower" so nobody is surprised.

**Idle boost (optional, off by default)**

- `--boost-when-idle`: if the system has been idle (no keyboard or mouse input for 10 minutes and CPU load under 20%) the job may run at `steady` regardless of its pace, dropping back within 2 seconds of input. Use the existing CPU probe; use `CGEventSourceSecondsSinceLastEventType` via a tiny helper for input idle. Skip this if it takes more than a day; it is a nicety.

**Battery and thermal**

- Reuse the existing 25% battery pause. Add: on battery at any level, `full` is silently demoted to `steady` and the status line says so. When the thermal pressure level reported by macOS reaches "serious" or above, pause for 60 seconds and retry.
- Pausing on battery must checkpoint the current item as `pending`, not `running`.

**Measurement harness (`mack bench`, minimal version)**

- A script that runs the existing memory-heavy and compute-heavy CPU probes for 60 seconds with no job, then for 60 seconds while a batch job runs at each pace, and prints a table: pace, measured tok/s, memory-heavy slowdown %, compute-heavy slowdown %. Use the fixed probe from the earlier research (pin away from efficiency cores).
- The published contract is: `gentle` ≤ 15%, `balanced` ≤ 25%, `steady` ≤ 30%, `full` unbounded, on memory-heavy work. Put these thresholds in one config file. The model test below checks them.

**Tests (stub engine with a fake clock)**

- At `gentle`, 100 items of 300 prefill + 100 decode tokens each take at least the time implied by the cap with the 0.25 weighting; at `full` they take the stub's unthrottled time.
- Starting a batch job sets the governor cap; finishing restores the prior value; pausing mid-job restores it too.
- A fake battery reading of 24% pauses the job with the current item `pending`; 30% resumes it.
- A fake "on battery" flag demotes `full` to `steady` and the status reflects it.

**Model tests (marked `model`, run by hand, each 5–10 minutes)**

- Run `mack bench` once on mains power with nothing else open. Record the table in `FINDINGS.md`. Every pace must meet its contract threshold; if one does not, lower that pace's tok/s until it does and record the new number. Do not silently loosen a threshold.
- Run a 30-item job at `gentle` while playing a 1080p YouTube video in Safari and scrolling a long web page. Note subjectively whether anything stutters. This is the demo scenario; it has to feel right.

**Checklist**

- [ ] Governor cap set and restored per job; prefill weighted into the cap
- [ ] Battery and thermal pauses checkpoint cleanly
- [ ] `mack bench` prints the four-row table
- [ ] Measured slowdowns recorded and all four paces meet their thresholds, or thresholds adjusted and the change explained
- [ ] Four stub tests pass

## Phase 5: recipes and built-in job types

Goal: five built-in recipes so a new user never writes a schema or a prompt on day one. A recipe bundles an instruction, a schema, file-type handling and an `apply` rule. Recipes live in `~/MACK/recipes/<name>.yaml`; users can copy and edit them, and `mack run --recipe` looks in the user folder first, then the built-ins.

**Recipe file format**

```yaml
name: tag
description: Pick 1-3 tags from a fixed list and write a one-line summary.
glob: "**/*.md"
instruction: |
  Choose up to three tags from the allowed list that describe this note.
  Write a one-sentence summary of what the note is about.
schema:
  type: object
  properties:
    tags: {type: array, items: {type: string, enum: ["meeting", "idea", "bug", "reference", "personal"]}, maxItems: 3}
    summary: {type: string, maxLength: 200}
apply:
  kind: frontmatter      # frontmatter | rename | sidecar | none
  fields: [tags, summary]
options:
  tags: ["meeting", "idea", "bug", "reference", "personal"]   # overridable with --option tags=a,b,c
```

**The five recipes**

| Recipe | Input | Output fields | Apply rule | Grounding |
| --- | --- | --- | --- | --- |
| `tag` | `.md`, `.txt` | `tags` (enum list), `summary` | merge into YAML frontmatter, never overwrite an existing key unless `--overwrite` | `summary` is free text; `tags` are enum |
| `summarize` | any text | `summary` (≤ 60 words), `key_points` (≤ 5 strings, each `x-quote`) | write `<name>.summary.md` sidecar | every key point carries a verified quote |
| `rename` | `.pdf` (OCR), `.png`/`.jpg` (OCR), `.txt`, `.md` | `date` (ISO, `x-grounded`), `source` (`x-grounded`), `kind` (enum: invoice, receipt, letter, statement, screenshot, note, other), `title` (≤ 6 words) | rename to `YYYY-MM-DD <source> <title>.<ext>`; missing date → `undated`; collisions get `-2` | date and source must appear in the text |
| `extract` | any text | user-supplied schema via `--schema`; recipe only sets the preamble and `sidecar` apply | write `<name>.json` sidecar or, with `--csv FILE`, append one row per item | all string fields default to `x-grounded` unless the user's schema says otherwise |
| `clean` | `.txt`, `.md` transcripts | `cleaned` (full text), `action_items` (list, each `x-quote`) | write `<name>.clean.md` sidecar; never modifies the original | `cleaned` must keep ≥ 85% of the original's distinct words (a token-overlap check) so the model cannot rewrite meaning |

**PDF and image inputs**

- Reuse the Apple Vision OCR path from the explainer. Extract text at enqueue time, cache it in the item row, and run the recipe on that text. If OCR returns under 20 characters, skip the item with `skipped: no_text`.
- PDFs: try the text layer first (`pdftotext` or PyMuPDF, whichever is already a dependency), fall back to OCR on the first 3 pages only.

**`mack recipes`**

- Lists built-in and user recipes with their descriptions. `mack recipes show NAME` prints the YAML.

**Tests (stub engine)**

- Each recipe loads, validates its own schema, and runs on three fixture files; `apply --dry-run` prints the expected changes.
- `tag` apply on a note with existing frontmatter keeps every existing key and adds `tags` and `summary`; on a note with no frontmatter, adds a well-formed block; `undo` restores bytes exactly.
- `rename` on a fixture with a known date and source produces the expected filename; a collision gets `-2`; a missing date gives `undated`.
- `clean` output that drops 30% of distinct words is flagged, not applied.
- `summarize` key point whose quote is not in the file is nulled and flagged.
- A user recipe in `~/MACK/recipes/` shadows a built-in of the same name.
- OCR path is called for a PNG fixture (mock Vision in tests); an image with no text is `skipped: no_text`.

**Model test (marked `model`)**

- Build a fixture set of 40 files: 15 markdown notes of mixed kinds, 10 synthetic invoices as PDFs, 10 screenshots with text, 5 transcripts with filler. Run all five recipes at `full`. Hand-check the review file. Record per recipe: correct / flagged / wrong-unflagged. Any recipe with more than 2 wrong-unflagged out of its inputs is not shipped as a built-in until its preamble or schema is fixed.

**Checklist**

- [ ] Five recipes in `recipes/` with the YAML format above
- [ ] OCR reused from the explainer, not reimplemented
- [ ] Seven stub tests pass
- [ ] Model test counts recorded in `FINDINGS.md`; every shipped recipe has ≤ 2 wrong-unflagged

## Phase 6: end-to-end tests on the real model

Goal: prove the three guarantees on the actual machine with the actual model, in the scenario the README will describe. Everything here is marked `model` and run by hand, one test at a time, with results written to `FINDINGS.md` under a dated heading. Check free disk before starting (needs 2 GB for outputs and logs).

**E2E-1: the Tuesday-morning run (durability + politeness)**

1. Generate a fixture vault of 300 markdown notes (script in `tests/fixtures/make_vault.py`, deterministic seed, mixed lengths 100–2,000 words, five note kinds).
2. `mack run --recipe tag --pace gentle --wait ~/MACK/tests/fixtures/vault` in one terminal; in another, run the memory-heavy probe continuously and log its throughput every 10 s.
3. After \~40 items, kill the worker with SIGKILL. Confirm `mack status` shows the job `paused` or `queued` with the killed item `pending`. Restart `mack work`; confirm it resumes at that item.
4. After \~80 items, put the Mac to sleep for 5 minutes. Confirm the job resumes after wake with no duplicate output lines.
5. Let it finish. Record: total wall time, average tok/s, probe slowdown (mean and 95th percentile), flagged count, and whether the output file has exactly 300 lines with no duplicates.
6. Pass if: 300 unique lines; mean probe slowdown ≤ 15%; no item has `attempts > 2`.

**E2E-2: the apply/undo round trip (reversibility)**

1. Copy the fixture vault; hash every file.
2. `mack apply` the E2E-1 job. Confirm only `done` items changed, frontmatter merged, no other bytes touched (diff the vault against the copy; every diff is inside a frontmatter block).
3. Edit one applied note by hand. `mack undo`: must refuse, name that file, and change nothing. `mack undo --force`: restores all others; the hand-edited file is untouched.
4. Pass if every non-edited file's hash equals its pre-apply hash.

**E2E-3: the rename inbox (OCR + grounding)**

1. Fixture inbox of 20 files: 8 synthetic invoice PDFs with known dates and vendors, 6 screenshots of text, 4 plain notes, 2 blank images.
2. `mack run --recipe rename --pace steady --wait`, then `mack review`, then `mack apply --dry-run`.
3. Pass if: both blank images are `skipped: no_text`; at least 12 of the remaining 18 get a filename with the correct date and source; zero files get a wrong date that was not flagged. The last condition is the one that matters.

**E2E-4: pace contract across all four paces**

- 20 items at each pace with the probe running. Record the table. Pass if each pace meets its Phase 4 threshold. Keep this table; it goes in the README.

**E2E-5: it does not answer from memory**

- 20 notes that each mention a well-known entity (a country, a company, a famous person) without stating a fact about it, with an `extract` schema asking for a fact about that entity (`capital`, `founder`, `birth_year`), all `x-grounded`. Pass if every field is `null` and flagged: the model may know the answer, and it must not be allowed to say so.

**Fresh-eyes review**

- Before closing Phase 6, start a separate agent (or a second Claude Code session) that has not seen this work. Give it only `README.md` and the fixture vault, and ask it to install and run the two-minute quick start. It must succeed without asking a question. Record where it got stuck.

**Checklist**

- [ ] E2E-1 through E2E-5 each run and recorded with date, numbers and pass/fail
- [ ] Any failure fixed and the test re-run; no threshold loosened without an explanation in `FINDINGS.md`
- [ ] Fresh-eyes quick-start run recorded

## Phase 7: docs, README and release prep

Goal: a stranger with a 16 GB Mac can go from the GitHub page to a finished first job in under five minutes, and understands exactly what Mack will and will not do.

**README.md** (rewrite; the current one describes the explainer)

1. One sentence at the top: "Run a local AI over thousands of your files without your Mac noticing."
2. The E2E-4 pace table, with the machine named (base M4, 16 GB, Qwen3.5-4B-4bit).
3. Install: a Homebrew tap or a single `pipx install` line, plus `mack model download` with the size and free-space requirement stated.
4. Two-minute quick start using `--limit 20` on the user's own notes folder, ending in `mack review`. Show the expected terminal output.
5. The five recipes, one line each, with the apply rule named.
6. **What Mack will not do**, as its own section: it will not write a report, answer questions from memory, run your code, or change a file before you review and apply. Say that it is built for narrow, grounded jobs and that a 4B model is wrong at open-ended ones.
7. The three guarantees, with a link to the test that proves each (`tests/test_batch_resume.py`, `tests/test_pace.py`, `tests/test_apply_undo.py`).
8. Pace, battery and thermal behavior in one short table.
9. Chat mode and the screen explainer as a short "Also included" section at the bottom, not the headline.
10. License (MIT or Apache-2.0; ask the user which), and a one-line author note.

**Other docs**

- `SPEC.md`: replace with the batch-job data model, the recipe format, the on-disk layout under `~/MACK/output/`, and the undo-log format. Delete the explainer-era content or move it to `docs/explainer.md`.
- `CLAUDE.md`: update so a future Claude Code session knows the main product is `mack run`, where the stub engine lives, how to run stub vs model tests, and the resource rules from this document.
- `docs/recipes.md`: how to write a recipe, with the YAML reference and a worked `extract` example that ends in a CSV.
- `docs/measurements.md`: the slowdown methodology from the earlier research, the `mack bench` script, and the raw numbers from E2E-4, so the headline claim is reproducible.
- `CONTRIBUTING.md`: how to run tests, the stub/model split, and a "good first recipe" invitation.

**Demo**

- A script `scripts/demo.sh` that creates the 300-note fixture vault in a temp folder and runs the quick start on it, so the README GIF can be recorded in one take: left pane `mack run --pace gentle --wait`, right pane Activity Monitor or the probe, with the pace switched from `full` to `gentle` partway and the slowdown visibly dropping. Record the GIF by hand later; the script just makes it repeatable.

**Release hygiene**

- `mack --version` reports a version; tag `v0.1.0` on the branch once Definition of done is met.
- Confirm nothing in the repo references the user's home-folder paths, personal notes or the old `gemma_app` prototype.
- `.gitignore` covers `output/`, model files and SQLite stores.
- The Swift menu-bar app is not part of this release; note in the README that it exists and is unverified.

**Checklist**

- [ ] README rewritten with all ten parts; quick start verified by the Phase 6 fresh-eyes run
- [ ] `SPEC.md`, `CLAUDE.md`, `docs/recipes.md`, `docs/measurements.md`, `CONTRIBUTING.md` written
- [ ] `scripts/demo.sh` runs end to end on the fixture vault
- [ ] Release hygiene items done; `v0.1.0` tagged

## Definition of done and the final report

The work is done when every item below is true. Do not report done with any item false; report what is false and why.

- [ ] All seven phase checklists ticked, in order, each with its commit
- [ ] Stub suite (original 46 plus every new test) passes in under 90 seconds with no model loaded
- [ ] E2E-1 through E2E-5 passed on the real model and recorded with dates and numbers
- [ ] All four paces meet their slowdown thresholds, or thresholds were changed with the reason recorded
- [ ] `mack run --recipe tag --limit 20 --wait` on a fresh folder works from a clean clone following only the README
- [ ] No file in the user's folders is ever modified before `mack apply`, and every apply has a working undo; proven by tests, not by inspection
- [ ] `FINDINGS.md`, `RELEASE_NOTES.md` and the five docs from Phase 7 exist and are current

**`RELEASE_NOTES.md` must contain**

1. What was built, in five sentences or fewer.
2. The E2E-4 pace table and the per-recipe correct / flagged / wrong-unflagged counts from Phase 5.
3. Every decision made where this document left a choice open (batch-as-one-task vs many, decoding route, prefill weighting, license), with one sentence of reasoning each.
4. Every place the real model behaved differently from this document's assumptions.
5. What remains unverified, honestly: at minimum the Swift app, the 9B, idle boost if skipped, and anything a `model` test could not be run for (disk, time).
6. The three things a maintainer should do next, in priority order.

**Order of operations if time is short**

Phases 0–3 and Phase 6's E2E-1 and E2E-2 are the product; ship those before anything else. Phase 4's measurement and Phase 5's `tag` and `rename` recipes are the differentiator and the demo; do them second. `summarize`, `extract`, `clean`, idle boost and the demo script can follow in a later release. Never skip a test to reach a later phase.
