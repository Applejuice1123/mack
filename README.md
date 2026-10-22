# Mack

**Run a local AI over thousands of your files without your Mac noticing.**

Mack applies one short instruction, or a ready-made recipe, to every file in a folder. It
reads one file at a time, keeps only answers it can trace back to that file, and runs at a
pace you choose so the rest of your Mac stays responsive. Nothing changes in your files until
you review the results and run `mack apply`, and every apply can be undone.

## How it stays out of your way

The default pace, **auto**, watches what you are doing once a second:

| you are | Mack runs at | extra power | memory-heavy apps slow by |
|---|---|---:|---:|
| away (no input for a minute, other apps quiet) | full speed (steady on battery) | uncapped (6 W on battery) | up to 45%, but you're not there |
| using the Mac | balanced, 4.5 tok/s | ≤ 3 W, about one busy browser tab | ~10% |
| running something heavy (≥ ~1.4 busy cores) | gentle, 2.3 tok/s | ≤ 1.5 W | ~5% |
| maxing the Mac, Mac hot, or memory critical | paused | 0 | 0 |

Compute-heavy work slows by under 3% at every pace. Prompt reading, which is what heats the Mac
(about 16 W flat out on this M4), happens in short bursts while you are present and spends the same
power budget. When there is nothing to do, Mack unloads the model and gives its ~3 GB of memory back.

Measured on a base M4 MacBook Air, 16 GB, Qwen3.5-4B-4bit. A real 12-note job while the owner kept
using the Mac (Chrome busy) finished in 6 minutes with the whole Mac averaging 2.5 W and thermal
pressure Nominal throughout. Fixed paces (`--pace gentle|balanced|steady|full`) are still available;
`mack bench` measures all of them on your Mac, see [docs/measurements.md](docs/measurements.md).

## Install

```bash
pipx install /path/to/mack          # or: python3 -m venv .venv && .venv/bin/pip install -e .
mack model download                 # Qwen3.5-4B-4bit, about 3.1 GB; needs 4 GB free
```

Apple Silicon only (MLX). macOS 14 or later. Recipes that read PDFs and images also need the
helper app: `app/build.sh` (Xcode command line tools, no Xcode needed).

## Two-minute quick start

Try it on 20 notes from your own notes folder, at full speed so it finishes quickly:

```bash
mack run --recipe tag --limit 20 --pace full --wait ~/Notes
```

```
20 items · about 1m at full · uncapped: up to 45% slower memory-heavy work while it runs
job 20261022-101500-3f2a · output ~/MACK/output/20261022-101500-3f2a.jsonl
20/20 · 31.4 tok/s · finished · 1 flagged
done. Review with: mack review 20261022-101500-3f2a
```

```bash
mack review last          # opens a Markdown file: flagged items first, then every result
mack apply last --dry-run # shows exactly what would change
mack apply last           # writes tags into each note's frontmatter
mack undo last            # puts every byte back
```

For the whole folder, start the worker once and let it run all day:

```bash
mack work &                                  # or open the menu-bar app, which starts it
mack run --recipe tag ~/Notes                # auto pace; returns immediately; `mack status` shows progress
```

## Recipes

| recipe | what it does | apply |
|---|---|---|
| `tag` | picks up to 3 tags from your list (`--option tags=work,home,idea`) and writes a one-line summary | merged into YAML frontmatter; existing keys are kept |
| `summarize` | a 60-word summary and up to 5 key points, each with a quote verified against the file | writes `name.summary.md` beside the file |
| `rename` | reads invoices, receipts, letters and screenshots (OCR for PDFs and images) | renames to `YYYY-MM-DD source title.ext` |
| `extract` | pulls the fields of your JSON Schema out of each file | writes `name.json`, or rows in a CSV with `--csv` |
| `clean` | removes filler words and false starts from transcripts and lists action items | writes `name.clean.md`; never edits the original |

`mack recipes` lists them; copy one into `~/MACK/recipes/` to change it. See [docs/recipes.md](docs/recipes.md).
Without a recipe: `mack run "List every person mentioned" ~/Notes`.

## What Mack will not do

Mack is built for narrow jobs where every answer can be checked against one file. A 4B model
is often wrong at open-ended work, so Mack does not try:

- It will not write a report or anything longer than its schema allows.
- It will not answer from memory. A field it cannot find in the file comes back empty and flagged.
- It will not run your code.
- It will not change a file before you review and `mack apply`, and it will not apply flagged results unless you ask.

## The three guarantees, and the tests that prove them

1. **Grounded**: outputs that cannot be traced to their file are removed and flagged. [tests/test_grounding.py](tests/test_grounding.py)
2. **Polite**: each pace holds its token rate, prompt reading included, pauses on low battery and heat, and never runs at background priority. [tests/test_pace.py](tests/test_pace.py)
3. **Durable and reversible**: progress survives kill, quit and reboot, and every apply has an undo written to disk first. [tests/test_batch_resume.py](tests/test_batch_resume.py), [tests/test_cli_batch.py](tests/test_cli_batch.py)

## Pace, battery and heat

| situation | what Mack does |
|---|---|
| any pace | prompt reading and generation share a power budget; generation uses the lower-energy decode loop; short GPU bursts while you're present |
| on battery, `full` | runs at `steady` instead |
| battery below 25% | pauses with the current file put back in the queue; resumes at 30% or on power |
| macOS reports serious heat | rests 60 seconds, then checks again |
| nothing to do for 5 minutes, or memory critical | unloads the model (~3 GB) until there is work again |

## Commands

`mack run`, `mack status [job]`, `mack review job`, `mack apply job [--dry-run] [--only done+flagged]`,
`mack undo job [--force]`, `mack recipes [show NAME]`, `mack work`, `mack bench`, `mack model download`.
Job ids accept a unique prefix or `last`. Exit codes: 0 ok, 1 your input, 2 engine or model, 3 refused.

## Also included

- **Chat** (`mack chat`, `mack serve`), with context compaction and recall.
- **Screen explainer**: a menu-bar app that explains text you drag a box over.
- **Long tasks** (`mack task add`): multi-section reports and small codebases. A 4B model makes factual errors at this, so it is not the headline.
- **Menu-bar app** (`app/build.sh`): shows job progress. It is not part of this release and is unverified.

Design notes: [SPEC.md](SPEC.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Not chosen yet (MIT or Apache-2.0).
