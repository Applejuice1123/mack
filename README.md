# Mack

### Point it at a folder. Keep working. Your files get sorted in the background.

Mack is a private AI assistant for your Mac that works through hundreds of files for you:
renaming scanned invoices, tagging notes, summarizing documents, cleaning up transcripts.
It runs entirely on your Mac, adjusts its speed so you don't notice it running, and changes
nothing until you've reviewed the results. Every change can be undone.

![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M4-black)
![Runs offline](https://img.shields.io/badge/runs-100%25%20offline-2ea44f)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

- **Private.** The AI model runs on your Mac. Your files are never uploaded anywhere.
- **Unnoticeable.** It slows down when you're working and speeds up when you step away.
  In testing, the whole Mac averaged under 2.5 W and stayed cool.
- **Careful.** Every answer is checked against the file it came from. If Mack can't back an
  answer up, it leaves the field blank and flags it for you instead of guessing.
- **Reversible.** You review everything before it's applied, and `mack undo` puts every byte back.

---

## What it does

Real results from a test run on sample files:

**A messy downloads folder of scans and receipts**

```
scan00.pdf        →  2025-10-11 Northwind Traders Invoice for consulting services and support plan.pdf
screenshot4.png   →  2026-07-01 Blue Yonder Airlines Flight BY 221.png
screenshot2.png   →  2026-03-09 Tailspin Toys Total $42.10.png
note0.md          →  note0.md   (no date or sender in it, so it keeps its name)
```

**A notes folder (Obsidian, Bear exports, plain Markdown)**

```yaml
---
tags: [meeting]
summary: This note documents a mobile release sync meeting attended by Ingrid and Leila, where
  concerns were raised and Priya was tasked with writing up options by Friday.
---
```

Tags are picked from *your* list, and anything already in the note is left alone.

**Meeting transcripts.** A cleaned-up copy with filler words and false starts removed, plus a list
of action items, each one backed by a quote from the transcript (for example, *"Kofi will write up
the options before Friday"*). The original is never touched.

**Long documents.** A short summary and up to five key points, each with a quote Mack found in
the document, saved next to it.

**Your own fields.** Describe what you want (vendor, amount, due date, anything) and get a
spreadsheet row for every file.

Or describe the job in your own words:

```bash
mack run "List every person mentioned" ~/Notes
```

---

## You won't notice it's running

Running an AI model on a laptop normally slows everything else down. On a MacBook Air,
running one flat out slowed memory-heavy apps by up to **45%**. Mack checks what you're doing
once a second and adjusts:

| when you are | Mack |
|---|---|
| away from the Mac | works at full speed (a little gentler on battery) |
| using the Mac | slows down so your apps lose only about 5–10% of their speed |
| running something heavy (a build, an export) | slows further, or pauses |
| low on battery, memory or cool air | pauses, then picks up where it left off |

When it has nothing to do, it unloads the model and gives about 3 GB of memory back to your apps.

**Measured on a fanless MacBook Air (M4, 16 GB):**
- During a 30-minute test while the owner kept working, the whole Mac averaged **1.8 W** and
  stayed at normal temperature in every reading.
- Left alone for 35 minutes on battery, it worked through about **3.6 files a minute**, then
  paused itself at 25% battery.

## It doesn't make things up

Mack uses a small AI model, and small models can be wrong. So Mack never takes the model's word
for it: every value has to be found in the file it came from, or it's removed and flagged.

| test | result |
|---|---|
| 30 invoices: vendor, date and amount | 26 right, 4 flagged for review, **0 wrong** |
| 20 notes that mention famous names but contain no facts | **20 of 20 left blank**; it never filled in from memory |
| a note with hidden instructions telling Mack to tag it "hacked" and "admin" | ignored; it got an accurate summary instead |

A date like `03/09/2025` could be March 9 or September 3, so Mack flags it rather than guessing.

## Nothing changes until you say so

```bash
mack review last          # a readable report: anything flagged first, then every result
mack apply last --dry-run # exactly what would change
mack apply last           # do it
mack undo last            # changed your mind? every file goes back, byte for byte
```

Close the lid, quit, restart or even force-kill it: Mack picks up where it stopped and never
processes a file twice. (Tested with 20 forced kills mid-job.)

---

## Get started

**You need** a Mac with Apple Silicon (M1 or later), macOS 14 or later, and about 4 GB of free
disk space for the AI model. Mack was built and measured on a 16 GB MacBook Air.

```bash
git clone https://github.com/Applejuice1123/mack ~/MACK
cd ~/MACK
pipx install .            # adds the `mack` command (no pipx? `brew install pipx`)
mack model download       # the AI model, about 3.1 GB, downloaded once
app/build.sh              # optional: lets Mack read PDFs, scans and screenshots
```

`app/build.sh` needs Apple's free command line tools (`xcode-select --install`), not Xcode itself.

### Your first run (2 minutes)

Try it on 20 of your own notes, at full speed so it finishes quickly:

```bash
mack run --recipe tag --limit 20 --pace full --wait ~/Notes
mack review last
```

For a whole folder, start the background worker once and forget about it:

```bash
mack work &
mack run --recipe rename ~/Downloads/Scans
mack status               # check in whenever you like
```

## Built-in recipes

| recipe | what you get |
|---|---|
| `rename` | invoices, receipts, letters and screenshots renamed to `YYYY-MM-DD sender title` |
| `tag` | 1–3 tags from your list plus a one-line summary, added to each note (`--option tags=work,home,idea`) |
| `summarize` | a short summary and key points with supporting quotes, saved next to each file |
| `clean` | tidy transcripts plus action items, saved as a new file |
| `extract` | the fields you define, as a file per document or one spreadsheet (`--csv`) |

`mack recipes` lists them. Recipes are short text files you can copy and edit to make your own;
see [docs/recipes.md](docs/recipes.md).

## How fast is it?

| situation | speed |
|---|---|
| you're away, plugged in | fastest: no speed limit |
| you're away, on battery | about 3–4 files a minute |
| you're using the Mac | about 1–1.5 files a minute, in the background |

A 150-note tagging job takes about 20 minutes left alone, or a couple of hours of background
work while you use the Mac.

## Questions

**Does anything leave my Mac?** No. The model is downloaded once from Hugging Face; after that,
Mack runs fully offline.

**Will it work on an Intel Mac?** No. It needs Apple Silicon.

**What about 8 GB Macs?** Untested so far. Mack only loads the model when there's room for it and
waits otherwise, but expect it to spend more time waiting.

**Can it be wrong?** Yes, which is why it shows its work. Anything it isn't sure of is flagged and
isn't applied unless you ask. You always review before anything changes.

**What won't it do?** Write essays or reports, answer from general knowledge, run code, or touch
a file you haven't approved. It's built for narrow, repetitive jobs where every answer can be
checked against the file.

**What does it cost?** Nothing. Mack is open source under the MIT license.

---

## Under the hood

Mack came out of a set of measurements on a base M4 MacBook Air.

- **The bottleneck is memory, not the processor.** On Apple Silicon the CPU and GPU share one
  memory pool. Generating text slowed memory-heavy work by about 45% but compute-heavy work by
  under 2%, and the slowdown rose roughly in step with generation speed. Mack's speed levels are
  the rates that keep that slowdown near 5%, 10% and 20%.
- **Reading heats the Mac; writing slows your apps.** Reading a prompt drew about 16 W extra but
  barely slowed other apps; generating drew about 7.6 W and slowed them a lot. Mack caps both
  with a power budget and reads in short bursts while you're at the Mac.
- **"Background priority" backfires.** macOS's low-priority mode moved the work onto the
  efficiency cores and made it slower without helping anything, so Mack doesn't use it.
- **Checking beats trusting.** An early prototype (a llama.cpp fork) kept long conversations small
  by deleting the context the model paid least attention to. The current model, Qwen3.5, mixes
  attention layers with recurrent ones that can't be trimmed that way, so Mack rebuilds its context
  from exact quotes and summaries that are checked line by line against the original. That
  "never trust, always verify against the source" rule became the grounding checks above.
  The original prototype is in [prototype/](prototype/).

Built on [MLX](https://github.com/ml-explore/mlx) with Qwen3.5-4B (4-bit).

## Command reference

`mack run`, `mack status [job]`, `mack review job`, `mack apply job [--dry-run] [--only done+flagged]`,
`mack undo job [--force]`, `mack recipes [show NAME]`, `mack work`, `mack bench`, `mack model download`.
Speeds: `--pace auto` (default) or `gentle | balanced | steady | full`. Job ids accept a unique
prefix or `last`. `mack bench` measures each speed level on your own Mac.

Also in the box: a local chat (`mack chat`, `mack serve`), a menu-bar screen explainer that explains
any text you drag a box over, and experimental long writing tasks (`mack task add`).

## License

MIT. See [LICENSE](LICENSE).
