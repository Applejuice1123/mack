# Mack: chat, explainer and long tasks (earlier design)

> This is the design document from before `mack run` became the main product. It still describes
> how chat mode, the screen explainer, the engine and the governor work. The current spec is ../SPEC.md.

Mack is a local assistant for Apple Silicon. It is an MLX system built on contention and prefill
measurements taken on a base M4 with 16 GB.

This document says what Mack does and why. Numbers marked *measured* come from
`mack phase0` and `mack calibrate` on this machine and are regenerated into
`reports/`; they are not constants. Standing rules for changing the code are in
`CLAUDE.md`.


## 1. What changed from the first prototype

| | first prototype (early 2026) | Mack |
|---|---|---|
| Engine | llama.cpp fork, Metal | MLX / mlx-lm |
| Model | Gemma-3-4B Q4_0 | Qwen3.5-4B-4bit now, Qwen3.5-9B-4bit when disk allows |
| Context control | delete the lowest-attention KV cell every 10 tokens | compaction: rebuild from exact quotes + verified summary + recent turns |
| What is lost | evicted tokens are gone (optional zlib dump) | nothing: every turn is archived and searchable |
| Process model | new process per request, full prompt re-evaluated | one resident engine, turns appended to a live cache |
| Contention | none | measured decode→CPU slowdown, rate cap, loop choice |

## 2. Engine: MLX

MLX and llama.cpp are separate engines; Mack uses MLX only. Basis, measured on
this machine (odi2027 S7 / N25, N27): a pipelined MLX loop idles the GPU 0.2% of
each token against llama.cpp's 1.8%, and llama.cpp spent 58% more energy per
token (0.290 J vs 0.184 J on Llama-3.2-3B).

`mack/engine.py` runs its own decode loop on top of the model call, because it
needs three modes and per-token control that `mlx_lm.generate` does not expose:

- **ahead1** — one token in flight: step *n+1* is submitted before token *n* is
  read. Default. *Measured* on Qwen3.5-4B: 36 tok/s against 34 for sync.
- **sync** — submit, wait, read.
- **spec** — draft with the small model, verify in one pass of the main model.

All three keep one invariant: `ctx.tokens` is exactly what the cache has
consumed, including after cancel, a length limit, or a rejected draft. After a
reply the cache ends right after the end-of-turn token, so the next user turn is
appended as a delta. History is never re-rendered through the chat template.

Prefill is chunked (`context.prefill_step`, default 512) and each chunk is
evaluated and the buffer pool cleared before the next.

## 3. Model

**Qwen3.5-9B-4bit is the target; Qwen3.5-4B-4bit is what ships today.** The disk
had 9.1 GB free and the 9B is 5.98 GB, which would have left about 2 GB for swap
on a 16 GB machine. The 4B has the same architecture, so everything below
transfers; switching is one line in `mack.toml` followed by `phase0` and
`calibrate`.

Qwen3.5 is a hybrid. *Measured* by `phase0` on the 4B: 32 layers, of which 8 are
full attention with a growing KV cache and 24 are Gated DeltaNet layers with a
fixed-size recurrent state. Consequences:

- The cache **cannot be trimmed**. mlx-lm's `can_trim_prompt_cache` is false.
  Token-level eviction, as in the first prototype, is not possible on this model.
- The cache is small: *measured* 32 KB per token (about 52 MB fixed), so 8K
  tokens of context is 320 MB.
- Earlier peak-memory slopes and the tiled-prefill result were taken on pure
  attention models and are **not assumed** here; section 4 is the re-measurement.

Fallback models (Qwen3-8B, Llama-3.1-8B, where the earlier kernels were
validated) are supported by the chat-format layer (`prompts.py` renders ChatML
and Llama-3) and by the cache rewind code (all-KV caches work); they have not
been downloaded or measured.

The KV cache stays fp16. There is no option to quantize it (F28: a 4-bit KV cache
raised peak memory during prefill, by 4 GB at 24K tokens).

## 4. Phase 0: what is measured before anything is relied on

`mack phase0` writes `state/phase0.json` and `reports/phase0.md`. Results for
Qwen3.5-4B-4bit, *measured* 2026-10-12:

| quantity | result |
|---|---|
| weights resident (main + draft) | 2.79 GB |
| GPU wired limit | 11.45 GB |
| cache growth | 32 KB/token, 52 MB fixed |
| prefill peak above resting, chunk 512 | 0.8 to 1.1 GB at every context from 1K to 8K: **flat** |
| prefill peak above resting, chunk 2048 | 1.1 GB at 1K rising to 2.2 GB at 8K |
| prefill speed | about 300 tok/s |
| context ceiling at 80% of the wired limit | about 169,000 tokens |
| decode, ahead1 / sync | 36.3 / 34.0 tok/s |
| speculative vs ahead1 | 24.0 vs 36.3 tok/s (**0.66×**), 51% acceptance |

What this settles:

- **Chunked prefill at 512 already gives a peak that is constant in context** on
  this hybrid, which is what tiling was built to achieve on attention models.
  Tiled attention is therefore not ported; chunk 512 is the default.
- **Memory is not what bounds the context here.** The ceiling is far above the
  configured budget. The budget (8,192 tokens) is set by time instead: a rebuild
  re-prefills at about 300 tok/s, and that cost is why compaction is rare.
  `Mack.budget` still takes `min(config, measured ceiling)`, which will matter
  for the 9B, whose weights leave much less headroom.
- **Speculative decoding loses on the 4B** (section 2, spec mode). See section 9.

## 5. Context: compaction, pins, verification, recall

`mack/session.py`. Eviction is compaction, never cache surgery: dropping
attention entries while the recurrent state still remembers those tokens puts a
hybrid model in a state it never saw in training.

**Trigger.** Before a user turn is appended, if the live context would pass
`high_water` (85%) of the budget, or the reply would not fit, Mack compacts.

**What a compaction does.**

1. Choose the verbatim window: the most recent turns that fit
   `recent_fraction` (25%) of the budget, starting on a user turn, always at
   least the last exchange.
2. **Pins.** From the turns leaving the window, extract sentences that carry a
   number, a name, a file/URL/identifier, a decision ("let's go with", "must",
   "deadline") or a fact about the user ("my name is"). They are kept as exact
   quotes with their message number. Selection is by score within
   `pin_fraction` (10%), user statements weighted above Mack's.
3. **Summary.** The model summarizes the leaving turns as one-fact bullets
   (in chunks of about 2,000 tokens, with the old cache already freed).
4. **Verification** (`verify.py`). Each bullet is kept only if every number in
   it appears in the source, every name and identifier appears in the source,
   and at least 60% of its content words do. Anything else is dropped, not
   repaired. `context.verify_with_model` adds a yes/no entailment check by the
   model per line. Bullets that only restate a pin are not stored twice.
5. Oldest summary bullets fall off when `summary_fraction` (8%) is exceeded.
   Summaries are never re-summarized.
6. **Rebuild**: a new cache is prefilled with system prompt + Memory section +
   the verbatim window. A context at 85% of 8,192 comes back at roughly 45%.

**Archive and recall** (`archive.py`). Every turn is written to a SQLite FTS5
index when it happens, split into passages. When a later message is a question
or refers back ("earlier", "what was", "remind me"), Mack searches the turns no
longer in the live context; a hit needs a shared number or name, or two shared
content words (one with an explicit back-reference). Hits whose relevant
sentences are already pinned are skipped. Up to `recall.max_tokens` (and at most
one eighth of the budget) of original text is placed in front of the user's message in a block marked
`[Recalled: …]`. Compaction runs before recall, recalled text is never clipped,
and recalled blocks are not carried through a rebuild.

**Oversized input.** A message that does not fit is archived whole and shown to
the model as head + tail with a marker.

## 6. Contention governor

`mack/governor.py`, `mack/calibrate.py`, `mack/probe.py`,
`mack/native/mack_probe.c`.

**The gap.** The research measured how CPU-side load slows decode. Mack needs the
opposite direction. `mack calibrate` measures it:

- *Rate cells.* Decode runs at several capped token rates while a CPU probe runs
  beside it: the STREAM triad from `odi2027/scripts/cpu_probe.c` (192 MB working
  set, persistent process) and a cache-resident FMA loop. Slowdown is
  `1 − rate / idle rate`, with the idle reference re-taken beside every cell.
  Readings use the second of two back-to-back bursts (N10: a burst after an idle
  gap measures the DVFS ramp). The probe runs at user-interactive QoS so it stands
  in for foreground work and stays on the performance cores.
- *Loop cells.* Under a synthetic stream load at duties 0 / 0.3 / 0.6 / 1.0, the
  pipelined loop is timed against the synchronous one.

Results are in `state/calibration.json` and summarized in section 6.1.

**Policy, per token.**

- **Rate.** The highest token rate whose measured slowdown is within
  `cpu_slowdown_target` (default 20%) on the chosen `ruler`, interpolated between
  calibration cells after forcing the curve monotone and anchoring it at zero, and
  never below `min_tps` (4 tok/s). `ruler = "triad"` is the
  worst case, work that is purely bandwidth-bound; F5 and F6 showed real
  applications are hit less than STREAM predicts, so this is conservative.
- **When.** `protect = "when_busy"` applies the cap only while the user's other
  processes are using at least `busy_duty` (0.25) of the performance cores, so an
  idle machine gets full speed. `always` and `never` are the other settings.
- **Loop.** Synchronous above the measured crossover duty; pipelined otherwise.
  The fallback when uncalibrated is 0.6 (N28: at stream duty 0.6 the two loops
  matched in wall time on the 8B while the pipelined one drew more power).
- **Speculative** only when uncapped, the machine is quiet, and `phase0`
  measured at least 1.05×.

**Other CPU load** is the summed CPU time of every other readable process per
second, in performance-core units. It is deliberately not the system-wide
counter: that one includes the kernel's own work while Mack decodes and thermal
management after sustained load, and read 0.5 with nothing else running. The
sweep costs about 6 ms per second.

**Priority.** Mack never runs at background priority. At startup it clears
Darwin background state on the process and sets the engine thread to
user-initiated QoS. Basis: N12, N22 to N24 (background placement puts the
submitting thread on the E-cluster; the penalty was +11.5 to +13.9 pp and
spread to the whole process).

### 6.1 Calibration results

Qwen3.5-4B-4bit, *measured* 2026-10-12, three interleaved rounds per cell
(`reports/calibration.md` has the full table). Unloaded decode is 36.3 tok/s.

| decode rate (pipelined) | slowdown of bandwidth-bound CPU work (triad) | slowdown of cache-resident CPU work |
|---:|---:|---:|
| 5.4 tok/s | 11.9% | 0.3% |
| 10.6 tok/s | 21.6% | 0.4% |
| 16.8 tok/s | 35.0% | 0.9% |
| 23.3 tok/s | 44.8% | 1.0% |
| uncapped (29 tok/s beside the probe) | 44.9% | 1.7% |

- **Decode costs memory-bound CPU work up to 45%, and costs compute-bound work
  almost nothing.** Mack's effect on the rest of the machine is a memory-bandwidth
  effect. The triad slowdown is close to linear in token rate up to about
  17 tok/s and then saturates.
- With the default 20% target on the triad ruler the cap is **9.7 tok/s**, applied
  only while other processes are busy. A 10% target gives 4.5 tok/s. On the
  compute ruler no cap is ever needed.
- The reverse direction shows up in the same data: one foreground triad thread
  slows decode from 36 to 29 tok/s.

| stream load duty (2 threads) | load traffic | ahead1 tok/s | sync tok/s | ratio |
|---:|---:|---:|---:|---:|
| 0 | 0 GB/s | 36.0 | 32.4 | 1.11 |
| 0.3 | 34 GB/s | 27.8 | 25.8 | 1.08 |
| 0.6 | 60 GB/s | 22.8 | 20.6 | 1.11 |
| 1.0 | 91 GB/s | 20.8 | 19.6 | 1.06 |

- **The pipelined loop stayed faster at every load**, so the measured threshold
  is "never" and Mack does not switch to the synchronous loop on this model.

Verified live through the server: with a two-thread stream load running, replies
came out at 9.3 tok/s under a 9.7 cap; five seconds after the load stopped,
35.1 tok/s uncapped.

## 7. Behaviour adapter (LoRA)

`mack/lora/`. The fine-tune is for behaviour, not intelligence. It teaches:

- identity (it is Mack, it runs locally), and
- the honesty rule: answer about earlier conversation only from what is visible
  (recent turns, the Memory section, a Recalled block); otherwise say it can't
  see that anymore and ask.

**Data** (`dataset.py`): 500 synthetic conversations rendered in the exact
runtime format, with real Memory and Recalled blocks. Every abstention case has
near-identical answerable twins (the detail pinned, recalled, or still
visible), plus hard negatives where the same kind of fact about a different
subject is in view, plus ordinary questions, so the model has to look rather
than learn to refuse. The 94 eval cases use fact types, subjects and phrasings
absent from training.

**Training** (`train.py`): mlx-lm trainer, rank 8 on the last 4 layers, batch 1,
gradient checkpointing. The loss is taken on reply tokens only and only those
positions are projected to the 248K vocabulary. A preflight check requires 6 GB
of free disk and 8 GB of available RAM, and a watchdog thread aborts the run if
swap grows by 0.75 GB.

**Gate** (`evals.py`): base and adapter both run the held-out set, greedy. The
adapter ships only if it fabricates less on unanswerable questions, refuses no
more on answerable ones (2 pp tolerance), and is no worse at answering from
memory, at ordinary questions, or at identity. The verdict is written to
`adapters/mack/SHIP.json`; the engine loads the adapter only if it says ship,
and training deletes it.

**Status: no adapter is trained for the 4B, and none is loaded.**

- *Base model, measured* (`reports/lora_baseline.json`, 94 held-out cases, with
  Mack's system prompt and no adapter): fabrication **7.5%** (3 of 40
  unanswerable; 2 of the 3 are hard negatives where it gave the other subject's
  value), unnecessary refusal **0%**, answering from memory 100%, identity 100%,
  ordinary questions 100%.
- *Training on the 4B was attempted and stopped.* With LoRA on 8 layers it put
  the machine 4 GB into swap within five minutes, on a disk with 3 GB free. The
  recurrent layers keep a per-token state for backprop, so training memory grows
  with sequence length times the number of adapted layers. Hence the leaner
  loss, the 4-layer default, the preflight and the watchdog. The same code path
  was then run on the 0.8B (60 iterations, 8 layers, peak 4.8 GB, validation
  loss 2.43 → 1.11), and the gate was run on that undertrained adapter and
  **refused to ship it**, which is the gate doing its job.
- *To finish:* free about 6 GB of disk, close other apps, then
  `mack lora train` and `mack lora eval`. Given the baseline, the adapter has
  3 fabrications in 40 to remove and nothing to give back, so a "do not ship"
  verdict is a plausible and acceptable outcome.

## 8. Interfaces

### 8.1 Explainer (primary)

Mack is used as a helper, not a chatbot: click it, drag over a region of the
screen, get that region explained.

- `app/Mack.swift` — a menu-bar app (no Dock icon), about 400 KB, built by
  `app/build.sh` with the command line tools only. A click on the icon or
  Control-Option-M runs the system crosshair selection (`screencapture -i -s`),
  recognizes the text in the captured image on-device with Vision, deletes the
  image, and posts the text to the engine. The answer streams into a floating,
  non-activating panel placed beside where the drag ended. Simpler / More detail
  / Summarize re-ask about the same text; one free-form question can be asked
  about it; nothing carries over to the next selection.
- The app starts `mack serve` if nothing is listening and stops it on quit. It
  checks the engine's `features` list and says so if an older server is running.
- `mack/explain.py`, `POST /api/explain` `{text, style, question?}` — a fresh
  context per request (system prompt + captured text), under the same governor
  as chat, streamed as server-sent events. Captured text is tidied (blank runs,
  line-end hyphenation) and clipped to about 3,000 tokens. Closing the panel or
  making a new selection drops the connection, which cancels generation.
- *Measured* on the 4B: about 1.1 s to the first word for a few lines of text,
  29 to 35 tok/s, replies of 130 to 220 tokens.
- Needs the Screen Recording permission, which only the user can grant. "Explain
  Copied Text" in the icon's menu works without it.
- *Measured:* without the permission the capture still succeeds (2560×1664 px)
  but contains only the desktop picture, 0 characters of text, and
  `CGPreflightScreenCaptureAccess` is not a reliable guard. Mack therefore
  checks whether it can see other apps' window titles before capturing, and
  "Check Screen Access" in the menu runs a test capture and reports counts.
- The app is ad-hoc signed with the designated requirement
  `identifier "ai.mack.explainer"`, so a rebuild does not silently invalidate
  the grant.

### 8.2 Chat

- `mack serve` — FastAPI on 127.0.0.1:8082. `POST /api/chat` streams
  server-sent events (`session`, `status`, `token`, `done`, `error`);
  `/api/status`, `/api/memory`, `/api/session`, `/api/cancel`. One worker thread
  owns the engine. A dropped connection cancels generation.
- Web UI (`mack/web/index.html`) — chat, context bar with the compaction mark,
  decode loop, other CPU load, rate cap, GPU memory, and the quotes and summary
  kept from compacted messages.
- `mack chat` — terminal chat with `/status`, `/memory`, `/new`.
- Sessions are saved to `state/sessions/` after each turn and resumed by
  rebuilding the context.

## 9. Where measurement changed the plan

| the plan said | what was measured here | what Mack does |
|---|---|---|
| Qwen3.5-9B-4bit as the main model | 9.1 GB of disk free; the 9B is 6 GB | ships on the 4B; the 9B is a config switch plus `phase0` and `calibrate` |
| Qwen3.5-0.8B as a draft for speculative decoding | mlx-lm refuses speculative decoding on this architecture (cache not trimmable). With Mack's own snapshot-rewind version it works but is **0.66× the speed** on 4B + 0.8B at 51% acceptance: the draft is only about 3× faster than the main model | implemented and tested, off under `speculative = "auto"`; the draft model is not loaded. Worth re-measuring on the 9B, where the speed ratio is larger |
| peak-memory slopes and tiled prefill need re-measuring on the hybrid | cache grows 32 KB/token; prefill peak at chunk 512 is flat in context | tiling not ported; chunk 512; memory ceiling about 169K tokens, far above the budget |
| pick the highest token rate that keeps CPU slowdown under a target | slowdown is up to 45% for bandwidth-bound work and under 2% for compute-bound work | target applies to a chosen ruler, default the worst case (triad) at 20%, giving 9.7 tok/s while other work runs |
| past about 0.6 duty, switch to the synchronous loop | pipelined stayed 6 to 11% faster at every load on this model. Energy was not measured (needs `powermetrics`, which needs sudo) | never switches on the 4B; the 0.6 rule remains the uncalibrated fallback |
| LoRA ships only if evals improve | base 4B already fabricates in only 7.5% of unanswerable cases with the system prompt alone; 4B training does not fit in memory beside other apps with 8 adapted layers | pipeline and gate built and validated on the 0.8B; no 4B adapter yet |

Two instrument problems were found and fixed while calibrating, and are worth
knowing if the numbers are ever re-derived by hand:

- At default QoS the probe process sometimes ran on an efficiency core and read
  about 50% slow regardless of what Mack was doing. It now runs at
  user-interactive QoS.
- The system-wide CPU counter read 0.5 "other load" while Mack decoded alone
  (kernel work and thermal management). Other load is now summed per process.

## 10. Verification

- `pytest -q`: 36 tests. Unit tests cover text helpers, pins, summary
  verification, archive search, rate-cap and loop-threshold math, config, the
  session (append without rebuild, compaction, recall, clipping, save/load,
  interrupted replies) and the LoRA dataset and gate. Live tests check the three
  decode modes for cache/token consistency on the 0.8B.
- `scripts/e2e_conversation.py`: a nine-turn conversation on the real 4B in a
  1,500-token budget. Eight of eight checks pass: compaction happened, pins are
  verbatim, the door code, budget and sister's name were recovered exactly after
  compaction, and a question about something never said got "I don't have a
  record of…" rather than an invention.
- The server and UI were exercised in a browser, including the governor capping
  and releasing the rate under a synthetic load.
- Explainer: `Mack --selftest` (text rendered to an image is read back exactly by
  the OCR path), `--render-test` (panel layout), and `--explain-text` (the app's
  streaming client against a live engine) pass. The app launches and stays up.

Not verified: the live click-and-drag capture (it needs the Screen Recording
permission, which was not granted during the build); anything on the 9B; a shipped adapter; energy per token;
behaviour on battery; conversations long enough to reach the default 8,192-token
budget in normal use (compaction was exercised at 1,500).
