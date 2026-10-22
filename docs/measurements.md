# How the slowdown numbers are measured

The headline claim, that Mack runs over your files without your Mac noticing, rests on two
measurements, both reproducible on any Apple Silicon Mac.

## The ruler

`mack/native/mack_probe.c` is a small C program with two kernels:

- **STREAM triad** over a 192 MB working set (far larger than the system-level cache), so every
  pass is real memory traffic. It stands in for memory-heavy work: builds, large spreadsheets,
  photo and video processing. Its rate in GB/s drops when something else competes for memory bandwidth.
- **A cache-resident FMA loop** (8 KB of state). It stands in for compute-heavy work and only drops if
  Mack takes CPU time or clock speed away.

Two rules keep the ruler honest, both learned the hard way:

1. Take **two back-to-back bursts and use the second**. A single burst after an idle moment
   measures the CPU's clock ramping up, not contention.
2. Run the probe at **user-interactive QoS**. At default QoS macOS sometimes put it on an
   efficiency core, where it read about 50% slow for reasons unrelated to Mack.

Slowdown = 1 − (rate while Mack runs ÷ idle rate), with the idle rate re-measured next to
every cell so drift does not accumulate.

## Calibration: what each token rate costs (`mack calibrate`)

Decode at several capped token rates beside the probe, three interleaved rounds per cell.
On a base M4 with 16 GB and Qwen3.5-4B-4bit (reports/calibration.md):

| decode rate | memory-heavy slowdown | compute-heavy slowdown |
|---:|---:|---:|
| 5.4 tok/s | 11.9% | 0.3% |
| 10.6 tok/s | 21.6% | 0.4% |
| 16.8 tok/s | 35.0% | 0.9% |
| 23.3 tok/s | 44.8% | 1.0% |
| uncapped | 44.9% | 1.7% |

Mack's effect on the rest of the Mac is a memory-bandwidth effect. The paces are the rates whose
interpolated memory-heavy slowdown is 5% (gentle, 2.3 tok/s), 10% (balanced, 4.5) and 20% (steady, 9.7).

## The benchmark: a real job at each pace (`mack bench`)

Calibration paces decode only. A batch job also reads prompts, rests between files and handles
JSON, so `mack bench` runs a real batch job on synthetic notes at each pace while the probe runs,
and prints:

```
| pace | tok/s (target) | tok/s (measured) | memory-heavy slowdown | compute-heavy slowdown | contract |
```

The contract (mack.toml `[pace]`): gentle ≤ 15%, balanced ≤ 25%, steady ≤ 30%, full unbounded.
If a pace breaks it, lower that pace's rate in `[pace]` (`gentle_tps = ...`) and record why in
FINDINGS.md; never loosen a threshold silently.

**Status:** `mack bench` is built and has not been run on the real model yet. Close other apps,
plug in, and run `mack bench` (about 6 minutes). Results go to reports/bench.md.

## The Tuesday-morning test (`scripts/e2e_tuesday.py`)

A 300-note vault tagged at gentle pace while the probe logs every 10 s, with the worker
SIGKILLed partway and the Mac put to sleep once. It passes if the output has exactly 300 unique lines,
mean probe slowdown is within gentle's contract, and no item needed more than two attempts.
At gentle pace this takes many hours; `--n 30` is a rehearsal.

## Why prefill is weighted at 0.25

Reading a prompt (prefill) is compute-bound and fast (about 300 tok/s); generating is
memory-bandwidth-bound. Counting a prompt token as a full generated token would make
long files crawl for no benefit; not counting it would let a long file run at full power under
a gentle pace. The weight is a config value (`[batch] prefill_weight`), and prefill also keeps
its duty-cycle rests between 512-token chunks. `mack bench` is what checks the result.
