# Calibration: how mlx-community/Qwen3.5-4B-4bit decode slows CPU work

Measured 2026-10-12T14:50:49 on Apple M4. Regenerate with `mack calibrate` (close other apps first).

Unloaded decode: 36.33 tok/s. Idle CPU probe: 91.2 GB/s triad.

## Rate cells: CPU slowdown at each token rate

Triad = memory-bandwidth-bound work (worst case). Compute = cache-resident work.

| loop | rate cap | achieved tok/s | triad GB/s | triad slowdown | compute slowdown | runs |
|---|---:|---:|---:|---:|---:|---:|
| ahead1 | none | 29.06 | 50.9 | 44.9% | 1.7% | 3 |
| ahead1 | 27.2 | 23.29 | 50.98 | 44.8% | 1.0% | 3 |
| ahead1 | 18.2 | 16.84 | 59.59 | 35.0% | 0.9% | 3 |
| ahead1 | 10.9 | 10.59 | 72.17 | 21.6% | 0.4% | 3 |
| ahead1 | 5.5 | 5.4 | 81.01 | 11.9% | 0.3% | 3 |
| sync | none | 27.61 | 53.58 | 42.0% | 1.4% | 3 |

## Rate cap this implies

| target | ruler | cap |
|---:|---|---|
| 5% | triad | 4.0 tok/s (target needs 2.3 tok/s; held at the 4 tok/s floor) |
| 10% | triad | 4.5 tok/s (interpolated) |
| 20% | triad | 9.7 tok/s (interpolated) |
| 30% | triad | 14.5 tok/s (interpolated) |
| 5% | compute | none (full speed is within target) |
| 10% | compute | none (full speed is within target) |
| 20% | compute | none (full speed is within target) |
| 30% | compute | none (full speed is within target) |

## Loop cells: pipelined vs synchronous under a stream load

| load duty | load GB/s | other CPU duty | ahead1 tok/s | sync tok/s | ahead1 / sync | runs |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.211 | 36.0 | 32.4 | 1.11 | 3 |
| 0.3 | 33.5 | 0.382 | 27.8 | 25.8 | 1.08 | 3 |
| 0.6 | 59.5 | 0.588 | 22.8 | 20.6 | 1.11 | 3 |
| 1.0 | 91.3 | 0.72 | 20.8 | 19.6 | 1.06 | 3 |

Synchronous-loop threshold: other CPU duty 1.01 (pipelined loop stayed faster at every measured load; synchronous loop never selected).
