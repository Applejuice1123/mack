# Phase 0 measurements: mlx-community/Qwen3.5-4B-4bit

Measured 2026-10-12T14:39:09 on Apple M4. Regenerate with `mack phase0`.

## Architecture
- 32 layers: 8 keep a growing KV cache, 24 carry fixed-size recurrent state.
- Cache trimmable: **False**. Token-level eviction is not available; context is reduced by rebuild.

## Memory
- Weights (main + draft): 2.791 GB. Wired limit: 11.45 GB.
- Cache growth: 31.99 KiB/token (base 51.5 MB).
- Prefill transient above the cache at chunk 512: 786.5 MB, bounded (see table: it does not grow with context).
- Peak at the largest context by chunk size (MB over resting): {128: 737.2, 512: 1085.8, 2048: 2227.4}.
- Context ceiling at 80% of the wired limit: **168,903 tokens** (weights + cache + transient). Available memory when measured: 4.48 GB.

| context | chunk | cache MB | peak over resting MB | prefill tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 85.1 | 415.7 | 315.2 |
| 1024 | 512 | 85.1 | 871.6 | 331.5 |
| 1024 | 2048 | 85.1 | 1088.4 | 331.8 |
| 2048 | 128 | 118.6 | 473.9 | 319.7 |
| 2048 | 512 | 118.6 | 904.0 | 326.7 |
| 2048 | 2048 | 118.6 | 1552.6 | 329.9 |
| 4096 | 128 | 185.7 | 581.1 | 314.6 |
| 4096 | 512 | 185.7 | 773.1 | 312.2 |
| 4096 | 2048 | 185.7 | 1824.8 | 309.4 |
| 8192 | 128 | 319.9 | 737.2 | 287.5 |
| 8192 | 512 | 319.9 | 1085.8 | 282.4 |
| 8192 | 2048 | 319.9 | 2227.4 | 247.5 |

## Decode
- ahead1 (one token in flight): 36.29 tok/s
- sync: 33.98 tok/s

## Speculative decoding (snapshot-rewind)
- 24.02 tok/s vs 36.34 tok/s plain: **x0.661**, acceptance 51% with 3 draft tokens.
- `speculative = "auto"` enables it only at x1.05 or better.
