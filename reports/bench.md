# mack bench: mlx-community/Qwen3.5-4B-4bit

Measured 2026-10-22T16:50:07, 60 s per pace.

| pace | tok/s (target) | tok/s (measured) | memory-heavy slowdown | compute-heavy slowdown | added power | thermal | contract |
|---|---:|---:|---:|---:|---:|---|---|
| gentle | 2.2651006711409396 | 2.1 | 5.1% | 4.4% | +-0.03 W | Nominal | ≤ 15% ✓ |
| balanced | 4.530201342281879 | 3.82 | 7.9% | 1.2% | +1.67 W | Nominal | ≤ 25% ✓ |
| steady | 9.727678018575851 | 6.8 | 21.0% | 0.3% | +3.67 W | Nominal | ≤ 30% ✓ |
| full | — | 6.8 | 17.7% | 0.4% | +3.32 W | Nominal | — |
| auto, your apps busy | — | 2.26 | 6.4% | 0.0% | +1.12 W | Nominal | — |
| auto, nothing else running | — | 4.43 | — | — | +1.82 W | Nominal | — |

**Noisy:** other apps were busy during gentle, full, auto, nothing else running; rerun with the Mac otherwise idle.
