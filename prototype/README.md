# The original prototype (March 2026)

Before Mack ran on MLX, it was a llama.cpp fork running Gemma-3-4B on Metal. The idea was
**token eviction**: after the first 30 generated tokens, and then every 10 tokens, it read the
attention scores out of the flash-attention kernel and evicted the KV-cache entry the model was
paying the least attention to, so long conversations stayed small.

| file | what it is |
|---|---|
| `llama.cpp-token-eviction.patch` | the fork's changes: Metal kernels and ops, the CPU path and `llama-context` |
| `main.cpp` | `gemma_chat`, a chat program with eviction on by default; `-no-kv-evict` turns it off for A/B runs, `-kvzlib` keeps a compressed copy of each evicted entry |
| `kv_zlib.cpp`, `kv_zlib.h` | lossless zlib compression of evicted KV entries |
| `Makefile` | builds `gemma_chat` against a copy of the patched llama.cpp sources |
| `benchmarks/` | a long-context stress test (facts planted early, mid and deep in a long prompt) comparing recall with eviction on and off, and its scorer |

To build it, check out llama.cpp at commit `f6b533d89` (2026-01-28), apply the patch with
`git apply llama.cpp-token-eviction.patch`, and copy `src/`, `ggml/` and `include/` next to
these files. The llama.cpp sources themselves are not included here.

## Why Mack moved on

Mack's current model, Qwen3.5, is a hybrid: only 8 of its 32 layers keep a KV cache, and the
other 24 carry a recurrent state that cannot be trimmed. Evicting attention entries while that
state still remembers the tokens would put the model somewhere it never trained. So Mack
replaced eviction with compaction, rebuilding the context from exact quotes and checked
summaries (see "Under the hood" in the main [README](../README.md)).

These files were added to the repository in October 2026 from the original project folders.
Each prototype commit is dated by when its files were last changed. llama.cpp is MIT licensed,
copyright the ggml authors; the patch is a derivative of it.
