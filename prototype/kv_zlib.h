#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/** zlib DEFLATE; lossless. Returns empty on failure. */
std::vector<uint8_t> kv_zlib_compress(const uint8_t * src, size_t src_len, int level = -1);

/** Inflate; `dst_cap` must be the exact uncompressed size (e.g. from llama_state_seq_get_size). */
bool kv_zlib_uncompress(const uint8_t * src, size_t src_len, uint8_t * dst, size_t dst_cap);
