#include "kv_zlib.h"

#include <zlib.h>

#include <cstring>

std::vector<uint8_t> kv_zlib_compress(const uint8_t * src, size_t src_len, int level) {
    if (!src && src_len > 0) {
        return {};
    }
    uLongf bound = compressBound(static_cast<uLong>(src_len));
    std::vector<uint8_t> dst(bound);
    uLongf dest_len = bound;
    const int zlev = level < 0 ? Z_DEFAULT_COMPRESSION : level;
    const int zerr =
        compress2(dst.data(), &dest_len, src, static_cast<uLong>(src_len), zlev);
    if (zerr != Z_OK) {
        return {};
    }
    dst.resize(static_cast<size_t>(dest_len));
    return dst;
}

bool kv_zlib_uncompress(const uint8_t * src, size_t src_len, uint8_t * dst, size_t dst_cap) {
    if (!src || !dst || dst_cap == 0) {
        return false;
    }
    uLongf out_len = static_cast<uLongf>(dst_cap);
    return uncompress(dst, &out_len, src, static_cast<uLong>(src_len)) == Z_OK &&
           out_len == static_cast<uLongf>(dst_cap);
}
