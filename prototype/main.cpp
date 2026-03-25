#include "src/llama-context.h"
#include "src/llama-graph.h"
#include "src/llama-impl.h"
#include "src/llama-kv-cache-iswa.h"
#include "src/llama-kv-cache.h"
#include "src/llama-memory-hybrid-iswa.h"
#include "src/llama-memory-hybrid.h"
#include "src/llama-memory.h"
#include "ggml-metal.h"
#include "kv_zlib.h"
#include "llama.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static bool mack_pack_iswa_cell(llama_kv_cache_iswa * iswa, llama_seq_id seq_id, llama_pos pos, std::vector<uint8_t> & out) {
    std::vector<uint8_t> base_blob;
    std::vector<uint8_t> swa_blob;
    const bool ok_b = iswa->get_base()->single_cell_kv_blob(seq_id, pos, base_blob);
    const bool ok_s = iswa->get_swa()->single_cell_kv_blob(seq_id, pos, swa_blob);
    if (!ok_b && !ok_s) {
        return false;
    }
    out.clear();
    auto pack_chunk = [&out](const std::vector<uint8_t> & chunk) {
        const uint32_t len = (uint32_t) chunk.size();
        const size_t   o   = out.size();
        out.resize(o + sizeof(len) + chunk.size());
        memcpy(out.data() + o, &len, sizeof len);
        if (!chunk.empty()) {
            memcpy(out.data() + o + sizeof len, chunk.data(), chunk.size());
        }
    };
    pack_chunk(base_blob);
    pack_chunk(swa_blob);
    return true;
}

/** Meta + K/V for the cell at `pos` (llama_pos), before seq_rm. */
static bool mack_collect_evicted_cell_blob(llama_memory_t mem, llama_seq_id seq_id, llama_pos pos, std::vector<uint8_t> & out) {
    out.clear();
    auto * mem_i = reinterpret_cast<llama_memory_i *>(mem);

    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem_i)) {
        return kv->single_cell_kv_blob(seq_id, pos, out);
    }
    if (auto * hy = dynamic_cast<llama_memory_hybrid *>(mem_i)) {
        llama_kv_cache * attn = hy->get_mem_attn();
        return attn && attn->single_cell_kv_blob(seq_id, pos, out);
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem_i)) {
        return mack_pack_iswa_cell(iswa, seq_id, pos, out);
    }
    if (auto * hyi = dynamic_cast<llama_memory_hybrid_iswa *>(mem_i)) {
        llama_kv_cache_iswa * attn = hyi->get_mem_attn();
        return attn && mack_pack_iswa_cell(attn, seq_id, pos, out);
    }
    return false;
}

static void llama_log_callback_simple(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level < GGML_LOG_LEVEL_INFO) {
        fputs(text, stderr);
    }
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-no-kv-evict] [-kvzlib] [prompt]\n", argv[0]);
    printf("\n    -no-kv-evict   disable Metal attention-based KV eviction (for A/B benchmarks).\n");
    printf("\n    -kvzlib     before each eviction, dump that cell's K/V, zlib-compress (lossless; slow).\n");
    printf("                Or set env MACK_KV_ZLIB=1. MACK_KV_ZLIB_CHECK=1 verifies round-trip.\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt    = "Testing gpu kernel efficiency";
    int         ngl       = 99;
    int         n_predict = 128;
    bool        opt_kv_zlib = false;
    bool        kv_evict  = true;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-kvzlib") == 0) {
                opt_kv_zlib = true;
            } else if (strcmp(argv[i], "-no-kv-evict") == 0) {
                kv_evict = false;
            } else {
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    {
        const char * e = std::getenv("MACK_KV_ZLIB");
        if (e && e[0] != '\0' && strcmp(e, "0") != 0) {
            opt_kv_zlib = true;
        }
    }

    ggml_backend_load_all();
    llama_log_set(llama_log_callback_simple, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model from '%s'\n", __func__, model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = n_prompt + n_predict - 1;
    ctx_params.n_batch              = n_prompt;
    ctx_params.no_perf              = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    ggml_backend_t backend_metal = nullptr;
    {
        auto * sched = ((struct llama_context *) ctx)->get_sched();
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
            auto * b = ggml_backend_sched_get_backend(sched, i);
            if (strcmp(ggml_backend_name(b), "Metal") == 0) {
                backend_metal = b;
                break;
            }
        }
    }

    int         n_decode = 0;
    int         n_evicted = 0;
    int         last_evict_pos = -1;
    llama_token new_token_id;
    std::vector<llama_token> gen_tokens;
    gen_tokens.reserve((size_t) n_predict + 4);
    std::string             prev_detok;
    size_t      kv_saved_actual_bytes = 0;
    const size_t seq_size_initial = llama_state_seq_get_size(ctx, 0);
    const auto t_gen_start = std::chrono::steady_clock::now();

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_head  = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    const int kv_dim = (n_head > 0) ? (n_embd * n_head_kv / n_head) : n_embd;
    const double bytes_per_token = (double) n_layer * (double) kv_dim * 2.0 /* K+V */ * 2.0 /* fp16 */;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
        if (kv_evict && backend_metal && n_decode >= 30 && n_decode % 10 == 0) {
            ggml_backend_metal_set_attention_extraction(backend_metal, true);
        }

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        if (kv_evict && backend_metal && n_decode >= 30 && n_decode % 10 == 0) {
            auto *               gf    = ((struct llama_context *) ctx)->get_gf_res_prev()->get_gf();
            struct ggml_tensor * fattn = nullptr;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                auto * node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    fattn = node;
                    break;
                }
            }
            if (fattn) {
                int                n_kv = (int) fattn->src[1]->ne[1];
                std::vector<float> importances(n_kv);
                ggml_backend_metal_get_attention_importances(backend_metal, importances.data(), n_kv);

                int   min_idx       = -1;
                float min_val       = 1e9f;
                int   protect_start = 4;
                int   protect_end   = n_pos - 4;

                if (protect_end > protect_start) {
                    for (int i = protect_start; i < protect_end; i++) {
                        if (importances[i] < min_val) {
                            min_val = importances[i];
                            min_idx = i;
                        }
                    }
                }

                if (min_idx != -1) {
                    llama_memory_t mem = llama_get_memory(ctx);
                    const size_t   seq_size_before = llama_state_seq_get_size(ctx, 0);
                    std::vector<uint8_t> evict_blob;
                    if (opt_kv_zlib) {
                        llama_synchronize(ctx);
                        if (!mack_collect_evicted_cell_blob(mem, 0, (llama_pos) min_idx, evict_blob)) {
                            fprintf(stderr, "[KV_ZLIB] evicted cell dump failed (pos=%d)\n", min_idx);
                        }
                    }

                    llama_memory_seq_rm(mem, 0, min_idx, min_idx + 1);
                    llama_memory_seq_add(mem, 0, min_idx + 1, -1, -1);
                    const size_t seq_size_after = llama_state_seq_get_size(ctx, 0);
                    if (seq_size_before > seq_size_after) {
                        kv_saved_actual_bytes += (seq_size_before - seq_size_after);
                    }
                    n_pos -= 1;
                    n_evicted += 1;
                    last_evict_pos = min_idx;
                    const double kv_saved_est_mb_now =
                        (n_evicted * bytes_per_token) / (1024.0 * 1024.0);
                    const double kv_saved_actual_mb_now =
                        (double) kv_saved_actual_bytes / (1024.0 * 1024.0);
                    const double kv_state_mb_now =
                        (double) seq_size_after / (1024.0 * 1024.0);
                    const double kv_state_saved_mb_now =
                        seq_size_initial > seq_size_after
                            ? (double) (seq_size_initial - seq_size_after) / (1024.0 * 1024.0)
                            : 0.0;
                    fprintf(stderr,
                            "[METRICS_UPDATE] evicted_tokens=%d evict_pos=%d kv_saved_mb=%.3f kv_saved_actual_mb=%.3f "
                            "kv_state_mb=%.3f kv_state_saved_mb=%.3f n_decode=%d\n",
                            n_evicted, min_idx, kv_saved_est_mb_now, kv_saved_actual_mb_now,
                            kv_state_mb_now, kv_state_saved_mb_now, n_decode);
                    fflush(stderr);

                    if (opt_kv_zlib && !evict_blob.empty()) {
                        const size_t           raw_sz = evict_blob.size();
                        const std::vector<uint8_t> comp = kv_zlib_compress(evict_blob.data(), raw_sz);
                        if (!comp.empty()) {
                            fprintf(stderr,
                                    "[KV_ZLIB] evicted_raw=%zu evicted_zlib=%zu ratio=%.4f pos=%d\n",
                                    raw_sz,
                                    comp.size(),
                                    (double) raw_sz / (double) comp.size(),
                                    min_idx);
                            const char * chk = std::getenv("MACK_KV_ZLIB_CHECK");
                            if (chk && chk[0] != '\0' && strcmp(chk, "0") != 0) {
                                std::vector<uint8_t> round(raw_sz);
                                if (kv_zlib_uncompress(comp.data(), comp.size(), round.data(), raw_sz) &&
                                    memcmp(evict_blob.data(), round.data(), raw_sz) == 0) {
                                    fprintf(stderr, "[KV_ZLIB] roundtrip ok\n");
                                } else {
                                    fprintf(stderr, "[KV_ZLIB] roundtrip FAILED\n");
                                }
                            }
                        } else {
                            fprintf(stderr, "[KV_ZLIB] compress failed\n");
                        }
                    }
                }
            }
            ggml_backend_metal_set_attention_extraction(backend_metal, false);
        }

        n_pos += batch.n_tokens;
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        gen_tokens.push_back(new_token_id);

        int32_t cap = 4096;
        std::vector<char> dt((size_t) cap);
        int32_t n = llama_detokenize(
            vocab,
            gen_tokens.data(),
            (int32_t) gen_tokens.size(),
            dt.data(),
            cap,
            false,
            false);
        while (n < 0) {
            cap     = -n + 256;
            dt.resize((size_t) cap);
            n = llama_detokenize(
                vocab,
                gen_tokens.data(),
                (int32_t) gen_tokens.size(),
                dt.data(),
                cap,
                false,
                false);
        }
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to detokenize output\n", __func__);
            return 1;
        }
        // Full detokenize each step (required for correct BPE + clean_spaces). Prefix-diff streaming
        // breaks when clean_spaces rewrites earlier bytes — causes "survi val" style gaps in the UI.
        // Framing: full UTF-8 text + RS (0x1E) terminator; client only renders on complete frames.
        if (n > 0) {
            fwrite(dt.data(), 1, (size_t) n, stdout);
            putchar((char) 0x1e);
            fflush(stdout);
            prev_detok.assign(dt.data(), (size_t) n);
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
    }

    printf("\n");
    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_gen_start).count();
    const double tps = (elapsed_s > 0.0 && n_decode > 0)
        ? (double) n_decode / elapsed_s : 0.0;
    const double kv_saved_mb = (n_evicted * bytes_per_token) / (1024.0 * 1024.0);
    const double kv_saved_actual_mb = (double) kv_saved_actual_bytes / (1024.0 * 1024.0);
    const size_t seq_size_final = llama_state_seq_get_size(ctx, 0);
    const double kv_state_mb = (double) seq_size_final / (1024.0 * 1024.0);
    const double kv_state_saved_mb =
        seq_size_initial > seq_size_final
            ? (double) (seq_size_initial - seq_size_final) / (1024.0 * 1024.0)
            : 0.0;
    fprintf(stderr,
            "[METRICS] evicted_tokens=%d evict_pos=%d kv_saved_mb=%.3f kv_saved_actual_mb=%.3f "
            "kv_state_mb=%.3f kv_state_saved_mb=%.3f n_decode=%d tps=%.2f\n",
            n_evicted, last_evict_pos, kv_saved_mb, kv_saved_actual_mb, kv_state_mb, kv_state_saved_mb, n_decode, tps);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
