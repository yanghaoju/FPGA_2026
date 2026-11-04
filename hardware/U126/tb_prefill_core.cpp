#include "unified_ip.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
static std::vector<T> read_binary(const std::string& path, std::size_t count) {
    std::vector<T> data(count);
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T)) {
        throw std::runtime_error("short read " + path);
    }
    return data;
}

static float bf16_to_float_sw(uint16_t x) {
    const uint32_t bits = static_cast<uint32_t>(x) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t float_to_bf16_rne_sw(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t upper = bits >> 16;
    const uint32_t lower = bits & 0xffffu;
    const uint32_t increment =
        lower > 0x8000u || (lower == 0x8000u && (upper & 1u));
    return static_cast<uint16_t>(upper + increment);
}

static std::vector<mem_word_t> pack_bf16(const std::vector<uint16_t>& raw) {
    if (raw.size() % VALUES_PER_WORD) throw std::runtime_error("unaligned input");
    std::vector<mem_word_t> packed(raw.size() / VALUES_PER_WORD);
    for (std::size_t word = 0; word < packed.size(); ++word) {
        mem_word_t value = 0;
        for (int item = 0; item < VALUES_PER_WORD; ++item) {
            value.range((item + 1) * BF16_BITS - 1, item * BF16_BITS) =
                raw[word * VALUES_PER_WORD + item];
        }
        packed[word] = value;
    }
    return packed;
}

static std::size_t raw_q_index(
    int kv_head,
    int query_block,
    int context,
    int word,
    int item
) {
    const int row = context / GQA_GROUP;
    const int lane = context % GQA_GROUP;
    const int q_head = kv_head * GQA_GROUP + lane;
    const int query = query_block * ROWS_PER_BLOCK + row;
    const int dim = word * VALUES_PER_WORD + item;
    return (static_cast<std::size_t>(q_head) * SEQ + query) * HEAD_DIM + dim;
}

static std::size_t block_io_word_index(
    int kv_head,
    int query_block,
    int context,
    int word
) {
    return ((static_cast<std::size_t>(kv_head) * BLOCKS_PER_KV_HEAD
             + query_block) * CONTEXTS_PER_BLOCK + context)
           * WORDS_PER_ROW + word;
}

static std::vector<mem_word_t> pack_q_block_major(
    const std::vector<uint16_t>& raw
) {
    const std::size_t expected =
        static_cast<std::size_t>(Q_HEADS) * SEQ * HEAD_DIM;
    if (raw.size() != expected) throw std::runtime_error("unexpected Q size");
    std::vector<mem_word_t> packed(expected / VALUES_PER_WORD);
    for (int kv_head = 0; kv_head < KV_HEADS; ++kv_head) {
        for (int query_block = 0;
             query_block < BLOCKS_PER_KV_HEAD;
             ++query_block) {
            for (int context = 0; context < CONTEXTS_PER_BLOCK; ++context) {
                for (int word = 0; word < WORDS_PER_ROW; ++word) {
                    mem_word_t value = 0;
                    for (int item = 0; item < VALUES_PER_WORD; ++item) {
                        value.range(
                            (item + 1) * BF16_BITS - 1,
                            item * BF16_BITS
                        ) = raw[raw_q_index(
                            kv_head, query_block, context, word, item)];
                    }
                    packed[block_io_word_index(
                        kv_head, query_block, context, word)] = value;
                }
            }
        }
    }
    return packed;
}

static std::vector<uint16_t> unpack_block_major(
    const std::vector<mem_word_t>& packed
) {
    const std::size_t expected =
        static_cast<std::size_t>(Q_HEADS) * SEQ * HEAD_DIM;
    if (packed.size() != expected / VALUES_PER_WORD) {
        throw std::runtime_error("unexpected block-major size");
    }
    std::vector<uint16_t> raw(expected);
    for (int kv_head = 0; kv_head < KV_HEADS; ++kv_head) {
        for (int query_block = 0;
             query_block < BLOCKS_PER_KV_HEAD;
             ++query_block) {
            for (int context = 0; context < CONTEXTS_PER_BLOCK; ++context) {
                for (int word = 0; word < WORDS_PER_ROW; ++word) {
                    const mem_word_t value = packed[block_io_word_index(
                        kv_head, query_block, context, word)];
                    for (int item = 0; item < VALUES_PER_WORD; ++item) {
                        raw[raw_q_index(
                            kv_head, query_block, context, word, item)] =
                            static_cast<uint16_t>(value.range(
                                (item + 1) * BF16_BITS - 1,
                                item * BF16_BITS));
                    }
                }
            }
        }
    }
    return raw;
}

static std::vector<mem_word_t> pack_v_dimension_major(
    const std::vector<uint16_t>& raw
) {
    const std::size_t expected =
        static_cast<std::size_t>(KV_HEADS) * STORAGE_SEQ * HEAD_DIM;
    if (raw.size() != expected) throw std::runtime_error("unexpected V size");
    std::vector<mem_word_t> packed(
        static_cast<std::size_t>(KV_HEADS) * TILES_PER_KV_HEAD * HEAD_DIM
    );
    for (int kv_head = 0; kv_head < KV_HEADS; ++kv_head) {
        for (int tile = 0; tile < TILES_PER_KV_HEAD; ++tile) {
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                mem_word_t word = 0;
                for (int key = 0; key < TILE_K; ++key) {
                    const std::size_t raw_index =
                        (static_cast<std::size_t>(kv_head) * STORAGE_SEQ
                         + tile * TILE_K + key) * HEAD_DIM + dim;
                    word.range((key + 1) * BF16_BITS - 1, key * BF16_BITS) =
                        raw[raw_index];
                }
                const std::size_t packed_index =
                    (static_cast<std::size_t>(kv_head) * TILES_PER_KV_HEAD
                     + tile) * HEAD_DIM + dim;
                packed[packed_index] = word;
            }
        }
    }
    return packed;
}

static bool verify_v_dimension_major(
    const std::vector<uint16_t>& raw,
    const std::vector<mem_word_t>& packed
) {
    for (int kv_head = 0; kv_head < KV_HEADS; ++kv_head) {
        for (int tile = 0; tile < TILES_PER_KV_HEAD; ++tile) {
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                const std::size_t packed_index =
                    (static_cast<std::size_t>(kv_head) * TILES_PER_KV_HEAD
                     + tile) * HEAD_DIM + dim;
                for (int key = 0; key < TILE_K; ++key) {
                    const std::size_t raw_index =
                        (static_cast<std::size_t>(kv_head) * STORAGE_SEQ
                         + tile * TILE_K + key) * HEAD_DIM + dim;
                    const uint16_t recovered = static_cast<uint16_t>(
                        packed[packed_index].range(
                            (key + 1) * BF16_BITS - 1, key * BF16_BITS)
                    );
                    if (recovered != raw[raw_index]) return false;
                }
            }
        }
    }
    return true;
}

static std::vector<uint16_t> unpack_bf16(const std::vector<mem_word_t>& packed) {
    std::vector<uint16_t> raw(packed.size() * VALUES_PER_WORD);
    for (std::size_t word = 0; word < packed.size(); ++word) {
        for (int item = 0; item < VALUES_PER_WORD; ++item) {
            raw[word * VALUES_PER_WORD + item] = static_cast<uint16_t>(
                packed[word].range((item + 1) * BF16_BITS - 1, item * BF16_BITS)
            );
        }
    }
    return raw;
}

static ap_int<DECODE_KV_QUANT_BITS> quantize_kv13_sw(float value, float scale) {
    int rounded = static_cast<int>(std::round(value / scale));
    rounded = std::max(-DECODE_KV_QUANT_LIMIT,
                       std::min(DECODE_KV_QUANT_LIMIT, rounded));
    return ap_int<DECODE_KV_QUANT_BITS>(rounded);
}

static bool verify_prompt_cache(
    const std::vector<uint16_t>& k_raw,
    const std::vector<uint16_t>& v_raw,
    const std::vector<mem_word_t>& k_cache,
    const std::vector<mem_word_t>& v_cache,
    unsigned prompt_tokens,
    int head_count = KV_HEADS
) {
    std::size_t k_data_mismatches = 0;
    std::size_t v_data_mismatches = 0;
    std::size_t scale_mismatches = 0;
    for (int head = 0; head < head_count; ++head) {
        for (unsigned token = 0; token < prompt_tokens; ++token) {
            float k_peak = 0.0f;
            float v_peak = 0.0f;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                const std::size_t raw_index =
                    (static_cast<std::size_t>(head) * STORAGE_SEQ + token)
                    * HEAD_DIM + dim;
                k_peak = std::max(k_peak,
                    std::abs(bf16_to_float_sw(k_raw[raw_index])));
                v_peak = std::max(v_peak,
                    std::abs(bf16_to_float_sw(v_raw[raw_index])));
            }
            const uint16_t k_scale_bits = float_to_bf16_rne_sw(
                k_peak > 0.0f ? k_peak / DECODE_KV_QUANT_LIMIT : 1.0f);
            const uint16_t v_scale_bits = float_to_bf16_rne_sw(
                v_peak > 0.0f ? v_peak / DECODE_KV_QUANT_LIMIT : 1.0f);
            const float k_scale = bf16_to_float_sw(k_scale_bits);
            const float v_scale = bf16_to_float_sw(v_scale_bits);
            mem_word_t expected_k[DECODE_CACHE_WORDS_PER_ROW];
            mem_word_t expected_v[DECODE_CACHE_WORDS_PER_ROW];
            for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
                expected_k[word] = 0;
                expected_v[word] = 0;
            }
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                const std::size_t raw_index =
                    (static_cast<std::size_t>(head) * STORAGE_SEQ + token)
                    * HEAD_DIM + dim;
                const ap_int<DECODE_KV_QUANT_BITS> k_value = quantize_kv13_sw(
                    bf16_to_float_sw(k_raw[raw_index]), k_scale);
                const ap_int<DECODE_KV_QUANT_BITS> v_value = quantize_kv13_sw(
                    bf16_to_float_sw(v_raw[raw_index]), v_scale);
                const int word = dim / DECODE_QUANT_VALUES_PER_CACHE_WORD;
                const int lane = dim % DECODE_QUANT_VALUES_PER_CACHE_WORD;
                expected_k[word].range(
                    (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                    lane * DECODE_KV_QUANT_BITS) = k_value;
                expected_v[word].range(
                    (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                    lane * DECODE_KV_QUANT_BITS) = v_value;
            }
            expected_k[0].range(DECODE_SCALE_LOW_BIT + BF16_BITS - 1,
                                DECODE_SCALE_LOW_BIT) = k_scale_bits;
            expected_v[0].range(DECODE_SCALE_LOW_BIT + BF16_BITS - 1,
                                DECODE_SCALE_LOW_BIT) = v_scale_bits;
            const int data_index =
                (head * DECODE_MAX_KV_TOKENS + token) *
                DECODE_CACHE_WORDS_PER_ROW;
            for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
                if (k_cache[data_index + word] != expected_k[word]) {
                    if (k_data_mismatches < 4) {
                        std::cout << "cache_k_data_mismatch head=" << head
                                  << " token=" << token
                                  << " word=" << word;
                        if (k_data_mismatches == 0) {
                            for (int lane = 0;
                                 lane < DECODE_QUANT_VALUES_PER_CACHE_WORD;
                                 ++lane) {
                                const ap_int<DECODE_KV_QUANT_BITS> actual =
                                    k_cache[data_index + word].range(
                                        (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                                        lane * DECODE_KV_QUANT_BITS);
                                const ap_int<DECODE_KV_QUANT_BITS> reference =
                                    expected_k[word].range(
                                        (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                                        lane * DECODE_KV_QUANT_BITS);
                                if (actual != reference) {
                                    std::cout << " first_lane_diff=" << lane
                                              << ":" << actual.to_int()
                                              << "/" << reference.to_int();
                                    break;
                                }
                            }
                            std::cout << " upper_equal="
                                      << (k_cache[data_index + word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT) ==
                                          expected_k[word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT))
                                      << " unused_actual="
                                      << ap_uint<MEM_WORD_BITS -
                                          DECODE_SCALE_LOW_BIT - BF16_BITS>(
                                          k_cache[data_index + word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT + BF16_BITS))
                                             .to_string(16)
                                      << " xor="
                                      << (k_cache[data_index + word] ^
                                          expected_k[word]).to_string(16);
                        }
                        std::cout << "\n";
                    }
                    ++k_data_mismatches;
                }
                if (v_cache[data_index + word] != expected_v[word]) {
                    if (v_data_mismatches < 4) {
                        std::cout << "cache_v_data_mismatch head=" << head
                                  << " token=" << token
                                  << " word=" << word;
                        if (v_data_mismatches == 0) {
                            for (int lane = 0;
                                 lane < DECODE_QUANT_VALUES_PER_CACHE_WORD;
                                 ++lane) {
                                const ap_int<DECODE_KV_QUANT_BITS> actual =
                                    v_cache[data_index + word].range(
                                        (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                                        lane * DECODE_KV_QUANT_BITS);
                                const ap_int<DECODE_KV_QUANT_BITS> reference =
                                    expected_v[word].range(
                                        (lane + 1) * DECODE_KV_QUANT_BITS - 1,
                                        lane * DECODE_KV_QUANT_BITS);
                                if (actual != reference) {
                                    std::cout << " first_lane_diff=" << lane
                                              << ":" << actual.to_int()
                                              << "/" << reference.to_int();
                                    break;
                                }
                            }
                            std::cout << " upper_equal="
                                      << (v_cache[data_index + word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT) ==
                                          expected_v[word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT))
                                      << " unused_actual="
                                      << ap_uint<MEM_WORD_BITS -
                                          DECODE_SCALE_LOW_BIT - BF16_BITS>(
                                          v_cache[data_index + word].range(
                                              MEM_WORD_BITS - 1,
                                              DECODE_SCALE_LOW_BIT + BF16_BITS))
                                             .to_string(16)
                                      << " xor="
                                      << (v_cache[data_index + word] ^
                                          expected_v[word]).to_string(16);
                        }
                        std::cout << "\n";
                    }
                    ++v_data_mismatches;
                }
            }
            const uint16_t actual_k_scale = static_cast<uint16_t>(
                k_cache[data_index].range(
                    DECODE_SCALE_LOW_BIT + BF16_BITS - 1,
                    DECODE_SCALE_LOW_BIT));
            const uint16_t actual_v_scale = static_cast<uint16_t>(
                v_cache[data_index].range(
                    DECODE_SCALE_LOW_BIT + BF16_BITS - 1,
                    DECODE_SCALE_LOW_BIT));
            if (actual_k_scale != k_scale_bits ||
                actual_v_scale != v_scale_bits) {
                if (scale_mismatches < 8) {
                    std::cout << "cache_scale_mismatch head=" << head
                              << " token=" << token
                              << " k_actual=" << actual_k_scale
                              << " k_expected=" << k_scale_bits
                              << " v_actual=" << actual_v_scale
                              << " v_expected=" << v_scale_bits << "\n";
                }
                ++scale_mismatches;
            }
        }
    }
    const bool pass = k_data_mismatches == 0 &&
        v_data_mismatches == 0 && scale_mismatches == 0;
    std::cout << "prompt_tokens = " << prompt_tokens << "\n";
    std::cout << "cache_k_data_mismatches = " << k_data_mismatches << "\n";
    std::cout << "cache_v_data_mismatches = " << v_data_mismatches << "\n";
    std::cout << "cache_scale_mismatches = " << scale_mismatches << "\n";
    std::cout << "prefill_to_decode_cache_pack = "
              << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

static bool run_runtime_length_sweep(const std::string& data_dir) {
    const std::size_t q_count =
        static_cast<std::size_t>(Q_HEADS) * SEQ * HEAD_DIM;
    const std::size_t kv_count =
        static_cast<std::size_t>(KV_HEADS) * STORAGE_SEQ * HEAD_DIM;
    const auto q_raw = read_binary<uint16_t>(data_dir + "/q.bf16", q_count);
    const auto k_raw = read_binary<uint16_t>(data_dir + "/k.bf16", kv_count);
    const auto v_raw = read_binary<uint16_t>(data_dir + "/v.bf16", kv_count);
    const auto golden_bf16 =
        read_binary<uint16_t>(data_dir + "/out_ref.bf16", q_count);
    const auto q = pack_q_block_major(q_raw);
    const auto k_input = pack_bf16(k_raw);
    const auto v_input = pack_v_dimension_major(v_raw);
    const unsigned lengths[] = {1, 31, 32, 33, 255, 256, 257, 511, 512};
    const uint16_t sentinel = 0x7e55u;
    mem_word_t sentinel_word = 0;
    for (int item = 0; item < VALUES_PER_WORD; ++item) {
        sentinel_word.range((item + 1) * BF16_BITS - 1,
                            item * BF16_BITS) = sentinel;
    }

    bool all_pass = true;
    for (unsigned prompt_tokens : lengths) {
        auto k = k_input;
        auto v = v_input;
        std::vector<mem_word_t> out(q.size(), sentinel_word);
        int8_gqa_prefill_decode_ip(
            FPT_MODE_PREFILL, q.data(), k.data(), v.data(), out.data(),
            prompt_tokens);
        const bool cache_pass = verify_prompt_cache(
            k_raw, v_raw, k, v, prompt_tokens);
        const auto actual_raw = unpack_block_major(out);
        std::size_t active_mismatches = 0;
        std::size_t inactive_writes = 0;
        double max_abs_error = 0.0;
        for (int head = 0; head < Q_HEADS; ++head) {
            for (int token = 0; token < SEQ; ++token) {
                for (int dim = 0; dim < HEAD_DIM; ++dim) {
                    const std::size_t index =
                        (static_cast<std::size_t>(head) * SEQ + token)
                        * HEAD_DIM + dim;
                    if (static_cast<unsigned>(token) < prompt_tokens) {
                        if (actual_raw[index] != golden_bf16[index]) {
                            ++active_mismatches;
                        }
                        max_abs_error = std::max(
                            max_abs_error,
                            std::abs(
                                static_cast<double>(bf16_to_float_sw(
                                    actual_raw[index]))
                                - static_cast<double>(bf16_to_float_sw(
                                    golden_bf16[index]))));
                    } else if (actual_raw[index] != sentinel) {
                        ++inactive_writes;
                    }
                }
            }
        }
        const std::size_t active_values =
            static_cast<std::size_t>(Q_HEADS) * prompt_tokens * HEAD_DIM;
        const std::size_t mismatch_limit =
            (active_values + 4095u) / 4096u;
        const bool length_pass = cache_pass && inactive_writes == 0
            && active_mismatches <= mismatch_limit
            && max_abs_error <= 0.015625;
        std::cout << "runtime_length=" << prompt_tokens
                  << " active_mismatches=" << active_mismatches
                  << " mismatch_limit=" << mismatch_limit
                  << " max_abs_error=" << max_abs_error
                  << " inactive_writes=" << inactive_writes
                  << " pass=" << (length_pass ? 1 : 0) << "\n";
        all_pass = all_pass && length_pass;
    }
    std::cout << "U126_RUNTIME_LENGTH_SWEEP_PASS="
              << (all_pass ? 1 : 0) << "\n";
    return all_pass;
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: tb_attention_ip DATA_DIR [pack-only|runtime-sweep]\n";
        return 2;
    }
    const std::string data_dir = argv[1];
    if (argc == 3) {
        const std::string mode = argv[2];
        if (mode == "runtime-sweep") {
            return run_runtime_length_sweep(data_dir) ? 0 : 1;
        }
        if (mode != "pack-only") {
            std::cerr << "unknown test mode\n";
            return 2;
        }
        const std::size_t kv_count =
            static_cast<std::size_t>(KV_HEADS) * STORAGE_SEQ * HEAD_DIM;
        const auto k_raw =
            read_binary<uint16_t>(data_dir + "/k.bf16", kv_count);
        const auto v_raw =
            read_binary<uint16_t>(data_dir + "/v.bf16", kv_count);
        const auto k_input = pack_bf16(k_raw);
        const auto v_input = pack_v_dimension_major(v_raw);
        std::vector<mem_word_t> k_cache(ATTENTION_IP_STORAGE_KV_WORDS);
        std::vector<mem_word_t> v_cache(ATTENTION_IP_STORAGE_KV_WORDS);
        fpt_test_pack_prompt_head_kv13(
            k_cache.data(), v_cache.data(), 0, 3,
            k_input.data(), v_input.data());
        return verify_prompt_cache(
            k_raw, v_raw, k_cache, v_cache, 3, 1) ? 0 : 1;
    }
    const std::size_t q_count = static_cast<std::size_t>(Q_HEADS) * SEQ * HEAD_DIM;
    // L512 candidate gate: scale the frozen BF16 mismatch policy with the
    // output size instead of reusing the U19 S256 constant.
    const std::size_t bf16_mismatch_limit = (q_count + 4095u) / 4096u;
    const std::size_t kv_count = static_cast<std::size_t>(KV_HEADS) * STORAGE_SEQ * HEAD_DIM;
    const auto q_raw = read_binary<uint16_t>(data_dir + "/q.bf16", q_count);
    const auto k_raw = read_binary<uint16_t>(data_dir + "/k.bf16", kv_count);
    const auto v_raw = read_binary<uint16_t>(data_dir + "/v.bf16", kv_count);
    const auto golden = read_binary<float>(data_dir + "/out_ref_fp32.f32", q_count);
    const auto golden_bf16 = read_binary<uint16_t>(data_dir + "/out_ref.bf16", q_count);
    const auto online = read_binary<float>(data_dir + "/out_online_fp32.f32", q_count);
    auto q = pack_q_block_major(q_raw);
    auto k = pack_bf16(k_raw);
    auto v = pack_v_dimension_major(v_raw);
    const bool host_pack_bit_exact = unpack_block_major(q) == q_raw && verify_v_dimension_major(v_raw, v);
    std::vector<mem_word_t> out(q.size());

    // The full-length run preserves the frozen S512 numerical gate. Boundary
    // and partial cache cases are covered by runtime-sweep.
    const unsigned prompt_tokens = STORAGE_SEQ;
    int8_gqa_prefill_decode_ip(
        FPT_MODE_PREFILL, q.data(), k.data(), v.data(), out.data(),
        prompt_tokens);
    const bool cache_pack_pass = verify_prompt_cache(
        k_raw, v_raw, k, v, prompt_tokens);
    const auto actual_raw = unpack_block_major(out);
    double max_abs_error = 0.0;
    double row0_max_abs_error = 0.0;
    double mae = 0.0;
    double mse = 0.0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t bf16_mismatch_count = 0;
    double bf16_max_abs_error = 0.0;
    double bf16_mae = 0.0;
    double bf16_mse = 0.0;
    std::size_t actual_online_mismatch_count = 0;
    std::size_t online_direct_mismatch_count = 0;
    std::size_t actual_direct_online_overlap_count = 0;
    std::size_t actual_direct_only_count = 0;
    double actual_online_max_abs_error = 0.0;
    double actual_online_mae = 0.0;
    double actual_online_mse = 0.0;
    std::size_t online_nan_count = 0;
    std::size_t online_inf_count = 0;
    for (std::size_t i = 0; i < q_count; ++i) {
        const double actual = bf16_to_float_sw(actual_raw[i]);
        const double golden_quantized = bf16_to_float_sw(golden_bf16[i]);
        const double bf16_abs_error = std::abs(actual - golden_quantized);
        bf16_max_abs_error = std::max(bf16_max_abs_error, bf16_abs_error);
        bf16_mae += bf16_abs_error;
        bf16_mse += bf16_abs_error * bf16_abs_error;
        if (actual_raw[i] != golden_bf16[i]) {
            ++bf16_mismatch_count;
            if (bf16_mismatch_count <= 32) {
                std::cout << "bf16_mismatch index=" << i
                          << " actual_raw=" << actual_raw[i]
                          << " golden_raw=" << golden_bf16[i]
                          << " abs_error=" << bf16_abs_error << "\n";
            }
        }
        const uint16_t online_raw = float_to_bf16_rne_sw(online[i]);
        const double online_quantized = bf16_to_float_sw(online_raw);
        const double actual_online_error = std::abs(actual - online_quantized);
        actual_online_max_abs_error = std::max(
            actual_online_max_abs_error, actual_online_error);
        actual_online_mae += actual_online_error;
        actual_online_mse += actual_online_error * actual_online_error;
        const bool actual_direct_mismatch = actual_raw[i] != golden_bf16[i];
        const bool online_direct_mismatch = online_raw != golden_bf16[i];
        if (actual_raw[i] != online_raw) ++actual_online_mismatch_count;
        if (online_direct_mismatch) ++online_direct_mismatch_count;
        if (actual_direct_mismatch && online_direct_mismatch) {
            ++actual_direct_online_overlap_count;
        }
        if (actual_direct_mismatch && !online_direct_mismatch) {
            ++actual_direct_only_count;
        }
        if (std::isnan(actual)) ++nan_count;
        if (std::isinf(actual)) ++inf_count;
        if (std::isnan(online[i])) ++online_nan_count;
        if (std::isinf(online[i])) ++online_inf_count;
        const double diff = actual - static_cast<double>(golden[i]);
        const double abs_diff = std::abs(diff);
        max_abs_error = std::max(max_abs_error, abs_diff);
        mae += abs_diff;
        mse += diff * diff;
    }
    for (int head = 0; head < Q_HEADS; ++head) {
        for (int dim = 0; dim < HEAD_DIM; ++dim) {
            const std::size_t index = static_cast<std::size_t>(head) * SEQ * HEAD_DIM + dim;
            row0_max_abs_error = std::max(
                row0_max_abs_error,
                std::abs(
                    static_cast<double>(bf16_to_float_sw(actual_raw[index]))
                    - static_cast<double>(golden[index])
                )
            );
        }
    }
    mae /= q_count;
    mse /= q_count;
    const double rmse = std::sqrt(mse);
    bf16_mae /= q_count;
    bf16_mse /= q_count;
    const double bf16_rmse = std::sqrt(bf16_mse);
    actual_online_mae /= q_count;
    actual_online_mse /= q_count;
    const double actual_online_rmse = std::sqrt(actual_online_mse);
    const bool direct_reference_gate = bf16_mismatch_count <= bf16_mismatch_limit
        && bf16_max_abs_error <= 0.015625 && bf16_mae < 5.0e-7
        && bf16_rmse < 7.5e-5 && max_abs_error < 0.02;
    const bool online_reference_gate = actual_online_mismatch_count <= bf16_mismatch_limit
        && actual_online_max_abs_error <= 0.015625
        && actual_online_mae < 5.0e-7
        && actual_online_rmse < 7.5e-5;
    const bool pass = direct_reference_gate && online_reference_gate
        && row0_max_abs_error < 1.0e-6 && nan_count == 0
        && inf_count == 0 && online_nan_count == 0
        && online_inf_count == 0 && host_pack_bit_exact;
    const bool u52_pass = pass && cache_pack_pass;
    std::cout << "configuration = " << Q_HEADS << "Q/" << KV_HEADS
              << "KV/" << SEQ << "SEQ/" << HEAD_DIM << "D\n";
    std::cout << "architecture = static_causal_frontier_2lane_shared_kv_block_contiguous_qo_norm2\n";
    std::cout << "host_pack_bit_exact = " << host_pack_bit_exact << "\n";
    std::cout << "tasks_per_kv_head = " << TASKS_PER_KV_HEAD << "\n";
    std::cout << "static_gqa_lanes = " << STATIC_GQA_LANES << "\n";
    std::cout << "qk_parallel = " << QK_PAR_D << "\n";
    std::cout << "pv_parallel = " << PV_PAR_K << "\n";
    std::cout << "row0_max_abs_error = " << row0_max_abs_error << "\n";
    std::cout << "max_abs_error = " << max_abs_error << "\n";
    std::cout << "mae = " << mae << "\n";
    std::cout << "rmse = " << rmse << "\n";
    std::cout << "nan_count = " << nan_count << "\n";
    std::cout << "inf_count = " << inf_count << "\n";
    std::cout << "bf16_mismatch_count = " << bf16_mismatch_count << "\n";
    std::cout << "bf16_mismatch_limit = " << bf16_mismatch_limit << "\n";
    std::cout << "bf16_mismatch_rate = "
              << static_cast<double>(bf16_mismatch_count) / q_count << "\n";
    std::cout << "bf16_max_abs_error = " << bf16_max_abs_error << "\n";
    std::cout << "bf16_mae = " << bf16_mae << "\n";
    std::cout << "bf16_rmse = " << bf16_rmse << "\n";
    std::cout << "actual_online_mismatch_count = "
              << actual_online_mismatch_count << "\n";
    std::cout << "actual_online_mismatch_rate = "
              << static_cast<double>(actual_online_mismatch_count) / q_count << "\n";
    std::cout << "actual_online_max_abs_error = "
              << actual_online_max_abs_error << "\n";
    std::cout << "actual_online_mae = " << actual_online_mae << "\n";
    std::cout << "actual_online_rmse = " << actual_online_rmse << "\n";
    std::cout << "online_direct_mismatch_count = "
              << online_direct_mismatch_count << "\n";
    std::cout << "actual_direct_online_overlap_count = "
              << actual_direct_online_overlap_count << "\n";
    std::cout << "actual_direct_only_count = " << actual_direct_only_count << "\n";
    std::cout << "online_nan_count = " << online_nan_count << "\n";
    std::cout << "online_inf_count = " << online_inf_count << "\n";
    std::cout << "direct_reference_gate = " << (direct_reference_gate ? 1 : 0) << "\n";
    std::cout << "online_reference_gate = " << (online_reference_gate ? 1 : 0) << "\n";
    std::cout << "HLS_V8S_IP_PASS = " << (u52_pass ? 1 : 0) << "\n";
    std::cout << "HLS_V8S_FINAL_PRECISION_GATE_PASS = "
              << (u52_pass ? 1 : 0) << "\n";
    return u52_pass ? 0 : 1;
}
