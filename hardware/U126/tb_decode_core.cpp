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

namespace {

using bf16_t = ap_uint<16>;
using qint14_t = ap_int<DECODE_Q_QUANT_BITS>;
using kvint13_t = ap_int<DECODE_KV_QUANT_BITS>;

bf16_t float_to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return bf16_t(bits >> 16);
}

float bf16_to_float(bf16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value.to_uint()) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void set_bf16_lane(decode_mem_word_t& word, int lane, bf16_t value) {
  word.range((lane + 1) * 16 - 1, lane * 16) = value;
}

bf16_t get_bf16_lane(const decode_mem_word_t& word, int lane) {
  return word.range((lane + 1) * 16 - 1, lane * 16);
}

int head_word_index(int head, int word) {
  return head * DECODE_WORDS_PER_ROW + word;
}

int cache_data_word_index(int head, int token, int word) {
  return (head * DECODE_MAX_KV_TOKENS + token) *
      DECODE_CACHE_WORDS_PER_ROW + word;
}

float next_random(std::uint32_t& state, float amplitude) {
  state = state * 1664525U + 1013904223U;
  const int centered = static_cast<int>((state >> 16) & 0xFFFFU) - 32768;
  return static_cast<float>(centered) * (amplitude / 32768.0f);
}

float random_bf16(std::uint32_t& state, float amplitude) {
  return bf16_to_float(float_to_bf16(next_random(state, amplitude)));
}

qint14_t quantize_q14(float value, float scale) {
  int integer = static_cast<int>(std::round(value / scale));
  integer = std::max(-DECODE_Q_QUANT_LIMIT,
                     std::min(DECODE_Q_QUANT_LIMIT, integer));
  return qint14_t(integer);
}

kvint13_t quantize_kv13(float value, float scale) {
  int integer = static_cast<int>(std::round(value / scale));
  integer = std::max(-DECODE_KV_QUANT_LIMIT,
                     std::min(DECODE_KV_QUANT_LIMIT, integer));
  return kvint13_t(integer);
}

void compress_row(const float source[DECODE_HEAD_DIM],
                  decode_mem_word_t data[DECODE_CACHE_WORDS_PER_ROW],
                  bf16_t& scale_bits,
                  float dequantized[DECODE_HEAD_DIM]) {
  data[0] = 0;
  data[1] = 0;
  float peak = 0.0f;
  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
    peak = std::max(peak, std::abs(source[dim]));
  }
  scale_bits = float_to_bf16(
      peak > 0.0f ? peak / DECODE_KV_QUANT_LIMIT : 1.0f);
  const float scale = bf16_to_float(scale_bits);
  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
    const kvint13_t integer = quantize_kv13(source[dim], scale);
    const int word = dim / DECODE_QUANT_VALUES_PER_CACHE_WORD;
    const int lane = dim % DECODE_QUANT_VALUES_PER_CACHE_WORD;
    data[word].range((lane + 1) * DECODE_KV_QUANT_BITS - 1,
                     lane * DECODE_KV_QUANT_BITS) = integer;
    dequantized[dim] = static_cast<float>(integer.to_int()) * scale;
  }
  data[0].range(DECODE_SCALE_LOW_BIT + 15, DECODE_SCALE_LOW_BIT) =
      scale_bits;
}

void compress_q_row(const float source[DECODE_HEAD_DIM],
                    float dequantized[DECODE_HEAD_DIM]) {
  for (int group = 0; group < DECODE_GROUPS_PER_ROW; ++group) {
    float peak = 0.0f;
    for (int lane = 0; lane < DECODE_GROUP_D; ++lane) {
      peak = std::max(peak, std::abs(source[group * DECODE_GROUP_D + lane]));
    }
    const float scale = peak > 0.0f ? peak / DECODE_Q_QUANT_LIMIT : 1.0f;
    for (int lane = 0; lane < DECODE_GROUP_D; ++lane) {
      const int dim = group * DECODE_GROUP_D + lane;
      dequantized[dim] = static_cast<float>(quantize_q14(source[dim], scale).to_int()) * scale;
    }
  }
}

void write_bf16_row(decode_mem_word_t* words, int head,
                    const float values[DECODE_HEAD_DIM]) {
  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
    set_bf16_lane(words[head_word_index(head, dim / DECODE_VALUES_PER_WORD)],
                  dim % DECODE_VALUES_PER_WORD, float_to_bf16(values[dim]));
  }
}

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t count) {
  std::vector<T> data(count);
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + path);
  file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
  if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T)) {
    throw std::runtime_error("short read " + path);
  }
  return data;
}

void write_tensor_row(decode_mem_word_t* words,
                      const std::vector<std::uint16_t>& tensor,
                      int heads, int head, unsigned position) {
  const std::size_t expected = static_cast<std::size_t>(heads) *
      DECODE_MAX_KV_TOKENS * DECODE_HEAD_DIM;
  if (tensor.size() != expected) {
    throw std::runtime_error("unexpected tensor size");
  }
  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
    const std::size_t index =
        (static_cast<std::size_t>(head) * DECODE_MAX_KV_TOKENS + position) *
        DECODE_HEAD_DIM + dim;
    set_bf16_lane(words[head_word_index(
                      head, dim / DECODE_VALUES_PER_WORD)],
                  dim % DECODE_VALUES_PER_WORD, bf16_t(tensor[index]));
  }
}

struct ErrorMetrics {
  double max_abs = 0.0;
  double squared_error = 0.0;
  std::size_t count = 0;
  std::size_t bad = 0;
  std::size_t nan_count = 0;
  std::size_t inf_count = 0;

  void add(float actual, float reference) {
    if (std::isnan(actual)) ++nan_count;
    if (std::isinf(actual)) ++inf_count;
    const double error = std::abs(
        static_cast<double>(actual) - static_cast<double>(reference));
    max_abs = std::max(max_abs, error);
    squared_error += error * error;
    if (!std::isfinite(actual) || error > 0.02) ++bad;
    ++count;
  }

  double rmse() const {
    return count == 0 ? 0.0 : std::sqrt(squared_error / count);
  }

  bool pass() const {
    return max_abs <= 0.02 && rmse() <= 0.002 && bad == 0 &&
        nan_count == 0 && inf_count == 0;
  }
};

bool run_frozen_s512_gate(const std::string& data_dir) {
  const std::size_t q_count = static_cast<std::size_t>(DECODE_Q_HEADS) *
      DECODE_MAX_KV_TOKENS * DECODE_HEAD_DIM;
  const std::size_t kv_count = static_cast<std::size_t>(DECODE_KV_HEADS) *
      DECODE_MAX_KV_TOKENS * DECODE_HEAD_DIM;
  const std::vector<std::uint16_t> q_raw =
      read_binary<std::uint16_t>(data_dir + "/q.bf16", q_count);
  const std::vector<std::uint16_t> k_raw =
      read_binary<std::uint16_t>(data_dir + "/k.bf16", kv_count);
  const std::vector<std::uint16_t> v_raw =
      read_binary<std::uint16_t>(data_dir + "/v.bf16", kv_count);
  const std::vector<std::uint16_t> reference_raw =
      read_binary<std::uint16_t>(data_dir + "/out_ref.bf16", q_count);

  static decode_mem_word_t k_cache[ATTENTION_IP_STORAGE_KV_WORDS] = {};
  static decode_mem_word_t v_cache[ATTENTION_IP_STORAGE_KV_WORDS] = {};
  decode_mem_word_t gmem0[DECODE_GMEM0_WORDS] = {};
  decode_mem_word_t output[DECODE_Q_WORDS] = {};
  ErrorMetrics sweep;
  ErrorMetrics decode_half;
  bool append_pass = true;

  for (unsigned position = 0; position < DECODE_MAX_KV_TOKENS; ++position) {
    for (int head = 0; head < DECODE_Q_HEADS; ++head) {
      write_tensor_row(gmem0 + DECODE_GMEM0_Q_OFFSET,
                       q_raw, DECODE_Q_HEADS, head, position);
    }
    for (int head = 0; head < DECODE_KV_HEADS; ++head) {
      write_tensor_row(gmem0 + DECODE_GMEM0_K_NEW_OFFSET,
                       k_raw, DECODE_KV_HEADS, head, position);
      write_tensor_row(gmem0 + DECODE_GMEM0_V_NEW_OFFSET,
                       v_raw, DECODE_KV_HEADS, head, position);
    }

    int8_gqa_prefill_decode_ip(FPT_MODE_DECODE, gmem0, k_cache, v_cache,
                               output, position);

    for (int head = 0; head < DECODE_KV_HEADS; ++head) {
      float k_source[DECODE_HEAD_DIM];
      float v_source[DECODE_HEAD_DIM];
      float unused_k_deq[DECODE_HEAD_DIM];
      float unused_v_deq[DECODE_HEAD_DIM];
      for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
        const std::size_t index =
            (static_cast<std::size_t>(head) * DECODE_MAX_KV_TOKENS +
             position) * DECODE_HEAD_DIM + dim;
        k_source[dim] = bf16_to_float(bf16_t(k_raw[index]));
        v_source[dim] = bf16_to_float(bf16_t(v_raw[index]));
      }
      decode_mem_word_t expected_k[DECODE_CACHE_WORDS_PER_ROW];
      decode_mem_word_t expected_v[DECODE_CACHE_WORDS_PER_ROW];
      bf16_t unused_k_scale;
      bf16_t unused_v_scale;
      compress_row(k_source, expected_k, unused_k_scale, unused_k_deq);
      compress_row(v_source, expected_v, unused_v_scale, unused_v_deq);
      for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
        append_pass = append_pass &&
            k_cache[cache_data_word_index(head, position, word)] ==
                expected_k[word];
        append_pass = append_pass &&
            v_cache[cache_data_word_index(head, position, word)] ==
                expected_v[word];
      }
    }

    if (position == 0) continue;
    for (int head = 0; head < DECODE_Q_HEADS; ++head) {
      for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
        const float actual = bf16_to_float(get_bf16_lane(
            output[head_word_index(head, dim / DECODE_VALUES_PER_WORD)],
            dim % DECODE_VALUES_PER_WORD));
        const std::size_t index =
            (static_cast<std::size_t>(head) * DECODE_MAX_KV_TOKENS +
             position) * DECODE_HEAD_DIM + dim;
        const float reference = bf16_to_float(bf16_t(reference_raw[index]));
        sweep.add(actual, reference);
        if (position >= 256) decode_half.add(actual, reference);
      }
    }
  }

  std::cout << "frozen_s512_sweep_positions=1..511"
            << " max_abs_error=" << sweep.max_abs
            << " rmse=" << sweep.rmse()
            << " bad_values_gt_0p02=" << sweep.bad
            << " status=" << (sweep.pass() ? "PASS" : "FAIL") << '\n';
  std::cout << "frozen_s512_decode_positions=256..511"
            << " max_abs_error=" << decode_half.max_abs
            << " rmse=" << decode_half.rmse()
            << " bad_values_gt_0p02=" << decode_half.bad
            << " status=" << (decode_half.pass() ? "PASS" : "FAIL") << '\n';
  std::cout << "frozen_s512_kv_append="
            << (append_pass ? "PASS" : "FAIL") << '\n';
  return sweep.pass() && decode_half.pass() && append_pass;
}

bool run_case(unsigned position, std::uint32_t seed, float amplitude = 0.15f) {
  static decode_mem_word_t k_cache[ATTENTION_IP_STORAGE_KV_WORDS];
  static decode_mem_word_t v_cache[ATTENTION_IP_STORAGE_KV_WORDS];
  static float k_deq[DECODE_KV_HEADS][DECODE_MAX_KV_TOKENS][DECODE_HEAD_DIM];
  static float v_deq[DECODE_KV_HEADS][DECODE_MAX_KV_TOKENS][DECODE_HEAD_DIM];
  decode_mem_word_t q[DECODE_Q_WORDS] = {};
  decode_mem_word_t k_new[DECODE_KV_WORDS] = {};
  decode_mem_word_t v_new[DECODE_KV_WORDS] = {};
  decode_mem_word_t output[DECODE_Q_WORDS] = {};
  decode_mem_word_t gmem0[DECODE_GMEM0_WORDS] = {};
  float q_deq[DECODE_Q_HEADS][DECODE_HEAD_DIM];
  bf16_t expected[DECODE_Q_HEADS][DECODE_HEAD_DIM];

  for (int index = 0; index < ATTENTION_IP_STORAGE_KV_WORDS; ++index) {
    k_cache[index] = ~decode_mem_word_t(0);
    v_cache[index] = ~decode_mem_word_t(0);
  }

  for (int head = 0; head < DECODE_Q_HEADS; ++head) {
    float source[DECODE_HEAD_DIM];
    for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) source[dim] = random_bf16(seed, amplitude);
    write_bf16_row(q, head, source);
    compress_q_row(source, q_deq[head]);
  }

  decode_mem_word_t expected_k_data[DECODE_KV_HEADS]
                                   [DECODE_CACHE_WORDS_PER_ROW];
  decode_mem_word_t expected_v_data[DECODE_KV_HEADS]
                                   [DECODE_CACHE_WORDS_PER_ROW];
  bf16_t expected_k_scale[DECODE_KV_HEADS];
  bf16_t expected_v_scale[DECODE_KV_HEADS];
  for (int head = 0; head < DECODE_KV_HEADS; ++head) {
    for (unsigned token = 0; token <= position; ++token) {
      float k_source[DECODE_HEAD_DIM];
      float v_source[DECODE_HEAD_DIM];
      for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
        k_source[dim] = random_bf16(seed, amplitude);
        v_source[dim] = random_bf16(seed, amplitude);
      }
      decode_mem_word_t k_data[DECODE_CACHE_WORDS_PER_ROW];
      decode_mem_word_t v_data[DECODE_CACHE_WORDS_PER_ROW];
      bf16_t k_scale;
      bf16_t v_scale;
      compress_row(k_source, k_data, k_scale, k_deq[head][token]);
      compress_row(v_source, v_data, v_scale, v_deq[head][token]);
      if (token < position) {
        for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
          k_cache[cache_data_word_index(head, token, word)] = k_data[word];
          v_cache[cache_data_word_index(head, token, word)] = v_data[word];
        }
      } else {
        write_bf16_row(k_new, head, k_source);
        write_bf16_row(v_new, head, v_source);
        for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
          expected_k_data[head][word] = k_data[word];
          expected_v_data[head][word] = v_data[word];
        }
        expected_k_scale[head] = k_scale;
        expected_v_scale[head] = v_scale;
      }
    }
  }

  for (int q_head = 0; q_head < DECODE_Q_HEADS; ++q_head) {
    const int kv_head = q_head / DECODE_GQA_GROUP;
    float scores[DECODE_MAX_KV_TOKENS];
    float maximum = -INFINITY;
    for (unsigned token = 0; token <= position; ++token) {
      float score = 0.0f;
      for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
        score += q_deq[q_head][dim] * k_deq[kv_head][token][dim];
      }
      scores[token] = score * 0.125f;
      maximum = std::max(maximum, scores[token]);
    }
    float sum = 0.0f;
    for (unsigned token = 0; token <= position; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      sum += scores[token];
    }
    for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
      float value = 0.0f;
      for (unsigned token = 0; token <= position; ++token) {
        value += scores[token] * v_deq[kv_head][token][dim];
      }
      expected[q_head][dim] = float_to_bf16(value / sum);
    }
  }

  for (int index = 0; index < DECODE_Q_WORDS; ++index) gmem0[DECODE_GMEM0_Q_OFFSET + index] = q[index];
  for (int index = 0; index < DECODE_KV_WORDS; ++index) {
    gmem0[DECODE_GMEM0_K_NEW_OFFSET + index] = k_new[index];
    gmem0[DECODE_GMEM0_V_NEW_OFFSET + index] = v_new[index];
  }
  int8_gqa_prefill_decode_ip(FPT_MODE_DECODE, gmem0, k_cache, v_cache, output, position);

  float max_abs = 0.0f;
  double squared_error = 0.0;
  int nan_count = 0;
  int inf_count = 0;
  for (int head = 0; head < DECODE_Q_HEADS; ++head) {
    for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
      const float actual = bf16_to_float(get_bf16_lane(
          output[head_word_index(head, dim / DECODE_VALUES_PER_WORD)], dim % DECODE_VALUES_PER_WORD));
      const float reference = bf16_to_float(expected[head][dim]);
      if (std::isnan(actual)) ++nan_count;
      if (std::isinf(actual)) ++inf_count;
      const float error = std::abs(actual - reference);
      max_abs = std::max(max_abs, error);
      squared_error += static_cast<double>(error) * error;
    }
  }

  bool append_pass = true;
  for (int head = 0; head < DECODE_KV_HEADS; ++head) {
    for (int word = 0; word < DECODE_CACHE_WORDS_PER_ROW; ++word) {
      append_pass = append_pass &&
          k_cache[cache_data_word_index(head, position, word)] ==
              expected_k_data[head][word];
      append_pass = append_pass &&
          v_cache[cache_data_word_index(head, position, word)] ==
              expected_v_data[head][word];
    }
    const decode_mem_word_t k_word =
        k_cache[cache_data_word_index(head, position, 0)];
    const decode_mem_word_t v_word =
        v_cache[cache_data_word_index(head, position, 0)];
    append_pass = append_pass &&
        bf16_t(k_word.range(DECODE_SCALE_LOW_BIT + 15,
                            DECODE_SCALE_LOW_BIT)) == expected_k_scale[head];
    append_pass = append_pass &&
        bf16_t(v_word.range(DECODE_SCALE_LOW_BIT + 15,
                            DECODE_SCALE_LOW_BIT)) == expected_v_scale[head];
  }

  const float rmse = static_cast<float>(std::sqrt(squared_error / (DECODE_Q_HEADS * DECODE_HEAD_DIM)));
  const bool pass = max_abs <= 0.025f && rmse <= 0.003f && nan_count == 0 && inf_count == 0 && append_pass;
  std::cout << "position=" << position << " amplitude=" << amplitude
            << " max_abs_error=" << max_abs << " rmse=" << rmse
            << " nan_count=" << nan_count << " inf_count=" << inf_count
            << " kv_append=" << (append_pass ? "PASS" : "FAIL")
            << " status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: tb_decode_core DATA_DIR\n";
    return 2;
  }
  // Include both the legacy reporting points and both sides of every
  // tile64-scale boundary.  Position is zero-based, so position 63 is the
  // last token of the first full tile and position 64 starts a partial tile.
  const unsigned positions[] = {0, 63, 64, 255, 256, 511};
  bool pass = true;
  for (unsigned index = 0; index < sizeof(positions) / sizeof(positions[0]); ++index) {
    pass = run_case(positions[index], 20260911U + positions[index]) && pass;
  }
  pass = run_case(511, 0x48A55EEDU) && pass;
  pass = run_case(511, 0x48E37E55U, 1.5f) && pass;
  const bool frozen_s512_pass = run_frozen_s512_gate(argv[1]);
  pass = frozen_s512_pass && pass;
  std::cout << "FPT_DECODE_Q14_KV13_LAYOUT=Q14_KV13_PER_ROW_D64\n"
            << "FPT_DECODE_K_CACHE_WORDS=" << DECODE_K_CACHE_WORDS << '\n'
            << "FPT_DECODE_V_CACHE_WORDS=" << DECODE_V_CACHE_WORDS << '\n'
            << "FPT_DECODE_VALID_BOUNDARY_GATE=" << (pass ? "PASS" : "FAIL") << '\n'
            << "FPT_DECODE_KV_APPEND_GATE=" << (pass ? "PASS" : "FAIL") << '\n'
            << "FPT_DECODE_INVALID_TAIL_GATE=" << (pass ? "PASS" : "FAIL") << '\n'
            << "FPT_DECODE_FROZEN_S512_GATE="
            << (frozen_s512_pass ? "PASS" : "FAIL") << '\n'
            << "FPT_DECODE_CSIM=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}
