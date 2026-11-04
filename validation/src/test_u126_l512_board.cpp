#include "fpt_attention_session.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kGqaGroup = 4;
constexpr int kSeq = 512;
constexpr int kHeadDim = 64;
constexpr int kTileK = 32;
constexpr int kRowsPerBlock = 2;
constexpr int kValuesPerWord = 32;
constexpr int kWordsPerRow = 2;
constexpr int kBlocksPerKvHead = kSeq / kRowsPerBlock;
constexpr int kContextsPerBlock = kGqaGroup * kRowsPerBlock;
constexpr int kTilesPerKvHead = kSeq / kTileK;

void configure_cache_sync(fpt_attention_session_t *session) {
  const char *value = std::getenv("FPT_BOARD_CACHE_SYNC");
  const auto strategy = value != nullptr && std::string(value) == "range"
                            ? FPT_ATTENTION_CACHE_SYNC_VALID_HEAD_RANGES
                            : FPT_ATTENTION_CACHE_SYNC_FULL_BO;
  if (fpt_attention_set_cache_sync_strategy(session, strategy) !=
      FPT_ATTENTION_OK) {
    throw std::runtime_error("set cache sync strategy failed");
  }
}

template <typename T>
std::vector<T> read_binary(const std::string &path, std::size_t count) {
  std::vector<T> data(count);
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + path);
  file.read(reinterpret_cast<char *>(data.data()), count * sizeof(T));
  if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T)) {
    throw std::runtime_error("short read " + path);
  }
  return data;
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::size_t raw_q_index(int kv_head, int query_block, int context,
                        int word, int item) {
  const int row = context / kGqaGroup;
  const int lane = context % kGqaGroup;
  const int q_head = kv_head * kGqaGroup + lane;
  const int query = query_block * kRowsPerBlock + row;
  const int dim = word * kValuesPerWord + item;
  return (static_cast<std::size_t>(q_head) * kSeq + query) * kHeadDim + dim;
}

std::size_t block_word_index(int kv_head, int query_block, int context,
                             int word) {
  return ((static_cast<std::size_t>(kv_head) * kBlocksPerKvHead +
           query_block) *
              kContextsPerBlock +
          context) *
             kWordsPerRow +
         word;
}

void pack_q_block_major(const std::vector<uint16_t> &raw, uint16_t *packed) {
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (int query_block = 0; query_block < kBlocksPerKvHead; ++query_block) {
      for (int context = 0; context < kContextsPerBlock; ++context) {
        for (int word = 0; word < kWordsPerRow; ++word) {
          const std::size_t base =
              block_word_index(kv_head, query_block, context, word) *
              kValuesPerWord;
          for (int item = 0; item < kValuesPerWord; ++item) {
            packed[base + item] =
                raw[raw_q_index(kv_head, query_block, context, word, item)];
          }
        }
      }
    }
  }
}

std::vector<uint16_t> unpack_q_block_major(const uint16_t *packed) {
  std::vector<uint16_t> raw(static_cast<std::size_t>(kQHeads) * kSeq *
                            kHeadDim);
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (int query_block = 0; query_block < kBlocksPerKvHead; ++query_block) {
      for (int context = 0; context < kContextsPerBlock; ++context) {
        for (int word = 0; word < kWordsPerRow; ++word) {
          const std::size_t base =
              block_word_index(kv_head, query_block, context, word) *
              kValuesPerWord;
          for (int item = 0; item < kValuesPerWord; ++item) {
            raw[raw_q_index(kv_head, query_block, context, word, item)] =
                packed[base + item];
          }
        }
      }
    }
  }
  return raw;
}

void pack_v_dimension_major(const std::vector<uint16_t> &raw,
                            uint16_t *packed) {
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (int tile = 0; tile < kTilesPerKvHead; ++tile) {
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t packed_base =
            ((static_cast<std::size_t>(kv_head) * kTilesPerKvHead + tile) *
                 kHeadDim +
             dim) *
            kValuesPerWord;
        for (int key = 0; key < kTileK; ++key) {
          const std::size_t raw_index =
              (static_cast<std::size_t>(kv_head) * kSeq +
               tile * kTileK + key) *
                  kHeadDim +
              dim;
          packed[packed_base + key] = raw[raw_index];
        }
      }
    }
  }
}

void require_status(fpt_attention_status_t status,
                    fpt_attention_session_t *session,
                    const char *operation) {
  if (status == FPT_ATTENTION_OK) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           fpt_attention_session_last_error(session));
}

std::size_t q_index(int head, int token, int dim) {
  return (static_cast<std::size_t>(head) * kSeq + token) * kHeadDim + dim;
}

std::size_t kv_index(int head, int token, int dim) {
  return (static_cast<std::size_t>(head) * kSeq + token) * kHeadDim + dim;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--runtime-length")) {
    std::cerr << "usage: " << argv[0]
              << " <xclbin> <vector_dir> [--runtime-length]\n";
    return 2;
  }
  const bool runtime_length = argc == 4;

  const std::size_t q_count =
      static_cast<std::size_t>(kQHeads) * kSeq * kHeadDim;
  const std::size_t kv_count =
      static_cast<std::size_t>(kKvHeads) * kSeq * kHeadDim;
  fpt_attention_session_t *session = nullptr;

  try {
    const std::string vector_dir = argv[2];
    const auto q = read_binary<uint16_t>(vector_dir + "/q.bf16", q_count);
    const auto k = read_binary<uint16_t>(vector_dir + "/k.bf16", kv_count);
    const auto v = read_binary<uint16_t>(vector_dir + "/v.bf16", kv_count);
    const auto golden_bf16 =
        read_binary<uint16_t>(vector_dir + "/out_ref.bf16", q_count);
    const auto golden_fp32 =
        read_binary<float>(vector_dir + "/out_ref_fp32.f32", q_count);

    require_status(fpt_attention_session_create(argv[1], 0U, 512U, 1U,
                                                 &session),
                   session, "create");
    configure_cache_sync(session);
    fpt_attention_prefill_io_t prefill{};
    require_status(fpt_attention_get_prefill_mapped_io(session, 0U, &prefill),
                   session, "get prefill I/O");
    if (prefill.k_prefill_native != prefill.k_cache_native ||
        prefill.v_prefill_native != prefill.v_cache_native) {
      throw std::runtime_error("U57 K/V prefill/cache alias is missing");
    }
    pack_q_block_major(q, prefill.q_native);
    std::copy(k.begin(), k.end(), prefill.k_prefill_native);
    pack_v_dimension_major(v, prefill.v_prefill_native);
    if (runtime_length) {
      std::fill_n(prefill.output_native, q_count, uint16_t{0x7e55u});
      require_status(fpt_attention_sync_prefill_output_to_device(session),
                     session, "prefill output sentinel sync");
    }

    fpt_attention_metrics_t prefill_metrics{};
    require_status(fpt_attention_run_prefill_mapped(session, 0U, 256U,
                                                     &prefill_metrics),
                   session, "run prefill");
    const auto prefill_actual = unpack_q_block_major(prefill.output_native);

    std::size_t prefill_mismatches = 0;
    std::size_t prefill_inactive_writes = 0;
    std::size_t prefill_active_values = 0;
    double prefill_max_bf16_abs = 0.0;
    double prefill_mae = 0.0;
    double prefill_max_fp32_abs = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      const std::size_t position = (i / kHeadDim) % kSeq;
      const bool active = !runtime_length || position < 256U;
      const double actual = bf16_to_float(prefill_actual[i]);
      if (active) {
        ++prefill_active_values;
        const double expected_bf16 = bf16_to_float(golden_bf16[i]);
        const double bf16_error = std::abs(actual - expected_bf16);
        if (prefill_actual[i] != golden_bf16[i]) ++prefill_mismatches;
        prefill_max_bf16_abs = std::max(prefill_max_bf16_abs, bf16_error);
        prefill_mae += bf16_error;
        prefill_max_fp32_abs = std::max(
            prefill_max_fp32_abs,
            std::abs(actual - static_cast<double>(golden_fp32[i])));
      } else if (prefill_actual[i] != 0x7e55u) {
        ++prefill_inactive_writes;
      }
    }
    prefill_mae /= prefill_active_values;
    const std::size_t mismatch_limit = (prefill_active_values + 4095U) / 4096U;
    const bool prefill_pass =
        prefill_mismatches <= mismatch_limit &&
        prefill_inactive_writes == 0 &&
        prefill_max_bf16_abs <= 0.015625 && prefill_mae < 5.0e-7 &&
        prefill_max_fp32_abs < 0.02;

    fpt_attention_metrics_t commit_metrics{};
    require_status(fpt_attention_commit_prompt_cache(session, 0U, 256U,
                                                      &commit_metrics),
                   session, "commit prompt cache");
    fpt_attention_decode_io_t decode{};
    require_status(fpt_attention_get_decode_mapped_io(session, &decode),
                   session, "get decode I/O");

    double decode_kernel_ms = 0.0;
    double decode_total_ms = 0.0;
    double decode_max_abs = 0.0;
    double decode_squared_error = 0.0;
    std::size_t decode_values = 0;
    std::size_t decode_bad_values = 0;
    for (int position = 256; position < kSeq; ++position) {
      for (int head = 0; head < kQHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          decode.q[static_cast<std::size_t>(head) * kHeadDim + dim] =
              q[q_index(head, position, dim)];
        }
      }
      for (int head = 0; head < kKvHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          const std::size_t row_index =
              static_cast<std::size_t>(head) * kHeadDim + dim;
          decode.k_new[row_index] = k[kv_index(head, position, dim)];
          decode.v_new[row_index] = v[kv_index(head, position, dim)];
        }
      }

      fpt_attention_metrics_t metrics{};
      require_status(fpt_attention_run_decode_mapped(
                         session, 0U, static_cast<uint32_t>(position),
                         &metrics),
                     session, "run decode");
      decode_kernel_ms += metrics.kernel_ms;
      decode_total_ms += metrics.total_ms;
      for (int head = 0; head < kQHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          const std::size_t output_index =
              static_cast<std::size_t>(head) * kHeadDim + dim;
          const double actual = bf16_to_float(decode.output[output_index]);
          const double expected =
              bf16_to_float(golden_bf16[q_index(head, position, dim)]);
          const double error = std::abs(actual - expected);
          decode_max_abs = std::max(decode_max_abs, error);
          decode_squared_error += error * error;
          if (error > 0.02) ++decode_bad_values;
          ++decode_values;
        }
      }
    }
    const double decode_rmse =
        std::sqrt(decode_squared_error / static_cast<double>(decode_values));
    const bool decode_pass = decode_max_abs <= 0.02 &&
                             decode_rmse <= 0.002 && decode_bad_values == 0;

    // A second session starts with one committed prompt token so Decode can
    // run sequentially through almost the complete context.  This preserves
    // the public session protocol and exposes tile-boundary behavior that a
    // 256..511 aggregate alone would hide.
    fpt_attention_session_destroy(session);
    session = nullptr;
    require_status(fpt_attention_session_create(argv[1], 0U, 512U, 1U,
                                                 &session),
                   session, "create full sweep session");
    configure_cache_sync(session);
    fpt_attention_prefill_io_t sweep_prefill{};
    require_status(
        fpt_attention_get_prefill_mapped_io(session, 0U, &sweep_prefill),
        session, "get full sweep prefill I/O");
    pack_q_block_major(q, sweep_prefill.q_native);
    std::copy(k.begin(), k.end(), sweep_prefill.k_prefill_native);
    pack_v_dimension_major(v, sweep_prefill.v_prefill_native);
    fpt_attention_metrics_t sweep_prefill_metrics{};
    require_status(fpt_attention_run_prefill_mapped(
                       session, 0U, 1U, &sweep_prefill_metrics),
                   session, "run one-token sweep prefill");
    fpt_attention_metrics_t sweep_commit_metrics{};
    require_status(fpt_attention_commit_prompt_cache(
                       session, 0U, 1U, &sweep_commit_metrics),
                   session, "commit one-token sweep cache");
    fpt_attention_decode_io_t sweep_decode{};
    require_status(fpt_attention_get_decode_mapped_io(session, &sweep_decode),
                   session, "get full sweep decode I/O");

    constexpr int kBucketCount = 5;
    const char *bucket_names[kBucketCount] = {
        "1_31", "32_63", "64_127", "128_255", "256_511"};
    double bucket_kernel_ms[kBucketCount] = {};
    double bucket_total_ms[kBucketCount] = {};
    int bucket_steps[kBucketCount] = {};
    const int sample_positions[] = {
        1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511};
    constexpr int kSampleCount =
        static_cast<int>(sizeof(sample_positions) / sizeof(sample_positions[0]));
    double sample_kernel_ms[kSampleCount] = {};
    double sample_total_ms[kSampleCount] = {};
    double sweep_max_abs = 0.0;
    double sweep_squared_error = 0.0;
    std::size_t sweep_values = 0;
    std::size_t sweep_bad_values = 0;

    for (int position = 1; position < kSeq; ++position) {
      for (int head = 0; head < kQHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          sweep_decode.q[static_cast<std::size_t>(head) * kHeadDim + dim] =
              q[q_index(head, position, dim)];
        }
      }
      for (int head = 0; head < kKvHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          const std::size_t row_index =
              static_cast<std::size_t>(head) * kHeadDim + dim;
          sweep_decode.k_new[row_index] = k[kv_index(head, position, dim)];
          sweep_decode.v_new[row_index] = v[kv_index(head, position, dim)];
        }
      }
      fpt_attention_metrics_t metrics{};
      require_status(fpt_attention_run_decode_mapped(
                         session, 0U, static_cast<uint32_t>(position),
                         &metrics),
                     session, "run full decode sweep");

      const int bucket = position <= 31 ? 0 :
                         position <= 63 ? 1 :
                         position <= 127 ? 2 :
                         position <= 255 ? 3 : 4;
      bucket_kernel_ms[bucket] += metrics.kernel_ms;
      bucket_total_ms[bucket] += metrics.total_ms;
      ++bucket_steps[bucket];
      for (int sample = 0; sample < kSampleCount; ++sample) {
        if (position == sample_positions[sample]) {
          sample_kernel_ms[sample] = metrics.kernel_ms;
          sample_total_ms[sample] = metrics.total_ms;
        }
      }
      for (int head = 0; head < kQHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          const std::size_t output_index =
              static_cast<std::size_t>(head) * kHeadDim + dim;
          const double actual =
              bf16_to_float(sweep_decode.output[output_index]);
          const double expected =
              bf16_to_float(golden_bf16[q_index(head, position, dim)]);
          const double error = std::abs(actual - expected);
          sweep_max_abs = std::max(sweep_max_abs, error);
          sweep_squared_error += error * error;
          if (error > 0.02) ++sweep_bad_values;
          ++sweep_values;
        }
      }
    }
    const double sweep_rmse = std::sqrt(
        sweep_squared_error / static_cast<double>(sweep_values));
    const bool sweep_pass = sweep_max_abs <= 0.02 &&
                            sweep_rmse <= 0.002 && sweep_bad_values == 0;

    fpt_attention_metrics_t full_prefill_metrics{};
    std::size_t full_prefill_mismatches = 0;
    bool full_prefill_pass = true;
    if (runtime_length) {
      fpt_attention_session_destroy(session);
      session = nullptr;
      require_status(fpt_attention_session_create(argv[1], 0U, 512U, 1U,
                                                   &session),
                     session, "create S512 prefill session");
      fpt_attention_prefill_io_t full_prefill{};
      require_status(fpt_attention_get_prefill_mapped_io(session, 0U,
                                                          &full_prefill),
                     session, "get S512 prefill I/O");
      pack_q_block_major(q, full_prefill.q_native);
      std::copy(k.begin(), k.end(), full_prefill.k_prefill_native);
      pack_v_dimension_major(v, full_prefill.v_prefill_native);
      require_status(fpt_attention_run_prefill_mapped(
                         session, 0U, 512U, &full_prefill_metrics),
                     session, "run S512 prefill");
      const auto full_actual = unpack_q_block_major(full_prefill.output_native);
      double full_max_bf16_abs = 0.0;
      double full_mae = 0.0;
      double full_max_fp32_abs = 0.0;
      for (std::size_t i = 0; i < q_count; ++i) {
        const double actual = bf16_to_float(full_actual[i]);
        const double bf16_error =
            std::abs(actual - bf16_to_float(golden_bf16[i]));
        if (full_actual[i] != golden_bf16[i]) ++full_prefill_mismatches;
        full_max_bf16_abs = std::max(full_max_bf16_abs, bf16_error);
        full_mae += bf16_error;
        full_max_fp32_abs = std::max(
            full_max_fp32_abs,
            std::abs(actual - static_cast<double>(golden_fp32[i])));
      }
      full_mae /= q_count;
      full_prefill_pass = full_prefill_mismatches <= (q_count + 4095U) / 4096U &&
                          full_max_bf16_abs <= 0.015625 &&
                          full_mae < 5.0e-7 && full_max_fp32_abs < 0.02;
      std::cout << "U126_BOARD_S512_PREFILL_BF16_MISMATCHES="
                << full_prefill_mismatches << '\n'
                << "U126_BOARD_S512_PREFILL_MAX_BF16_ABS="
                << full_max_bf16_abs << '\n'
                << "U126_BOARD_S512_PREFILL_MAE=" << full_mae << '\n'
                << "U126_BOARD_S512_PREFILL_MAX_FP32_ABS="
                << full_max_fp32_abs << '\n'
                << "U126_BOARD_S512_PREFILL_KERNEL_MS="
                << full_prefill_metrics.kernel_ms << '\n'
                << "U126_BOARD_S512_PREFILL_TOTAL_MS="
                << full_prefill_metrics.total_ms << '\n'
                << "U126_BOARD_S512_PREFILL_CORRECTNESS="
                << (full_prefill_pass ? "PASS" : "FAIL") << '\n';
    }

    std::cout << "U126_BOARD_SHAPE=Q32_KV8_GQA4_D64_S512\n"
              << "U126_BOARD_XCLBIN_LOADS=1\n"
              << "U126_BOARD_PREFILL_BF16_MISMATCHES=" << prefill_mismatches
              << '\n'
              << "U126_BOARD_PREFILL_ACTIVE_VALUES=" << prefill_active_values
              << '\n'
              << "U126_BOARD_PREFILL_INACTIVE_WRITES="
              << prefill_inactive_writes << '\n'
              << "U126_BOARD_PREFILL_BF16_MISMATCH_LIMIT=" << mismatch_limit
              << '\n'
              << "U126_BOARD_PREFILL_MAX_BF16_ABS=" << prefill_max_bf16_abs
              << '\n'
              << "U126_BOARD_PREFILL_MAE=" << prefill_mae << '\n'
              << "U126_BOARD_PREFILL_MAX_FP32_ABS=" << prefill_max_fp32_abs
              << '\n'
              << "U126_BOARD_PREFILL_KERNEL_MS=" << prefill_metrics.kernel_ms
              << '\n'
              << "U126_BOARD_PREFILL_TOTAL_MS=" << prefill_metrics.total_ms
              << '\n'
              << "U126_BOARD_COMMIT_H2D_SYNC_CALLS="
              << commit_metrics.host_to_device_sync_calls << '\n'
              << "U126_BOARD_COMMIT_H2D_BYTES="
              << commit_metrics.host_to_device_bytes << '\n'
              << "U126_BOARD_COMMIT_TOTAL_MS=" << commit_metrics.total_ms
              << '\n'
              << "U126_BOARD_PREFILL_CORRECTNESS="
              << (prefill_pass ? "PASS" : "FAIL") << '\n'
              << "U126_BOARD_DECODE_POSITIONS=256..511\n"
              << "U126_BOARD_DECODE_STEPS=256\n"
              << "U126_BOARD_DECODE_MAX_ABS=" << decode_max_abs << '\n'
              << "U126_BOARD_DECODE_RMSE=" << decode_rmse << '\n'
              << "U126_BOARD_DECODE_BAD_VALUES=" << decode_bad_values << '\n'
              << "U126_BOARD_DECODE_KERNEL_MS_SUM=" << decode_kernel_ms << '\n'
              << "U126_BOARD_DECODE_KERNEL_MS_AVG="
              << decode_kernel_ms / 256.0 << '\n'
              << "U126_BOARD_DECODE_TOTAL_MS_AVG=" << decode_total_ms / 256.0
              << '\n'
              << "U126_BOARD_DECODE_CORRECTNESS="
              << (decode_pass ? "PASS" : "FAIL") << '\n'
              << "U126_BOARD_SWEEP_POSITIONS=1..511\n"
              << "U126_BOARD_POSITION_0_PATH=PREFILL_PROTOCOL_TOKEN\n"
              << "U126_BOARD_SWEEP_MAX_ABS=" << sweep_max_abs << '\n'
              << "U126_BOARD_SWEEP_RMSE=" << sweep_rmse << '\n'
              << "U126_BOARD_SWEEP_BAD_VALUES=" << sweep_bad_values << '\n'
              << "U126_BOARD_SWEEP_CORRECTNESS="
              << (sweep_pass ? "PASS" : "FAIL") << '\n'
              << "U126_BOARD_MODE_SEQUENCE=0_prefill,1_decode\n"
              << "U126_BOARD_TEST="
              << (prefill_pass && decode_pass && sweep_pass && full_prefill_pass
                      ? "PASS" : "FAIL")
              << '\n';

    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
      std::cout << "U126_BOARD_SWEEP_BUCKET_" << bucket_names[bucket]
                << "_STEPS=" << bucket_steps[bucket] << '\n'
                << "U126_BOARD_SWEEP_BUCKET_" << bucket_names[bucket]
                << "_KERNEL_MS_AVG="
                << bucket_kernel_ms[bucket] / bucket_steps[bucket] << '\n'
                << "U126_BOARD_SWEEP_BUCKET_" << bucket_names[bucket]
                << "_TOTAL_MS_AVG="
                << bucket_total_ms[bucket] / bucket_steps[bucket] << '\n';
    }
    for (int sample = 0; sample < kSampleCount; ++sample) {
      std::cout << "U126_BOARD_POSITION_" << sample_positions[sample]
                << "_KERNEL_MS=" << sample_kernel_ms[sample] << '\n'
                << "U126_BOARD_POSITION_" << sample_positions[sample]
                << "_TOTAL_MS=" << sample_total_ms[sample] << '\n';
    }

    fpt_attention_session_destroy(session);
    return prefill_pass && decode_pass && sweep_pass && full_prefill_pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "U126_BOARD_ERROR=" << error.what() << '\n';
    std::cout << "U126_BOARD_TEST=FAIL\n";
    fpt_attention_session_destroy(session);
    return 1;
  }
}
