#include "fpt_llama_pl_hook.h"

#include "fpt_attention_session.h"
#include "fpt_prefill_internal.hpp"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(__aarch64__) && __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define FPT_LLAMA_NEON_AVAILABLE 1
#else
#define FPT_LLAMA_NEON_AVAILABLE 0
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point &begin,
                  const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

constexpr uint32_t kLayers = 16U;
constexpr uint32_t kQHeads = FPT_ATTENTION_Q_HEADS;
constexpr uint32_t kKvHeads = FPT_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = FPT_ATTENTION_HEAD_DIM;
constexpr uint32_t kMaxKv = FPT_ATTENTION_MAX_KV_TOKENS;

struct MetricsAccumulator {
  double pack_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double unpack_ms = 0.0;
  double total_ms = 0.0;
  uint64_t h2d_calls = 0;
  uint64_t d2h_calls = 0;
  uint64_t h2d_bytes = 0;
  uint64_t d2h_bytes = 0;
};

struct HookState {
  bool enabled = false;
  bool strict = true;
  bool shadow = false;
  bool initialized = false;
  bool seen_prefill = false;
  bool seen_decode = false;
  bool report_registered = false;
  bool app_profile_verified = false;
  bool direct_unpack = true;
  double session_create_ms = 0.0;
  uint32_t bucket = 0;
  uint32_t prompt_tokens = 0;
  uint32_t last_decode_position = 0;
  uint32_t current_layer = 0;
  uint64_t calls = 0;
  uint64_t prefill_calls = 0;
  uint64_t decode_calls = 0;
  uint64_t fallback_calls = 0;
  uint64_t failure_calls = 0;
  uint64_t shadow_compares = 0;
  uint64_t shadow_nonfinite = 0;
  double shadow_max_abs = 0.0;
  double shadow_sq_sum = 0.0;
  uint64_t shadow_elements = 0;
  MetricsAccumulator prefill;
  MetricsAccumulator commit;
  MetricsAccumulator decode;
  uint64_t prefill_k_cache_alias_calls = 0;
  uint64_t prefill_k_cache_valid_copy_bytes = 0;
  uint64_t prefill_v_cache_valid_copy_bytes = 0;
  std::string profile;
  std::string xclbin;
  std::string report_path;
  fpt_attention_session_t *session = nullptr;
  // Reused across layers to avoid allocator churn on the PS side.
  std::vector<uint16_t> q_compact;
  std::vector<uint16_t> k_compact;
  std::vector<uint16_t> v_compact;
  std::vector<uint16_t> out_compact;
  std::vector<uint16_t> decode_q;
  std::vector<uint16_t> decode_k;
  std::vector<uint16_t> decode_v;
};

HookState &state() {
  static HookState s;
  return s;
}

std::once_flag g_init_once;
std::mutex g_call_mutex;

bool env_enabled(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0;
}

std::string env_string(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

void write_report();

void initialize() {
  HookState &s = state();
  s.enabled = env_enabled("FPT_LLAMA_PL_ENABLE", false);
  s.strict = env_enabled("FPT_LLAMA_PL_STRICT", true);
  s.shadow = env_enabled("FPT_LLAMA_PL_SHADOW", false);
  s.direct_unpack = env_enabled("FPT_LLAMA_PL_DIRECT_UNPACK", true);
  s.report_path = env_string("FPT_LLAMA_PL_REPORT");
  if (s.enabled && !s.report_registered) {
    std::atexit(write_report);
    s.report_registered = true;
  }
}

bool enabled() {
  std::call_once(g_init_once, initialize);
  return state().enabled;
}

void report_error(const std::string &message) {
  HookState &s = state();
  ++s.failure_calls;
  std::cerr << "FPT_LLAMA_PL_ERROR=" << message << '\n';
  if (s.strict) {
    std::abort();
  }
}

bool is_supported_shape(const ggml_tensor *dst) {
  if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr ||
      dst->src[2] == nullptr) {
    return false;
  }
  const ggml_tensor *q = dst->src[0];
  const ggml_tensor *k = dst->src[1];
  const ggml_tensor *v = dst->src[2];
  return q->ne[0] == kHeadDim && k->ne[0] == kHeadDim &&
         v->ne[0] == kHeadDim && q->ne[2] == kQHeads &&
         k->ne[2] == kKvHeads && v->ne[2] == kKvHeads &&
         q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
         dst->ne[0] == kHeadDim && dst->ne[1] == kQHeads &&
         dst->ne[2] == q->ne[1] && dst->ne[3] == 1 &&
         q->ne[1] > 0 && k->ne[1] > 0 && k->ne[1] == v->ne[1] &&
         q->type != GGML_TYPE_COUNT &&
         k->type != GGML_TYPE_COUNT && v->type != GGML_TYPE_COUNT;
}

float load_mask_value(const ggml_tensor *mask, int64_t kv, int64_t query) {
  const char *base = static_cast<const char *>(mask->data) +
                     kv * mask->nb[0] + query * mask->nb[1];
  switch (mask->type) {
  case GGML_TYPE_F32: return *reinterpret_cast<const float *>(base);
  case GGML_TYPE_F16: return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
  case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(base));
  default: return -std::numeric_limits<float>::infinity();
  }
}

uint32_t valid_kv_tokens(const ggml_tensor *mask, uint32_t query,
                         uint32_t kv_extent) {
  if (mask == nullptr) return kv_extent;
  if (mask->data == nullptr || query >= static_cast<uint32_t>(mask->ne[1]) ||
      mask->ne[0] < kv_extent) return 0U;
  for (uint32_t kv = kv_extent; kv > 0U; --kv) {
    if (std::isfinite(load_mask_value(mask, kv - 1U, query))) return kv;
  }
  return 0U;
}
float load_value(const ggml_tensor *tensor, int64_t dim, int64_t token,
                 int64_t head) {
  const char *base = static_cast<const char *>(tensor->data) +
                     dim * tensor->nb[0] + token * tensor->nb[1] +
                     head * tensor->nb[2];
  switch (tensor->type) {
  case GGML_TYPE_F32:
    return *reinterpret_cast<const float *>(base);
  case GGML_TYPE_F16:
    return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
  case GGML_TYPE_BF16:
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(base));
  default:
    return std::numeric_limits<float>::quiet_NaN();
  }
}

uint16_t to_bf16(float value) {
  return ggml_fp32_to_bf16(value).bits;
}

void read_compact(const ggml_tensor *tensor, uint32_t tokens,
                  uint32_t heads, std::vector<uint16_t> *out) {
  out->resize(static_cast<size_t>(tokens) * heads * kHeadDim);

  // Preserve directly exposed BF16 bits and avoid a BF16->F32->BF16 round
  // trip on the PS. The scalar path remains for F32/F16 and strided views.
  if (tensor->type == GGML_TYPE_BF16 &&
      tensor->nb[0] == sizeof(ggml_bf16_t)) {
    for (uint32_t token = 0; token < tokens; ++token) {
      for (uint32_t head = 0; head < heads; ++head) {
        const char *base = static_cast<const char *>(tensor->data) +
                           token * tensor->nb[1] + head * tensor->nb[2];
        const size_t index =
            (static_cast<size_t>(token) * heads + head) * kHeadDim;
        std::memcpy(out->data() + index, base,
                    static_cast<size_t>(kHeadDim) * sizeof(ggml_bf16_t));
      }
    }
    return;
  }
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        (*out)[(static_cast<size_t>(token) * heads + head) * kHeadDim + dim] =
            to_bf16(load_value(tensor, dim, token, head));
      }
    }
  }
}

void read_decode_q(const ggml_tensor *tensor, std::vector<uint16_t> *out) {
  out->resize(static_cast<size_t>(kQHeads) * kHeadDim);
  if (tensor->type == GGML_TYPE_BF16 &&
      tensor->nb[0] == sizeof(ggml_bf16_t)) {
    for (uint32_t head = 0; head < kQHeads; ++head) {
      const char *base = static_cast<const char *>(tensor->data) +
                         head * tensor->nb[2];
      std::memcpy(out->data() + static_cast<size_t>(head) * kHeadDim, base,
                  static_cast<size_t>(kHeadDim) * sizeof(ggml_bf16_t));
    }
    return;
  }
  for (uint32_t head = 0; head < kQHeads; ++head) {
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      (*out)[static_cast<size_t>(head) * kHeadDim + dim] =
          to_bf16(load_value(tensor, dim, 0, head));
    }
  }
}

void read_decode_kv(const ggml_tensor *tensor, uint32_t position,
                    std::vector<uint16_t> *out) {
  out->resize(static_cast<size_t>(kKvHeads) * kHeadDim);
  if (tensor->type == GGML_TYPE_BF16 &&
      tensor->nb[0] == sizeof(ggml_bf16_t)) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const char *base = static_cast<const char *>(tensor->data) +
                         position * tensor->nb[1] + head * tensor->nb[2];
      std::memcpy(out->data() + static_cast<size_t>(head) * kHeadDim, base,
                  static_cast<size_t>(kHeadDim) * sizeof(ggml_bf16_t));
    }
    return;
  }
  for (uint32_t head = 0; head < kKvHeads; ++head) {
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      (*out)[static_cast<size_t>(head) * kHeadDim + dim] =
          to_bf16(load_value(tensor, dim, position, head));
    }
  }
}

void write_output(const ggml_tensor *src, ggml_tensor *dst, uint32_t tokens,
                  uint32_t heads, const std::vector<uint16_t> &bf16) {
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const size_t index =
            (static_cast<size_t>(token) * heads + head) * kHeadDim + dim;
        const ggml_bf16_t word{bf16[index]};
        const float value = ggml_bf16_to_fp32(word);
        char *base = static_cast<char *>(dst->data) + dim * dst->nb[0] +
                     head * dst->nb[1] + token * dst->nb[2];
        switch (dst->type) {
        case GGML_TYPE_F32:
          *reinterpret_cast<float *>(base) = value;
          break;
        case GGML_TYPE_F16:
          *reinterpret_cast<ggml_fp16_t *>(base) = ggml_fp32_to_fp16(value);
          break;
        case GGML_TYPE_BF16:
          *reinterpret_cast<ggml_bf16_t *>(base) = ggml_fp32_to_bf16(value);
          break;
        default:
          report_error("unsupported FLASH_ATTN_EXT output type");
          return;
        }
      }
    }
  }
  (void)src;
}

bool write_f32_bf16_row_if_contiguous(const ggml_tensor *dst,
                                      uint32_t token, uint32_t head,
                                      const uint16_t *bf16,
                                      uint32_t elements);
bool write_bf16_row_if_contiguous(const ggml_tensor *dst, uint32_t token,
                                  uint32_t head, const uint16_t *bf16,
                                  uint32_t elements) {
  if (dst->type != GGML_TYPE_BF16 ||
      dst->nb[0] != sizeof(ggml_bf16_t) || bf16 == nullptr ||
      elements == 0U) {
    return false;
  }
  char *base = static_cast<char *>(dst->data) + head * dst->nb[1] +
               token * dst->nb[2];
  std::memcpy(base, bf16,
              static_cast<size_t>(elements) * sizeof(ggml_bf16_t));
  return true;
}
void write_decode_output(ggml_tensor *dst, const uint16_t *bf16,
                         size_t bf16_elements) {
  const size_t required = static_cast<size_t>(kQHeads) * kHeadDim;
  if (bf16 == nullptr || bf16_elements < required) {
    report_error("decode output view is smaller than Q32/D64");
    return;
  }
  for (uint32_t head = 0; head < kQHeads; ++head) {
    const uint16_t *row = bf16 + static_cast<size_t>(head) * kHeadDim;
    if (write_bf16_row_if_contiguous(dst, 0U, head, row, kHeadDim) ||
        write_f32_bf16_row_if_contiguous(dst, 0U, head, row, kHeadDim)) {
      continue;
    }
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      const ggml_bf16_t word{
          bf16[static_cast<size_t>(head) * kHeadDim + dim]};
      const float value = ggml_bf16_to_fp32(word);
      char *base = static_cast<char *>(dst->data) + dim * dst->nb[0] +
                   head * dst->nb[1];
      switch (dst->type) {
      case GGML_TYPE_F32:
        *reinterpret_cast<float *>(base) = value;
        break;
      case GGML_TYPE_F16:
        *reinterpret_cast<ggml_fp16_t *>(base) = ggml_fp32_to_fp16(value);
        break;
      case GGML_TYPE_BF16:
        *reinterpret_cast<ggml_bf16_t *>(base) = ggml_fp32_to_bf16(value);
        break;
      default:
        report_error("unsupported FLASH_ATTN_EXT decode output type");
        return;
      }
    }
  }
}

bool write_f32_bf16_row_if_contiguous(const ggml_tensor *dst,
                                      uint32_t token, uint32_t head,
                                      const uint16_t *bf16,
                                      uint32_t elements) {
  if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) ||
      bf16 == nullptr || elements == 0U) {
    return false;
  }
  char *base = static_cast<char *>(dst->data) + head * dst->nb[1] +
               token * dst->nb[2];
  float *output = reinterpret_cast<float *>(base);
#if FPT_LLAMA_NEON_AVAILABLE
  uint32_t dim = 0U;
  for (; dim + 8U <= elements; dim += 8U) {
    const uint32x4_t lo =
        vshll_n_u16(vld1_u16(bf16 + dim), 16);
    const uint32x4_t hi =
        vshll_n_u16(vld1_u16(bf16 + dim + 4U), 16);
    vst1q_f32(output + dim, vreinterpretq_f32_u32(lo));
    vst1q_f32(output + dim + 4U, vreinterpretq_f32_u32(hi));
  }
  for (; dim < elements; ++dim) {
    output[dim] = ggml_bf16_to_fp32(ggml_bf16_t{bf16[dim]});
  }
#else
  for (uint32_t dim = 0U; dim < elements; ++dim) {
    output[dim] = ggml_bf16_to_fp32(ggml_bf16_t{bf16[dim]});
  }
#endif
  return true;
}

std::size_t prefill_output_index(std::size_t kv_head, std::size_t block,
                                 std::size_t context, std::size_t dim,
                                 std::size_t bucket) {
  const std::size_t blocks = bucket / 2U;
  return (((kv_head * blocks + block) * 8U + context) * kHeadDim) + dim;
}

void write_prefill_output(const uint16_t *packed, size_t packed_elements,
                          ggml_tensor *dst, uint32_t tokens,
                          uint32_t bucket) {
  const size_t required = static_cast<size_t>(kQHeads) * bucket * kHeadDim;
  if (packed == nullptr || packed_elements < required) {
    report_error("prefill output view is smaller than Q32/D64 bucket");
    return;
  }
  for (uint32_t token = 0; token < tokens; ++token) {
    const std::size_t block = token / 2U;
    const std::size_t row = token % 2U;
    for (uint32_t head = 0; head < kQHeads; ++head) {
      const std::size_t kv_head = head / 4U;
      const std::size_t context = row * 4U + head % 4U;
      const std::size_t source =
          prefill_output_index(kv_head, block, context, 0, bucket);
      if (write_bf16_row_if_contiguous(dst, token, head, packed + source,
                                       kHeadDim) ||
          write_f32_bf16_row_if_contiguous(dst, token, head, packed + source,
                                           kHeadDim)) {
        continue;
      }
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const float value =
            ggml_bf16_to_fp32(ggml_bf16_t{packed[source + dim]});
        char *base = static_cast<char *>(dst->data) + dim * dst->nb[0] +
                     head * dst->nb[1] + token * dst->nb[2];
        switch (dst->type) {
        case GGML_TYPE_F32:
          *reinterpret_cast<float *>(base) = value;
          break;
        case GGML_TYPE_F16:
          *reinterpret_cast<ggml_fp16_t *>(base) = ggml_fp32_to_fp16(value);
          break;
        case GGML_TYPE_BF16:
          *reinterpret_cast<ggml_bf16_t *>(base) = ggml_fp32_to_bf16(value);
          break;
        default:
          report_error("unsupported FLASH_ATTN_EXT prefill output type");
          return;
        }
      }
    }
  }
}
void fill_cache_views(const std::vector<uint16_t> &compact, uint32_t tokens,
                      uint32_t heads, uint16_t *cache, size_t cache_elements) {
  const size_t required = static_cast<size_t>(heads) * kMaxKv * kHeadDim;
  if (cache == nullptr || cache_elements < required || tokens > kMaxKv) {
    report_error("invalid persistent cache view");
    return;
  }

  // U19 decode receives the valid position and only loads rows [0, position].
  // Clearing the unused [tokens, 256) tail cannot affect the result, so omit
  // that full-cache PS write while retaining every valid K/V row.
  for (uint32_t head = 0; head < heads; ++head) {
    for (uint32_t token = 0; token < tokens; ++token) {
      const size_t src =
          (static_cast<size_t>(token) * heads + head) * kHeadDim;
      const size_t dst =
          (static_cast<size_t>(head) * kMaxKv + token) * kHeadDim;
      std::copy_n(compact.data() + src, kHeadDim, cache + dst);
    }
  }
}

uint32_t choose_bucket(uint32_t tokens) {
  if (tokens <= 64U) return 64U;
  if (tokens <= 128U) return 128U;
  if (tokens <= 256U) return 256U;
  return 0U;
}

std::string profile_name(uint32_t bucket) {
  return bucket == 64U ? "S64" : bucket == 128U ? "S128" : "S256";
}

std::string xclbin_for_bucket(uint32_t bucket) {
  const char *name = bucket == 64U
                         ? "FPT_LLAMA_PL_XCLBIN_S64"
                         : bucket == 128U ? "FPT_LLAMA_PL_XCLBIN_S128"
                                          : "FPT_LLAMA_PL_XCLBIN_S256";
  std::string path = env_string(name);
  if (path.empty()) path = env_string("FPT_LLAMA_PL_XCLBIN");
  return path;
}

void accumulate(MetricsAccumulator *dst, const fpt_attention_metrics_t &m) {
  dst->pack_ms += m.pack_ms;
  dst->h2d_ms += m.host_to_device_ms;
  dst->kernel_ms += m.kernel_ms;
  dst->d2h_ms += m.device_to_host_ms;
  dst->unpack_ms += m.unpack_ms;
  dst->total_ms += m.total_ms;
  dst->h2d_calls += m.host_to_device_sync_calls;
  dst->d2h_calls += m.device_to_host_sync_calls;
  dst->h2d_bytes += m.host_to_device_bytes;
  dst->d2h_bytes += m.device_to_host_bytes;
}

bool create_session(uint32_t prompt_tokens) {
  HookState &s = state();
  uint32_t bucket = choose_bucket(prompt_tokens);
  // The host can select a profile from the complete prompt length before
  // llama.cpp splits it into micro-batches. The override changes only the
  // capacity/xclbin; prompt_tokens remains the actual committed token count.
  const std::string forced_profile = env_string("FPT_LLAMA_PL_PROFILE");
  if (forced_profile == "S64") bucket = 64U;
  else if (forced_profile == "S128") bucket = 128U;
  else if (forced_profile == "S256") bucket = 256U;
  if (bucket == 0U) return false;
  s.bucket = bucket;
  s.prompt_tokens = prompt_tokens;
  s.profile = profile_name(bucket);
  s.xclbin = xclbin_for_bucket(bucket);
  if (s.xclbin.empty()) {
    report_error("no xclbin configured for " + s.profile);
    return false;
  }
  const auto init_begin = Clock::now();
  fpt_attention_status_t status = fpt_attention_session_create(
      s.xclbin.c_str(), 0U, bucket, kLayers, &s.session);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("session_create: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  const std::string strategy = env_string("FPT_LLAMA_PL_CACHE_SYNC");
  const auto cache_strategy = strategy == "full"
                                   ? FPT_ATTENTION_CACHE_SYNC_FULL_BO
                                   : FPT_ATTENTION_CACHE_SYNC_VALID_HEAD_RANGES;
  status = fpt_attention_set_cache_sync_strategy(s.session, cache_strategy);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("set_cache_sync_strategy: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  s.session_create_ms = elapsed_ms(init_begin, Clock::now());
  s.initialized = true;
  s.app_profile_verified = true;
  std::cerr << "FPT_LLAMA_PL_PROFILE_SELECTED=" << s.profile << '\n'
            << "FPT_LLAMA_PL_DECODE_RESELECT=NO\n"
            << "FPT_LLAMA_PL_PREFILL_TOKENS=" << prompt_tokens << '\n'
            << "FPT_LLAMA_PL_BUCKET_TOKENS=" << bucket << '\n';
  return true;
}

bool prefill_call(const ggml_tensor *q, const ggml_tensor *k,
                  const ggml_tensor *v, ggml_tensor *dst,
                  uint32_t layer) {
  HookState &s = state();
  const uint32_t tokens = static_cast<uint32_t>(q->ne[1]);
  if (!s.initialized && !create_session(tokens)) return false;
  if (s.bucket == 0U || tokens > s.bucket || k->ne[1] < tokens) return false;

  const auto pack_begin = Clock::now();
  std::vector<uint16_t> &q_compact = s.q_compact;
  std::vector<uint16_t> &k_compact = s.k_compact;
  std::vector<uint16_t> &v_compact = s.v_compact;
  read_compact(q, tokens, kQHeads, &q_compact);
  read_compact(k, tokens, kKvHeads, &k_compact);
  read_compact(v, tokens, kKvHeads, &v_compact);

  fpt_attention_prefill_io_t io{};
  fpt_attention_status_t status =
      fpt_attention_get_prefill_mapped_io(s.session, layer, &io);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("get_prefill_mapped_io: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  fpt_prefill::pack_q_into(q_compact.data(), tokens, s.bucket,
                           FPT_PREFILL_TOKEN_HEAD_DIM, io.q_native);
  fpt_prefill::pack_k_into(k_compact.data(), tokens, s.bucket,
                           FPT_PREFILL_TOKEN_HEAD_DIM, io.k_prefill_native);
  fpt_prefill::pack_v_into(v_compact.data(), tokens, s.bucket,
                           FPT_PREFILL_TOKEN_HEAD_DIM, io.v_prefill_native);
  // At S256 the prefill K view aliases the persistent decode-cache K view.
  // pack_k_into already writes the complete [kv_head][bucket][dim] region,
  // including zero padding, so copying K into the cache a second time only
  // burns PS bandwidth and host time. S64/S128 keep the explicit conversion
  // because their prefill and decode strides are different.
  const uint64_t cache_valid_bytes =
      static_cast<uint64_t>(tokens) * kKvHeads * kHeadDim * sizeof(uint16_t);
  if (io.k_prefill_native != io.k_cache_native) {
    fill_cache_views(k_compact, tokens, kKvHeads, io.k_cache_native,
                     io.k_cache_elements);
    s.prefill_k_cache_valid_copy_bytes += cache_valid_bytes;
  } else {
    ++s.prefill_k_cache_alias_calls;
  }
  fill_cache_views(v_compact, tokens, kKvHeads, io.v_cache_native,
                   io.v_cache_elements);
  s.prefill_v_cache_valid_copy_bytes += cache_valid_bytes;
  const auto pack_end = Clock::now();

  fpt_attention_metrics_t metrics{};
  status = fpt_attention_run_prefill_mapped(s.session, layer, &metrics);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("run_prefill_mapped: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  metrics.pack_ms = elapsed_ms(pack_begin, pack_end);
  const auto unpack_begin = Clock::now();
  if (s.direct_unpack) {
    write_prefill_output(io.output_native, io.output_elements, dst,
                         tokens, s.bucket);
  } else {
    s.out_compact.assign(
        static_cast<size_t>(tokens) * kQHeads * kHeadDim, 0U);
    fpt_prefill::unpack_out(io.output_native, tokens, s.bucket,
                            FPT_PREFILL_TOKEN_HEAD_DIM,
                            s.out_compact.data());
    write_output(q, dst, tokens, kQHeads, s.out_compact);
  }
  const auto unpack_end = Clock::now();
  metrics.unpack_ms = elapsed_ms(unpack_begin, unpack_end);
  metrics.total_ms += metrics.pack_ms + metrics.unpack_ms;
  accumulate(&s.prefill, metrics);
  ++s.prefill_calls;

  // Each layer owns a distinct persistent K/V BO. Commit it immediately
  // after its prefill run so decode can begin at layer 0 without a hidden
  // CPU fallback or a later all-layer synchronization pass.
  fpt_attention_metrics_t commit{};
  status = fpt_attention_commit_prompt_cache(s.session, layer, tokens,
                                              &commit);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("commit_prompt_cache: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  accumulate(&s.commit, commit);
  if (layer + 1U == kLayers) {
    s.seen_prefill = true;
  }
  return true;
}

bool decode_call(const ggml_tensor *q, const ggml_tensor *k,
                 const ggml_tensor *v, ggml_tensor *dst, uint32_t layer) {
  HookState &s = state();
  if (!s.initialized || !s.seen_prefill || q->ne[1] != 1 ||
      k->ne[1] == 0 || k->ne[1] > kMaxKv) {
    return false;
  }
  const uint32_t valid_kv = valid_kv_tokens(dst->src[3], 0U, static_cast<uint32_t>(k->ne[1]));
  if (valid_kv == 0U) return false;
  const uint32_t position = valid_kv - 1U;
  if (position != s.prompt_tokens && s.last_decode_position != position) {
    report_error("non-sequential decode position");
    return false;
  }
  fpt_attention_decode_io_t io{};
  fpt_attention_status_t status =
      fpt_attention_get_decode_mapped_io(s.session, &io);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("get_decode_mapped_io: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  const auto pack_begin = Clock::now();
  std::vector<uint16_t> &q_values = s.decode_q;
  std::vector<uint16_t> &k_values = s.decode_k;
  std::vector<uint16_t> &v_values = s.decode_v;
  read_decode_q(q, &q_values);
  read_decode_kv(k, position, &k_values);
  read_decode_kv(v, position, &v_values);
  std::copy(q_values.begin(), q_values.end(), io.q);
  std::copy(k_values.begin(), k_values.end(), io.k_new);
  std::copy(v_values.begin(), v_values.end(), io.v_new);
  const auto pack_end = Clock::now();

  fpt_attention_metrics_t metrics{};
  status = fpt_attention_run_decode_mapped(s.session, layer, position,
                                            &metrics);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("run_decode_mapped: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  metrics.pack_ms = elapsed_ms(pack_begin, pack_end);
  const auto unpack_begin = Clock::now();
  write_decode_output(dst, io.output, io.output_elements);
  const auto unpack_end = Clock::now();
  metrics.unpack_ms = elapsed_ms(unpack_begin, unpack_end);
  metrics.total_ms += metrics.pack_ms + metrics.unpack_ms;
  accumulate(&s.decode, metrics);
  ++s.decode_calls;
  if (layer + 1U == kLayers) {
    s.last_decode_position = position + 1U;
    s.seen_decode = true;
  }
  return true;
}

void compare_shadow(const ggml_tensor *reference, const ggml_tensor *actual) {
  HookState &s = state();
  const size_t elements = static_cast<size_t>(actual->ne[0]) * actual->ne[1] *
                          actual->ne[2] * actual->ne[3];
  double local_sq = 0.0;
  double local_max = 0.0;
  uint64_t nonfinite = 0;
  for (int64_t b = 0; b < actual->ne[3]; ++b) {
    for (int64_t h = 0; h < actual->ne[2]; ++h) {
      for (int64_t t = 0; t < actual->ne[1]; ++t) {
        for (int64_t d = 0; d < actual->ne[0]; ++d) {
          const char *rbase = static_cast<const char *>(reference->data) +
                              d * reference->nb[0] + t * reference->nb[1] +
                              h * reference->nb[2] + b * reference->nb[3];
          const char *abase = static_cast<const char *>(actual->data) +
                              d * actual->nb[0] + t * actual->nb[1] +
                              h * actual->nb[2] + b * actual->nb[3];
          const float rv = load_value(reference, d, t, h);
          const float av = load_value(actual, d, t, h);
          if (!std::isfinite(rv) || !std::isfinite(av)) {
            ++nonfinite;
            continue;
          }
          const double diff = static_cast<double>(av) - rv;
          local_max = std::max(local_max, std::abs(diff));
          local_sq += diff * diff;
        }
      }
    }
  }
  ++s.shadow_compares;
  s.shadow_nonfinite += nonfinite;
  s.shadow_max_abs = std::max(s.shadow_max_abs, local_max);
  s.shadow_sq_sum += local_sq;
  s.shadow_elements += elements;
}

void write_metric(std::ostream &out, const char *prefix,
                  const MetricsAccumulator &m) {
  out << prefix << "_PACK_MS=" << std::setprecision(10) << m.pack_ms << '\n'
      << prefix << "_H2D_MS=" << m.h2d_ms << '\n'
      << prefix << "_KERNEL_MS=" << m.kernel_ms << '\n'
      << prefix << "_D2H_MS=" << m.d2h_ms << '\n'
      << prefix << "_UNPACK_MS=" << m.unpack_ms << '\n'
      << prefix << "_TOTAL_MS=" << m.total_ms << '\n'
      << prefix << "_H2D_CALLS=" << m.h2d_calls << '\n'
      << prefix << "_D2H_CALLS=" << m.d2h_calls << '\n'
      << prefix << "_H2D_BYTES=" << m.h2d_bytes << '\n'
      << prefix << "_D2H_BYTES=" << m.d2h_bytes << '\n';
}

void write_report() {
  HookState &s = state();
  if (!s.enabled || s.report_path.empty()) return;
  std::ofstream out(s.report_path, std::ios::trunc);
  if (!out) return;
  out << "SCHEMA=FPT-LLAMA32-1B-PS-PL-U19-v1\n"
      << "STATUS=" << (s.failure_calls == 0 && s.calls > 0 ? "PASS" : "FAIL")
      << '\n'
      << "MODEL=Llama-3.2-1B\n"
      << "Q_HEADS=32\nKV_HEADS=8\nHEAD_DIM=64\nCACHE_LAYERS=16\n"
      << "PROFILE_SELECTED=" << s.profile << '\n'
      << "PROFILE_BUCKET_TOKENS=" << s.bucket << '\n'
      << "PREFILL_TOKENS=" << s.prompt_tokens << '\n'
      << "DECODE_RESELECT=NO\n"
      << "XCLBIN=" << s.xclbin << '\n'
      << "PL_ATTENTION_CALLS=" << s.calls << '\n'
      << "PL_PREFILL_CALLS=" << s.prefill_calls << '\n'
      << "PL_DECODE_CALLS=" << s.decode_calls << '\n'
      << "PL_FALLBACK_CALLS=" << s.fallback_calls << '\n'
      << "PL_FAILURE_CALLS=" << s.failure_calls << '\n'
      << "SESSION_CREATE_MS=" << s.session_create_ms << '\n'
      << "SHADOW_COMPARE_CALLS=" << s.shadow_compares << '\n'
      << "SHADOW_NONFINITE=" << s.shadow_nonfinite << '\n'
      << "SHADOW_MAX_ABS=" << std::setprecision(10) << s.shadow_max_abs << '\n'
      << "SHADOW_RMSE="
      << (s.shadow_elements == 0
              ? 0.0
              : std::sqrt(s.shadow_sq_sum /
                          static_cast<double>(s.shadow_elements)))
      << '\n';
  write_metric(out, "PREFILL", s.prefill);
  write_metric(out, "PROMPT_CACHE_COMMIT", s.commit);
  write_metric(out, "DECODE", s.decode);
  out << "PREFILL_K_CACHE_ALIAS_CALLS=" << s.prefill_k_cache_alias_calls
      << '\n'
      << "PREFILL_K_CACHE_VALID_COPY_BYTES="
      << s.prefill_k_cache_valid_copy_bytes << '\n'
      << "PREFILL_V_CACHE_VALID_COPY_BYTES="
      << s.prefill_v_cache_valid_copy_bytes << '\n';
  out << "E2E_PREFILL_WITH_CACHE_COMMIT_MS="
      << s.prefill.total_ms + s.commit.total_ms << '\n'
      << "E2E_PL_TOTAL_MS="
      << s.prefill.total_ms + s.commit.total_ms + s.decode.total_ms << '\n'
      << "E2E_PL_COLD_TOTAL_MS="
      << s.session_create_ms + s.prefill.total_ms + s.commit.total_ms +
             s.decode.total_ms << '\n';
  out << "FPT_LLAMA_PL_REPORT=PASS\n";
}

} // namespace

extern "C" bool fpt_llama_pl_try_compute(
    const struct ggml_compute_params *params, struct ggml_tensor *dst,
    fpt_llama_pl_cpu_attention_fn cpu_reference) {
  if (!enabled()) {
    return false;
  }
  if (!is_supported_shape(dst)) {
    return false;
  }
  const ggml_tensor *q = dst->src[0];
  const ggml_tensor *k = dst->src[1];
  const ggml_tensor *v = dst->src[2];
  HookState &s = state();
  // Do not partially enter the PL path when no bitstream is configured.
  // Returning false before the worker split keeps the native CPU graph fully
  // parallel and makes disabled-board tests deterministic.
  const bool any_xclbin = !env_string("FPT_LLAMA_PL_XCLBIN").empty() ||
      !env_string("FPT_LLAMA_PL_XCLBIN_S64").empty() ||
      !env_string("FPT_LLAMA_PL_XCLBIN_S128").empty() ||
      !env_string("FPT_LLAMA_PL_XCLBIN_S256").empty();
  if (!any_xclbin) return false;
  // A q=1 graph is also used while the prompt is still being evaluated
  // (small micro-batches and warmup). It cannot be treated as decode until
  // all 16 prefill layers have committed the prompt cache.
  if (q->ne[1] == 1 && !s.seen_prefill) return false;
  // A shape can be recognized by the gate but still be intentionally left to
  // ggml CPU. The real llama.cpp prefill view keeps K/V at the selected
  // bucket extent (often 256) and uses the mask to mark valid prompt tokens.
  // Only the first prompt operation with a causal mask is eligible; q=1 is
  // decode only after all prefill layers have committed the cache.
  const uint32_t q_tokens = static_cast<uint32_t>(q->ne[1]);
  const uint32_t kv_extent = static_cast<uint32_t>(k->ne[1]);
  const ggml_tensor *mask = dst->src[3];
  const bool exact_prefill = q_tokens > 1U && !s.seen_prefill &&
      (kv_extent == q_tokens ||
       (mask != nullptr && valid_kv_tokens(mask, 0U, kv_extent) == 1U &&
        valid_kv_tokens(mask, q_tokens - 1U, kv_extent) == q_tokens));
  // A q=1 graph can appear during prompt continuation. Do not let it enter
  // decode unless its masked position follows the committed prompt or the
  // previous PL decode position; otherwise leave the whole op to CPU.
  if (q_tokens == 1U && s.seen_prefill && mask != nullptr) {
    const uint32_t valid = valid_kv_tokens(mask, 0U, kv_extent);
    const uint32_t position = valid == 0U ? 0U : valid - 1U;
    if (position != s.prompt_tokens && position != s.last_decode_position) {
      return false;
    }
  }
  const bool cpu_candidate = exact_prefill || q_tokens == 1U;
  if (!cpu_candidate) return false;
  if (params->ith != 0) return true;
  std::lock_guard<std::mutex> guard(g_call_mutex);
  const uint32_t layer = static_cast<uint32_t>(dst->op_params[4]);
  if (layer >= kLayers) {
    report_error("missing/invalid layer id in FLASH_ATTN_EXT op_params[4]");
    return false;
  }
  if (env_enabled("FPT_LLAMA_PL_DEBUG_SEQUENCE", false)) {
    static std::atomic<uint32_t> printed = 0U;
    if (printed.fetch_add(1U, std::memory_order_relaxed) < 256U) {
      const uint32_t q_tokens = static_cast<uint32_t>(q->ne[1]);
      const uint32_t kv_extent = static_cast<uint32_t>(k->ne[1]);
      const uint32_t first_valid = valid_kv_tokens(dst->src[3], 0U, kv_extent);
      const uint32_t last_valid = valid_kv_tokens(dst->src[3], q_tokens - 1U, kv_extent);
      std::cerr << "FPT_LLAMA_PL_SEQUENCE layer=" << layer
                << " q_tokens=" << q_tokens << " kv_extent=" << kv_extent
                << " valid_first=" << first_valid
                << " valid_last=" << last_valid;
      if (dst->src[3] != nullptr) {
        std::cerr << " mask=" << dst->src[3]->ne[0] << ',' << dst->src[3]->ne[1]
                  << ',' << dst->src[3]->ne[2] << ',' << dst->src[3]->ne[3]
                  << ":type=" << dst->src[3]->type;
      }
      std::cerr << '\n';
    }
  }
  std::vector<float> reference_output;
  std::vector<uint8_t> reference_bytes;
  if (s.shadow && cpu_reference != nullptr) {
    reference_bytes.resize(ggml_nbytes(dst));
    ggml_compute_params ref_params = *params;
    ref_params.ith = 0;
    ref_params.nth = 1;
    cpu_reference(&ref_params, dst);
    std::memcpy(reference_bytes.data(), dst->data, ggml_nbytes(dst));
  }

  bool handled = false;
  if (exact_prefill) {
    handled = prefill_call(q, k, v, dst, layer);
  } else if (q->ne[1] == 1) {
    handled = decode_call(q, k, v, dst, layer);
  }
  if (!handled) {
    ++s.fallback_calls;
    // Keep all ggml workers on the same decision. Once a candidate shape has
    // passed the gate, non-zero workers have returned true, so worker 0 must
    // complete a whole CPU fallback instead of returning false alone.
    if (cpu_reference != nullptr) {
      ggml_compute_params fallback_params = *params;
      fallback_params.ith = 0;
      fallback_params.nth = 1;
      cpu_reference(&fallback_params, dst);
      return true;
    }
    if (!s.strict) return false;
    report_error("recognized Llama shape could not be handled by U19");
    return false;
  }
  ++s.calls;

  if (s.shadow && !reference_bytes.empty()) {
    ggml_tensor ref = *dst;
    ref.data = reference_bytes.data();
    compare_shadow(&ref, dst);
  }
  return true;
}
