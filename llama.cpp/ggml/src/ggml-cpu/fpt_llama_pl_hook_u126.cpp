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
  bool cache_stale = false;
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
  uint64_t cache_rebase_calls = 0;
  uint64_t fallback_calls = 0;
  uint64_t failure_calls = 0;
  uint64_t shadow_compares = 0;
  uint64_t shadow_nonfinite = 0;
  double shadow_max_abs = 0.0;
  double shadow_sq_sum = 0.0;
  uint64_t shadow_elements = 0;
  MetricsAccumulator prefill;
  MetricsAccumulator rebase;
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
  const bool supported = q->ne[0] == kHeadDim && k->ne[0] == kHeadDim &&
         v->ne[0] == kHeadDim && q->ne[2] == kQHeads &&
         k->ne[2] == kKvHeads && v->ne[2] == kKvHeads &&
         q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
         dst->ne[0] == kHeadDim && dst->ne[1] == kQHeads &&
         dst->ne[2] == q->ne[1] && dst->ne[3] == 1 &&
         q->ne[1] > 0 && k->ne[1] > 0 && k->ne[1] == v->ne[1] &&
         q->type != GGML_TYPE_COUNT &&
         k->type != GGML_TYPE_COUNT && v->type != GGML_TYPE_COUNT;
  if (!supported && env_enabled("FPT_LLAMA_PL_DEBUG_SHAPES", false)) {
    static std::atomic<uint32_t> printed = 0U;
    if (printed.fetch_add(1U, std::memory_order_relaxed) < 64U) {
      std::cerr << "FPT_LLAMA_PL_UNSUPPORTED_SHAPE"
                << " q=" << q->ne[0] << ',' << q->ne[1] << ',' << q->ne[2] << ',' << q->ne[3]
                << ":t" << q->type
                << " k=" << k->ne[0] << ',' << k->ne[1] << ',' << k->ne[2] << ',' << k->ne[3]
                << ":t" << k->type
                << " v=" << v->ne[0] << ',' << v->ne[1] << ',' << v->ne[2] << ',' << v->ne[3]
                << ":t" << v->type
                << " dst=" << dst->ne[0] << ',' << dst->ne[1] << ',' << dst->ne[2] << ',' << dst->ne[3]
                << ":t" << dst->type << '\n';
    }
  }
  return supported;
}

bool shadow_layer_enabled(uint32_t layer) {
  const std::string only = env_string("FPT_LLAMA_PL_SHADOW_LAYER");
  if (only.empty()) return true;
  char *end = nullptr;
  const long selected = std::strtol(only.c_str(), &end, 10);
  return end != only.c_str() && *end == '\0' && selected >= 0 &&
         layer == static_cast<uint32_t>(selected);
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

bool mask_covers_complete_prompt(const ggml_tensor *mask, uint32_t q_tokens,
                                 uint32_t kv_extent) {
  if (mask == nullptr || q_tokens == 0U ||
      mask->ne[1] < static_cast<int64_t>(q_tokens)) {
    return false;
  }
  for (uint32_t query = 0; query < q_tokens; ++query) {
    if (valid_kv_tokens(mask, query, kv_extent) != query + 1U) {
      return false;
    }
  }
  return true;
}

bool decode_mask_position(const ggml_tensor *mask, uint32_t kv_extent,
                          uint32_t *position) {
  if (mask == nullptr || mask->data == nullptr || mask->ne[0] < kv_extent ||
      mask->ne[1] != 1) return false;
  const uint32_t visible = valid_kv_tokens(mask, 0U, kv_extent);
  if (visible == 0U || visible > kMaxKv) return false;
  for (uint32_t key = 0; key < kv_extent; ++key) {
    if (std::isfinite(load_mask_value(mask, key, 0)) != (key < visible)) {
      return false;
    }
  }
  *position = visible - 1U;
  return true;
}
float load_value(const ggml_tensor *tensor, int64_t dim, int64_t token,
                 int64_t head, int64_t batch = 0) {
  const char *base = static_cast<const char *>(tensor->data) +
                     dim * tensor->nb[0] + token * tensor->nb[1] +
                     head * tensor->nb[2] + batch * tensor->nb[3];
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

// FLASH_ATTN_EXT stores its result as [head_dim, q_heads, tokens, batch],
// unlike its Q/K/V inputs which are [head_dim, tokens, heads, batch].
float load_output_value(const ggml_tensor *tensor, int64_t dim,
                        int64_t token, int64_t head, int64_t batch = 0) {
  const char *base = static_cast<const char *>(tensor->data) +
                     dim * tensor->nb[0] + head * tensor->nb[1] +
                     token * tensor->nb[2] + batch * tensor->nb[3];
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

bool row_memcpy_safe(const ggml_tensor *tensor, uint32_t elements,
                     size_t item_bytes) {
  if (tensor == nullptr || tensor->data == nullptr || elements == 0U ||
      tensor->nb[0] != item_bytes) {
    return false;
  }
  // nb[0] alone does not prove that a full dim-0 row is contiguous. Views
  // with padding or unusual strides can otherwise make memcpy cross rows.
  const size_t row_bytes = static_cast<size_t>(elements) * item_bytes;
  if (tensor->ne[1] > 1 && tensor->nb[1] < row_bytes) return false;
  if (tensor->ne[2] > 1 && tensor->nb[2] < row_bytes) return false;
  return true;
}

// Included inside the hook's existing anonymous namespace.
bool read_converted_rows(const ggml_tensor *tensor, uint32_t first_token,
                         uint32_t tokens, uint32_t heads,
                         std::vector<uint16_t> *out) {
  const bool f32 = tensor->type == GGML_TYPE_F32;
  const bool f16 = tensor->type == GGML_TYPE_F16;
  const size_t item_bytes = f32 ? sizeof(float) : sizeof(ggml_fp16_t);
  if ((!f32 && !f16) || !row_memcpy_safe(tensor, kHeadDim, item_bytes) ||
      env_enabled("FPT_LLAMA_PL_DISABLE_ROW_CONVERT", false)) {
    return false;
  }
  float intermediate[kHeadDim];
  ggml_bf16_t converted[kHeadDim];
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      const char *base = static_cast<const char *>(tensor->data) +
                         (first_token + token) * tensor->nb[1] +
                         head * tensor->nb[2];
      if (f32) {
        ggml_fp32_to_bf16_row(reinterpret_cast<const float *>(base),
                             converted, kHeadDim);
      } else {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(base),
                              intermediate, kHeadDim);
        ggml_fp32_to_bf16_row(intermediate, converted, kHeadDim);
      }
      const size_t index = (static_cast<size_t>(token) * heads + head) * kHeadDim;
      std::memcpy(out->data() + index, converted, sizeof(converted));
    }
  }
  return true;
}

void read_compact(const ggml_tensor *tensor, uint32_t tokens,
                  uint32_t heads, std::vector<uint16_t> *out) {
  out->resize(static_cast<size_t>(tokens) * heads * kHeadDim);
  if (read_converted_rows(tensor, 0U, tokens, heads, out)) return;

  // Preserve directly exposed BF16 bits and avoid a BF16->F32->BF16 round
  // trip on the PS. The scalar path remains for F32/F16 and strided views.
  if (tensor->type == GGML_TYPE_BF16 &&
      row_memcpy_safe(tensor, kHeadDim, sizeof(ggml_bf16_t)) &&
      !env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
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
  if (read_converted_rows(tensor, 0U, 1U, kQHeads, out)) return;
  if (tensor->type == GGML_TYPE_BF16 &&
      row_memcpy_safe(tensor, kHeadDim, sizeof(ggml_bf16_t)) &&
      !env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
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
  if (read_converted_rows(tensor, position, 1U, kKvHeads, out)) return;
  if (tensor->type == GGML_TYPE_BF16 &&
      row_memcpy_safe(tensor, kHeadDim, sizeof(ggml_bf16_t)) &&
      !env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
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
      elements == 0U || !row_memcpy_safe(dst, elements, sizeof(ggml_bf16_t)) ||
      env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
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
      bf16 == nullptr || elements == 0U ||
      !row_memcpy_safe(dst, elements, sizeof(float)) ||
      env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
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
  const std::size_t blocks = bucket / fpt_prefill::kRowsPerBlock;
  return (((kv_head * blocks + block) * fpt_prefill::kContextsPerBlock +
           context) * kHeadDim) + dim;
}

template <bool F32>
void store_bf16_output_row(char *destination, const uint16_t *source) {
  if (!F32) {
    std::memcpy(destination, source, kHeadDim * sizeof(uint16_t));
    return;
  }
  float *output = reinterpret_cast<float *>(destination);
#if FPT_LLAMA_NEON_AVAILABLE
  for (uint32_t dim = 0; dim < kHeadDim; dim += 8U) {
    const uint32x4_t lo = vshll_n_u16(vld1_u16(source + dim), 16);
    const uint32x4_t hi = vshll_n_u16(vld1_u16(source + dim + 4U), 16);
    vst1q_f32(output + dim, vreinterpretq_f32_u32(lo));
    vst1q_f32(output + dim + 4U, vreinterpretq_f32_u32(hi));
  }
#else
  for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
    output[dim] = ggml_bf16_to_fp32(ggml_bf16_t{source[dim]});
  }
#endif
}

template <bool F32, bool NativeOrder>
void write_prefill_rows_fast(const uint16_t *packed, ggml_tensor *dst,
                             uint32_t tokens, uint32_t bucket) {
  if (NativeOrder) {
    const uint32_t blocks = (tokens + fpt_prefill::kRowsPerBlock - 1U) /
                            fpt_prefill::kRowsPerBlock;
    for (uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      for (uint32_t block = 0; block < blocks; ++block) {
        for (uint32_t row = 0; row < fpt_prefill::kRowsPerBlock; ++row) {
          const uint32_t token = block * fpt_prefill::kRowsPerBlock + row;
          if (token >= tokens) break;
          for (uint32_t group = 0; group < fpt_prefill::kGqaGroup; ++group) {
            const uint32_t head = kv_head * fpt_prefill::kGqaGroup + group;
            const uint32_t context = row * fpt_prefill::kGqaGroup + group;
            const uint16_t *source = packed +
                prefill_output_index(kv_head, block, context, 0, bucket);
            char *destination = static_cast<char *>(dst->data) +
                                head * dst->nb[1] + token * dst->nb[2];
            store_bf16_output_row<F32>(destination, source);
          }
        }
      }
    }
  } else {
    for (uint32_t token = 0; token < tokens; ++token) {
      for (uint32_t head = 0; head < kQHeads; ++head) {
        const uint16_t *source = packed + prefill_output_index(
            head / fpt_prefill::kGqaGroup,
            token / fpt_prefill::kRowsPerBlock,
            (token % fpt_prefill::kRowsPerBlock) * fpt_prefill::kGqaGroup +
                head % fpt_prefill::kGqaGroup,
            0, bucket);
        char *destination = static_cast<char *>(dst->data) +
                            head * dst->nb[1] + token * dst->nb[2];
        store_bf16_output_row<F32>(destination, source);
      }
    }
  }
}

template <bool NativeOrder>
bool write_prefill_rows_if_contiguous(const uint16_t *packed,
                                      size_t packed_elements,
                                      ggml_tensor *dst, uint32_t tokens,
                                      uint32_t bucket) {
  if (dst == nullptr || dst->data == nullptr) return false;
  const bool f32 = dst->type == GGML_TYPE_F32;
  const bool bf16 = dst->type == GGML_TYPE_BF16;
  const size_t item_bytes = f32 ? sizeof(float) : sizeof(ggml_bf16_t);
  const size_t row_bytes = kHeadDim * item_bytes;
  const bool disjoint_rows = tokens != 0U &&
      (dst->nb[2] >= (kQHeads - 1U) * dst->nb[1] + row_bytes ||
       dst->nb[1] >= (tokens - 1U) * dst->nb[2] + row_bytes);
  if ((!f32 && !bf16) || packed == nullptr || tokens == 0U ||
      tokens > bucket || bucket % fpt_prefill::kRowsPerBlock != 0U ||
      dst->ne[0] < kHeadDim || dst->ne[1] != kQHeads ||
      dst->ne[2] < tokens || !disjoint_rows ||
      packed_elements < static_cast<size_t>(kQHeads) * bucket * kHeadDim ||
      !row_memcpy_safe(dst, kHeadDim, item_bytes) ||
      env_enabled("FPT_LLAMA_PL_DISABLE_ROW_MEMCPY", false)) {
    return false;
  }
  // Dispatch once per tensor; retain the original fallback for other views.
  if (f32) write_prefill_rows_fast<true, NativeOrder>(packed, dst, tokens, bucket);
  else write_prefill_rows_fast<false, NativeOrder>(packed, dst, tokens, bucket);
  return true;
}

void write_prefill_output(const uint16_t *packed, size_t packed_elements,
                          ggml_tensor *dst, uint32_t tokens,
                          uint32_t bucket) {
  const size_t required = static_cast<size_t>(kQHeads) * bucket * kHeadDim;
  if (packed == nullptr || packed_elements < required) {
    report_error("prefill output view is smaller than Q32/D64 bucket");
    return;
  }
  if (write_prefill_rows_if_contiguous<false>(packed, packed_elements, dst, tokens, bucket)) return;
  for (uint32_t token = 0; token < tokens; ++token) {
    const std::size_t block = token / fpt_prefill::kRowsPerBlock;
    const std::size_t row = token % fpt_prefill::kRowsPerBlock;
    for (uint32_t head = 0; head < kQHeads; ++head) {
      const std::size_t kv_head = head / 4U;
      const std::size_t context = row * fpt_prefill::kGqaGroup +
                                  head % fpt_prefill::kGqaGroup;
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
  // U126 is the frozen single-profile L512 implementation.
  return tokens > 0U && tokens <= kMaxKv ? kMaxKv : 0U;
}

void debug_tensor_contract(const char *name, const ggml_tensor *tensor,
                           uint32_t tokens, uint32_t heads) {
  if (!env_enabled("FPT_LLAMA_PL_DEBUG_TENSORS", false) || tensor == nullptr) {
    return;
  }
  static std::atomic<uint32_t> printed = 0U;
  if (printed.fetch_add(1U, std::memory_order_relaxed) >= 3U) {
    return;
  }
  std::cerr << "FPT_LLAMA_PL_TENSOR name=" << name
            << " type=" << tensor->type
            << " ne=" << tensor->ne[0] << ',' << tensor->ne[1] << ','
            << tensor->ne[2] << ',' << tensor->ne[3]
            << " nb=" << tensor->nb[0] << ',' << tensor->nb[1] << ','
            << tensor->nb[2] << ',' << tensor->nb[3]
            << " data=" << tensor->data << '\n';
  const uint32_t show_tokens = std::min<uint32_t>(tokens, 2U);
  const uint32_t show_heads = std::min<uint32_t>(heads, 2U);
  const uint32_t show_dims = 8U;
  for (uint32_t token = 0; token < show_tokens; ++token) {
    for (uint32_t head = 0; head < show_heads; ++head) {
      std::cerr << "FPT_LLAMA_PL_TENSOR_VALUES name=" << name
                << " token=" << token << " head=" << head << ':';
      for (uint32_t dim = 0; dim < show_dims; ++dim) {
        std::cerr << ' ' << std::setprecision(9)
                  << load_value(tensor, dim, token, head);
      }
      std::cerr << '\n';
    }
  }
}

// A thread-pool-free reference used only for diagnosing the PS/PL boundary.
// Calling ggml's normal implementation from inside this hook is unsafe: the
// other ggml workers have already returned to the caller while worker 0 is
// running the hook, so its internal barriers and scratch partitioning do not
// describe a valid CPU execution.
void scalar_prefill_shadow(const ggml_tensor *q, const ggml_tensor *k,
                           const ggml_tensor *v, const ggml_tensor *mask,
                           const ggml_tensor *actual, uint32_t tokens,
                           uint32_t layer, bool decode = false) {
  const bool diagnose = decode
      ? env_enabled("FPT_LLAMA_PL_SCALAR_DECODE_SHADOW", false)
      : env_enabled("FPT_LLAMA_PL_SCALAR_SHADOW", false);
  if (!diagnose || q == nullptr ||
      k == nullptr || v == nullptr || actual == nullptr || tokens == 0U) {
    return;
  }
  float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  std::memcpy(&scale, actual->op_params + 0, sizeof(scale));
  float max_bias = 0.0f;
  float logit_softcap = 0.0f;
  std::memcpy(&max_bias, actual->op_params + 1, sizeof(max_bias));
  std::memcpy(&logit_softcap, actual->op_params + 2, sizeof(logit_softcap));
  if (logit_softcap != 0.0f) scale /= logit_softcap;

  const uint32_t q_heads = static_cast<uint32_t>(q->ne[2]);
  const uint32_t kv_heads = static_cast<uint32_t>(k->ne[2]);
  const uint32_t kv_extent = static_cast<uint32_t>(k->ne[1]);
  const uint32_t actual_tokens = static_cast<uint32_t>(actual->ne[2]);
  const uint32_t n_tokens = std::min(tokens, actual_tokens);
  const uint32_t group = q_heads / kv_heads;
  std::vector<float> reference(static_cast<size_t>(n_tokens) * q_heads *
                               kHeadDim, 0.0f);
  double sq = 0.0;
  double max_abs = 0.0;
  uint64_t count = 0;
  uint64_t nonfinite = 0;
  for (uint32_t token = 0; token < n_tokens; ++token) {
    for (uint32_t head = 0; head < q_heads; ++head) {
      const uint32_t kv_head = group == 0U ? 0U : head / group;
      std::vector<float> scores(kv_extent, -INFINITY);
      float max_score = -INFINITY;
      for (uint32_t key = 0; key < kv_extent; ++key) {
        const float mv = mask == nullptr
                             ? 0.0f
                             : load_mask_value(mask, key, token);
        if (!std::isfinite(mv)) continue;
        float dot = 0.0f;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
          dot += load_value(q, dim, token, head) *
                 load_value(k, dim, key, kv_head);
        }
        float score = dot * scale;
        if (logit_softcap != 0.0f) {
          score = logit_softcap * std::tanh(score);
        }
        float mask_score = mv;
        if (max_bias > 0.0f) {
          const uint32_t log2_heads = 1U << static_cast<uint32_t>(
              std::floor(std::log2(static_cast<float>(q_heads))));
          const float m0 = std::pow(2.0f, -max_bias / log2_heads);
          const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / log2_heads);
          const float slope = head < log2_heads
                                  ? std::pow(m0, head + 1U)
                                  : std::pow(m1, 2.0f * (head - log2_heads) +
                                                 1.0f);
          mask_score *= slope;
        }
        score += mask_score;
        scores[key] = score;
        max_score = std::max(max_score, score);
      }
      float sum = 0.0f;
      for (float score : scores) {
        if (std::isfinite(score)) sum += std::exp(score - max_score);
      }
      const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        float value = 0.0f;
        for (uint32_t key = 0; key < kv_extent; ++key) {
          if (std::isfinite(scores[key])) {
            value += std::exp(scores[key] - max_score) *
                     load_value(v, dim, key, kv_head);
          }
        }
        reference[(static_cast<size_t>(token) * q_heads + head) * kHeadDim +
                  dim] = value * inv_sum;
      }
    }
  }
  for (uint32_t token = 0; token < n_tokens; ++token) {
    for (uint32_t head = 0; head < q_heads; ++head) {
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const float rv = reference[(static_cast<size_t>(token) * q_heads +
                                    head) * kHeadDim + dim];
        const float av = load_output_value(actual, dim, token, head);
        if (!std::isfinite(rv) || !std::isfinite(av)) {
          ++nonfinite;
          continue;
        }
        const double diff = static_cast<double>(av) - rv;
        max_abs = std::max(max_abs, std::abs(diff));
        sq += diff * diff;
        ++count;
      }
    }
  }
  std::cerr << (decode ? "FPT_LLAMA_PL_SCALAR_DECODE" :
                          "FPT_LLAMA_PL_SCALAR_SHADOW") << " layer=" << layer
            << " q_tokens=" << n_tokens << " max_abs=" << max_abs
            << " finite=" << count << " nonfinite=" << nonfinite
            << " rmse=" << (count == 0 ? 0.0
                                      : std::sqrt(sq / static_cast<double>(count)))
            << "\n";
  if (env_enabled("FPT_LLAMA_PL_DEBUG_VALUES", false)) {
    std::cerr << "FPT_LLAMA_PL_SCALAR_VALUES REF=";
    for (uint32_t dim = 0; dim < 8U; ++dim) {
      if (dim != 0U) std::cerr << ',';
      std::cerr << reference[dim];
    }
    std::cerr << " ACT=";
    for (uint32_t dim = 0; dim < 8U; ++dim) {
      if (dim != 0U) std::cerr << ',';
      std::cerr << load_output_value(actual, dim, 0, 0);
    }
    std::cerr << '\n';
    const uint32_t show_tokens = std::min<uint32_t>(n_tokens, 2U);
    const uint32_t show_heads = std::min<uint32_t>(q_heads, 8U);
    for (uint32_t token = 0; token < show_tokens; ++token) {
      for (uint32_t head = 0; head < show_heads; ++head) {
        const size_t offset =
            (static_cast<size_t>(token) * q_heads + head) * kHeadDim;
        std::cerr << "FPT_LLAMA_PL_SCALAR_ROW token=" << token
                  << " head=" << head << " REF=";
        for (uint32_t dim = 0; dim < 4U; ++dim) {
          if (dim != 0U) std::cerr << ',';
          std::cerr << reference[offset + dim];
        }
        std::cerr << " ACT=";
        for (uint32_t dim = 0; dim < 4U; ++dim) {
          if (dim != 0U) std::cerr << ',';
          std::cerr << load_output_value(actual, dim, token, head);
        }
        std::cerr << '\n';
      }
    }
  }
}

void token_major_to_head_major(const std::vector<uint16_t> &token_major,
                               uint32_t tokens, uint32_t heads,
                               std::vector<uint16_t> *head_major) {
  head_major->resize(static_cast<size_t>(tokens) * heads * kHeadDim);
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      const size_t src = (static_cast<size_t>(token) * heads + head) * kHeadDim;
      const size_t dst = (static_cast<size_t>(head) * tokens + token) * kHeadDim;
      std::copy_n(token_major.data() + src, kHeadDim, head_major->data() + dst);
    }
  }
}

void head_major_to_token_major(const std::vector<uint16_t> &head_major,
                               uint32_t tokens, uint32_t heads,
                               std::vector<uint16_t> *token_major) {
  token_major->resize(static_cast<size_t>(tokens) * heads * kHeadDim);
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      const size_t src = (static_cast<size_t>(head) * tokens + token) * kHeadDim;
      const size_t dst = (static_cast<size_t>(token) * heads + head) * kHeadDim;
      std::copy_n(head_major.data() + src, kHeadDim, token_major->data() + dst);
    }
  }
}

std::string profile_name(uint32_t bucket) {
  return bucket == kMaxKv ? "U126_L512" : "UNSUPPORTED";
}

std::string xclbin_for_bucket(uint32_t bucket) {
  return bucket == kMaxKv ? env_string("FPT_LLAMA_PL_XCLBIN") : std::string{};
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
  if (!forced_profile.empty() && forced_profile != "U126_L512") {
    report_error("U126 requires FPT_LLAMA_PL_PROFILE=U126_L512");
    return false;
  }
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
                  uint32_t layer, uint32_t cache_prefix = 0U) {
  HookState &s = state();
  const bool cache_only = cache_prefix != 0U;
  const uint32_t tokens = cache_only ? cache_prefix : static_cast<uint32_t>(q->ne[1]);
  if (!s.initialized && !create_session(tokens)) return false;
  if (s.bucket == 0U || tokens > s.bucket || k->ne[1] < tokens) return false;

  const auto pack_begin = Clock::now();
  std::vector<uint16_t> &q_compact = s.q_compact;
  std::vector<uint16_t> &k_compact = s.k_compact;
  std::vector<uint16_t> &v_compact = s.v_compact;
  debug_tensor_contract("q", q, cache_only ? 1U : tokens, kQHeads);
  debug_tensor_contract("k", k, tokens, kKvHeads);
  debug_tensor_contract("v", v, tokens, kKvHeads);
  if (cache_only) {
    q_compact.assign(static_cast<size_t>(tokens) * kQHeads * kHeadDim, 0U);
  } else {
    read_compact(q, tokens, kQHeads, &q_compact);
  }
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
  // U126 prefill K aliases the persistent decode-cache K view. Avoid a second
  // host copy when the mapped views are identical.
  const uint64_t cache_valid_bytes =
      static_cast<uint64_t>(tokens) * kKvHeads * kHeadDim * sizeof(uint16_t);
  if (io.k_prefill_native != io.k_cache_native) {
    fill_cache_views(k_compact, tokens, kKvHeads, io.k_cache_native,
                     io.k_cache_elements);
    s.prefill_k_cache_valid_copy_bytes += cache_valid_bytes;
  } else {
    ++s.prefill_k_cache_alias_calls;
  }
  // U126 consumes V in tile-major [head][tile][dim][key] form, then
  // converts the same BO in-place to the Decode cache format. Copying the
  // ordinary head-major cache here would overwrite the Prefill input.
  if (io.v_prefill_native != io.v_cache_native) {
    fill_cache_views(v_compact, tokens, kKvHeads, io.v_cache_native,
                     io.v_cache_elements);
    s.prefill_v_cache_valid_copy_bytes += cache_valid_bytes;
  }
  const auto pack_end = Clock::now();

  fpt_attention_metrics_t metrics{};
  status = fpt_attention_run_prefill_mapped(s.session, layer, tokens, &metrics);
  if (status != FPT_ATTENTION_OK) {
    report_error(std::string("run_prefill_mapped: ") +
                 fpt_attention_session_last_error(s.session));
    return false;
  }
  metrics.pack_ms = elapsed_ms(pack_begin, pack_end);
  const auto unpack_begin = Clock::now();
  if (!cache_only) {
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
    if (env_enabled("FPT_LLAMA_PL_DEBUG_RAW_OUTPUT", false) && layer == 0U) {
      const size_t show = std::min<size_t>(32U, io.output_elements);
      std::cerr << "FPT_LLAMA_PL_RAW_OUTPUT layer=0 words=";
      for (size_t i = 0; i < show; ++i) {
        if (i != 0U) std::cerr << ',';
        std::cerr << io.output_native[i];
      }
      std::cerr << '\n';
      for (uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (uint32_t context = 0; context < 8U; ++context) {
          const size_t source =
              (static_cast<size_t>(kv_head) * (s.bucket / fpt_prefill::kRowsPerBlock) *
                   fpt_prefill::kContextsPerBlock + context) * kHeadDim;
          std::cerr << "FPT_LLAMA_PL_RAW_ROW kv_head=" << kv_head
                    << " context=" << context << ':';
          for (uint32_t dim = 0; dim < 4U; ++dim) {
            if (dim != 0U) std::cerr << ',';
            std::cerr << io.output_native[source + dim];
          }
          std::cerr << '\n';
        }
      }
      std::cerr << "FPT_LLAMA_PL_OUTPUT_VALUES mode="
                << (s.direct_unpack ? "direct" : "explicit") << "=";
      for (uint32_t dim = 0; dim < 8U; ++dim) {
        if (dim != 0U) std::cerr << ',';
        std::cerr << load_value(dst, dim, 0, 0);
      }
      std::cerr << '\n';
    }
    scalar_prefill_shadow(q, k, v, dst->src[3], dst, tokens, layer);
  }
  const auto unpack_end = Clock::now();
  metrics.unpack_ms = elapsed_ms(unpack_begin, unpack_end);
  metrics.total_ms += metrics.pack_ms + metrics.unpack_ms;
  accumulate(cache_only ? &s.rebase : &s.prefill, metrics);
  if (cache_only) ++s.cache_rebase_calls;
  else ++s.prefill_calls;

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
  if (!cache_only && layer + 1U == kLayers) {
    s.seen_prefill = true;
    s.cache_stale = false;
    s.seen_decode = false;
    s.prompt_tokens = tokens;
    // The next decode must observe the committed prompt's KV prefix.
    s.last_decode_position = tokens;
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
  uint32_t position = 0U;
  if (!decode_mask_position(dst->src[3], static_cast<uint32_t>(k->ne[1]),
                            &position) || position == 0U || position >= kMaxKv) {
    return false;
  }
  if (s.cache_stale || position != s.last_decode_position) {
    // Reconstruct this layer's persistent KV13 cache from the CPU K/V prefix.
    if (!prefill_call(q, k, v, dst, layer, position)) return false;
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
  if (!s.seen_decode) {
    scalar_prefill_shadow(q, k, v, dst->src[3], dst, 1U, layer, true);
  }
  const auto unpack_end = Clock::now();
  metrics.unpack_ms = elapsed_ms(unpack_begin, unpack_end);
  metrics.total_ms += metrics.pack_ms + metrics.unpack_ms;
  accumulate(&s.decode, metrics);
  ++s.decode_calls;
  if (layer + 1U == kLayers) {
    s.last_decode_position = position + 1U;
    s.seen_decode = true;
    s.cache_stale = false;
  }
  return true;
}

void compare_shadow(const ggml_tensor *reference, const ggml_tensor *actual,
                    uint32_t layer, uint32_t q_tokens) {
  HookState &s = state();
  double local_sq = 0.0;
  double local_max = 0.0;
  uint64_t nonfinite = 0;
  uint64_t finite = 0;
  uint64_t ref_nonfinite = 0;
  uint64_t actual_nonfinite = 0;
  // FLASH_ATTN_EXT uses [head_dim, n_heads, n_tokens, batch].
  for (int64_t b = 0; b < actual->ne[3]; ++b) {
    for (int64_t token = 0; token < actual->ne[2]; ++token) {
      for (int64_t head = 0; head < actual->ne[1]; ++head) {
        for (int64_t d = 0; d < actual->ne[0]; ++d) {
          const float rv = load_output_value(reference, d, token, head, b);
          const float av = load_output_value(actual, d, token, head, b);
          if (!std::isfinite(rv) || !std::isfinite(av)) {
            ++nonfinite;
            if (!std::isfinite(rv)) ++ref_nonfinite;
            if (!std::isfinite(av)) ++actual_nonfinite;
            continue;
          }
          const double diff = static_cast<double>(av) - rv;
          local_max = std::max(local_max, std::abs(diff));
          local_sq += diff * diff;
          ++finite;
        }
      }
    }
  }
  ++s.shadow_compares;
  s.shadow_nonfinite += nonfinite;
  s.shadow_max_abs = std::max(s.shadow_max_abs, local_max);
  s.shadow_sq_sum += local_sq;
  s.shadow_elements += finite;
  if (env_enabled("FPT_LLAMA_PL_DEBUG_SHADOW", false)) {
    std::cerr << "FPT_LLAMA_PL_SHADOW layer=" << layer
              << " q_tokens=" << q_tokens << " max_abs=" << local_max
              << " finite=" << finite << " nonfinite=" << nonfinite
              << " ref_nonfinite=" << ref_nonfinite
              << " actual_nonfinite=" << actual_nonfinite
              << " rmse=" << (finite == 0 ? 0.0 :
                                  std::sqrt(local_sq / static_cast<double>(finite)))
              << '\n';
  }
  if (env_enabled("FPT_LLAMA_PL_DEBUG_VALUES", false)) {
    auto print_contract = [](const char *name, const ggml_tensor *tensor) {
      std::cerr << "FPT_LLAMA_PL_SHADOW_DST " << name
                << " type=" << tensor->type
                << " ne=" << tensor->ne[0] << ',' << tensor->ne[1] << ','
                << tensor->ne[2] << ',' << tensor->ne[3]
                << " nb=" << tensor->nb[0] << ',' << tensor->nb[1] << ','
                << tensor->nb[2] << ',' << tensor->nb[3] << '\n';
    };
    print_contract("REF", reference);
    print_contract("PL", actual);
    auto print_values = [](const char *name, const ggml_tensor *tensor) {
      std::cerr << "FPT_LLAMA_PL_VALUES " << name << "=";
      const uint32_t count = 8U;
      for (uint32_t i = 0; i < count; ++i) {
        const float value = load_output_value(tensor, i, 0, 0);
        std::cerr << (i == 0U ? "" : ",") << value;
      }
      std::cerr << '\n';
    };
    print_values("REF", reference);
    print_values("PL", actual);
    // ggml's FLASH_ATTN_EXT output is [head_dim, q_heads, tokens, batch].
    // The previous diagnostic assumed a head-major [head_dim, tokens, heads]
    // shape and read beyond ne[2], producing misleading garbage samples.
    const uint32_t valid_heads = static_cast<uint32_t>(
        std::min<int64_t>(actual->ne[1], 8));
    std::cerr << "FPT_LLAMA_PL_HEAD0=";
    for (uint32_t head = 0; head < valid_heads; ++head) {
      if (head != 0U) std::cerr << ",";
      std::cerr << load_output_value(reference, 0, 0, head) << "/"
                << load_output_value(actual, 0, 0, head);
    }
    std::cerr << '\n';
  }
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
  out << "SCHEMA=FPT-LLAMA32-1B-PS-PL-U126-v1\n"
      << "STATUS=" << (s.failure_calls == 0 && s.calls > 0 ? "PASS" : "FAIL")
      << '\n'
      << "PRECISION_VERIFIED=NO\n"
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
      << "PL_CACHE_REBASE_CALLS=" << s.cache_rebase_calls << '\n'
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
  write_metric(out, "CACHE_REBASE", s.rebase);
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
  out << "FPT_LLAMA_PL_REPORT="
      << (s.failure_calls == 0 && s.calls > 0 ? "PASS" : "FAIL") << '\n';
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
  const bool any_xclbin = !env_string("FPT_LLAMA_PL_XCLBIN").empty();
  if (!any_xclbin) return false;
  // llama-cli/server issue one- or two-token startup probes even with model
  // warmup disabled. They are not user prompts and must not initialize the
  // persistent 16-layer PL cache or turn later q=1 work into Decode.
  if (q->ne[1] <= 2 && !s.seen_prefill) return false;
  // K/V tensors expose the cache capacity, not the number of valid prompt
  // tokens. Use the causal mask to distinguish a complete prompt from a
  // partial chunk before committing its cache to the PL session.
  const uint32_t q_tokens = static_cast<uint32_t>(q->ne[1]);
  const uint32_t kv_extent = static_cast<uint32_t>(k->ne[1]);
  const ggml_tensor *mask = dst->src[3];
  if (env_enabled("FPT_LLAMA_PL_DEBUG_MASK", false)) {
    static std::atomic<uint32_t> printed_mask = 0U;
    if (printed_mask.fetch_add(1U, std::memory_order_relaxed) < 64U) {
      const uint32_t vf = valid_kv_tokens(mask, 0U, kv_extent);
      const uint32_t vl = valid_kv_tokens(mask, q_tokens - 1U, kv_extent);
      std::cerr << "FPT_LLAMA_PL_MASK q_tokens=" << q_tokens
                << " kv_extent=" << kv_extent
                << " valid_first=" << vf << " valid_last=" << vl;
      if (mask != nullptr) {
        std::cerr << " mask=" << mask->ne[0] << ',' << mask->ne[1]
                  << ',' << mask->ne[2] << ',' << mask->ne[3]
                  << ":type=" << mask->type;
      }
      std::cerr << '\n';
    }
  }
  const bool exact_prefill = q_tokens > 1U &&
      mask_covers_complete_prompt(mask, q_tokens, kv_extent);
  if (q_tokens > 1U && s.seen_prefill && !exact_prefill) {
    if (params->ith == 0) {
      std::lock_guard<std::mutex> guard(g_call_mutex);
      s.cache_stale = true;
    }
    return false;
  }
  const bool prefill_only = env_enabled("FPT_LLAMA_PL_PREFILL_ONLY", false);
  const bool decode_only = env_enabled("FPT_LLAMA_PL_DECODE_ONLY", false);
  const bool cpu_candidate = (exact_prefill && !decode_only) ||
      (q_tokens == 1U && !prefill_only);
  if (!cpu_candidate) return false;
  const uint32_t layer = static_cast<uint32_t>(dst->op_params[4]);
  if (layer >= kLayers) {
    if (params->ith == 0) {
      report_error("missing/invalid layer id in FLASH_ATTN_EXT op_params[4]");
    }
    return false;
  }
  // All workers must take the native path together: it contains barriers.
  const std::string max_pl_layer = env_string("FPT_LLAMA_PL_MAX_LAYER");
  if (!max_pl_layer.empty()) {
    char *end = nullptr;
    const long limit = std::strtol(max_pl_layer.c_str(), &end, 10);
    if (end != max_pl_layer.c_str() && *end == '\0' &&
        (limit <= 0 || layer >= static_cast<uint32_t>(limit))) {
      return false;
    }
  }
  if (params->ith != 0) return true;
  std::lock_guard<std::mutex> guard(g_call_mutex);
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
  const bool shadow_this_layer =
      s.shadow && cpu_reference != nullptr && shadow_layer_enabled(layer);
  if (shadow_this_layer) {
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
    report_error("recognized Llama shape could not be handled by U126");
    return false;
  }
  ++s.calls;

  if (shadow_this_layer && !reference_bytes.empty()) {
    ggml_tensor ref = *dst;
    ref.data = reference_bytes.data();
    compare_shadow(&ref, dst, layer, static_cast<uint32_t>(q->ne[1]));
  }
  return true;
}
