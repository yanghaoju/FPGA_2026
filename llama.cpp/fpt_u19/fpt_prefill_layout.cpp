#include "fpt_prefill_internal.hpp"

#include <algorithm>
#include <array>

#if defined(__aarch64__) && __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define FPT_PREFILL_NEON_AVAILABLE 1
#else
#define FPT_PREFILL_NEON_AVAILABLE 0
#endif

namespace {

constexpr std::array<fpt_prefill_profile_t, 3> kProfiles = {{
    {"gqa_32q_8kv_d64_s64_llama32_1b",
     "fpt-v8s-d64-s64-200mhz-ef1a414a",
     "ef1a414a32740f0c9011c7c667c41199e851f4f171adc6a9897023379c9421d1",
     64, 262144, 65536, 65536, 262144, 2.52221057, 2.53471615},
    {"gqa_32q_8kv_d64_s128_llama32_1b",
     "fpt-v8s-d64-s128-200mhz-4cec6a45",
     "4cec6a456a0ee377b992b3f68729de586fc9df7de3f2fc77bb7edbe1ee1d1d50",
     128, 524288, 131072, 131072, 524288, 7.26415687, 7.27776820},
    {"gqa_32q_8kv_d64_s256_llama32_1b",
     "fpt-v8s-d64-s256-200mhz-b9e58175",
     "b9e58175e611290d672f5e9aa4053cc999ac0000140b0fd8f1d73462e728e5c7",
     256, 1048576, 262144, 262144, 1048576, 23.56229714, 23.57517957},
}};

std::size_t tensor_index(std::size_t token, std::size_t head,
                         std::size_t dim, std::size_t heads,
                         std::size_t valid_tokens,
                         fpt_prefill_layout_t layout) {
  if (layout == FPT_PREFILL_TOKEN_HEAD_DIM) {
    return (token * heads + head) * FPT_PREFILL_HEAD_DIM + dim;
  }
  return (head * valid_tokens + token) * FPT_PREFILL_HEAD_DIM + dim;
}

std::size_t packed_q_index(std::size_t kv_head, std::size_t query_block,
                           std::size_t context, std::size_t dim,
                           std::size_t bucket) {
  const std::size_t blocks = bucket / fpt_prefill::kRowsPerBlock;
  return (((kv_head * blocks + query_block) *
           fpt_prefill::kContextsPerBlock + context) *
          FPT_PREFILL_HEAD_DIM) + dim;
}

#if FPT_PREFILL_NEON_AVAILABLE
inline void neon_transpose_store_8x8(const std::uint16_t *source,
                                     std::size_t source_stride,
                                     std::uint16_t *destination) {
  const uint16x8_t r0 = vld1q_u16(source + 0U * source_stride);
  const uint16x8_t r1 = vld1q_u16(source + 1U * source_stride);
  const uint16x8_t r2 = vld1q_u16(source + 2U * source_stride);
  const uint16x8_t r3 = vld1q_u16(source + 3U * source_stride);
  const uint16x8_t r4 = vld1q_u16(source + 4U * source_stride);
  const uint16x8_t r5 = vld1q_u16(source + 5U * source_stride);
  const uint16x8_t r6 = vld1q_u16(source + 6U * source_stride);
  const uint16x8_t r7 = vld1q_u16(source + 7U * source_stride);
  const uint16x8_t a0 = vtrn1q_u16(r0, r1);
  const uint16x8_t a1 = vtrn2q_u16(r0, r1);
  const uint16x8_t b0 = vtrn1q_u16(r2, r3);
  const uint16x8_t b1 = vtrn2q_u16(r2, r3);
  const uint16x8_t c0 = vtrn1q_u16(r4, r5);
  const uint16x8_t c1 = vtrn2q_u16(r4, r5);
  const uint16x8_t d0 = vtrn1q_u16(r6, r7);
  const uint16x8_t d1 = vtrn2q_u16(r6, r7);
  const uint32x4_t ab0 = vtrn1q_u32(vreinterpretq_u32_u16(a0),
                                    vreinterpretq_u32_u16(b0));
  const uint32x4_t ab1 = vtrn2q_u32(vreinterpretq_u32_u16(a0),
                                    vreinterpretq_u32_u16(b0));
  const uint32x4_t ab2 = vtrn1q_u32(vreinterpretq_u32_u16(a1),
                                    vreinterpretq_u32_u16(b1));
  const uint32x4_t ab3 = vtrn2q_u32(vreinterpretq_u32_u16(a1),
                                    vreinterpretq_u32_u16(b1));
  const uint32x4_t cd0 = vtrn1q_u32(vreinterpretq_u32_u16(c0),
                                    vreinterpretq_u32_u16(d0));
  const uint32x4_t cd1 = vtrn2q_u32(vreinterpretq_u32_u16(c0),
                                    vreinterpretq_u32_u16(d0));
  const uint32x4_t cd2 = vtrn1q_u32(vreinterpretq_u32_u16(c1),
                                    vreinterpretq_u32_u16(d1));
  const uint32x4_t cd3 = vtrn2q_u32(vreinterpretq_u32_u16(c1),
                                    vreinterpretq_u32_u16(d1));
  const uint64x2_t out0 = vtrn1q_u64(vreinterpretq_u64_u32(ab0),
                                     vreinterpretq_u64_u32(cd0));
  const uint64x2_t out4 = vtrn2q_u64(vreinterpretq_u64_u32(ab0),
                                     vreinterpretq_u64_u32(cd0));
  const uint64x2_t out2 = vtrn1q_u64(vreinterpretq_u64_u32(ab1),
                                     vreinterpretq_u64_u32(cd1));
  const uint64x2_t out6 = vtrn2q_u64(vreinterpretq_u64_u32(ab1),
                                     vreinterpretq_u64_u32(cd1));
  const uint64x2_t out1 = vtrn1q_u64(vreinterpretq_u64_u32(ab2),
                                     vreinterpretq_u64_u32(cd2));
  const uint64x2_t out5 = vtrn2q_u64(vreinterpretq_u64_u32(ab2),
                                     vreinterpretq_u64_u32(cd2));
  const uint64x2_t out3 = vtrn1q_u64(vreinterpretq_u64_u32(ab3),
                                     vreinterpretq_u64_u32(cd3));
  const uint64x2_t out7 = vtrn2q_u64(vreinterpretq_u64_u32(ab3),
                                     vreinterpretq_u64_u32(cd3));
  vst1q_u16(destination + 0U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out0));
  vst1q_u16(destination + 1U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out1));
  vst1q_u16(destination + 2U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out2));
  vst1q_u16(destination + 3U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out3));
  vst1q_u16(destination + 4U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out4));
  vst1q_u16(destination + 5U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out5));
  vst1q_u16(destination + 6U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out6));
  vst1q_u16(destination + 7U * fpt_prefill::kTileK,
            vreinterpretq_u16_u64(out7));
}
#endif
}  // namespace

extern "C" const fpt_prefill_profile_t *fpt_prefill_select_profile(
    uint32_t valid_tokens) {
  if (valid_tokens == 0) {
    return nullptr;
  }
  for (const auto &profile : kProfiles) {
    if (valid_tokens <= profile.sequence) {
      return &profile;
    }
  }
  return nullptr;
}

extern "C" size_t fpt_prefill_q_elements(uint32_t valid_tokens) {
  if (valid_tokens == 0 || valid_tokens > 256) {
    return 0;
  }
  return static_cast<size_t>(valid_tokens) * FPT_PREFILL_Q_HEADS *
         FPT_PREFILL_HEAD_DIM;
}

extern "C" size_t fpt_prefill_kv_elements(uint32_t valid_tokens) {
  if (valid_tokens == 0 || valid_tokens > 256) {
    return 0;
  }
  return static_cast<size_t>(valid_tokens) * FPT_PREFILL_KV_HEADS *
         FPT_PREFILL_HEAD_DIM;
}

namespace fpt_prefill {

bool valid_layout(fpt_prefill_layout_t layout) {
  return layout == FPT_PREFILL_TOKEN_HEAD_DIM ||
         layout == FPT_PREFILL_HEAD_TOKEN_DIM;
}

void pack_q_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed) {
  const std::size_t packed_elements =
      FPT_PREFILL_Q_HEADS * bucket * FPT_PREFILL_HEAD_DIM;
  if (valid_tokens < bucket) {
    std::fill(packed, packed + packed_elements, std::uint16_t{0});
  }
  const std::size_t blocks = bucket / kRowsPerBlock;
  if (layout == FPT_PREFILL_TOKEN_HEAD_DIM) {
    constexpr std::size_t kGroupElements =
        kGqaGroup * FPT_PREFILL_HEAD_DIM;
    for (std::size_t token = 0; token < valid_tokens; ++token) {
      const std::size_t block = token / kRowsPerBlock;
      const std::size_t row = token % kRowsPerBlock;
      for (std::size_t kv_head = 0; kv_head < FPT_PREFILL_KV_HEADS;
           ++kv_head) {
        const std::size_t source =
            (token * FPT_PREFILL_Q_HEADS + kv_head * kGqaGroup) *
            FPT_PREFILL_HEAD_DIM;
        const std::size_t destination =
            packed_q_index(kv_head, block, row * kGqaGroup, 0, bucket);
        std::copy_n(input + source, kGroupElements, packed + destination);
      }
    }
    return;
  }
  for (std::size_t kv_head = 0; kv_head < FPT_PREFILL_KV_HEADS; ++kv_head) {
    for (std::size_t block = 0; block < blocks; ++block) {
      for (std::size_t context = 0; context < kContextsPerBlock; ++context) {
        const std::size_t token = block * kRowsPerBlock + context / kGqaGroup;
        const std::size_t head = kv_head * kGqaGroup + context % kGqaGroup;
        if (token >= valid_tokens) {
          continue;
        }
        const std::size_t source =
            tensor_index(token, head, 0, FPT_PREFILL_Q_HEADS,
                         valid_tokens, layout);
        const std::size_t destination =
            packed_q_index(kv_head, block, context, 0, bucket);
        std::copy_n(input + source, FPT_PREFILL_HEAD_DIM,
                    packed + destination);
      }
    }
  }
}

void pack_k_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed) {
  const std::size_t packed_elements =
      FPT_PREFILL_KV_HEADS * bucket * FPT_PREFILL_HEAD_DIM;
  if (valid_tokens < bucket) {
    std::fill(packed, packed + packed_elements, std::uint16_t{0});
  }
  for (std::size_t head = 0; head < FPT_PREFILL_KV_HEADS; ++head) {
    for (std::size_t token = 0; token < valid_tokens; ++token) {
      const std::size_t source =
          tensor_index(token, head, 0, FPT_PREFILL_KV_HEADS,
                       valid_tokens, layout);
      const std::size_t destination =
          (head * bucket + token) * FPT_PREFILL_HEAD_DIM;
      std::copy_n(input + source, FPT_PREFILL_HEAD_DIM,
                  packed + destination);
    }
  }
}

void pack_v_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed) {
  const std::size_t tiles = bucket / kTileK;
  const std::size_t packed_elements =
      FPT_PREFILL_KV_HEADS * bucket * FPT_PREFILL_HEAD_DIM;
  if (valid_tokens < bucket) {
    std::fill(packed, packed + packed_elements, std::uint16_t{0});
  }
  constexpr std::size_t kTransposeBlock = 8;
#if FPT_PREFILL_NEON_AVAILABLE
  const std::size_t source_stride =
      layout == FPT_PREFILL_TOKEN_HEAD_DIM
          ? FPT_PREFILL_KV_HEADS * FPT_PREFILL_HEAD_DIM
          : FPT_PREFILL_HEAD_DIM;
#endif
  std::array<std::array<std::uint16_t, kTransposeBlock>, kTransposeBlock>
      block{};
  for (std::size_t head = 0; head < FPT_PREFILL_KV_HEADS; ++head) {
    for (std::size_t tile = 0; tile < tiles; ++tile) {
      for (std::size_t dim_base = 0; dim_base < FPT_PREFILL_HEAD_DIM;
           dim_base += kTransposeBlock) {
        for (std::size_t key_base = 0; key_base < kTileK;
             key_base += kTransposeBlock) {
          const std::size_t token_base = tile * kTileK + key_base;
          const std::size_t valid_keys =
              token_base < valid_tokens
                  ? std::min(kTransposeBlock, valid_tokens - token_base)
                  : 0;
#if FPT_PREFILL_NEON_AVAILABLE
          if (valid_keys == kTransposeBlock) {
            const std::size_t source =
                tensor_index(token_base, head, dim_base, FPT_PREFILL_KV_HEADS,
                             valid_tokens, layout);
            const std::size_t destination =
                (((head * tiles + tile) * FPT_PREFILL_HEAD_DIM + dim_base) *
                 kTileK) +
                key_base;
            neon_transpose_store_8x8(input + source, source_stride,
                                     packed + destination);
            continue;
          }
#endif
          for (std::size_t key = 0; key < valid_keys; ++key) {
            const std::size_t source =
                tensor_index(token_base + key, head, dim_base,
                             FPT_PREFILL_KV_HEADS, valid_tokens, layout);
            std::copy_n(input + source, kTransposeBlock,
                        block[key].begin());
          }
          for (std::size_t dim = 0; dim < kTransposeBlock; ++dim) {
            const std::size_t destination =
                (((head * tiles + tile) * FPT_PREFILL_HEAD_DIM +
                  dim_base + dim) *
                 kTileK) +
                key_base;
            for (std::size_t key = 0; key < valid_keys; ++key) {
              packed[destination + key] = block[key][dim];
            }
          }
        }
      }
    }
  }
}
void unpack_out(const std::uint16_t *packed, std::size_t valid_tokens,
                std::size_t bucket, fpt_prefill_layout_t layout,
                std::uint16_t *output) {
  if (layout == FPT_PREFILL_TOKEN_HEAD_DIM) {
    constexpr std::size_t kGroupElements =
        kGqaGroup * FPT_PREFILL_HEAD_DIM;
    for (std::size_t token = 0; token < valid_tokens; ++token) {
      const std::size_t block = token / kRowsPerBlock;
      const std::size_t row = token % kRowsPerBlock;
      for (std::size_t kv_head = 0; kv_head < FPT_PREFILL_KV_HEADS;
           ++kv_head) {
        const std::size_t source =
            packed_q_index(kv_head, block, row * kGqaGroup, 0, bucket);
        const std::size_t destination =
            (token * FPT_PREFILL_Q_HEADS + kv_head * kGqaGroup) *
            FPT_PREFILL_HEAD_DIM;
        std::copy_n(packed + source, kGroupElements,
                    output + destination);
      }
    }
    return;
  }
  for (std::size_t token = 0; token < valid_tokens; ++token) {
    const std::size_t block = token / kRowsPerBlock;
    const std::size_t row = token % kRowsPerBlock;
    for (std::size_t head = 0; head < FPT_PREFILL_Q_HEADS; ++head) {
      const std::size_t kv_head = head / kGqaGroup;
      const std::size_t context = row * kGqaGroup + head % kGqaGroup;
      const std::size_t source =
          packed_q_index(kv_head, block, context, 0, bucket);
      const std::size_t destination =
          tensor_index(token, head, 0, FPT_PREFILL_Q_HEADS,
                       valid_tokens, layout);
      std::copy_n(packed + source, FPT_PREFILL_HEAD_DIM,
                  output + destination);
    }
  }
}

}  // namespace fpt_prefill
