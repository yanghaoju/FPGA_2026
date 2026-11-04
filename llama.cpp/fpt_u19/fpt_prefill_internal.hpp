#ifndef FPT_PREFILL_INTERNAL_HPP
#define FPT_PREFILL_INTERNAL_HPP

#include "fpt_prefill.h"

#include <cstddef>
#include <cstdint>

namespace fpt_prefill {

constexpr std::size_t kRowsPerBlock = 2;
constexpr std::size_t kGqaGroup = 4;
constexpr std::size_t kContextsPerBlock = kRowsPerBlock * kGqaGroup;
constexpr std::size_t kTileK = 32;

bool valid_layout(fpt_prefill_layout_t layout);

void pack_q_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed);

void pack_k_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed);

void pack_v_into(const std::uint16_t *input, std::size_t valid_tokens,
                 std::size_t bucket, fpt_prefill_layout_t layout,
                 std::uint16_t *packed);

void unpack_out(const std::uint16_t *packed, std::size_t valid_tokens,
                std::size_t bucket, fpt_prefill_layout_t layout,
                std::uint16_t *output);

}  // namespace fpt_prefill

#endif
