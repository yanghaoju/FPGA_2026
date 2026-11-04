#ifndef FPT_UNIFIED_PREFILL_DECODE_IP_HPP
#define FPT_UNIFIED_PREFILL_DECODE_IP_HPP

#include <ap_int.h>

#include "decode_core.hpp"
#include "prefill_core.hpp"

static const unsigned FPT_MODE_PREFILL = 0;
static const unsigned FPT_MODE_DECODE = 1;

using unified_mem_word_t = ap_uint<ATTENTION_IP_MEM_WORD_BITS>;

static_assert(ATTENTION_IP_MEM_WORD_BITS == DECODE_MEM_WORD_BITS,
              "Prefill and Decode must use the same external word width");
static_assert(ATTENTION_IP_HEAD_DIM == DECODE_HEAD_DIM,
              "Prefill and Decode head dimensions must match");
static_assert(ATTENTION_IP_Q_HEADS == DECODE_Q_HEADS,
              "Prefill and Decode query-head counts must match");
static_assert(ATTENTION_IP_KV_HEADS == DECODE_KV_HEADS,
              "Prefill and Decode KV-head counts must match");

static const int DECODE_GMEM0_Q_OFFSET = 0;
static const int DECODE_GMEM0_K_NEW_OFFSET = DECODE_Q_WORDS;
static const int DECODE_GMEM0_V_NEW_OFFSET = DECODE_Q_WORDS + DECODE_KV_WORDS;
static const int DECODE_GMEM0_WORDS = DECODE_Q_WORDS + 2 * DECODE_KV_WORDS;

static_assert(DECODE_GMEM0_WORDS <= ATTENTION_IP_LOGICAL_Q_WORDS,
              "gmem0 must hold Decode Q, Knew, and Vnew");
static_assert(DECODE_CACHE_WORDS <= ATTENTION_IP_STORAGE_KV_WORDS,
              "gmem1/gmem2 must hold the Decode K/V cache");
static_assert(DECODE_Q_WORDS <= ATTENTION_IP_LOGICAL_Q_WORDS,
              "gmem3 must hold the Decode output");

extern "C" void int8_gqa_prefill_decode_ip(
    unsigned mode,
    const unified_mem_word_t gmem0[ATTENTION_IP_LOGICAL_Q_WORDS],
    unified_mem_word_t gmem1[ATTENTION_IP_STORAGE_KV_WORDS],
    unified_mem_word_t gmem2[ATTENTION_IP_STORAGE_KV_WORDS],
    unified_mem_word_t gmem3[ATTENTION_IP_LOGICAL_Q_WORDS],
    // Prefill: number of valid prompt tokens. Decode: appended token index.
    unsigned decode_position);

#endif
