#ifndef FPT_DECODE_Q14_KV13_ROW_IP_HPP
#define FPT_DECODE_Q14_KV13_ROW_IP_HPP

#include <ap_int.h>

#include "decode_ip_profile.hpp"

using decode_mem_word_t = ap_uint<DECODE_MEM_WORD_BITS>;
using decode_tile_word_t = ap_uint<DECODE_TILE_WORD_BITS>;

extern "C" void fpt_decode_q14_kv13_row_core(
    const decode_mem_word_t q_current[DECODE_Q_WORDS],
    decode_mem_word_t k_cache[DECODE_CACHE_WORDS],
    decode_mem_word_t v_cache[DECODE_CACHE_WORDS],
    const decode_mem_word_t k_new[DECODE_KV_WORDS],
    const decode_mem_word_t v_new[DECODE_KV_WORDS],
    decode_mem_word_t output[DECODE_Q_WORDS],
    unsigned position);

#endif
