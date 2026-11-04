#ifndef FPT_EXACT_GQA_ATTENTION_IP_HPP
#define FPT_EXACT_GQA_ATTENTION_IP_HPP

#include <ap_int.h>
#include "attention_ip_profile.hpp"

static const int BATCH = 1;
static const int Q_HEADS = ATTENTION_IP_Q_HEADS;
static const int KV_HEADS = ATTENTION_IP_KV_HEADS;
static const int GQA_GROUP = ATTENTION_IP_GQA_GROUP;
// SEQ is the physical compiled extent; LOGICAL_SEQ is the model contract.
static const int LOGICAL_SEQ = ATTENTION_IP_LOGICAL_SEQ;
static const int SEQ = ATTENTION_IP_LOGICAL_SEQ;
static const int STORAGE_SEQ = ATTENTION_IP_STORAGE_SEQ;
static const int HEAD_DIM = ATTENTION_IP_HEAD_DIM;
static const int TILE_K = ATTENTION_IP_TILE_K;
static const int TILES_PER_KV_HEAD = ATTENTION_IP_STATIC_FRONTIERS;
static const int MEM_WORD_BITS = ATTENTION_IP_MEM_WORD_BITS;
static const int BF16_BITS = 16;
static const int VALUES_PER_WORD = ATTENTION_IP_VALUES_PER_WORD;
static const int WORDS_PER_ROW = ATTENTION_IP_WORDS_PER_ROW;
static const int ROWS_PER_BLOCK = ATTENTION_IP_ROWS_PER_BLOCK;
static const int CONTEXTS_PER_BLOCK = ROWS_PER_BLOCK * GQA_GROUP;
static const int STATIC_GQA_LANES = ATTENTION_IP_STATIC_GQA_LANES;
static const int GROUPS_PER_BLOCK = CONTEXTS_PER_BLOCK / STATIC_GQA_LANES;
static const int QK_PAR_D = ATTENTION_IP_QK_PAR_D;
static const int QK_CHUNKS = HEAD_DIM / QK_PAR_D;
static const int PV_PAR_K = ATTENTION_IP_PV_PAR_K;
static const int BLOCKS_PER_KV_HEAD = SEQ / ROWS_PER_BLOCK;
static const int LOGICAL_BLOCKS_PER_KV_HEAD = LOGICAL_SEQ / ROWS_PER_BLOCK;
static const int NORM_PAR_D = ATTENTION_IP_NORM_PAR_D;
static const int FRONTIER_GROUPS = TILES_PER_KV_HEAD;
static const int BLOCKS_PER_FRONTIER = BLOCKS_PER_KV_HEAD / FRONTIER_GROUPS;
static const int QK_STEPS_PER_PHASE = GROUPS_PER_BLOCK * QK_CHUNKS * TILE_K;
static const int PV_STEPS_PER_PHASE = WORDS_PER_ROW * VALUES_PER_WORD * GROUPS_PER_BLOCK;
static const int DOT_STEPS_PER_PHASE = QK_STEPS_PER_PHASE;
static const int TASKS_PER_KV_HEAD = 0;

static_assert(Q_HEADS == KV_HEADS * GQA_GROUP, "GQA ratio mismatch");
static_assert(STORAGE_SEQ % TILE_K == 0, "physical sequence must tile exactly");
static_assert(LOGICAL_SEQ <= SEQ, "logical sequence exceeds physical storage");
static_assert(LOGICAL_SEQ % ROWS_PER_BLOCK == 0, "logical query blocks must be integral");
static_assert(BLOCKS_PER_KV_HEAD % FRONTIER_GROUPS == 0,
              "frontier groups must partition physical query blocks");
static_assert(HEAD_DIM % QK_PAR_D == 0, "QK lanes must divide head_dim");
static_assert(TILE_K == PV_PAR_K, "PV consumes one full key tile");
static_assert(QK_STEPS_PER_PHASE == PV_STEPS_PER_PHASE,
              "QK and PV phases must have identical fixed trip counts");

// U126 keeps the BF16 Prefill datapath and emits the row-scaled signed-INT13
// cache layout in-place for Decode.
typedef ap_uint<16> bf16_t;
typedef ap_uint<MEM_WORD_BITS> mem_word_t;

extern "C" void fpt_prefill_p2_core(
    const mem_word_t q[ATTENTION_IP_LOGICAL_Q_WORDS],
    mem_word_t k[ATTENTION_IP_STORAGE_KV_WORDS],
    mem_word_t v[ATTENTION_IP_STORAGE_KV_WORDS],
    mem_word_t out[ATTENTION_IP_LOGICAL_Q_WORDS],
    unsigned prompt_tokens
);

#ifndef __SYNTHESIS__
void fpt_test_pack_prompt_head_kv13(
    mem_word_t k[ATTENTION_IP_STORAGE_KV_WORDS],
    mem_word_t v[ATTENTION_IP_STORAGE_KV_WORDS],
    int kv_head,
    unsigned prompt_tokens,
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]);
#endif

#endif
