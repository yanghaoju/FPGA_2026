#ifndef FPT_DECODE_Q14_KV13_ROW_IP_PROFILE_HPP
#define FPT_DECODE_Q14_KV13_ROW_IP_PROFILE_HPP

#define DECODE_Q_HEADS 32
#define DECODE_KV_HEADS 8
#define DECODE_GQA_GROUP 4
#define DECODE_HEAD_DIM 64
#define DECODE_MAX_KV_TOKENS 512

// U78 uses signed INT14 for transient Q and signed INT13 for cached K/V.
// Each operand uses one scale per token/head row. QK performs a full D64
// integer reduction before applying the Q*K scale once per score.
#define DECODE_TILE_K 64
#define DECODE_QK_PAR 8
#define DECODE_PV_PAR 8
#define DECODE_GROUP_D 64
#define DECODE_GROUPS_PER_ROW (DECODE_HEAD_DIM / DECODE_GROUP_D)
#define DECODE_SCHEDULED_Q_LANES 1
#define DECODE_ACCUM_SLOTS 4
#define DECODE_TILE_BUFFERS 2
#define DECODE_TILE_PLANES 2
#define DECODE_COLOCATED_TILE_K (DECODE_TILE_PLANES * DECODE_TILE_K)
#define DECODE_V_TILE_OFFSET DECODE_TILE_K

#define DECODE_MEM_WORD_BITS 512
// Q, Knew, Vnew, and output retain the U46 BF16 ABI.
#define DECODE_VALUES_PER_WORD 32
#define DECODE_WORDS_PER_ROW 2
#define DECODE_Q_QUANT_BITS 14
#define DECODE_Q_QUANT_LIMIT 8191
#define DECODE_KV_QUANT_BITS 13
#define DECODE_KV_QUANT_LIMIT 4095
#define DECODE_QUANT_VALUES_PER_CACHE_WORD 32
#define DECODE_CACHE_WORDS_PER_ROW 2
#define DECODE_SCALE_LOW_BIT \
  (DECODE_QUANT_VALUES_PER_CACHE_WORD * DECODE_KV_QUANT_BITS)
#define DECODE_TILE_WORD_BITS 104
#define DECODE_VALUES_PER_TILE_WORD 8
#define DECODE_TILE_CHUNKS_PER_CACHE_WORD 4
#define DECODE_TILE_CHUNKS_PER_ROW \
  (DECODE_HEAD_DIM / DECODE_VALUES_PER_TILE_WORD)
#define DECODE_Q_WORDS (DECODE_Q_HEADS * DECODE_WORDS_PER_ROW)
#define DECODE_KV_WORDS (DECODE_KV_HEADS * DECODE_WORDS_PER_ROW)

// Each row uses two words. The low 416 bits of each word hold 32 INT13
// elements; bits [431:416] of word 0 hold the BF16 row scale. This keeps the
// U46 8192-word cache ABI and uses only compile-time bit positions.
#define DECODE_CACHE_DATA_WORDS \
  (DECODE_KV_HEADS * DECODE_MAX_KV_TOKENS * DECODE_CACHE_WORDS_PER_ROW)
#define DECODE_K_CACHE_WORDS DECODE_CACHE_DATA_WORDS
#define DECODE_V_CACHE_WORDS DECODE_CACHE_DATA_WORDS
#define DECODE_CACHE_WORDS DECODE_K_CACHE_WORDS

static_assert(DECODE_Q_HEADS == DECODE_KV_HEADS * DECODE_GQA_GROUP,
              "GQA ratio mismatch");
static_assert(DECODE_GQA_GROUP % DECODE_SCHEDULED_Q_LANES == 0,
              "scheduled query lanes must divide the GQA group");
static_assert(DECODE_HEAD_DIM % DECODE_QK_PAR == 0,
              "QK parallelism must divide head dimension");
static_assert(DECODE_HEAD_DIM % DECODE_PV_PAR == 0,
              "PV parallelism must divide head dimension");
static_assert(DECODE_HEAD_DIM % DECODE_GROUP_D == 0,
              "quantization group must divide head dimension");
static_assert(DECODE_MAX_KV_TOKENS % DECODE_TILE_K == 0,
              "cache extent must tile exactly");
static_assert(DECODE_TILE_K == 64,
              "U60A is specialized for a 64-token KV tile");
static_assert(DECODE_TILE_CHUNKS_PER_ROW == DECODE_QK_PAR,
              "D64 QK reduction requires eight D8 chunks");
static_assert(DECODE_TILE_WORD_BITS ==
                  DECODE_VALUES_PER_TILE_WORD * DECODE_KV_QUANT_BITS,
              "tile word must contain eight INT13 values");
static_assert(DECODE_TILE_CHUNKS_PER_CACHE_WORD * DECODE_TILE_WORD_BITS ==
                  DECODE_SCALE_LOW_BIT,
              "four INT13 chunks must occupy the low 416 cache bits");
static_assert(DECODE_SCALE_LOW_BIT + 16 <= DECODE_MEM_WORD_BITS,
              "embedded BF16 scale must fit cache word 0");
static_assert(DECODE_K_CACHE_WORDS <= 8192 && DECODE_V_CACHE_WORDS <= 8192,
              "INT13 caches must fit the shared U46 AXI buffers");
static_assert(DECODE_ACCUM_SLOTS == 4,
              "D8-R17 slot reduction is specialized for four slots");
static_assert(DECODE_TILE_BUFFERS == 2,
              "U78 retains the U76 load/compute ping-pong schedule");
static_assert(DECODE_COLOCATED_TILE_K == 128,
              "U78 stores K and V in disjoint halves of one tile RAM");
static_assert(DECODE_V_TILE_OFFSET == 64,
              "U78 V tile offset must follow the complete K tile");

#endif
