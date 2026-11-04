#include "prefill_core.hpp"
#include "decode_ip_profile.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <hls_math.h>
#include <hls_stream.h>

static float bf16_to_float(bf16_t value) {
#pragma HLS INLINE
    uint32_t bits = static_cast<uint32_t>(value.to_uint()) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static bf16_t float_to_bf16_rne(float value) {
#pragma HLS INLINE
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return bf16_t(bits >> 16);
}

static int block_io_word_index(
    int kv_head,
    int query_block,
    int context,
    int word
) {
#pragma HLS INLINE
    return ((kv_head * BLOCKS_PER_KV_HEAD + query_block)
            * CONTEXTS_PER_BLOCK + context) * WORDS_PER_ROW + word;
}

static int kv_word_index(int head, int sequence, int word) {
#pragma HLS INLINE
    return (head * STORAGE_SEQ + sequence) * WORDS_PER_ROW + word;
}

static float reduce32(const float value[32]) {
#pragma HLS INLINE
    float l0[16];
    float l1[8];
    float l2[4];
    float l3[2];
#pragma HLS ARRAY_PARTITION variable=l0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l3 complete dim=1
V8S_IP_REDUCE32_L0:
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        l0[i] = value[2 * i] + value[2 * i + 1];
    }
V8S_IP_REDUCE32_L1:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        l1[i] = l0[2 * i] + l0[2 * i + 1];
    }
V8S_IP_REDUCE32_L2:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
V8S_IP_REDUCE32_L3:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l3[i] = l2[2 * i] + l2[2 * i + 1];
    }
    return l3[0] + l3[1];
}

static uint32_t ordered_max_bits(uint32_t a, uint32_t b) {
#pragma HLS INLINE
    const bool a_negative = (a & 0x80000000U) != 0U;
    const bool b_negative = (b & 0x80000000U) != 0U;
    const bool choose_a = a_negative != b_negative
        ? !a_negative
        : (((a & 0x7FFFFFFFU) > (b & 0x7FFFFFFFU)) ^ a_negative);
    return choose_a ? a : b;
}

static void shared_dual_dot32(
    const float left0[QK_PAR_D],
    const float left1[QK_PAR_D],
    const float right[QK_PAR_D],
    float &sum0,
    float &sum1
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=left0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=right complete dim=1
    float product0[QK_PAR_D];
    float product1[QK_PAR_D];
#pragma HLS ARRAY_PARTITION variable=product0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product1 complete dim=1
#pragma HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6
V8S_IP_SHARED_DOT_PRODUCTS:
    for (int item = 0; item < QK_PAR_D; ++item) {
#pragma HLS UNROLL
        product0[item] = left0[item] * right[item];
        product1[item] = left1[item] * right[item];
    }
    sum0 = reduce32(product0);
    sum1 = reduce32(product1);
}

static float reduce8(const float value[8]) {
#pragma HLS INLINE
    float l0[4];
    float l1[2];
#pragma HLS ARRAY_PARTITION variable=l0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
V8S_IP_REDUCE8_L0:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l0[i] = value[2 * i] + value[2 * i + 1];
    }
V8S_IP_REDUCE8_L1:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l1[i] = l0[2 * i] + l0[2 * i + 1];
    }
    return l1[0] + l1[1];
}

// The producer remains faster than the PV consumer, so the two lanes can use
// one physical reduction tree without changing the dataflow interval.
static void shared_reduce8(
    const float value[8],
    float &sum
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=2
#pragma HLS ALLOCATION operation instances=fadd limit=4
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
    sum = reduce8(value);
}

static void shared_reduce4(
    const float value[4],
    float &sum
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=3
#pragma HLS ALLOCATION operation instances=fadd limit=1
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
    sum = (value[0] + value[1]) + (value[2] + value[3]);
}

typedef ap_uint<64> packed_weight_pair_t;
typedef ap_uint<64> packed_score_pair_t;
typedef ap_uint<128> packed_q_bank_t;
typedef ap_uint<32> packed_weight_single_t;
static const int TRIPLE_BATCHES = (CONTEXTS_PER_BLOCK + 2) / 3;
static const int TRIPLE_BANK_DEPTH = (TRIPLE_BATCHES + 1) / 2;
typedef ap_uint<96> packed_weight_triple_t;

static void local_dual_dot8_bank0(
    const packed_weight_pair_t left01[8],
    const packed_weight_pair_t left23[8],
    const float right[8],
    float &sum0,
    float &sum1,
    float &sum2,
    float &sum3
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=left01 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left23 complete dim=1
#pragma HLS ARRAY_PARTITION variable=right complete dim=1
    float product0[8];
    float product1[8];
    float product2[8];
    float product3[8];
#pragma HLS ARRAY_PARTITION variable=product0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product3 complete dim=1
#pragma HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product2 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product3 op=mul impl=fulldsp latency=6
V8S_IP_LOCAL_DOT8_BANK0_PRODUCTS:
    for (int item = 0; item < 8; ++item) {
#pragma HLS UNROLL
        const packed_weight_pair_t packed01 = left01[item];
        const packed_weight_pair_t packed23 = left23[item];
        const uint32_t left0_bits = static_cast<uint32_t>(packed01.range(31, 0));
        const uint32_t left1_bits = static_cast<uint32_t>(packed01.range(63, 32));
        const uint32_t left2_bits = static_cast<uint32_t>(packed23.range(31, 0));
        const uint32_t left3_bits = static_cast<uint32_t>(packed23.range(63, 32));
        float left0;
        float left1;
        float left2;
        float left3;
        std::memcpy(&left0, &left0_bits, sizeof(left0));
        std::memcpy(&left1, &left1_bits, sizeof(left1));
        std::memcpy(&left2, &left2_bits, sizeof(left2));
        std::memcpy(&left3, &left3_bits, sizeof(left3));
        product0[item] = left0 * right[item];
        product1[item] = left1 * right[item];
        product2[item] = left2 * right[item];
        product3[item] = left3 * right[item];
    }
    sum0 = reduce8(product0);
    sum1 = reduce8(product1);
    sum2 = reduce8(product2);
    sum3 = reduce8(product3);
}

static void local_dual_dot8_bank1(
    const packed_weight_pair_t left01[8],
    const packed_weight_pair_t left23[8],
    const float right[8],
    float &sum0,
    float &sum1,
    float &sum2,
    float &sum3
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=left01 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left23 complete dim=1
#pragma HLS ARRAY_PARTITION variable=right complete dim=1
    float product0[8];
    float product1[8];
    float product2[8];
    float product3[8];
#pragma HLS ARRAY_PARTITION variable=product0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product3 complete dim=1
#pragma HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product2 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product3 op=mul impl=fulldsp latency=6
V8S_IP_LOCAL_DOT8_BANK1_PRODUCTS:
    for (int item = 0; item < 8; ++item) {
#pragma HLS UNROLL
        const packed_weight_pair_t packed01 = left01[item];
        const packed_weight_pair_t packed23 = left23[item];
        const uint32_t left0_bits = static_cast<uint32_t>(packed01.range(31, 0));
        const uint32_t left1_bits = static_cast<uint32_t>(packed01.range(63, 32));
        const uint32_t left2_bits = static_cast<uint32_t>(packed23.range(31, 0));
        const uint32_t left3_bits = static_cast<uint32_t>(packed23.range(63, 32));
        float left0;
        float left1;
        float left2;
        float left3;
        std::memcpy(&left0, &left0_bits, sizeof(left0));
        std::memcpy(&left1, &left1_bits, sizeof(left1));
        std::memcpy(&left2, &left2_bits, sizeof(left2));
        std::memcpy(&left3, &left3_bits, sizeof(left3));
        product0[item] = left0 * right[item];
        product1[item] = left1 * right[item];
        product2[item] = left2 * right[item];
        product3[item] = left3 * right[item];
    }
    sum0 = reduce8(product0);
    sum1 = reduce8(product1);
    sum2 = reduce8(product2);
    sum3 = reduce8(product3);
}

static void local_dual_dot8_bank2(
    const packed_weight_pair_t left01[8],
    const packed_weight_pair_t left23[8],
    const float right[8],
    float &sum0,
    float &sum1,
    float &sum2,
    float &sum3
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=left01 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left23 complete dim=1
#pragma HLS ARRAY_PARTITION variable=right complete dim=1
    float product0[8];
    float product1[8];
    float product2[8];
    float product3[8];
#pragma HLS ARRAY_PARTITION variable=product0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product3 complete dim=1
#pragma HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product2 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product3 op=mul impl=fulldsp latency=6
V8S_IP_LOCAL_DOT8_BANK2_PRODUCTS:
    for (int item = 0; item < 8; ++item) {
#pragma HLS UNROLL
        const packed_weight_pair_t packed01 = left01[item];
        const packed_weight_pair_t packed23 = left23[item];
        const uint32_t left0_bits = static_cast<uint32_t>(packed01.range(31, 0));
        const uint32_t left1_bits = static_cast<uint32_t>(packed01.range(63, 32));
        const uint32_t left2_bits = static_cast<uint32_t>(packed23.range(31, 0));
        const uint32_t left3_bits = static_cast<uint32_t>(packed23.range(63, 32));
        float left0;
        float left1;
        float left2;
        float left3;
        std::memcpy(&left0, &left0_bits, sizeof(left0));
        std::memcpy(&left1, &left1_bits, sizeof(left1));
        std::memcpy(&left2, &left2_bits, sizeof(left2));
        std::memcpy(&left3, &left3_bits, sizeof(left3));
        product0[item] = left0 * right[item];
        product1[item] = left1 * right[item];
        product2[item] = left2 * right[item];
        product3[item] = left3 * right[item];
    }
    sum0 = reduce8(product0);
    sum1 = reduce8(product1);
    sum2 = reduce8(product2);
    sum3 = reduce8(product3);
}

static void local_dual_dot8_bank3(
    const packed_weight_pair_t left01[8],
    const packed_weight_pair_t left23[8],
    const float right[8],
    float &sum0,
    float &sum1,
    float &sum2,
    float &sum3
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=left01 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left23 complete dim=1
#pragma HLS ARRAY_PARTITION variable=right complete dim=1
    float product0[8];
    float product1[8];
    float product2[8];
    float product3[8];
#pragma HLS ARRAY_PARTITION variable=product0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=product3 complete dim=1
#pragma HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product2 op=mul impl=fulldsp latency=6
#pragma HLS BIND_OP variable=product3 op=mul impl=fulldsp latency=6
V8S_IP_LOCAL_DOT8_BANK3_PRODUCTS:
    for (int item = 0; item < 8; ++item) {
#pragma HLS UNROLL
        const packed_weight_pair_t packed01 = left01[item];
        const packed_weight_pair_t packed23 = left23[item];
        const uint32_t left0_bits = static_cast<uint32_t>(packed01.range(31, 0));
        const uint32_t left1_bits = static_cast<uint32_t>(packed01.range(63, 32));
        const uint32_t left2_bits = static_cast<uint32_t>(packed23.range(31, 0));
        const uint32_t left3_bits = static_cast<uint32_t>(packed23.range(63, 32));
        float left0;
        float left1;
        float left2;
        float left3;
        std::memcpy(&left0, &left0_bits, sizeof(left0));
        std::memcpy(&left1, &left1_bits, sizeof(left1));
        std::memcpy(&left2, &left2_bits, sizeof(left2));
        std::memcpy(&left3, &left3_bits, sizeof(left3));
        product0[item] = left0 * right[item];
        product1[item] = left1 * right[item];
        product2[item] = left2 * right[item];
        product3[item] = left3 * right[item];
    }
    sum0 = reduce8(product0);
    sum1 = reduce8(product1);
    sum2 = reduce8(product2);
    sum3 = reduce8(product3);
}

#define DEFINE_TRIPLE_DOT8_BANK(NAME)                                         \
static void NAME(                                                             \
    const packed_weight_pair_t left01[8],                                     \
    const packed_weight_single_t left2[8],                                    \
    const float right[8], float &sum0, float &sum1, float &sum2) {            \
  _Pragma("HLS INLINE off")                                                   \
  _Pragma("HLS PIPELINE II=1")                                               \
  _Pragma("HLS ARRAY_PARTITION variable=left01 complete dim=1")              \
  _Pragma("HLS ARRAY_PARTITION variable=left2 complete dim=1")               \
  _Pragma("HLS ARRAY_PARTITION variable=right complete dim=1")               \
  float product0[8], product1[8], product2[8];                                \
  _Pragma("HLS ARRAY_PARTITION variable=product0 complete dim=1")            \
  _Pragma("HLS ARRAY_PARTITION variable=product1 complete dim=1")            \
  _Pragma("HLS ARRAY_PARTITION variable=product2 complete dim=1")            \
  _Pragma("HLS BIND_OP variable=product0 op=mul impl=fulldsp latency=6")      \
  _Pragma("HLS BIND_OP variable=product1 op=mul impl=fulldsp latency=6")      \
  _Pragma("HLS BIND_OP variable=product2 op=mul impl=fulldsp latency=6")      \
  for (int item = 0; item < 8; ++item) {                                      \
    _Pragma("HLS UNROLL")                                                     \
    const packed_weight_pair_t packed01 = left01[item];                       \
    const uint32_t left0_bits =                                               \
        static_cast<uint32_t>(packed01.range(31, 0));                         \
    const uint32_t left1_bits =                                               \
        static_cast<uint32_t>(packed01.range(63, 32));                        \
    const uint32_t left2_bits = left2[item].to_uint();                        \
    float left0, left1, left2_value;                                          \
    std::memcpy(&left0, &left0_bits, sizeof(left0));                          \
    std::memcpy(&left1, &left1_bits, sizeof(left1));                          \
    std::memcpy(&left2_value, &left2_bits, sizeof(left2_value));              \
    product0[item] = left0 * right[item];                                     \
    product1[item] = left1 * right[item];                                     \
    product2[item] = left2_value * right[item];                               \
  }                                                                            \
  sum0 = reduce8(product0);                                                    \
  sum1 = reduce8(product1);                                                    \
  sum2 = reduce8(product2);                                                    \
}

DEFINE_TRIPLE_DOT8_BANK(local_triple_dot8_bank0)
DEFINE_TRIPLE_DOT8_BANK(local_triple_dot8_bank1)
DEFINE_TRIPLE_DOT8_BANK(local_triple_dot8_bank2)
DEFINE_TRIPLE_DOT8_BANK(local_triple_dot8_bank3)

#undef DEFINE_TRIPLE_DOT8_BANK

static void commit_triple_result(
    bool pv_phase,
    int qk_chunk,
    int qk_key,
    int pv_dim,
    bool pv_sum_step,
    int tile_start,
    int query,
    float scale,
    float dot,
    float partial_lane[TILE_K],
    uint32_t &tile_max_bits,
    uint32_t score_lane[TILE_K],
    float previous_scale,
    float base_sum,
    float &merged_sum,
    bool had_previous,
    float output_state_lane[HEAD_DIM]
) {
#pragma HLS INLINE
    if (pv_phase) {
        if (pv_sum_step) {
            merged_sum = base_sum + dot;
        } else {
            output_state_lane[pv_dim] = had_previous
                ? previous_scale * output_state_lane[pv_dim] + dot
                : dot;
        }
    } else {
        const float accumulated = qk_chunk == 0
            ? dot : partial_lane[qk_key] + dot;
        partial_lane[qk_key] = accumulated;
        if (qk_chunk == QK_CHUNKS - 1) {
            const bool active = tile_start + qk_key <= query;
            const float score = active ? accumulated * scale : -INFINITY;
            uint32_t score_bits;
            std::memcpy(&score_bits, &score, sizeof(score_bits));
            score_lane[qk_key] = score_bits;
            tile_max_bits = qk_key == 0
                ? score_bits : ordered_max_bits(tile_max_bits, score_bits);
        }
    }
}

static void shared_dot_phase_quad_reference(
    bool pv_phase,
    int query_base,
    int tile_start,
    float scale,
    const packed_q_bank_t q_bank[STATIC_GQA_LANES][4]
        [GROUPS_PER_BLOCK * WORDS_PER_ROW],
    const packed_weight_pair_t weight01[GROUPS_PER_BLOCK][4][8],
    const packed_weight_pair_t weight23[GROUPS_PER_BLOCK][4][8],
    const float previous_scale[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    const float base_sum[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float merged_sum[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    const bool had_previous[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float output_state[CONTEXTS_PER_BLOCK][HEAD_DIM],
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM],
    float partial[GROUPS_PER_BLOCK][STATIC_GQA_LANES][TILE_K],
    uint32_t tile_max_bits[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    packed_score_pair_t score_pair01[GROUPS_PER_BLOCK][TILE_K],
    packed_score_pair_t score_pair23[GROUPS_PER_BLOCK][TILE_K]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_bank complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight01 complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight01 complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight23 complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight23 complete dim=3
#pragma HLS ARRAY_PARTITION variable=previous_scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=previous_scale complete dim=2
#pragma HLS ARRAY_PARTITION variable=base_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=base_sum complete dim=2
#pragma HLS ARRAY_PARTITION variable=merged_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=merged_sum complete dim=2
#pragma HLS ARRAY_PARTITION variable=had_previous complete dim=1
#pragma HLS ARRAY_PARTITION variable=had_previous complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=2
    const int phase_steps = pv_phase
        ? GROUPS_PER_BLOCK * (HEAD_DIM + 1)
        : QK_STEPS_PER_PHASE;
U109_SHARED_DOT_PHASE:
    for (int step = 0; step < phase_steps; ++step) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=partial inter RAW distance=32 true
#pragma HLS DEPENDENCE variable=tile_max_bits inter RAW distance=2 true
#pragma HLS DEPENDENCE variable=merged_sum inter false
#pragma HLS DEPENDENCE variable=output_state inter false
        packed_weight_pair_t left_bank01[4][8];
        packed_weight_pair_t left_bank23[4][8];
        float right_bank[4][8];
#pragma HLS ARRAY_PARTITION variable=left_bank01 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left_bank01 complete dim=2
#pragma HLS ARRAY_PARTITION variable=left_bank23 complete dim=1
#pragma HLS ARRAY_PARTITION variable=left_bank23 complete dim=2
#pragma HLS ARRAY_PARTITION variable=right_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=right_bank complete dim=2

        int qk_group = 0;
        int qk_chunk = 0;
        int qk_key = 0;
        bool qk_active = false;
        int pv_group = 0;
        int pv_dim = 0;
        bool pv_sum_step = false;

        if (pv_phase) {
            pv_group = step / (HEAD_DIM + 1);
            pv_dim = step % (HEAD_DIM + 1);
            pv_sum_step = pv_dim == HEAD_DIM;
            const int v_address =
                (tile_start / TILE_K) * HEAD_DIM
                + (pv_sum_step ? 0 : pv_dim);
            const mem_word_t packed_v = v_head_cache[v_address];
U108_SHARED_PV_BANK_INPUT:
            for (int lane = 0; lane < TILE_K; ++lane) {
#pragma HLS UNROLL
                const int bank = lane / 8;
                const int local_key = lane % 8;
                const bf16_t v_bits = packed_v.range(
                    (lane + 1) * BF16_BITS - 1, lane * BF16_BITS);
                left_bank01[bank][local_key] =
                    weight01[pv_group][bank][local_key];
                left_bank23[bank][local_key] =
                    weight23[pv_group][bank][local_key];
                right_bank[bank][local_key] = pv_sum_step
                    ? 1.0f : bf16_to_float(v_bits);
            }
        } else {
            qk_group = step / (QK_CHUNKS * TILE_K);
            const int within_group = step % (QK_CHUNKS * TILE_K);
            qk_chunk = within_group / TILE_K;
            qk_key = within_group % TILE_K;
            const int row =
                qk_group / (GQA_GROUP / STATIC_GQA_LANES);
            const int query = query_base + row;
            qk_active = tile_start + qk_key <= query;
            const int q_address = qk_group * WORDS_PER_ROW + qk_chunk;
            const int k_address =
                (tile_start + qk_key) * WORDS_PER_ROW + qk_chunk;
            const mem_word_t packed_k = k_head_cache[k_address];
U108_SHARED_QK_BANK_INPUT:
            for (int lane = 0; lane < QK_PAR_D; ++lane) {
#pragma HLS UNROLL
                const int bank = lane / 8;
                const int local_key = lane % 8;
                const int bank_bit = local_key * BF16_BITS;
                const bf16_t k_bits = packed_k.range(
                    (lane + 1) * BF16_BITS - 1, lane * BF16_BITS);
                const packed_q_bank_t packed_q0 = q_bank[0][bank][q_address];
                const packed_q_bank_t packed_q1 = q_bank[1][bank][q_address];
                const packed_q_bank_t packed_q2 = q_bank[2][bank][q_address];
                const packed_q_bank_t packed_q3 = q_bank[3][bank][q_address];
                const bf16_t q0_bits = packed_q0.range(
                    bank_bit + BF16_BITS - 1, bank_bit);
                const bf16_t q1_bits = packed_q1.range(
                    bank_bit + BF16_BITS - 1, bank_bit);
                const bf16_t q2_bits = packed_q2.range(
                    bank_bit + BF16_BITS - 1, bank_bit);
                const bf16_t q3_bits = packed_q3.range(
                    bank_bit + BF16_BITS - 1, bank_bit);
                const float q0_value = bf16_to_float(q0_bits);
                const float q1_value = bf16_to_float(q1_bits);
                const float q2_value = bf16_to_float(q2_bits);
                const float q3_value = bf16_to_float(q3_bits);
                uint32_t q0_raw;
                uint32_t q1_raw;
                uint32_t q2_raw;
                uint32_t q3_raw;
                std::memcpy(&q0_raw, &q0_value, sizeof(q0_raw));
                std::memcpy(&q1_raw, &q1_value, sizeof(q1_raw));
                std::memcpy(&q2_raw, &q2_value, sizeof(q2_raw));
                std::memcpy(&q3_raw, &q3_value, sizeof(q3_raw));
                packed_weight_pair_t packed_q01;
                packed_weight_pair_t packed_q23;
                packed_q01.range(31, 0) = q0_raw;
                packed_q01.range(63, 32) = q1_raw;
                packed_q23.range(31, 0) = q2_raw;
                packed_q23.range(63, 32) = q3_raw;
                left_bank01[bank][local_key] = packed_q01;
                left_bank23[bank][local_key] = packed_q23;
                right_bank[bank][local_key] = qk_active
                    ? bf16_to_float(k_bits) : 0.0f;
            }
        }

        float bank_dot0[4];
        float bank_dot1[4];
        float bank_dot2[4];
        float bank_dot3[4];
#pragma HLS ARRAY_PARTITION variable=bank_dot0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_dot1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_dot2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_dot3 complete dim=1
        local_dual_dot8_bank0(
            left_bank01[0], left_bank23[0], right_bank[0],
            bank_dot0[0], bank_dot1[0], bank_dot2[0], bank_dot3[0]);
        local_dual_dot8_bank1(
            left_bank01[1], left_bank23[1], right_bank[1],
            bank_dot0[1], bank_dot1[1], bank_dot2[1], bank_dot3[1]);
        local_dual_dot8_bank2(
            left_bank01[2], left_bank23[2], right_bank[2],
            bank_dot0[2], bank_dot1[2], bank_dot2[2], bank_dot3[2]);
        local_dual_dot8_bank3(
            left_bank01[3], left_bank23[3], right_bank[3],
            bank_dot0[3], bank_dot1[3], bank_dot2[3], bank_dot3[3]);
        const float dot0 = (bank_dot0[0] + bank_dot0[1])
            + (bank_dot0[2] + bank_dot0[3]);
        const float dot1 = (bank_dot1[0] + bank_dot1[1])
            + (bank_dot1[2] + bank_dot1[3]);
        const float dot2 = (bank_dot2[0] + bank_dot2[1])
            + (bank_dot2[2] + bank_dot2[3]);
        const float dot3 = (bank_dot3[0] + bank_dot3[1])
            + (bank_dot3[2] + bank_dot3[3]);

        if (pv_phase) {
            const int context0 = pv_group * STATIC_GQA_LANES;
            const int context1 = context0 + 1;
            const int context2 = context0 + 2;
            const int context3 = context0 + 3;
            if (pv_sum_step) {
                merged_sum[pv_group][0] = base_sum[pv_group][0] + dot0;
                merged_sum[pv_group][1] = base_sum[pv_group][1] + dot1;
                merged_sum[pv_group][2] = base_sum[pv_group][2] + dot2;
                merged_sum[pv_group][3] = base_sum[pv_group][3] + dot3;
            } else {
                output_state[context0][pv_dim] = had_previous[pv_group][0]
                    ? previous_scale[pv_group][0]
                        * output_state[context0][pv_dim] + dot0
                    : dot0;
                output_state[context1][pv_dim] = had_previous[pv_group][1]
                    ? previous_scale[pv_group][1]
                        * output_state[context1][pv_dim] + dot1
                    : dot1;
                output_state[context2][pv_dim] = had_previous[pv_group][2]
                    ? previous_scale[pv_group][2]
                        * output_state[context2][pv_dim] + dot2
                    : dot2;
                output_state[context3][pv_dim] = had_previous[pv_group][3]
                    ? previous_scale[pv_group][3]
                        * output_state[context3][pv_dim] + dot3
                    : dot3;
            }
        } else {
            const float accumulated0 = qk_chunk == 0
                ? dot0 : partial[qk_group][0][qk_key] + dot0;
            const float accumulated1 = qk_chunk == 0
                ? dot1 : partial[qk_group][1][qk_key] + dot1;
            const float accumulated2 = qk_chunk == 0
                ? dot2 : partial[qk_group][2][qk_key] + dot2;
            const float accumulated3 = qk_chunk == 0
                ? dot3 : partial[qk_group][3][qk_key] + dot3;
            partial[qk_group][0][qk_key] = accumulated0;
            partial[qk_group][1][qk_key] = accumulated1;
            partial[qk_group][2][qk_key] = accumulated2;
            partial[qk_group][3][qk_key] = accumulated3;
            if (qk_chunk == QK_CHUNKS - 1) {
                const float score0 = qk_active
                    ? accumulated0 * scale : -INFINITY;
                const float score1 = qk_active
                    ? accumulated1 * scale : -INFINITY;
                const float score2 = qk_active
                    ? accumulated2 * scale : -INFINITY;
                const float score3 = qk_active
                    ? accumulated3 * scale : -INFINITY;
                uint32_t score0_bits;
                uint32_t score1_bits;
                uint32_t score2_bits;
                uint32_t score3_bits;
                std::memcpy(&score0_bits, &score0, sizeof(score0_bits));
                std::memcpy(&score1_bits, &score1, sizeof(score1_bits));
                std::memcpy(&score2_bits, &score2, sizeof(score2_bits));
                std::memcpy(&score3_bits, &score3, sizeof(score3_bits));
                packed_score_pair_t packed_score01;
                packed_score_pair_t packed_score23;
                packed_score01.range(31, 0) = score0_bits;
                packed_score01.range(63, 32) = score1_bits;
                packed_score23.range(31, 0) = score2_bits;
                packed_score23.range(63, 32) = score3_bits;
                score_pair01[qk_group][qk_key] = packed_score01;
                score_pair23[qk_group][qk_key] = packed_score23;
                tile_max_bits[qk_group][0] = qk_key == 0
                    ? score0_bits
                    : ordered_max_bits(tile_max_bits[qk_group][0], score0_bits);
                tile_max_bits[qk_group][1] = qk_key == 0
                    ? score1_bits
                    : ordered_max_bits(tile_max_bits[qk_group][1], score1_bits);
                tile_max_bits[qk_group][2] = qk_key == 0
                    ? score2_bits
                    : ordered_max_bits(tile_max_bits[qk_group][2], score2_bits);
                tile_max_bits[qk_group][3] = qk_key == 0
                    ? score3_bits
                    : ordered_max_bits(tile_max_bits[qk_group][3], score3_bits);
            }
        }
    }
}

static void shared_dot_phase(
    bool pv_phase,
    int query_base,
    int tile_start,
    float scale,
    const packed_q_bank_t q_triple_bank[3][4]
        [TRIPLE_BATCHES * WORDS_PER_ROW],
    const packed_weight_triple_t weight_batch[TRIPLE_BATCHES][4][8],
    const float previous_scale[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    const float base_sum[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    float merged_sum[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    const bool had_previous[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    float output_state[STATIC_GQA_LANES][GROUPS_PER_BLOCK][HEAD_DIM],
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM],
    float partial[STATIC_GQA_LANES][GROUPS_PER_BLOCK][TILE_K],
    uint32_t tile_max_bits[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    uint32_t score_lane[STATIC_GQA_LANES][GROUPS_PER_BLOCK][TILE_K]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=3
#pragma HLS ARRAY_PARTITION variable=previous_scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=base_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=merged_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=had_previous complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=score_lane complete dim=1
    const int phase_steps = pv_phase
        ? TRIPLE_BATCHES * (HEAD_DIM + 1)
        : TRIPLE_BATCHES * QK_CHUNKS * TILE_K;
U117_TRIPLE_SHARED_DOT_PHASE:
    for (int step = 0; step < phase_steps; ++step) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=partial inter RAW distance=32 true
#pragma HLS DEPENDENCE variable=tile_max_bits inter RAW distance=1 true
#pragma HLS DEPENDENCE variable=merged_sum inter false
#pragma HLS DEPENDENCE variable=output_state inter false
        packed_weight_pair_t left_bank01[4][8];
        packed_weight_single_t left_bank2[4][8];
        float right_bank[4][8];
#pragma HLS ARRAY_PARTITION variable=left_bank01 complete dim=0
#pragma HLS ARRAY_PARTITION variable=left_bank2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=right_bank complete dim=0

        int batch = 0;
        int qk_chunk = 0;
        int qk_key = 0;
        int pv_dim = 0;
        bool pv_sum_step = false;
        if (pv_phase) {
            batch = step / (HEAD_DIM + 1);
            pv_dim = step % (HEAD_DIM + 1);
            pv_sum_step = pv_dim == HEAD_DIM;
            const int v_address = (tile_start / TILE_K) * HEAD_DIM
                + (pv_sum_step ? 0 : pv_dim);
            const mem_word_t packed_v = v_head_cache[v_address];
U118_TRIPLE_PV_BATCH_INPUT:
            for (int key = 0; key < TILE_K; ++key) {
#pragma HLS UNROLL
                const int bank = key / 8;
                const int local_key = key % 8;
                const packed_weight_triple_t selected =
                    weight_batch[batch][bank][local_key];
                const packed_weight_pair_t selected01 =
                    selected.range(63, 0);
                const packed_weight_single_t selected2 =
                    selected.range(95, 64);
                const bf16_t v_bits = packed_v.range(
                    (key + 1) * BF16_BITS - 1, key * BF16_BITS);
                left_bank01[bank][local_key] = selected01;
                left_bank2[bank][local_key] = selected2;
                right_bank[bank][local_key] = pv_sum_step
                    ? 1.0f : bf16_to_float(v_bits);
            }
        } else {
            batch = step / (QK_CHUNKS * TILE_K);
            const int within_batch = step % (QK_CHUNKS * TILE_K);
            qk_chunk = within_batch / TILE_K;
            qk_key = within_batch % TILE_K;
            const int context0 = batch * 3;
            const int context1 = context0 + 1;
            const int context2 = context0 + 2;
            const bool valid0 = context0 < CONTEXTS_PER_BLOCK;
            const bool valid1 = context1 < CONTEXTS_PER_BLOCK;
            const bool valid2 = context2 < CONTEXTS_PER_BLOCK;
            const int group0 = valid0 ? context0 / STATIC_GQA_LANES : 0;
            const int group1 = valid1 ? context1 / STATIC_GQA_LANES : 0;
            const int group2 = valid2 ? context2 / STATIC_GQA_LANES : 0;
            const bool active0 = valid0 &&
                tile_start + qk_key <= query_base + group0;
            const bool active1 = valid1 &&
                tile_start + qk_key <= query_base + group1;
            const bool active2 = valid2 &&
                tile_start + qk_key <= query_base + group2;
            const int k_address =
                (tile_start + qk_key) * WORDS_PER_ROW + qk_chunk;
            const mem_word_t packed_k = k_head_cache[k_address];
            const int q_address = batch * WORDS_PER_ROW + qk_chunk;
U116_TRIPLE_QK_BANK_INPUT:
            for (int item = 0; item < QK_PAR_D; ++item) {
#pragma HLS UNROLL
                const int bank = item / 8;
                const int local_item = item % 8;
                const int bank_bit = local_item * BF16_BITS;
                const bf16_t k_bits = packed_k.range(
                    (item + 1) * BF16_BITS - 1, item * BF16_BITS);
                const packed_q_bank_t packed_q0 =
                    q_triple_bank[0][bank][q_address];
                const packed_q_bank_t packed_q1 =
                    q_triple_bank[1][bank][q_address];
                const packed_q_bank_t packed_q2 =
                    q_triple_bank[2][bank][q_address];
                const bf16_t q0_bits = active0 ? packed_q0.range(
                    bank_bit + BF16_BITS - 1, bank_bit) : bf16_t(0);
                const bf16_t q1_bits = active1 ? packed_q1.range(
                    bank_bit + BF16_BITS - 1, bank_bit) : bf16_t(0);
                const bf16_t q2_bits = active2 ? packed_q2.range(
                    bank_bit + BF16_BITS - 1, bank_bit) : bf16_t(0);
                const float q0_value = bf16_to_float(q0_bits);
                const float q1_value = bf16_to_float(q1_bits);
                const float q2_value = bf16_to_float(q2_bits);
                uint32_t q0_raw, q1_raw, q2_raw;
                std::memcpy(&q0_raw, &q0_value, sizeof(q0_raw));
                std::memcpy(&q1_raw, &q1_value, sizeof(q1_raw));
                std::memcpy(&q2_raw, &q2_value, sizeof(q2_raw));
                packed_weight_pair_t packed_q01;
                packed_q01.range(31, 0) = q0_raw;
                packed_q01.range(63, 32) = q1_raw;
                left_bank01[bank][local_item] = packed_q01;
                left_bank2[bank][local_item] = q2_raw;
                right_bank[bank][local_item] = bf16_to_float(k_bits);
            }
        }

        float bank_dot0[4], bank_dot1[4], bank_dot2[4];
#pragma HLS ARRAY_PARTITION variable=bank_dot0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_dot1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_dot2 complete dim=1
        local_triple_dot8_bank0(left_bank01[0], left_bank2[0], right_bank[0],
                                bank_dot0[0], bank_dot1[0], bank_dot2[0]);
        local_triple_dot8_bank1(left_bank01[1], left_bank2[1], right_bank[1],
                                bank_dot0[1], bank_dot1[1], bank_dot2[1]);
        local_triple_dot8_bank2(left_bank01[2], left_bank2[2], right_bank[2],
                                bank_dot0[2], bank_dot1[2], bank_dot2[2]);
        local_triple_dot8_bank3(left_bank01[3], left_bank2[3], right_bank[3],
                                bank_dot0[3], bank_dot1[3], bank_dot2[3]);
        const float dot[3] = {
            (bank_dot0[0] + bank_dot0[1]) + (bank_dot0[2] + bank_dot0[3]),
            (bank_dot1[0] + bank_dot1[1]) + (bank_dot1[2] + bank_dot1[3]),
            (bank_dot2[0] + bank_dot2[1]) + (bank_dot2[2] + bank_dot2[3])};
#pragma HLS ARRAY_PARTITION variable=dot complete dim=1

        int lane_group[STATIC_GQA_LANES] = {0, 0, 0, 0};
        int lane_slot[STATIC_GQA_LANES] = {0, 0, 0, 0};
        bool lane_valid[STATIC_GQA_LANES] = {false, false, false, false};
#pragma HLS ARRAY_PARTITION variable=lane_group complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_slot complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_valid complete dim=1
        if (batch == 0) {
            lane_valid[0] = true; lane_slot[0] = 0;
            lane_valid[1] = true; lane_slot[1] = 1;
            lane_valid[2] = true; lane_slot[2] = 2;
        } else if (batch == 1) {
            lane_valid[3] = true; lane_slot[3] = 0;
            lane_valid[0] = true; lane_slot[0] = 1; lane_group[0] = 1;
            lane_valid[1] = true; lane_slot[1] = 2; lane_group[1] = 1;
        } else if (batch == 2) {
            lane_valid[2] = true; lane_slot[2] = 0; lane_group[2] = 1;
            lane_valid[3] = true; lane_slot[3] = 1; lane_group[3] = 1;
            lane_valid[0] = true; lane_slot[0] = 2; lane_group[0] = 2;
        } else if (batch == 3) {
            lane_valid[1] = true; lane_slot[1] = 0; lane_group[1] = 2;
            lane_valid[2] = true; lane_slot[2] = 1; lane_group[2] = 2;
            lane_valid[3] = true; lane_slot[3] = 2; lane_group[3] = 2;
        } else if (batch == 4) {
            lane_valid[0] = true; lane_slot[0] = 0; lane_group[0] = 3;
            lane_valid[1] = true; lane_slot[1] = 1; lane_group[1] = 3;
            lane_valid[2] = true; lane_slot[2] = 2; lane_group[2] = 3;
        } else {
            lane_valid[3] = true; lane_slot[3] = 0; lane_group[3] = 3;
        }
U115_COMMIT_PHYSICAL_LANES:
        for (int lane = 0; lane < STATIC_GQA_LANES; ++lane) {
#pragma HLS UNROLL
            const int group = lane_group[lane];
            const int slot = lane_slot[lane];
            if (lane_valid[lane]) {
                commit_triple_result(
                    pv_phase, qk_chunk, qk_key, pv_dim, pv_sum_step,
                    tile_start, query_base + group, scale, dot[slot],
                    partial[lane][group], tile_max_bits[lane][group],
                    score_lane[lane][group],
                    previous_scale[lane][group], base_sum[lane][group],
                    merged_sum[lane][group], had_previous[lane][group],
                    output_state[lane][group]);
            }
        }
    }
}

struct weight_pair_t {
    float lane0;
    float lane1;
};

static void produce_group_weights(
    const packed_score_pair_t score_pair[GROUPS_PER_BLOCK][TILE_K],
    const float new_max[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    hls::stream<weight_pair_t> &weight_stream
) {
#pragma HLS INLINE off
V8S_IP_WEIGHT_PRODUCER_GROUP:
    for (int group = 0; group < GROUPS_PER_BLOCK; ++group) {
        const float safe_max0 = new_max[group][0] == -INFINITY
            ? 0.0f : new_max[group][0];
        const float safe_max1 = new_max[group][1] == -INFINITY
            ? 0.0f : new_max[group][1];
V8S_IP_WEIGHT_PRODUCER_EXP:
        for (int key = 0; key < TILE_K; ++key) {
#pragma HLS PIPELINE II=1
            const packed_score_pair_t packed = score_pair[group][key];
            const uint32_t score0_bits =
                static_cast<uint32_t>(packed.range(31, 0));
            const uint32_t score1_bits =
                static_cast<uint32_t>(packed.range(63, 32));
            float score0;
            float score1;
            std::memcpy(&score0, &score0_bits, sizeof(score0));
            std::memcpy(&score1, &score1_bits, sizeof(score1));
            weight_pair_t pair;
            const float weight_exp0 = hls::expf(
                score0 - safe_max0);
            const float weight_exp1 = hls::expf(
                score1 - safe_max1);
#pragma HLS BIND_OP variable=weight_exp0 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp1 op=fexp impl=fulldsp latency=20
            pair.lane0 = weight_exp0;
            pair.lane1 = weight_exp1;
            weight_stream.write(pair);
        }
    }
}

static void produce_group_weights_array(
    const packed_score_pair_t score_pair01[GROUPS_PER_BLOCK][TILE_K],
    const packed_score_pair_t score_pair23[GROUPS_PER_BLOCK][TILE_K],
    const float new_max[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    packed_weight_pair_t weight01[GROUPS_PER_BLOCK][4][8],
    packed_weight_pair_t weight23[GROUPS_PER_BLOCK][4][8]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=new_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=new_max complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight01 complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight01 complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight23 complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight23 complete dim=3
U108_WEIGHT_ARRAY_GROUP:
    for (int group = 0; group < GROUPS_PER_BLOCK; ++group) {
        const float safe_max0 = new_max[group][0] == -INFINITY
            ? 0.0f : new_max[group][0];
        const float safe_max1 = new_max[group][1] == -INFINITY
            ? 0.0f : new_max[group][1];
        const float safe_max2 = new_max[group][2] == -INFINITY
            ? 0.0f : new_max[group][2];
        const float safe_max3 = new_max[group][3] == -INFINITY
            ? 0.0f : new_max[group][3];
U108_WEIGHT_ARRAY_EXP:
        for (int key = 0; key < TILE_K; ++key) {
#pragma HLS PIPELINE II=1
            const packed_score_pair_t packed_score01 =
                score_pair01[group][key];
            const packed_score_pair_t packed_score23 =
                score_pair23[group][key];
            const uint32_t score0_bits =
                static_cast<uint32_t>(packed_score01.range(31, 0));
            const uint32_t score1_bits =
                static_cast<uint32_t>(packed_score01.range(63, 32));
            const uint32_t score2_bits =
                static_cast<uint32_t>(packed_score23.range(31, 0));
            const uint32_t score3_bits =
                static_cast<uint32_t>(packed_score23.range(63, 32));
            float score0;
            float score1;
            float score2;
            float score3;
            std::memcpy(&score0, &score0_bits, sizeof(score0));
            std::memcpy(&score1, &score1_bits, sizeof(score1));
            std::memcpy(&score2, &score2_bits, sizeof(score2));
            std::memcpy(&score3, &score3_bits, sizeof(score3));
            const float weight_exp0 = hls::expf(score0 - safe_max0);
            const float weight_exp1 = hls::expf(score1 - safe_max1);
            const float weight_exp2 = hls::expf(score2 - safe_max2);
            const float weight_exp3 = hls::expf(score3 - safe_max3);
#pragma HLS BIND_OP variable=weight_exp0 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp1 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp2 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp3 op=fexp impl=fulldsp latency=20
            uint32_t lane0_bits;
            uint32_t lane1_bits;
            uint32_t lane2_bits;
            uint32_t lane3_bits;
            std::memcpy(&lane0_bits, &weight_exp0, sizeof(lane0_bits));
            std::memcpy(&lane1_bits, &weight_exp1, sizeof(lane1_bits));
            std::memcpy(&lane2_bits, &weight_exp2, sizeof(lane2_bits));
            std::memcpy(&lane3_bits, &weight_exp3, sizeof(lane3_bits));
            packed_weight_pair_t packed_weight01;
            packed_weight_pair_t packed_weight23;
            packed_weight01.range(31, 0) = lane0_bits;
            packed_weight01.range(63, 32) = lane1_bits;
            packed_weight23.range(31, 0) = lane2_bits;
            packed_weight23.range(63, 32) = lane3_bits;
            weight01[group][key / 8][key % 8] = packed_weight01;
            weight23[group][key / 8][key % 8] = packed_weight23;
        }
    }
}

static void produce_batch_weights_lane_array(
    const uint32_t score_lane[STATIC_GQA_LANES][GROUPS_PER_BLOCK][TILE_K],
    const float new_max[STATIC_GQA_LANES][GROUPS_PER_BLOCK],
    packed_weight_triple_t weight_batch[TRIPLE_BATCHES][4][8]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=score_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=new_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=3
U118_WEIGHT_BATCH_EXP:
    for (int step = 0; step < TRIPLE_BATCHES * TILE_K; ++step) {
#pragma HLS PIPELINE II=1
        const int batch = step / TILE_K;
        const int key = step % TILE_K;
        uint32_t score0_bits = 0;
        uint32_t score1_bits = 0;
        uint32_t score2_bits = 0;
        float max0 = 0.0f;
        float max1 = 0.0f;
        float max2 = 0.0f;
        if (batch == 0) {
            score0_bits = score_lane[0][0][key]; max0 = new_max[0][0];
            score1_bits = score_lane[1][0][key]; max1 = new_max[1][0];
            score2_bits = score_lane[2][0][key]; max2 = new_max[2][0];
        } else if (batch == 1) {
            score0_bits = score_lane[3][0][key]; max0 = new_max[3][0];
            score1_bits = score_lane[0][1][key]; max1 = new_max[0][1];
            score2_bits = score_lane[1][1][key]; max2 = new_max[1][1];
        } else if (batch == 2) {
            score0_bits = score_lane[2][1][key]; max0 = new_max[2][1];
            score1_bits = score_lane[3][1][key]; max1 = new_max[3][1];
            score2_bits = score_lane[0][2][key]; max2 = new_max[0][2];
        } else if (batch == 3) {
            score0_bits = score_lane[1][2][key]; max0 = new_max[1][2];
            score1_bits = score_lane[2][2][key]; max1 = new_max[2][2];
            score2_bits = score_lane[3][2][key]; max2 = new_max[3][2];
        } else if (batch == 4) {
            score0_bits = score_lane[0][3][key]; max0 = new_max[0][3];
            score1_bits = score_lane[1][3][key]; max1 = new_max[1][3];
            score2_bits = score_lane[2][3][key]; max2 = new_max[2][3];
        } else {
            score0_bits = score_lane[3][3][key]; max0 = new_max[3][3];
        }
        const float safe_max0 = max0 == -INFINITY ? 0.0f : max0;
        const float safe_max1 = max1 == -INFINITY ? 0.0f : max1;
        const float safe_max2 = max2 == -INFINITY ? 0.0f : max2;
        float score0, score1, score2;
        std::memcpy(&score0, &score0_bits, sizeof(score0));
        std::memcpy(&score1, &score1_bits, sizeof(score1));
        std::memcpy(&score2, &score2_bits, sizeof(score2));
        const float weight_exp0 = hls::expf(score0 - safe_max0);
        const float weight_exp1 = hls::expf(score1 - safe_max1);
        const float weight_exp2 = hls::expf(score2 - safe_max2);
#pragma HLS BIND_OP variable=weight_exp0 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp1 op=fexp impl=fulldsp latency=20
#pragma HLS BIND_OP variable=weight_exp2 op=fexp impl=fulldsp latency=20
        uint32_t lane0_bits, lane1_bits, lane2_bits;
        std::memcpy(&lane0_bits, &weight_exp0, sizeof(lane0_bits));
        std::memcpy(&lane1_bits, &weight_exp1, sizeof(lane1_bits));
        std::memcpy(&lane2_bits, &weight_exp2, sizeof(lane2_bits));
        packed_weight_triple_t word = 0;
        word.range(31, 0) = lane0_bits;
        word.range(63, 32) = batch < TRIPLE_BATCHES - 1 ? lane1_bits : 0;
        word.range(95, 64) = batch < TRIPLE_BATCHES - 1 ? lane2_bits : 0;
        weight_batch[batch][key / 8][key % 8] = word;
    }
}

static void consume_group_weights(
    int tile_start,
    hls::stream<weight_pair_t> &weight_stream,
    const float previous_scale[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float merged_sum[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    const bool had_previous[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float output_state[CONTEXTS_PER_BLOCK][HEAD_DIM],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
    packed_weight_pair_t weight[GROUPS_PER_BLOCK][4][8];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3

U108_WEIGHT_CONSUMER_LOAD_ALL_GROUPS:
    for (int load_step = 0;
         load_step < GROUPS_PER_BLOCK * TILE_K;
         ++load_step) {
#pragma HLS PIPELINE II=1
            const int group = load_step / TILE_K;
            const int key = load_step % TILE_K;
            const int bank = key / 8;
            const int local_key = key % 8;
            const weight_pair_t pair = weight_stream.read();
            uint32_t lane0_bits;
            uint32_t lane1_bits;
            std::memcpy(&lane0_bits, &pair.lane0, sizeof(lane0_bits));
            std::memcpy(&lane1_bits, &pair.lane1, sizeof(lane1_bits));
            packed_weight_pair_t packed;
            packed.range(31, 0) = lane0_bits;
            packed.range(63, 32) = lane1_bits;
            weight[group][bank][local_key] = packed;
    }

    float base_sum0[GROUPS_PER_BLOCK];
    float base_sum1[GROUPS_PER_BLOCK];
#pragma HLS ARRAY_PARTITION variable=base_sum0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=base_sum1 complete dim=1
U108_CAPTURE_BASE_SUMS:
    for (int group = 0; group < GROUPS_PER_BLOCK; ++group) {
#pragma HLS UNROLL
        base_sum0[group] = merged_sum[group][0];
        base_sum1[group] = merged_sum[group][1];
    }

    mem_word_t packed_ones = 0;
U108_PACK_SYNTHETIC_ONES:
    for (int item = 0; item < VALUES_PER_WORD; ++item) {
#pragma HLS UNROLL
        packed_ones.range(
            (item + 1) * BF16_BITS - 1,
            item * BF16_BITS) = bf16_t(0x3f80);
    }

    int group = 0;
    int dim = 0;
U108_FLAT_GROUP_PV:
    for (int pv_step = 0;
         pv_step < GROUPS_PER_BLOCK * (HEAD_DIM + 1);
         ++pv_step) {
#pragma HLS PIPELINE II=1
            const bool sum_step = dim == HEAD_DIM;
            const int context0 = group * STATIC_GQA_LANES;
            const int context1 = context0 + 1;
            const int v_address =
                (tile_start / TILE_K) * HEAD_DIM + (sum_step ? 0 : dim);
            const mem_word_t packed_v = sum_step
                ? packed_ones : v_head_cache[v_address];
            bf16_t v_bits[VALUES_PER_WORD];
#pragma HLS ARRAY_PARTITION variable=v_bits complete dim=1
V8S_IP_DATAFLOW_UNPACK_V_WORD:
            for (int item = 0; item < VALUES_PER_WORD; ++item) {
#pragma HLS UNROLL
                v_bits[item] = packed_v.range(
                    (item + 1) * BF16_BITS - 1,
                    item * BF16_BITS);
            }
            float v_bank0[8];
            float v_bank1[8];
            float v_bank2[8];
            float v_bank3[8];
#pragma HLS ARRAY_PARTITION variable=v_bank0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_bank1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_bank2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_bank3 complete dim=1
V8S_IP_DATAFLOW_UNPACK_V_BANKS:
            for (int local_key = 0; local_key < 8; ++local_key) {
#pragma HLS UNROLL
                v_bank0[local_key] = bf16_to_float(v_bits[local_key]);
                v_bank1[local_key] = bf16_to_float(v_bits[8 + local_key]);
                v_bank2[local_key] = bf16_to_float(v_bits[16 + local_key]);
                v_bank3[local_key] = bf16_to_float(v_bits[24 + local_key]);
            }

            float bank0_dot0;
            float bank0_dot1;
            float bank1_dot0;
            float bank1_dot1;
            float bank2_dot0;
            float bank2_dot1;
            float bank3_dot0;
            float bank3_dot1;
            float bank0_dot2;
            float bank0_dot3;
            float bank1_dot2;
            float bank1_dot3;
            float bank2_dot2;
            float bank2_dot3;
            float bank3_dot2;
            float bank3_dot3;
            local_dual_dot8_bank0(
                weight[group][0], weight[group][0], v_bank0,
                bank0_dot0, bank0_dot1, bank0_dot2, bank0_dot3);
            local_dual_dot8_bank1(
                weight[group][1], weight[group][1], v_bank1,
                bank1_dot0, bank1_dot1, bank1_dot2, bank1_dot3);
            local_dual_dot8_bank2(
                weight[group][2], weight[group][2], v_bank2,
                bank2_dot0, bank2_dot1, bank2_dot2, bank2_dot3);
            local_dual_dot8_bank3(
                weight[group][3], weight[group][3], v_bank3,
                bank3_dot0, bank3_dot1, bank3_dot2, bank3_dot3);

            const float dot0 = (bank0_dot0 + bank1_dot0)
                + (bank2_dot0 + bank3_dot0);
            const float dot1 = (bank0_dot1 + bank1_dot1)
                + (bank2_dot1 + bank3_dot1);
            if (sum_step) {
#pragma HLS DEPENDENCE variable=merged_sum inter false
                merged_sum[group][0] = base_sum0[group] + dot0;
                merged_sum[group][1] = base_sum1[group] + dot1;
            } else {
#pragma HLS DEPENDENCE variable=output_state inter false
                output_state[context0][dim] = had_previous[group][0]
                    ? previous_scale[group][0] * output_state[context0][dim]
                        + dot0
                    : dot0;
                output_state[context1][dim] = had_previous[group][1]
                    ? previous_scale[group][1] * output_state[context1][dim]
                        + dot1
                    : dot1;
            }

            if (sum_step) {
                dim = 0;
                ++group;
            } else {
                ++dim;
            }
    }
}

static void process_group_weights_dataflow(
    int tile_start,
    const packed_score_pair_t score_pair[GROUPS_PER_BLOCK][TILE_K],
    const float new_max[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    const float previous_scale[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float merged_sum[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    const bool had_previous[GROUPS_PER_BLOCK][STATIC_GQA_LANES],
    float output_state[CONTEXTS_PER_BLOCK][HEAD_DIM],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<weight_pair_t> weight_stream("weight_stream");
#pragma HLS STREAM variable=weight_stream depth=64
#pragma HLS BIND_STORAGE variable=weight_stream type=fifo impl=bram

    produce_group_weights(score_pair, new_max, weight_stream);
    consume_group_weights(
        tile_start, weight_stream,
        previous_scale, merged_sum, had_previous,
        output_state, v_head_cache);
}

static void load_query_block(
    const mem_word_t q[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    int kv_head,
    int query_base,
    packed_q_bank_t q_triple_bank[3][4]
        [TRIPLE_BATCHES * WORDS_PER_ROW],
    float running_max[CONTEXTS_PER_BLOCK],
    float running_sum[CONTEXTS_PER_BLOCK]
) {
#pragma HLS INLINE off
    const int query_block = query_base / ROWS_PER_BLOCK;
V8S_IP_LOAD_Q_BLOCK:
    for (int step = 0;
         step < CONTEXTS_PER_BLOCK * WORDS_PER_ROW;
         ++step) {
#pragma HLS PIPELINE II=1
        const int context = step / WORDS_PER_ROW;
        const int word = step % WORDS_PER_ROW;
        const int batch = context / 3;
        const int slot = context % 3;
        const mem_word_t packed = q[block_io_word_index(
            kv_head, query_block, context, word)];
        const int address = batch * WORDS_PER_ROW + word;
U116_LOAD_TRIPLE_Q_BANKS:
        for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
            q_triple_bank[slot][bank][address] = packed.range(
                (bank + 1) * 128 - 1, bank * 128);
        }
        if (word == 0) {
            running_max[context] = -INFINITY;
            running_sum[context] = 0.0f;
        }
    }
U116_ZERO_PAD_LAST_TRIPLE_BATCH:
    for (int word = 0; word < WORDS_PER_ROW; ++word) {
#pragma HLS PIPELINE II=1
U116_ZERO_PAD_LAST_TRIPLE_BANKS:
        for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
            const int address = (TRIPLE_BATCHES - 1) * WORDS_PER_ROW + word;
            q_triple_bank[1][bank][address] = 0;
            q_triple_bank[2][bank][address] = 0;
        }
    }
}

static void load_kv_head(
    const mem_word_t k[(KV_HEADS * STORAGE_SEQ * HEAD_DIM) / VALUES_PER_WORD],
    const mem_word_t v_dimension_major[(KV_HEADS * STORAGE_SEQ * HEAD_DIM) / VALUES_PER_WORD],
    int kv_head,
    unsigned prompt_tokens,
    int active_tiles,
    mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
    const int k_head_base = kv_head * STORAGE_SEQ * WORDS_PER_ROW;
    const int v_head_base = kv_head * TILES_PER_KV_HEAD * HEAD_DIM;
    const int valid_k_words = static_cast<int>(prompt_tokens) * WORDS_PER_ROW;
    const int valid_v_words = active_tiles * HEAD_DIM;
    const int load_words = valid_k_words > valid_v_words
        ? valid_k_words : valid_v_words;
V8S_IP_LOAD_KV_HEAD_ONCE:
    for (int step = 0; step < load_words; ++step) {
#pragma HLS PIPELINE II=1
        if (step < valid_k_words) {
            k_head_cache[step] = k[k_head_base + step];
        }
        if (step < valid_v_words) {
            v_head_cache[step] = v_dimension_major[v_head_base + step];
        }
    }
}

static ap_int<DECODE_KV_QUANT_BITS> pack_quantize_kv13(
    float value, float inverse_scale) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    const float scaled = value * inverse_scale;
    int rounded = static_cast<int>(hls::roundf(scaled));
    if (rounded > DECODE_KV_QUANT_LIMIT) rounded = DECODE_KV_QUANT_LIMIT;
    if (rounded < -DECODE_KV_QUANT_LIMIT) rounded = -DECODE_KV_QUANT_LIMIT;
    return ap_int<DECODE_KV_QUANT_BITS>(rounded);
}

static float pack_row_abs_max(const float values[HEAD_DIM]) {
#pragma HLS INLINE off
    float peak = 0.0f;
V8S_IP_PACK_ROW_ABS_MAX:
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
        const float magnitude = hls::fabsf(values[dim]);
        peak = magnitude > peak ? magnitude : peak;
    }
    return peak;
}

static void pack_prompt_head_kv13(
    mem_word_t k[ATTENTION_IP_STORAGE_KV_WORDS],
    mem_word_t v[ATTENTION_IP_STORAGE_KV_WORDS],
    int kv_head,
    unsigned prompt_tokens,
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
    float k_values[HEAD_DIM];
    bf16_t v_tile[TILE_K][HEAD_DIM];
    float v_peaks[TILE_K];
#pragma HLS BIND_STORAGE variable=k_values type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=v_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=v_peaks type=ram_2p impl=lutram

V8S_IP_PACK_V_TILE:
    for (int tile = 0; tile < TILES_PER_KV_HEAD; ++tile) {
        const unsigned tile_base = static_cast<unsigned>(tile * TILE_K);
        if (tile_base < prompt_tokens) {
V8S_IP_PACK_CLEAR_V_PEAK:
            for (int lane = 0; lane < TILE_K; ++lane) {
#pragma HLS PIPELINE II=1
                v_peaks[lane] = 0.0f;
            }
V8S_IP_PACK_TRANSPOSE_V:
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                mem_word_t packed_v =
                    v_head_cache[tile * HEAD_DIM + dim];
V8S_IP_PACK_TRANSPOSE_V_LANE:
                for (int lane = 0; lane < TILE_K; ++lane) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=v_peaks inter RAW distance=32 true
                    const bf16_t v_bits =
                        bf16_t(packed_v.range(BF16_BITS - 1, 0));
                    packed_v >>= BF16_BITS;
                    v_tile[lane][dim] = v_bits;
                    const float magnitude =
                        hls::fabsf(bf16_to_float(v_bits));
                    const float old_peak = v_peaks[lane];
                    v_peaks[lane] = magnitude > old_peak
                        ? magnitude : old_peak;
                }
            }

V8S_IP_PACK_TOKEN:
            for (int lane = 0; lane < TILE_K; ++lane) {
                const unsigned token = tile_base + static_cast<unsigned>(lane);
                if (token < prompt_tokens) {
                    mem_word_t k_shift0 =
                        k_head_cache[token * WORDS_PER_ROW];
                    mem_word_t k_shift1 =
                        k_head_cache[token * WORDS_PER_ROW + 1];

V8S_IP_PACK_LOAD_ROWS:
                    for (int dim = 0; dim < HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
                        const bf16_t k_bits = dim < VALUES_PER_WORD
                            ? bf16_t(k_shift0.range(BF16_BITS - 1, 0))
                            : bf16_t(k_shift1.range(BF16_BITS - 1, 0));
                        if (dim < VALUES_PER_WORD) {
                            k_shift0 >>= BF16_BITS;
                        } else {
                            k_shift1 >>= BF16_BITS;
                        }
                        k_values[dim] = bf16_to_float(k_bits);
                    }

                    const float k_peak = pack_row_abs_max(k_values);
                    const bf16_t k_scale_bits = float_to_bf16_rne(
                        k_peak > 0.0f
                            ? k_peak * (1.0f / DECODE_KV_QUANT_LIMIT)
                            : 1.0f);
                    const float k_inverse_scale =
                        1.0f / bf16_to_float(k_scale_bits);
                    mem_word_t k_data0 = 0;
                    mem_word_t k_data1 = 0;
V8S_IP_PACK_QUANTIZE_K:
                    for (int item = 0;
                         item < DECODE_QUANT_VALUES_PER_CACHE_WORD; ++item) {
#pragma HLS PIPELINE II=1
                        const ap_int<DECODE_KV_QUANT_BITS> quantized0 =
                            pack_quantize_kv13(k_values[item], k_inverse_scale);
                        const ap_int<DECODE_KV_QUANT_BITS> quantized1 =
                            pack_quantize_kv13(
                                k_values[DECODE_QUANT_VALUES_PER_CACHE_WORD + item],
                                k_inverse_scale);
                        k_data0 >>= DECODE_KV_QUANT_BITS;
                        k_data1 >>= DECODE_KV_QUANT_BITS;
                        k_data0.range(MEM_WORD_BITS - 1,
                                      MEM_WORD_BITS - DECODE_KV_QUANT_BITS) =
                            quantized0;
                        k_data1.range(MEM_WORD_BITS - 1,
                                      MEM_WORD_BITS - DECODE_KV_QUANT_BITS) =
                            quantized1;
                    }
                    k_data0 >>= MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
                    k_data1 >>= MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
                    k_data0.range(MEM_WORD_BITS - 1,
                                  DECODE_SCALE_LOW_BIT) = 0;
                    k_data1.range(MEM_WORD_BITS - 1,
                                  DECODE_SCALE_LOW_BIT) = 0;
                    k_data0.range(DECODE_SCALE_LOW_BIT + 15,
                                  DECODE_SCALE_LOW_BIT) = k_scale_bits;

                    const float v_peak = v_peaks[lane];
                    const bf16_t v_scale_bits = float_to_bf16_rne(
                        v_peak > 0.0f
                            ? v_peak * (1.0f / DECODE_KV_QUANT_LIMIT)
                            : 1.0f);
                    const float v_inverse_scale =
                        1.0f / bf16_to_float(v_scale_bits);
                    mem_word_t v_data0 = 0;
                    mem_word_t v_data1 = 0;
V8S_IP_PACK_QUANTIZE_V:
                    for (int item = 0;
                         item < DECODE_QUANT_VALUES_PER_CACHE_WORD; ++item) {
#pragma HLS PIPELINE II=1
                        const ap_int<DECODE_KV_QUANT_BITS> quantized0 =
                            pack_quantize_kv13(
                                bf16_to_float(v_tile[lane][item]),
                                v_inverse_scale);
                        const ap_int<DECODE_KV_QUANT_BITS> quantized1 =
                            pack_quantize_kv13(
                                bf16_to_float(v_tile[lane]
                                    [DECODE_QUANT_VALUES_PER_CACHE_WORD + item]),
                                v_inverse_scale);
                        v_data0 >>= DECODE_KV_QUANT_BITS;
                        v_data1 >>= DECODE_KV_QUANT_BITS;
                        v_data0.range(MEM_WORD_BITS - 1,
                                      MEM_WORD_BITS - DECODE_KV_QUANT_BITS) =
                            quantized0;
                        v_data1.range(MEM_WORD_BITS - 1,
                                      MEM_WORD_BITS - DECODE_KV_QUANT_BITS) =
                            quantized1;
                    }
                    v_data0 >>= MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
                    v_data1 >>= MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
                    v_data0.range(MEM_WORD_BITS - 1,
                                  DECODE_SCALE_LOW_BIT) = 0;
                    v_data1.range(MEM_WORD_BITS - 1,
                                  DECODE_SCALE_LOW_BIT) = 0;
                    v_data0.range(DECODE_SCALE_LOW_BIT + 15,
                                  DECODE_SCALE_LOW_BIT) = v_scale_bits;

                    const int data_index =
                        (kv_head * DECODE_MAX_KV_TOKENS + token) *
                        DECODE_CACHE_WORDS_PER_ROW;
                    k[data_index] = k_data0;
                    k[data_index + 1] = k_data1;
                    v[data_index] = v_data0;
                    v[data_index + 1] = v_data1;
                }
            }
        }
    }
}

#ifndef __SYNTHESIS__
void fpt_test_pack_prompt_head_kv13(
    mem_word_t k[ATTENTION_IP_STORAGE_KV_WORDS],
    mem_word_t v[ATTENTION_IP_STORAGE_KV_WORDS],
    int kv_head,
    unsigned prompt_tokens,
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]) {
    pack_prompt_head_kv13(
        k, v, kv_head, prompt_tokens, k_head_cache, v_head_cache);
}
#endif

static void compute_tile_static_4lane(
    int query_base,
    int tile_start,
    float scale,
    const packed_q_bank_t q_triple_bank[3][4]
        [TRIPLE_BATCHES * WORDS_PER_ROW],
    float output_state[STATIC_GQA_LANES][GROUPS_PER_BLOCK][HEAD_DIM],
    float running_max[CONTEXTS_PER_BLOCK],
    float running_sum[CONTEXTS_PER_BLOCK],
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_sum complete dim=1

    float partial[STATIC_GQA_LANES][GROUPS_PER_BLOCK][TILE_K];
    uint32_t score_lane[STATIC_GQA_LANES][GROUPS_PER_BLOCK][TILE_K];
    packed_weight_triple_t weight_batch[TRIPLE_BATCHES][4][8];
    uint32_t tile_max_bits[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
    float new_max[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
    float previous_scale[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
    float base_sum[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
    float merged_sum[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
    bool had_previous[STATIC_GQA_LANES][GROUPS_PER_BLOCK];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=score_lane complete dim=1
#pragma HLS BIND_STORAGE variable=score_lane type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_batch complete dim=3
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_max_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=new_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=previous_scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=base_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=merged_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=had_previous complete dim=1
#pragma HLS ALLOCATION function instances=shared_dot_phase limit=1

    shared_dot_phase(
        false, query_base, tile_start, scale,
        q_triple_bank, weight_batch,
        previous_scale, base_sum, merged_sum, had_previous, output_state,
        k_head_cache, v_head_cache,
        partial, tile_max_bits, score_lane);

V8S_IP_PREPARE_MERGE:
    for (int step = 0; step < CONTEXTS_PER_BLOCK; ++step) {
#pragma HLS PIPELINE II=1
        const int group = step / STATIC_GQA_LANES;
        const int lane = step % STATIC_GQA_LANES;
        const int row = group / (GQA_GROUP / STATIC_GQA_LANES);
        const int pair = group % (GQA_GROUP / STATIC_GQA_LANES);
        const int context = row * GQA_GROUP
            + pair * STATIC_GQA_LANES + lane;
        const float old_max = running_max[context];
        const float old_sum = running_sum[context];
        const uint32_t tile_bits = tile_max_bits[lane][group];
        float tile_max;
        std::memcpy(&tile_max, &tile_bits, sizeof(tile_max));
        const float merged = old_max > tile_max ? old_max : tile_max;
        const float rescale = old_sum == 0.0f
            ? 0.0f : hls::expf(old_max - merged);
        new_max[lane][group] = merged;
        previous_scale[lane][group] = rescale;
        merged_sum[lane][group] = old_sum * rescale;
        had_previous[lane][group] = old_sum != 0.0f;
    }

U108_CAPTURE_BASE_SUMS:
    for (int group = 0; group < GROUPS_PER_BLOCK; ++group) {
#pragma HLS UNROLL
U108_CAPTURE_BASE_SUM_LANES:
        for (int lane = 0; lane < STATIC_GQA_LANES; ++lane) {
#pragma HLS UNROLL
            base_sum[lane][group] = merged_sum[lane][group];
        }
    }

    produce_batch_weights_lane_array(
        score_lane, new_max, weight_batch);
    shared_dot_phase(
        true, query_base, tile_start, scale,
        q_triple_bank, weight_batch,
        previous_scale, base_sum, merged_sum, had_previous, output_state,
        k_head_cache, v_head_cache,
        partial, tile_max_bits, score_lane);
V8S_IP_COMMIT_STATE:
    for (int group = 0; group < GROUPS_PER_BLOCK; ++group) {
#pragma HLS UNROLL
        const int row = group / (GQA_GROUP / STATIC_GQA_LANES);
        const int pair = group % (GQA_GROUP / STATIC_GQA_LANES);
        const int context_base =
            row * GQA_GROUP + pair * STATIC_GQA_LANES;
U108_COMMIT_LANES:
        for (int lane = 0; lane < STATIC_GQA_LANES; ++lane) {
#pragma HLS UNROLL
            const int context = context_base + lane;
            running_max[context] = new_max[lane][group];
            running_sum[context] = merged_sum[lane][group];
        }
    }
}

static void normalize_and_store(
    mem_word_t out[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    int kv_head,
    int query_base,
    int valid_query_rows,
    const float output_state[STATIC_GQA_LANES][GROUPS_PER_BLOCK][HEAD_DIM],
    const float running_sum[CONTEXTS_PER_BLOCK]
) {
#pragma HLS INLINE off
    float reciprocal_sum[CONTEXTS_PER_BLOCK];
#pragma HLS ARRAY_PARTITION variable=reciprocal_sum complete dim=1
V8S_IP_RECIP_SUM:
    for (int context = 0;
         context < valid_query_rows * GQA_GROUP;
         ++context) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=CONTEXTS_PER_BLOCK avg=16
        reciprocal_sum[context] = 1.0f / running_sum[context];
    }
V8S_IP_STORE_CONTEXT:
    for (int context = 0;
         context < valid_query_rows * GQA_GROUP;
         ++context) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=CONTEXTS_PER_BLOCK avg=16
        const int group = context / STATIC_GQA_LANES;
        const int lane = context % STATIC_GQA_LANES;

V8S_IP_STORE_WORD:
        for (int word = 0; word < WORDS_PER_ROW; ++word) {
            mem_word_t packed = 0;
V8S_IP_STORE_CHUNK:
            for (int chunk = 0; chunk < VALUES_PER_WORD / NORM_PAR_D; ++chunk) {
#pragma HLS PIPELINE II=1
V8S_IP_STORE_ITEMS:
                for (int item = 0; item < NORM_PAR_D; ++item) {
#pragma HLS UNROLL
                    const int within_word = chunk * NORM_PAR_D + item;
                    const int dim = word * VALUES_PER_WORD + within_word;
                    packed.range(
                        (within_word + 1) * BF16_BITS - 1,
                        within_word * BF16_BITS
                    ) = float_to_bf16_rne(
                        output_state[lane][group][dim] * reciprocal_sum[context]
                    );
                }
            }
            out[block_io_word_index(kv_head, query_base / ROWS_PER_BLOCK, context, word)] = packed;
        }
    }
}

static void process_frontier_group(
    const mem_word_t q[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    mem_word_t out[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    int kv_head,
    int frontier_tiles,
    unsigned prompt_tokens,
    float scale,
    packed_q_bank_t q_triple_bank[3][4]
        [TRIPLE_BATCHES * WORDS_PER_ROW],
    float output_state[STATIC_GQA_LANES][GROUPS_PER_BLOCK][HEAD_DIM],
    float running_max[CONTEXTS_PER_BLOCK],
    float running_sum[CONTEXTS_PER_BLOCK],
    const mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW],
    const mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=2
    const int frontier_query_base = (frontier_tiles - 1) * TILE_K;
    const int remaining_queries = static_cast<int>(prompt_tokens)
        - frontier_query_base;
    const int active_blocks_unclamped =
        (remaining_queries + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int active_blocks = active_blocks_unclamped < BLOCKS_PER_FRONTIER
        ? active_blocks_unclamped : BLOCKS_PER_FRONTIER;
V8S_IP_BLOCK_IN_FRONTIER:
    for (int local_block = 0;
         local_block < active_blocks;
         ++local_block) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=BLOCKS_PER_FRONTIER avg=8
        const int query_block =
            (frontier_tiles - 1) * BLOCKS_PER_FRONTIER + local_block;
        const int query_base = query_block * ROWS_PER_BLOCK;
        const int block_remaining = static_cast<int>(prompt_tokens) - query_base;
        const int valid_query_rows = block_remaining < ROWS_PER_BLOCK
            ? block_remaining : ROWS_PER_BLOCK;
        load_query_block(
            q, kv_head, query_base,
            q_triple_bank, running_max, running_sum
        );
V8S_IP_RUNTIME_FRONTIER_TILE:
        for (int tile_index = 0;
             tile_index < frontier_tiles;
             ++tile_index) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILES_PER_KV_HEAD avg=8
            compute_tile_static_4lane(
                query_base, tile_index * TILE_K, scale,
                q_triple_bank, output_state,
                running_max, running_sum,
                k_head_cache, v_head_cache
            );
        }
        normalize_and_store(
            out, kv_head, query_base, valid_query_rows,
            output_state, running_sum
        );
    }
}
extern "C" void fpt_prefill_p2_core(
    const mem_word_t q[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    mem_word_t k[(KV_HEADS * STORAGE_SEQ * HEAD_DIM) / VALUES_PER_WORD],
    mem_word_t v[(KV_HEADS * STORAGE_SEQ * HEAD_DIM) / VALUES_PER_WORD],
    mem_word_t out[(Q_HEADS * SEQ * HEAD_DIM) / VALUES_PER_WORD],
    unsigned prompt_tokens
) {

    if (prompt_tokens == 0 || prompt_tokens > STORAGE_SEQ) {
        return;
    }

    const float scale = 1.0f / hls::sqrtf(static_cast<float>(HEAD_DIM));
    const int active_tiles =
        (static_cast<int>(prompt_tokens) + TILE_K - 1) / TILE_K;
    packed_q_bank_t q_triple_bank[3][4]
        [TRIPLE_BATCHES * WORDS_PER_ROW];
    float output_state[STATIC_GQA_LANES][GROUPS_PER_BLOCK][HEAD_DIM];
    float running_max[CONTEXTS_PER_BLOCK];
    float running_sum[CONTEXTS_PER_BLOCK];
    mem_word_t k_head_cache[STORAGE_SEQ * WORDS_PER_ROW];
    mem_word_t v_head_cache[TILES_PER_KV_HEAD * HEAD_DIM];

#pragma HLS ARRAY_PARTITION variable=output_state complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_triple_bank complete dim=2
#pragma HLS BIND_STORAGE variable=q_triple_bank type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=output_state type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=k_head_cache type=ram_1p impl=uram
#pragma HLS BIND_STORAGE variable=v_head_cache type=ram_1p impl=uram

V8S_IP_KV_HEAD:
    for (int kv_head = 0; kv_head < KV_HEADS; ++kv_head) {
        load_kv_head(
            k, v, kv_head, prompt_tokens, active_tiles,
            k_head_cache, v_head_cache);
V8S_IP_RUNTIME_FRONTIER:
        for (int frontier_tiles = 1;
             frontier_tiles <= active_tiles;
             ++frontier_tiles) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILES_PER_KV_HEAD avg=8
            process_frontier_group(
                q, out, kv_head, frontier_tiles, prompt_tokens,
                scale, q_triple_bank,
                output_state, running_max, running_sum,
                k_head_cache, v_head_cache
            );
        }
        pack_prompt_head_kv13(
            k, v, kv_head, prompt_tokens,
            k_head_cache, v_head_cache);
    }
}
