#include "unified_ip.hpp"

extern "C" void int8_gqa_prefill_decode_ip(
    unsigned mode,
    const unified_mem_word_t gmem0[32768],
    unified_mem_word_t gmem1[8192],
    unified_mem_word_t gmem2[8192],
    unified_mem_word_t gmem3[32768],
    unsigned decode_position) {
#pragma HLS INTERFACE m_axi port=gmem0 bundle=gmem0 offset=slave depth=32768 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem1 bundle=gmem1 offset=slave depth=8192 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=gmem2 bundle=gmem2 offset=slave depth=8192 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=gmem3 bundle=gmem3 offset=slave depth=32768 max_write_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=gmem0 bundle=control
#pragma HLS INTERFACE s_axilite port=gmem1 bundle=control
#pragma HLS INTERFACE s_axilite port=gmem2 bundle=control
#pragma HLS INTERFACE s_axilite port=gmem3 bundle=control
#pragma HLS INTERFACE s_axilite port=decode_position bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  if (mode == FPT_MODE_PREFILL) {
    fpt_prefill_p2_core(gmem0, gmem1, gmem2, gmem3, decode_position);
  } else if (mode == FPT_MODE_DECODE) {
    fpt_decode_q14_kv13_row_core(
        gmem0 + DECODE_GMEM0_Q_OFFSET, gmem1, gmem2,
        gmem0 + DECODE_GMEM0_K_NEW_OFFSET,
        gmem0 + DECODE_GMEM0_V_NEW_OFFSET, gmem3, decode_position);
  }
}
