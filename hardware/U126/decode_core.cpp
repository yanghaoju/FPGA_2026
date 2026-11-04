#include "decode_core.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <hls_math.h>

namespace {

using bf16_t = ap_uint<16>;
using qint14_t = ap_int<DECODE_Q_QUANT_BITS>;
using kvint13_t = ap_int<DECODE_KV_QUANT_BITS>;
// Keep one guard bit over the mathematical product and the same reduction
// widths used by the U75 tree: eight products add three bits and D64 adds six.
using qint28_t = ap_int<DECODE_Q_QUANT_BITS + DECODE_KV_QUANT_BITS + 1>;
using qint31_t = ap_int<DECODE_Q_QUANT_BITS + DECODE_KV_QUANT_BITS + 4>;
using qint34_t = ap_int<DECODE_Q_QUANT_BITS + DECODE_KV_QUANT_BITS + 7>;

float bf16_to_float(bf16_t value) {
#pragma HLS INLINE
  const std::uint32_t bits = static_cast<std::uint32_t>(value.to_uint()) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bf16_t float_to_bf16_rne(float value) {
#pragma HLS INLINE
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return bf16_t(bits >> 16);
}

// row_abs_max() returns a non-negative finite value.  For that restricted
// domain, testing the IEEE-754 magnitude bits is exactly equivalent to
// `peak > 0.0f`, while avoiding a full floating-point comparator on the
// query-scale control path.
bool nonzero_nonnegative_float(float value) {
#pragma HLS INLINE
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7FFFFFFFU) != 0U;
}

float softmax_exp_lut(float value) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
  static const float table[257] = {
    1.0f, 0.93941306281347581f, 0.88249690258459546f, 0.82902911818040037f,
    0.77880078307140488f, 0.73161562894664178f, 0.68728927879097224f, 0.645648526427892f,
    0.60653065971263342f, 0.569782824730923f, 0.53526142851899028f, 0.50283157797094091f,
    0.47236655274101469f, 0.44374731008107987f, 0.41686201967850839f, 0.391605626676799f,
    0.36787944117144233f, 0.34559075257697452f, 0.32465246735834974f, 0.30498276871105928f,
    0.28650479686019009f, 0.26914634872918386f, 0.25283959580474646f, 0.23752081909545814f,
    0.22313016014842982f, 0.20961138715109781f, 0.19691167520419406f, 0.18498139990730428f,
    0.17377394345044514f, 0.16324551245395841f, 0.15335496684492847f, 0.14406365910145327f,
    0.1353352832366127f, 0.12713573293203559f, 0.11943296826671962f, 0.11219689052034373f,
    0.10539922456186433f, 0.0990134083638263f, 0.093014489210663492f, 0.087379026195420387f,
    0.0820849986238988f, 0.077111719968316714f, 0.072439757034251456f, 0.0680508540250102f,
    0.06392786120670757f, 0.060054667895307945f, 0.05641613950377735f, 0.0529980584033558f,
    0.049787068367863944f, 0.04677062238395898f, 0.04393693362340742f, 0.041274929385797549f,
    0.038774207831722009f, 0.036424997337364234f, 0.034218118311666032f, 0.032144947326876068f,
    0.0301973834223185f, 0.0283678164497131f, 0.026649097336355485f, 0.025034510149960148f,
    0.023517745856009107f, 0.022092877665062443f, 0.020754337873699742f, 0.019496896108597995f,
    0.018315638888734179f, 0.017205950425851383f, 0.016163494588165874f, 0.015184197956837946f,
    0.014264233908999256f, 0.013400007665140828f, 0.012588142242433998f, 0.011825465259096618f,
    0.011108996538242306f, 0.010435936462774504f, 0.0098036550358218278f, 0.00920968160396814f,
    0.0086516952031206341f, 0.008127515489292211f, 0.0076350942188599617f, 0.0071725072450086989f,
    0.006737946999085467f, 0.006329715427485747f, 0.0059462173564720942f, 0.0055859542589981f,
    0.0052475183991813846f, 0.0049295873315450519f, 0.0046309187335332458f, 0.0043503455511087691f,
    0.0040867714384640666f, 0.0038391664740261636f, 0.0036065631360157305f, 0.0033880525218347116f,
    0.0031827807965096669f, 0.0029899458563130603f, 0.0028087941945255128f, 0.0026386179570919216f,
    0.0024787521766663585f, 0.0023285721742377138f, 0.0021874911181828851f, 0.0020549577312094589f,
    0.0019304541362277093f, 0.0018134938327346152f, 0.001703619795802574f, 0.0016004026902445643f,
    0.0015034391929775724f, 0.0014123504170288816f, 0.0013267804310269915f, 0.0012463948683920495f,
    0.0011708796207911744f, 0.0010999396107533182f, 0.001033297638647637f, 0.00097069329951990889f,
    0.00091188196555451624f, 0.00085663383018594048f, 0.00080473301012461325f, 0.00075597670178827073f,
    0.000710174388842549f, 0.00066714709775426734f, 0.00062672669844845757f, 0.00058875524733644315f,
    0.00055308437014783363f, 0.00051957468215483844f, 0.000488095243523415f, 0.00045852304766302063f,
    0.00043074254057568753f, 0.00040464516932626452f, 0.00038012895786946368f, 0.00035709810857624758f,
    0.00033546262790251185f, 0.000315137974737356f, 0.00029604473005685538f, 0.00027810828659249914f,
    0.00026125855730166754f, 0.00024542970150098948f, 0.00023055986759244163f, 0.00021659095137688503f,
    0.00020346836901064417f, 0.00019114084371795175f, 0.00017956020542583297f, 0.00016868120253849865f,
    0.00015846132511575126f, 0.00014886063876446985f, 0.00013984162859410104f, 0.00013136905262640898f,
    0.00012340980408667956f, 0.00011593278203827863f, 0.00010890876985506645f, 0.00010231032105679591f,
    9.61116520613947E-05f, 9.0288541435057907E-05f, 8.4818235246469155E-05f, 7.96793581553195E-05f,
    7.48518298877006E-05f, 7.0316786771998079E-05f, 6.6056508028684814E-05f, 6.205434652598976E-05f,
    5.8294663730868811E-05f, 5.476276860109711E-05f, 5.1444860179702278E-05f, 4.8327973667425135E-05f,
    4.5399929762484854E-05f, 4.2649287069692573E-05f, 4.0065297392951069E-05f, 3.7637863736444928E-05f,
    3.5357500850409981E-05f, 3.3215298167313715E-05f, 3.1202884983619004E-05f, 2.9312397751078144E-05f,
    2.7536449349747158E-05f, 2.586810022265412E-05f, 2.4300831259329461E-05f, 2.2828518322240145E-05f,
    2.1445408316589164E-05f, 2.0146096709972611E-05f, 1.892550641405186E-05f, 1.7778867945720539E-05f,
    1.6701700790245659E-05f, 1.5689795893558925E-05f, 1.4739199215286483E-05f, 1.3846196278250254E-05f,
    1.300729765406762E-05f, 1.2219225328134202E-05f, 1.147889989071055E-05f, 1.0783428504061669E-05f,
    1.0130093598630711E-05f, 9.51634225407686E-06f, 8.93977622368364E-06f, 8.3981425631577358E-06f,
    7.8893248272002229E-06f, 7.4113347994505574E-06f, 6.9623047234879457E-06f, 6.54048000453254E-06f,
    6.14421235332821E-06f, 5.7719533454164476E-06f, 5.4222483706341527E-06f, 5.093730949192808E-06f,
    4.7851173921290088E-06f, 4.4952017852619439E-06f, 4.2228512770575271E-06f, 3.9670016519864093E-06f,
    3.7266531720786709E-06f, 3.5008666704259795E-06f, 3.2887598813664843E-06f, 3.0895039930125723E-06f,
    2.9023204086504041E-06f, 2.7264777043563346E-06f, 2.5612887709420388E-06f, 2.4061081290604234E-06f,
    2.2603294069810542E-06f, 2.1233829711794397E-06f, 1.994733700481656E-06f, 1.8738788950667307E-06f,
    1.7603463121561693E-06f, 1.6536923207150339E-06f, 1.5535001679540347E-06f, 1.4593783508589488E-06f,
    1.3709590863840845E-06f, 1.2878968743320372E-06f, 1.2098671473041613E-06f, 1.1365650024464047E-06f,
    1.0677040100347827E-06f, 1.0030150942450052E-06f, 9.4224548173284748E-07f, 8.8515771391681325E-07f,
    8.3152871910356788E-07f, 7.811489408304491E-07f, 7.3382151901903473E-07f, 6.8936152074010868E-07f,
    6.4759521758422093E-07f, 6.0835940681415228E-07f, 5.7150077364667207E-07f, 5.3687529217169119E-07f,
    5.04347662567888E-07f, 4.7379078241571714E-07f, 4.4508525004194189E-07f, 4.1811889795500234E-07f,
    3.9278635454810392E-07f, 3.6898863235737409E-07f, 3.4663274126619643E-07f, 3.2563132514430866E-07f,
    3.0590232050182579E-07f, 2.8736863582436966E-07f, 2.6995785033630141E-07f, 2.5360193101496682E-07f,
    2.382369667501818E-07f, 2.2380291861018046E-07f, 2.1024338523818466E-07f, 1.9750538246287656E-07f,
    1.8553913626159784E-07f, 1.7429788826727445E-07f, 1.6373771305908126E-07f, 1.5381734652290559E-07f,
    1.4449802461092448E-07f, 1.3574333187024558E-07f, 1.2751905914873347E-07f, 1.197930699220045E-07f,
    1.1253517471925912E-07f
  };
#pragma HLS BIND_STORAGE variable=table type=rom_2p impl=lutram

  if (value >= 0.0f) return 1.0f;
  if (value <= -16.0f) return 0.0f;
  const float scaled = -value * 16.0f;
  const int index = static_cast<int>(scaled);
  const float fraction = scaled - static_cast<float>(index);
  const float lower = table[index];
  const float upper = table[index + 1];
  return lower + (upper - lower) * fraction;
}
bf16_t read_lane(const decode_mem_word_t& word, int lane) {
#pragma HLS INLINE
  return word.range((lane + 1) * 16 - 1, lane * 16);
}

void write_lane(decode_mem_word_t& word, int lane, bf16_t value) {
#pragma HLS INLINE
  word.range((lane + 1) * 16 - 1, lane * 16) = value;
}

int cache_data_word_index(int kv_head, int token, int word) {
#pragma HLS INLINE
  return (kv_head * DECODE_MAX_KV_TOKENS + token) *
      DECODE_CACHE_WORDS_PER_ROW + word;
}

int head_word_index(int head, int word) {
#pragma HLS INLINE
  return head * DECODE_WORDS_PER_ROW + word;
}

void load_head_vector(const decode_mem_word_t* words, int head,
                      float out[DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  for (int word = 0; word < DECODE_WORDS_PER_ROW; ++word) {
#pragma HLS PIPELINE II=1
    const decode_mem_word_t packed = words[head_word_index(head, word)];
    for (int lane = 0; lane < DECODE_VALUES_PER_WORD; ++lane) {
#pragma HLS UNROLL
      out[word * DECODE_VALUES_PER_WORD + lane] =
          bf16_to_float(read_lane(packed, lane));
    }
  }
}

float row_abs_max(const float values[DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  float peak = 0.0f;
  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
    const float magnitude = hls::fabsf(values[dim]);
    peak = magnitude > peak ? magnitude : peak;
  }
  return peak;
}

qint14_t quantize_q14_with_inv(float value, float inverse_scale) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
  const float scaled = value * inverse_scale;
  int rounded = static_cast<int>(hls::roundf(scaled));
  if (rounded > DECODE_Q_QUANT_LIMIT) rounded = DECODE_Q_QUANT_LIMIT;
  if (rounded < -DECODE_Q_QUANT_LIMIT) rounded = -DECODE_Q_QUANT_LIMIT;
  return qint14_t(rounded);
}

kvint13_t quantize_kv13_with_inv(float value, float inverse_scale) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
  const float scaled = value * inverse_scale;
  int rounded = static_cast<int>(hls::roundf(scaled));
  if (rounded > DECODE_KV_QUANT_LIMIT) rounded = DECODE_KV_QUANT_LIMIT;
  if (rounded < -DECODE_KV_QUANT_LIMIT) rounded = -DECODE_KV_QUANT_LIMIT;
  return kvint13_t(rounded);
}

void load_bf16_row(const decode_mem_word_t* source, int head,
                   float values[DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  for (int word = 0; word < DECODE_WORDS_PER_ROW; ++word) {
#pragma HLS PIPELINE II=1
    const decode_mem_word_t packed = source[head_word_index(head, word)];
    for (int lane = 0; lane < DECODE_VALUES_PER_WORD; ++lane) {
#pragma HLS UNROLL
      values[word * DECODE_VALUES_PER_WORD + lane] =
          bf16_to_float(read_lane(packed, lane));
    }
  }
}

void store_cache_rows_kv13(decode_mem_word_t* k_cache,
                           decode_mem_word_t* v_cache,
                           int kv_head, int token,
                           const decode_mem_word_t* k_source,
                           const decode_mem_word_t* v_source) {
#pragma HLS INLINE off
  float k_values[DECODE_HEAD_DIM];
  float v_values[DECODE_HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=k_values cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=v_values cyclic factor=8 dim=1
  load_bf16_row(k_source, kv_head, k_values);
  load_bf16_row(v_source, kv_head, v_values);

  decode_mem_word_t k_data0 = 0;
  decode_mem_word_t k_data1 = 0;
  decode_mem_word_t v_data0 = 0;
  decode_mem_word_t v_data1 = 0;
  const float k_peak = row_abs_max(k_values);
  const bf16_t k_scale_bits = float_to_bf16_rne(
      k_peak > 0.0f ? k_peak * (1.0f / DECODE_KV_QUANT_LIMIT) : 1.0f);
  const float k_scale = bf16_to_float(k_scale_bits);
  const float k_inverse_scale = 1.0f / k_scale;

  // Each 512-bit word stores 32 INT13 values in bits [415:0]. The BF16
  // scale occupies bits [431:416] of word 0.
  for (int lane = 0; lane < DECODE_QUANT_VALUES_PER_CACHE_WORD; ++lane) {
#pragma HLS PIPELINE II=1
    const kvint13_t quantized0 = quantize_kv13_with_inv(
        k_values[lane], k_inverse_scale);
    const kvint13_t quantized1 = quantize_kv13_with_inv(
        k_values[DECODE_QUANT_VALUES_PER_CACHE_WORD + lane], k_inverse_scale);
    k_data0 >>= DECODE_KV_QUANT_BITS;
    k_data1 >>= DECODE_KV_QUANT_BITS;
    k_data0.range(DECODE_MEM_WORD_BITS - 1,
                  DECODE_MEM_WORD_BITS - DECODE_KV_QUANT_BITS) = quantized0;
    k_data1.range(DECODE_MEM_WORD_BITS - 1,
                  DECODE_MEM_WORD_BITS - DECODE_KV_QUANT_BITS) = quantized1;
  }
  k_data0 >>= DECODE_MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
  k_data1 >>= DECODE_MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
  k_data0.range(DECODE_SCALE_LOW_BIT + 15, DECODE_SCALE_LOW_BIT) =
      k_scale_bits;

  const float v_peak = row_abs_max(v_values);
  const bf16_t v_scale_bits = float_to_bf16_rne(
      v_peak > 0.0f ? v_peak * (1.0f / DECODE_KV_QUANT_LIMIT) : 1.0f);
  const float v_scale = bf16_to_float(v_scale_bits);
  const float v_inverse_scale = 1.0f / v_scale;
  for (int lane = 0; lane < DECODE_QUANT_VALUES_PER_CACHE_WORD; ++lane) {
#pragma HLS PIPELINE II=1
    const kvint13_t quantized0 = quantize_kv13_with_inv(
        v_values[lane], v_inverse_scale);
    const kvint13_t quantized1 = quantize_kv13_with_inv(
        v_values[DECODE_QUANT_VALUES_PER_CACHE_WORD + lane], v_inverse_scale);
    v_data0 >>= DECODE_KV_QUANT_BITS;
    v_data1 >>= DECODE_KV_QUANT_BITS;
    v_data0.range(DECODE_MEM_WORD_BITS - 1,
                  DECODE_MEM_WORD_BITS - DECODE_KV_QUANT_BITS) = quantized0;
    v_data1.range(DECODE_MEM_WORD_BITS - 1,
                  DECODE_MEM_WORD_BITS - DECODE_KV_QUANT_BITS) = quantized1;
  }
  v_data0 >>= DECODE_MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
  v_data1 >>= DECODE_MEM_WORD_BITS - DECODE_SCALE_LOW_BIT;
  v_data0.range(DECODE_SCALE_LOW_BIT + 15, DECODE_SCALE_LOW_BIT) =
      v_scale_bits;

  k_cache[cache_data_word_index(kv_head, token, 0)] = k_data0;
  k_cache[cache_data_word_index(kv_head, token, 1)] = k_data1;
  v_cache[cache_data_word_index(kv_head, token, 0)] = v_data0;
  v_cache[cache_data_word_index(kv_head, token, 1)] = v_data1;
}

float reduce4_tree(const float values[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
  return (values[0] + values[1]) + (values[2] + values[3]);
}

float reduce8_tree(const float values[8]) {
#pragma HLS INLINE
  const float pair0 = values[0] + values[1];
  const float pair1 = values[2] + values[3];
  const float pair2 = values[4] + values[5];
  const float pair3 = values[6] + values[7];
  return (pair0 + pair1) + (pair2 + pair3);
}

float reduce16_tree(const float values[16]) {
#pragma HLS INLINE
  return reduce8_tree(values) + reduce8_tree(values + 8);
}

void load_tile(
    const decode_mem_word_t* k_cache,
    const decode_mem_word_t* v_cache,
    int kv_head,
    int tile_base,
    int tile_count,
    decode_tile_word_t
        kv_tile[DECODE_TILE_CHUNKS_PER_ROW][DECODE_COLOCATED_TILE_K],
    float k_scale[DECODE_TILE_K][DECODE_GROUPS_PER_ROW],
    float v_scale[DECODE_TILE_K]) {
#pragma HLS INLINE off
  for (int token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE II=1
    const int row = tile_base + token;
    const decode_mem_word_t k0 =
        k_cache[cache_data_word_index(kv_head, row, 0)];
    const decode_mem_word_t k1 =
        k_cache[cache_data_word_index(kv_head, row, 1)];
    kv_tile[0][token] = k0.range(DECODE_TILE_WORD_BITS - 1, 0);
    kv_tile[1][token] = k0.range(2 * DECODE_TILE_WORD_BITS - 1, DECODE_TILE_WORD_BITS);
    kv_tile[2][token] = k0.range(3 * DECODE_TILE_WORD_BITS - 1, 2 * DECODE_TILE_WORD_BITS);
    kv_tile[3][token] = k0.range(4 * DECODE_TILE_WORD_BITS - 1, 3 * DECODE_TILE_WORD_BITS);
    kv_tile[4][token] = k1.range(DECODE_TILE_WORD_BITS - 1, 0);
    kv_tile[5][token] = k1.range(2 * DECODE_TILE_WORD_BITS - 1, DECODE_TILE_WORD_BITS);
    kv_tile[6][token] = k1.range(3 * DECODE_TILE_WORD_BITS - 1, 2 * DECODE_TILE_WORD_BITS);
    kv_tile[7][token] = k1.range(4 * DECODE_TILE_WORD_BITS - 1, 3 * DECODE_TILE_WORD_BITS);
    k_scale[token][0] = bf16_to_float(
        bf16_t(k0.range(DECODE_SCALE_LOW_BIT + 15, DECODE_SCALE_LOW_BIT)));
  }

  for (int token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE II=1
    const int row = tile_base + token;
    const decode_mem_word_t v0 =
        v_cache[cache_data_word_index(kv_head, row, 0)];
    const decode_mem_word_t v1 =
        v_cache[cache_data_word_index(kv_head, row, 1)];
    const int v_token = DECODE_V_TILE_OFFSET + token;
    kv_tile[0][v_token] = v0.range(DECODE_TILE_WORD_BITS - 1, 0);
    kv_tile[1][v_token] = v0.range(2 * DECODE_TILE_WORD_BITS - 1, DECODE_TILE_WORD_BITS);
    kv_tile[2][v_token] = v0.range(3 * DECODE_TILE_WORD_BITS - 1, 2 * DECODE_TILE_WORD_BITS);
    kv_tile[3][v_token] = v0.range(4 * DECODE_TILE_WORD_BITS - 1, 3 * DECODE_TILE_WORD_BITS);
    kv_tile[4][v_token] = v1.range(DECODE_TILE_WORD_BITS - 1, 0);
    kv_tile[5][v_token] = v1.range(2 * DECODE_TILE_WORD_BITS - 1, DECODE_TILE_WORD_BITS);
    kv_tile[6][v_token] = v1.range(3 * DECODE_TILE_WORD_BITS - 1, 2 * DECODE_TILE_WORD_BITS);
    kv_tile[7][v_token] = v1.range(4 * DECODE_TILE_WORD_BITS - 1, 3 * DECODE_TILE_WORD_BITS);
    v_scale[token] = bf16_to_float(
        bf16_t(v0.range(DECODE_SCALE_LOW_BIT + 15, DECODE_SCALE_LOW_BIT)));
  }
}

void compute_head_tile_pair(
    const decode_tile_word_t
        kv_tile[DECODE_TILE_CHUNKS_PER_ROW][DECODE_COLOCATED_TILE_K],
    const float k_scale[DECODE_TILE_K][DECODE_GROUPS_PER_ROW],
    const float v_scale[DECODE_TILE_K],
    int tile_count,
    bool first_tile,
    int pair_base,
    const qint14_t q_group[DECODE_GQA_GROUP][DECODE_HEAD_DIM],
    const float q_scale[DECODE_GQA_GROUP][DECODE_GROUPS_PER_ROW],
    float running_max[DECODE_GQA_GROUP],
    float running_sum[DECODE_GQA_GROUP],
    float running_output[DECODE_GQA_GROUP][DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  float scores[2][DECODE_TILE_K];
  float weights[2][DECODE_TILE_K];
  qint31_t qk_chunk_sum[2][DECODE_TILE_K][DECODE_TILE_CHUNKS_PER_ROW];
  float partial_sum[2][DECODE_ACCUM_SLOTS];
  float partial_output[2][DECODE_ACCUM_SLOTS][DECODE_HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=scores complete dim=0
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=qk_chunk_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=qk_chunk_sum complete dim=3
#pragma HLS BIND_STORAGE variable=qk_chunk_sum type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=partial_sum complete dim=0
#pragma HLS ARRAY_PARTITION variable=partial_output complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial_output complete dim=2
#pragma HLS ARRAY_PARTITION variable=partial_output cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=partial_output type=ram_2p impl=lutram

  float tile_max[2] = {-INFINITY, -INFINITY};
#pragma HLS ARRAY_PARTITION variable=tile_max complete dim=1

  for (int flat = 0;
       flat < tile_count * DECODE_TILE_CHUNKS_PER_ROW; ++flat) {
#pragma HLS PIPELINE II=1
    const int token = flat / DECODE_TILE_CHUNKS_PER_ROW;
    const int chunk = flat % DECODE_TILE_CHUNKS_PER_ROW;
    const decode_tile_word_t packed = kv_tile[chunk][token];
    qint28_t products[2][DECODE_VALUES_PER_TILE_WORD];
#pragma HLS ARRAY_PARTITION variable=products complete dim=0
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      const int group_q = pair_base + query_lane;
      for (int lane = 0; lane < DECODE_VALUES_PER_TILE_WORD; ++lane) {
#pragma HLS UNROLL
        const int dim = chunk * DECODE_VALUES_PER_TILE_WORD + lane;
        const kvint13_t element = packed.range(
            (lane + 1) * DECODE_KV_QUANT_BITS - 1,
            lane * DECODE_KV_QUANT_BITS);
        products[query_lane][lane] = q_group[group_q][dim] * element;
      }
      const qint31_t level4_0 =
          products[query_lane][0] + products[query_lane][1];
      const qint31_t level4_1 =
          products[query_lane][2] + products[query_lane][3];
      const qint31_t level4_2 =
          products[query_lane][4] + products[query_lane][5];
      const qint31_t level4_3 =
          products[query_lane][6] + products[query_lane][7];
      qk_chunk_sum[query_lane][token][chunk] =
          (level4_0 + level4_1) + (level4_2 + level4_3);
    }
  }

  for (int token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE II=1
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      const int group_q = pair_base + query_lane;
      const qint34_t level4_0 =
          qint34_t(qk_chunk_sum[query_lane][token][0]) +
          qint34_t(qk_chunk_sum[query_lane][token][1]);
      const qint34_t level4_1 =
          qint34_t(qk_chunk_sum[query_lane][token][2]) +
          qint34_t(qk_chunk_sum[query_lane][token][3]);
      const qint34_t level4_2 =
          qint34_t(qk_chunk_sum[query_lane][token][4]) +
          qint34_t(qk_chunk_sum[query_lane][token][5]);
      const qint34_t level4_3 =
          qint34_t(qk_chunk_sum[query_lane][token][6]) +
          qint34_t(qk_chunk_sum[query_lane][token][7]);
      const qint34_t integer_sum =
          (level4_0 + level4_1) + (level4_2 + level4_3);
      scores[query_lane][token] =
          static_cast<float>(integer_sum.to_int64()) *
          q_scale[group_q][0] * k_scale[token][0] * 0.125f;
    }
  }

  for (int token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE II=1
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      if (scores[query_lane][token] > tile_max[query_lane]) {
        tile_max[query_lane] = scores[query_lane][token];
      }
    }
  }

  for (int flat = 0;
       flat < DECODE_ACCUM_SLOTS * DECODE_HEAD_DIM; ++flat) {
#pragma HLS PIPELINE II=1
    const int slot = flat / DECODE_HEAD_DIM;
    const int dim = flat % DECODE_HEAD_DIM;
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      partial_output[query_lane][slot][dim] = 0.0f;
    }
  }

  for (int token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE II=1
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      weights[query_lane][token] = softmax_exp_lut(
          scores[query_lane][token] - tile_max[query_lane]);
    }
  }

  for (int slot = 0; slot < DECODE_ACCUM_SLOTS; ++slot) {
#pragma HLS PIPELINE II=1
    float slot_weights[2][DECODE_TILE_K / DECODE_ACCUM_SLOTS];
#pragma HLS ARRAY_PARTITION variable=slot_weights complete dim=0
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      for (int token_group = 0;
           token_group < DECODE_TILE_K / DECODE_ACCUM_SLOTS;
           ++token_group) {
#pragma HLS UNROLL
        const int token = slot + token_group * DECODE_ACCUM_SLOTS;
        slot_weights[query_lane][token_group] = token < tile_count
            ? weights[query_lane][token] : 0.0f;
      }
      partial_sum[query_lane][slot] =
          reduce16_tree(slot_weights[query_lane]);
    }
  }

  for (int flat = 0;
       flat < tile_count * DECODE_TILE_CHUNKS_PER_ROW; ++flat) {
#pragma HLS PIPELINE II=1
    const int token = flat / DECODE_TILE_CHUNKS_PER_ROW;
    const int chunk = flat % DECODE_TILE_CHUNKS_PER_ROW;
    const int slot = token % DECODE_ACCUM_SLOTS;
    const decode_tile_word_t packed =
        kv_tile[chunk][DECODE_V_TILE_OFFSET + token];
    float scaled_weight[2];
#pragma HLS ARRAY_PARTITION variable=scaled_weight complete dim=1
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      scaled_weight[query_lane] =
          weights[query_lane][token] * v_scale[token];
      for (int lane = 0; lane < DECODE_VALUES_PER_TILE_WORD; ++lane) {
#pragma HLS UNROLL
        const int dim = chunk * DECODE_VALUES_PER_TILE_WORD + lane;
        const kvint13_t element = packed.range(
            (lane + 1) * DECODE_KV_QUANT_BITS - 1,
            lane * DECODE_KV_QUANT_BITS);
        partial_output[query_lane][slot][dim] +=
            static_cast<float>(element.to_int()) *
            scaled_weight[query_lane];
      }
    }
  }

  float alpha[2];
  float beta[2];
  float merged_max[2];
#pragma HLS ARRAY_PARTITION variable=alpha complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=merged_max complete dim=1
  for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
    const int group_q = pair_base + query_lane;
    if (first_tile) {
      merged_max[query_lane] = tile_max[query_lane];
      alpha[query_lane] = 0.0f;
      beta[query_lane] = 1.0f;
    } else if (running_max[group_q] >= tile_max[query_lane]) {
      merged_max[query_lane] = running_max[group_q];
      alpha[query_lane] = 1.0f;
      beta[query_lane] = softmax_exp_lut(
          tile_max[query_lane] - running_max[group_q]);
    } else {
      merged_max[query_lane] = tile_max[query_lane];
      alpha[query_lane] = softmax_exp_lut(
          running_max[group_q] - tile_max[query_lane]);
      beta[query_lane] = 1.0f;
    }
    float slot_sum[DECODE_ACCUM_SLOTS];
#pragma HLS ARRAY_PARTITION variable=slot_sum complete dim=1
    for (int slot = 0; slot < DECODE_ACCUM_SLOTS; ++slot) {
#pragma HLS UNROLL
      slot_sum[slot] = partial_sum[query_lane][slot];
    }
    const float tile_sum = reduce4_tree(slot_sum);
    running_sum[group_q] = alpha[query_lane] * running_sum[group_q] +
        beta[query_lane] * tile_sum;
  }

  for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
    float slot_value[2][DECODE_ACCUM_SLOTS];
#pragma HLS ARRAY_PARTITION variable=slot_value complete dim=0
    for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
      const int group_q = pair_base + query_lane;
      for (int slot = 0; slot < DECODE_ACCUM_SLOTS; ++slot) {
#pragma HLS UNROLL
        slot_value[query_lane][slot] =
            partial_output[query_lane][slot][dim];
      }
      const float tile_value = reduce4_tree(slot_value[query_lane]);
      running_output[group_q][dim] =
          alpha[query_lane] * running_output[group_q][dim] +
          beta[query_lane] * tile_value;
    }
  }
  for (int query_lane = 0; query_lane < 2; ++query_lane) {
#pragma HLS UNROLL
    running_max[pair_base + query_lane] = merged_max[query_lane];
  }
}

void compute_tile(
    const decode_tile_word_t
        kv_tile[DECODE_TILE_CHUNKS_PER_ROW][DECODE_COLOCATED_TILE_K],
    const float k_scale[DECODE_TILE_K][DECODE_GROUPS_PER_ROW],
    const float v_scale[DECODE_TILE_K],
    int tile_count,
    bool first_tile,
    const qint14_t q_group[DECODE_GQA_GROUP][DECODE_HEAD_DIM],
    const float q_scale[DECODE_GQA_GROUP][DECODE_GROUPS_PER_ROW],
    float running_max[DECODE_GQA_GROUP],
    float running_sum[DECODE_GQA_GROUP],
    float running_output[DECODE_GQA_GROUP][DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  for (int query_pair = 0;
       query_pair < DECODE_GQA_GROUP / 2; ++query_pair) {
    compute_head_tile_pair(
        kv_tile, k_scale, v_scale, tile_count, first_tile,
        query_pair * 2, q_group, q_scale,
        running_max, running_sum, running_output);
  }
}
void load_next_compute_current(
    const decode_mem_word_t* k_cache,
    const decode_mem_word_t* v_cache,
    int kv_head,
    int next_base,
    int next_count,
    decode_tile_word_t
        next_kv[DECODE_TILE_CHUNKS_PER_ROW][DECODE_COLOCATED_TILE_K],
    float next_k_scale[DECODE_TILE_K][DECODE_GROUPS_PER_ROW],
    float next_v_scale[DECODE_TILE_K],
    const decode_tile_word_t
        current_kv[DECODE_TILE_CHUNKS_PER_ROW][DECODE_COLOCATED_TILE_K],
    const float current_k_scale[DECODE_TILE_K][DECODE_GROUPS_PER_ROW],
    const float current_v_scale[DECODE_TILE_K],
    int current_count,
    bool first_tile,
    const qint14_t q_group[DECODE_GQA_GROUP][DECODE_HEAD_DIM],
    const float q_scale[DECODE_GQA_GROUP][DECODE_GROUPS_PER_ROW],
    float running_max[DECODE_GQA_GROUP],
    float running_sum[DECODE_GQA_GROUP],
    float running_output[DECODE_GQA_GROUP][DECODE_HEAD_DIM]) {
#pragma HLS INLINE off
  load_tile(k_cache, v_cache, kv_head, next_base, next_count,
            next_kv, next_k_scale, next_v_scale);
  compute_tile(current_kv, current_k_scale, current_v_scale,
               current_count, first_tile, q_group, q_scale,
               running_max, running_sum, running_output);
}

}  // namespace

extern "C" void fpt_decode_q14_kv13_row_core(
    const decode_mem_word_t q_current[DECODE_Q_WORDS],
    decode_mem_word_t k_cache[DECODE_CACHE_WORDS],
    decode_mem_word_t v_cache[DECODE_CACHE_WORDS],
    const decode_mem_word_t k_new[DECODE_KV_WORDS],
    const decode_mem_word_t v_new[DECODE_KV_WORDS],
    decode_mem_word_t output[DECODE_Q_WORDS],
    unsigned position) {

  if (position >= DECODE_MAX_KV_TOKENS) return;

  const int valid_tokens = static_cast<int>(position) + 1;
  const int tile_count =
      (valid_tokens + DECODE_TILE_K - 1) / DECODE_TILE_K;

  for (int kv_head = 0; kv_head < DECODE_KV_HEADS; ++kv_head) {
    store_cache_rows_kv13(k_cache, v_cache, kv_head,
                          static_cast<int>(position), k_new, v_new);
  }

  decode_tile_word_t kv_tile[DECODE_TILE_BUFFERS][DECODE_TILE_CHUNKS_PER_ROW]
                             [DECODE_COLOCATED_TILE_K];
  float k_scale[DECODE_TILE_BUFFERS][DECODE_TILE_K]
               [DECODE_GROUPS_PER_ROW];
  float v_scale[DECODE_TILE_BUFFERS][DECODE_TILE_K];
#pragma HLS ARRAY_PARTITION variable=kv_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=kv_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_scale complete dim=1
#pragma HLS BIND_STORAGE variable=kv_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=k_scale type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=v_scale type=ram_2p impl=bram

  const int first_count =
      valid_tokens < DECODE_TILE_K ? valid_tokens : DECODE_TILE_K;
  load_tile(k_cache, v_cache, 0, 0, first_count,
            kv_tile[0], k_scale[0], v_scale[0]);

  int work_index = 0;
  for (int kv_head = 0; kv_head < DECODE_KV_HEADS; ++kv_head) {
    qint14_t q_group[DECODE_GQA_GROUP][DECODE_HEAD_DIM];
    float q_scale[DECODE_GQA_GROUP][DECODE_GROUPS_PER_ROW];
    float q_values[DECODE_GQA_GROUP][DECODE_HEAD_DIM];
    float running_max[DECODE_GQA_GROUP];
    float running_sum[DECODE_GQA_GROUP];
    float running_output[DECODE_GQA_GROUP][DECODE_HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=q_group complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=q_scale complete dim=0
#pragma HLS ARRAY_PARTITION variable=q_values complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_values cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=running_max complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=running_output complete dim=1

    // Load the four contiguous GQA query heads once. Keeping one AXI read
    // schedule avoids four helper start/drain penalties at every KV head.
    for (int q_word = 0;
         q_word < DECODE_GQA_GROUP * DECODE_WORDS_PER_ROW; ++q_word) {
#pragma HLS PIPELINE II=1
      const int group_q = q_word / DECODE_WORDS_PER_ROW;
      const int word = q_word % DECODE_WORDS_PER_ROW;
      const int q_head = kv_head * DECODE_GQA_GROUP + group_q;
      const decode_mem_word_t packed =
          q_current[head_word_index(q_head, word)];
      for (int lane = 0; lane < DECODE_VALUES_PER_WORD; ++lane) {
#pragma HLS UNROLL
        q_values[group_q][word * DECODE_VALUES_PER_WORD + lane] =
            bf16_to_float(read_lane(packed, lane));
      }
    }

    for (int group_q = 0; group_q < DECODE_GQA_GROUP; ++group_q) {
      const float peak = row_abs_max(q_values[group_q]);
      const float scale = nonzero_nonnegative_float(peak)
                              ? peak * (1.0f / DECODE_Q_QUANT_LIMIT)
                              : 1.0f;
      const float inverse_scale = 1.0f / scale;
      q_scale[group_q][0] = scale;
      for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
        q_group[group_q][dim] =
            quantize_q14_with_inv(q_values[group_q][dim], inverse_scale);
      }
    }

    for (int group_q = 0; group_q < DECODE_GQA_GROUP; ++group_q) {
#pragma HLS UNROLL
      running_max[group_q] = -INFINITY;
      running_sum[group_q] = 0.0f;
    }
    for (int dim = 0; dim < DECODE_HEAD_DIM; ++dim) {
#pragma HLS PIPELINE II=1
      for (int group_q = 0; group_q < DECODE_GQA_GROUP; ++group_q) {
#pragma HLS UNROLL
        running_output[group_q][dim] = 0.0f;
      }
    }

    for (int tile = 0; tile < tile_count; ++tile) {
      const int current_bank = work_index & 1;
      const int next_bank = current_bank ^ 1;
      const int current_base = tile * DECODE_TILE_K;
      const int remaining = valid_tokens - current_base;
      const int current_count =
          remaining < DECODE_TILE_K ? remaining : DECODE_TILE_K;
      const bool has_next_tile = tile + 1 < tile_count;
      const bool has_next_head = kv_head + 1 < DECODE_KV_HEADS;
      const bool has_next_work = has_next_tile || has_next_head;
      const int next_kv_head =
          has_next_tile ? kv_head : (has_next_head ? kv_head + 1 : 0);
      const int next_base = has_next_tile ? current_base + DECODE_TILE_K : 0;
      const int next_remaining = valid_tokens - next_base;
      const int bounded_next_count = next_remaining < DECODE_TILE_K
                                         ? next_remaining
                                         : DECODE_TILE_K;
      const int next_count = has_next_work ? bounded_next_count : 0;
      load_next_compute_current(
          k_cache, v_cache, next_kv_head, next_base, next_count,
          kv_tile[next_bank],
          k_scale[next_bank], v_scale[next_bank],
          kv_tile[current_bank],
          k_scale[current_bank], v_scale[current_bank],
          current_count, tile == 0, q_group, q_scale,
          running_max, running_sum, running_output);
      ++work_index;
    }

    // Build and commit one AXI word at a time.  The previous fully
    // partitioned 4x2x512-bit staging array created a 4098-fanout common
    // clear/enable net after implementation.  A static 16-bit shift keeps
    // the same single-divider schedule without retaining all eight words.
    for (int group_q = 0; group_q < DECODE_GQA_GROUP; ++group_q) {
      for (int word = 0; word < DECODE_WORDS_PER_ROW; ++word) {
        decode_mem_word_t packed_output = 0;
        for (int lane = 0; lane < DECODE_VALUES_PER_WORD; ++lane) {
#pragma HLS PIPELINE II=1
          const int dim = word * DECODE_VALUES_PER_WORD + lane;
          const float value =
              running_output[group_q][dim] / running_sum[group_q];
          packed_output >>= 16;
          packed_output.range(DECODE_MEM_WORD_BITS - 1,
                              DECODE_MEM_WORD_BITS - 16) =
              float_to_bf16_rne(value);
        }
        const int q_head = kv_head * DECODE_GQA_GROUP + group_q;
        output[head_word_index(q_head, word)] = packed_output;
      }
    }
  }
}
