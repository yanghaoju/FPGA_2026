#ifndef FPT_ATTENTION_SESSION_H
#define FPT_ATTENTION_SESSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPT_ATTENTION_SESSION_ABI_VERSION 5u
#define FPT_ATTENTION_Q_HEADS 32u
#define FPT_ATTENTION_KV_HEADS 8u
#define FPT_ATTENTION_HEAD_DIM 64u
#define FPT_ATTENTION_MAX_KV_TOKENS 512u

typedef enum {
  FPT_ATTENTION_OK = 0,
  FPT_ATTENTION_INVALID_ARGUMENT = 1,
  FPT_ATTENTION_UNSUPPORTED_SHAPE = 2,
  FPT_ATTENTION_CACHE_STATE_ERROR = 3,
  FPT_ATTENTION_XRT_ERROR = 4
} fpt_attention_status_t;

typedef struct {
  double pack_ms;
  double host_to_device_ms;
  double kernel_ms;
  double device_to_host_ms;
  double unpack_ms;
  double total_ms;
  uint32_t host_to_device_sync_calls;
  uint32_t device_to_host_sync_calls;
  uint64_t host_to_device_bytes;
  uint64_t device_to_host_bytes;
} fpt_attention_metrics_t;

typedef enum {
  FPT_ATTENTION_CACHE_SYNC_FULL_BO = 0,
  FPT_ATTENTION_CACHE_SYNC_VALID_HEAD_RANGES = 1
} fpt_attention_cache_sync_strategy_t;

typedef struct {
  uint16_t *q_native;
  size_t q_elements;
  uint16_t *k_prefill_native;
  size_t k_prefill_elements;
  uint16_t *v_prefill_native;
  size_t v_prefill_elements;
  uint16_t *k_cache_native;
  size_t k_cache_elements;
  uint16_t *v_cache_native;
  size_t v_cache_elements;
  uint16_t *output_native;
  size_t output_elements;
  uint32_t bucket_tokens;
} fpt_attention_prefill_io_t;

typedef struct {
  uint16_t *q;
  size_t q_elements;
  uint16_t *k_new;
  size_t k_new_elements;
  uint16_t *v_new;
  size_t v_new_elements;
  uint16_t *output;
  size_t output_elements;
} fpt_attention_decode_io_t;

typedef struct fpt_attention_session fpt_attention_session_t;

/*
 * One session owns one xclbin load, one kernel/run handle, shared gmem0/gmem3
 * buffers, and persistent K/V cache BOs for every layer. The session is not
 * thread-safe; mapped views remain valid until destroy.
 */
fpt_attention_status_t fpt_attention_session_create(
    const char *xclbin_path, uint32_t device_index, uint32_t bucket_tokens,
    uint32_t cache_layers, fpt_attention_session_t **out_session);

/* Benchmark both strategies on the target board; XRT call overhead varies. */
fpt_attention_status_t fpt_attention_set_cache_sync_strategy(
    fpt_attention_session_t *session,
    fpt_attention_cache_sync_strategy_t strategy);

/*
 * q/output and K/V use the accelerator-native Prefill layouts before launch.
 * Both K and V Prefill views alias their persistent layer BOs. U57 overwrites
 * those BOs in-place with the packed signed-INT13 Decode layout (BF16 row
 * scale) before returning. Historical internal labels may still say row-INT8.
 */
fpt_attention_status_t fpt_attention_get_prefill_mapped_io(
    fpt_attention_session_t *session, uint32_t layer_index,
    fpt_attention_prefill_io_t *out_view);

/* Optional prefill output initialization for kernels that skip inactive rows. */
fpt_attention_status_t fpt_attention_sync_prefill_output_to_device(
    fpt_attention_session_t *session);

fpt_attention_status_t fpt_attention_run_prefill_mapped(
    fpt_attention_session_t *session, uint32_t layer_index,
    uint32_t prompt_tokens, fpt_attention_metrics_t *metrics);

/*
 * Commit the cache already packed by the U57 Prefill kernel. No host cache
 * upload is performed; prompt_tokens must match the preceding Prefill call.
 */
fpt_attention_status_t fpt_attention_commit_prompt_cache(
    fpt_attention_session_t *session, uint32_t layer_index,
    uint32_t prompt_tokens, fpt_attention_metrics_t *metrics);

fpt_attention_status_t fpt_attention_get_decode_mapped_io(
    fpt_attention_session_t *session,
    fpt_attention_decode_io_t *out_view);

fpt_attention_status_t fpt_attention_run_decode_mapped(
    fpt_attention_session_t *session, uint32_t layer_index,
    uint32_t position, fpt_attention_metrics_t *metrics);

uint32_t fpt_attention_session_bucket_tokens(
    const fpt_attention_session_t *session);
uint32_t fpt_attention_session_cache_layers(
    const fpt_attention_session_t *session);
const char *fpt_attention_session_last_error(
    const fpt_attention_session_t *session);
void fpt_attention_session_destroy(fpt_attention_session_t *session);

#ifdef __cplusplus
}
#endif

#endif
