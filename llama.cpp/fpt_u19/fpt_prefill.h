#ifndef FPT_PREFILL_H
#define FPT_PREFILL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPT_PREFILL_ABI_VERSION 2u
#define FPT_PREFILL_Q_HEADS 32u
#define FPT_PREFILL_KV_HEADS 8u
#define FPT_PREFILL_HEAD_DIM 64u

typedef enum {
    FPT_PREFILL_OK = 0,
    FPT_PREFILL_INVALID_ARGUMENT = 1,
    FPT_PREFILL_UNSUPPORTED_SHAPE = 2,
    FPT_PREFILL_BUFFER_TOO_SMALL = 3,
    FPT_PREFILL_XRT_ERROR = 4
} fpt_prefill_status_t;

typedef enum {
    /* [token][head][dimension], recommended at the model boundary. */
    FPT_PREFILL_TOKEN_HEAD_DIM = 0,
    /* [head][token][dimension], used by the frozen verification vectors. */
    FPT_PREFILL_HEAD_TOKEN_DIM = 1
} fpt_prefill_layout_t;

typedef struct {
    const char *profile_id;
    const char *kria_app_name;
    const char *binary_sha256;
    uint32_t sequence;
    uint64_t q_bytes;
    uint64_t k_bytes;
    uint64_t v_bytes;
    uint64_t out_bytes;
    double measured_kernel_ms;
    double measured_end_to_end_ms;
} fpt_prefill_profile_t;

typedef struct {
    double pack_ms;
    double host_to_device_ms;
    double kernel_ms;
    double device_to_host_ms;
    double unpack_ms;
    double total_ms;
} fpt_prefill_metrics_t;
typedef struct {
    uint16_t *q_native;
    size_t q_elements;
    uint16_t *k_native;
    size_t k_elements;
    uint16_t *v_native;
    size_t v_elements;
    uint16_t *out_native;
    size_t out_elements;
    uint32_t bucket_tokens;
} fpt_prefill_mapped_io_t;

typedef struct fpt_prefill_context fpt_prefill_context_t;

/* Select the smallest compiled bucket that contains valid_tokens. */
const fpt_prefill_profile_t *fpt_prefill_select_profile(uint32_t valid_tokens);

size_t fpt_prefill_q_elements(uint32_t valid_tokens);
size_t fpt_prefill_kv_elements(uint32_t valid_tokens);

/*
 * The matching Kria app must already be loaded with xmutil. xclbin_path points
 * to that app's binary_container_1.bin. The context owns persistent XRT BOs.
 */
fpt_prefill_status_t fpt_prefill_create(
    uint32_t valid_tokens,
    const char *xclbin_path,
    uint32_t device_index,
    fpt_prefill_context_t **out_context);
/*
 * Native persistent BO views. q/out use the accelerator GQA-blocked layout;
 * k is [kv_head][bucket_token][dim]; v is tile-major [dim][key].
 * The view remains valid until destroy and must not be used concurrently.
 */
fpt_prefill_status_t fpt_prefill_get_mapped_io(
    fpt_prefill_context_t *context,
    fpt_prefill_mapped_io_t *out_view);

/* Execute directly from/to the native mapped views without layout conversion. */
fpt_prefill_status_t fpt_prefill_run_mapped(
    fpt_prefill_context_t *context,
    fpt_prefill_metrics_t *metrics);

/*
 * q/out contain 32 heads; k/v contain 8 heads. Element counts describe the
 * compact valid-token tensors, not the padded bucket. RoPE must already have
 * been applied to q and k. Values are raw IEEE-754 BF16 words.
 */
fpt_prefill_status_t fpt_prefill_run(
    fpt_prefill_context_t *context,
    const uint16_t *q,
    size_t q_elements,
    const uint16_t *k,
    size_t k_elements,
    const uint16_t *v,
    size_t v_elements,
    uint32_t valid_tokens,
    fpt_prefill_layout_t layout,
    uint16_t *out,
    size_t out_elements,
    fpt_prefill_metrics_t *metrics);

const fpt_prefill_profile_t *fpt_prefill_context_profile(
    const fpt_prefill_context_t *context);

const char *fpt_prefill_last_error(const fpt_prefill_context_t *context);

void fpt_prefill_destroy(fpt_prefill_context_t *context);

#ifdef __cplusplus
}
#endif

#endif
