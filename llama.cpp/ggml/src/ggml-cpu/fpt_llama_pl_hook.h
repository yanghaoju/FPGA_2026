#ifndef FPT_LLAMA_PL_HOOK_H
#define FPT_LLAMA_PL_HOOK_H

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*fpt_llama_pl_cpu_attention_fn)(
    const struct ggml_compute_params *params, struct ggml_tensor *dst);

/*
 * Returns true when the operation was handled by the U19 PL session.  For a
 * recognized shape, non-zero worker threads return true after thread 0 has
 * completed the single PL call.  Unsupported shapes return false so the
 * normal ggml CPU implementation remains authoritative.
 */
bool fpt_llama_pl_try_compute(
    const struct ggml_compute_params *params, struct ggml_tensor *dst,
    fpt_llama_pl_cpu_attention_fn cpu_reference);

#ifdef __cplusplus
}
#endif

#endif
