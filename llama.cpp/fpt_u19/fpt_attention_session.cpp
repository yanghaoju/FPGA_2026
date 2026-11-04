#include "fpt_attention_session.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDecodeQElements =
    static_cast<size_t>(FPT_ATTENTION_Q_HEADS) * FPT_ATTENTION_HEAD_DIM;
constexpr size_t kDecodeKvElements =
    static_cast<size_t>(FPT_ATTENTION_KV_HEADS) * FPT_ATTENTION_HEAD_DIM;
constexpr size_t kDecodeKNewOffset = kDecodeQElements;
constexpr size_t kDecodeVNewOffset = kDecodeQElements + kDecodeKvElements;
constexpr size_t kDecodeInputElements =
    kDecodeQElements + 2 * kDecodeKvElements;
constexpr size_t kCacheElements =
    static_cast<size_t>(FPT_ATTENTION_KV_HEADS) *
    FPT_ATTENTION_MAX_KV_TOKENS * FPT_ATTENTION_HEAD_DIM;
constexpr size_t kCacheBytes = kCacheElements * sizeof(uint16_t);
constexpr size_t kCacheHeadStrideBytes =
    static_cast<size_t>(FPT_ATTENTION_MAX_KV_TOKENS) *
    FPT_ATTENTION_HEAD_DIM * sizeof(uint16_t);

static_assert((kDecodeKNewOffset * sizeof(uint16_t)) % 64 == 0,
              "Knew must be 512-bit aligned");
static_assert((kDecodeVNewOffset * sizeof(uint16_t)) % 64 == 0,
              "Vnew must be 512-bit aligned");

double elapsed_ms(const Clock::time_point &begin,
                  const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool valid_bucket(uint32_t tokens) {
  return tokens == 64U || tokens == 128U || tokens == 256U;
}

void clear_metrics(fpt_attention_metrics_t *metrics) {
  if (metrics != nullptr) {
    *metrics = {};
  }
}

struct SyncStats {
  uint32_t calls = 0;
  uint64_t bytes = 0;
};

SyncStats sync_prompt_cache_to_device(
    xrt::bo &buffer, uint32_t prompt_tokens,
    fpt_attention_cache_sync_strategy_t strategy) {
  if (strategy == FPT_ATTENTION_CACHE_SYNC_FULL_BO) {
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, kCacheBytes, 0);
    return {1U, kCacheBytes};
  }
  const size_t valid_head_bytes =
      static_cast<size_t>(prompt_tokens) * FPT_ATTENTION_HEAD_DIM *
      sizeof(uint16_t);
  for (uint32_t head = 0; head < FPT_ATTENTION_KV_HEADS; ++head) {
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, valid_head_bytes,
                static_cast<size_t>(head) * kCacheHeadStrideBytes);
  }
  return {FPT_ATTENTION_KV_HEADS,
          static_cast<uint64_t>(valid_head_bytes) *
              FPT_ATTENTION_KV_HEADS};
}

struct LayerCache {
  LayerCache(xrt::device &device, xrt::memory_group k_group,
             xrt::memory_group v_group)
      : k_bo(device, kCacheElements * sizeof(uint16_t), k_group),
        v_bo(device, kCacheElements * sizeof(uint16_t), v_group),
        k_map(k_bo.map<uint16_t *>()),
        v_map(v_bo.map<uint16_t *>()) {}

  xrt::bo k_bo;
  xrt::bo v_bo;
  uint16_t *k_map;
  uint16_t *v_map;
  uint32_t next_position = 0;
  bool prompt_committed = false;
  bool k_synced_by_prefill = false;
};

struct SessionRuntime {
  SessionRuntime(const char *xclbin_path, uint32_t device_index,
                 uint32_t bucket, uint32_t cache_layers)
      : bucket_tokens(bucket),
        prefill_q_elements(static_cast<size_t>(FPT_ATTENTION_Q_HEADS) *
                           bucket * FPT_ATTENTION_HEAD_DIM),
        prefill_kv_elements(static_cast<size_t>(FPT_ATTENTION_KV_HEADS) *
                            bucket * FPT_ATTENTION_HEAD_DIM),
        device(device_index),
        uuid(device.load_xclbin(xclbin_path)),
        kernel(device, uuid, "exact_gqa_prefill_decode_ip"),
        run(kernel),
        io0_bo(device,
               std::max(prefill_q_elements, kDecodeInputElements) *
                   sizeof(uint16_t),
               static_cast<xrt::memory_group>(kernel.group_id(1))),
        k_prefill_bo(device, prefill_kv_elements * sizeof(uint16_t),
                     static_cast<xrt::memory_group>(kernel.group_id(2))),
        v_prefill_bo(device, prefill_kv_elements * sizeof(uint16_t),
                     static_cast<xrt::memory_group>(kernel.group_id(3))),
        out_bo(device,
               std::max(prefill_q_elements, kDecodeQElements) *
                   sizeof(uint16_t),
               static_cast<xrt::memory_group>(kernel.group_id(4))),
        io0_map(io0_bo.map<uint16_t *>()),
        k_prefill_map(k_prefill_bo.map<uint16_t *>()),
        v_prefill_map(v_prefill_bo.map<uint16_t *>()),
        out_map(out_bo.map<uint16_t *>()) {
    const auto k_group =
        static_cast<xrt::memory_group>(kernel.group_id(2));
    const auto v_group =
        static_cast<xrt::memory_group>(kernel.group_id(3));
    layers.reserve(cache_layers);
    for (uint32_t index = 0; index < cache_layers; ++index) {
      layers.emplace_back(
          std::make_unique<LayerCache>(device, k_group, v_group));
    }
  }

  uint32_t bucket_tokens;
  size_t prefill_q_elements;
  size_t prefill_kv_elements;
  xrt::device device;
  xrt::uuid uuid;
  xrt::kernel kernel;
  xrt::run run;
  xrt::bo io0_bo;
  xrt::bo k_prefill_bo;
  xrt::bo v_prefill_bo;
  xrt::bo out_bo;
  uint16_t *io0_map;
  uint16_t *k_prefill_map;
  uint16_t *v_prefill_map;
  uint16_t *out_map;
  std::vector<std::unique_ptr<LayerCache>> layers;
  fpt_attention_cache_sync_strategy_t cache_sync_strategy =
      FPT_ATTENTION_CACHE_SYNC_FULL_BO;
};

thread_local std::string g_last_create_error;

}  // namespace

struct fpt_attention_session {
  uint32_t bucket_tokens = 0;
  uint32_t cache_layers = 0;
  std::unique_ptr<SessionRuntime> runtime;
  std::string last_error;
};

extern "C" fpt_attention_status_t fpt_attention_session_create(
    const char *xclbin_path, uint32_t device_index, uint32_t bucket_tokens,
    uint32_t cache_layers, fpt_attention_session_t **out_session) {
  if (out_session == nullptr) {
    g_last_create_error = "out_session is null";
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  *out_session = nullptr;
  if (xclbin_path == nullptr || xclbin_path[0] == '\0') {
    g_last_create_error = "xclbin_path is empty";
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (!valid_bucket(bucket_tokens) || cache_layers == 0 ||
      cache_layers > 64U) {
    g_last_create_error = "bucket must be 64/128/256 and layers 1..64";
    return FPT_ATTENTION_UNSUPPORTED_SHAPE;
  }
  try {
    auto session = std::make_unique<fpt_attention_session>();
    session->bucket_tokens = bucket_tokens;
    session->cache_layers = cache_layers;
    session->runtime = std::make_unique<SessionRuntime>(
        xclbin_path, device_index, bucket_tokens, cache_layers);
    *out_session = session.release();
    g_last_create_error.clear();
    return FPT_ATTENTION_OK;
  } catch (const std::exception &error) {
    g_last_create_error = error.what();
    return FPT_ATTENTION_XRT_ERROR;
  }
}

extern "C" fpt_attention_status_t fpt_attention_set_cache_sync_strategy(
    fpt_attention_session_t *session,
    fpt_attention_cache_sync_strategy_t strategy) {
  if (session == nullptr || session->runtime == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (strategy != FPT_ATTENTION_CACHE_SYNC_FULL_BO &&
      strategy != FPT_ATTENTION_CACHE_SYNC_VALID_HEAD_RANGES) {
    session->last_error = "unsupported cache sync strategy";
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  session->runtime->cache_sync_strategy = strategy;
  session->last_error.clear();
  return FPT_ATTENTION_OK;
}
extern "C" fpt_attention_status_t fpt_attention_get_prefill_mapped_io(
    fpt_attention_session_t *session, uint32_t layer_index,
    fpt_attention_prefill_io_t *out_view) {
  if (session == nullptr || session->runtime == nullptr ||
      out_view == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (layer_index >= session->cache_layers) {
    session->last_error = "layer index is out of range";
    return FPT_ATTENTION_UNSUPPORTED_SHAPE;
  }
  auto &runtime = *session->runtime;
  auto &layer = *runtime.layers[layer_index];
  uint16_t *k_prefill = runtime.bucket_tokens == 256U
                            ? layer.k_map
                            : runtime.k_prefill_map;
  *out_view = {
      runtime.io0_map,
      runtime.prefill_q_elements,
      k_prefill,
      runtime.prefill_kv_elements,
      runtime.v_prefill_map,
      runtime.prefill_kv_elements,
      layer.k_map,
      kCacheElements,
      layer.v_map,
      kCacheElements,
      runtime.out_map,
      runtime.prefill_q_elements,
      runtime.bucket_tokens};
  session->last_error.clear();
  return FPT_ATTENTION_OK;
}

extern "C" fpt_attention_status_t fpt_attention_run_prefill_mapped(
    fpt_attention_session_t *session, uint32_t layer_index,
    fpt_attention_metrics_t *metrics) {
  clear_metrics(metrics);
  if (session == nullptr || session->runtime == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (layer_index >= session->cache_layers) {
    session->last_error = "layer index is out of range";
    return FPT_ATTENTION_UNSUPPORTED_SHAPE;
  }
  try {
    auto &runtime = *session->runtime;
    auto &layer = *runtime.layers[layer_index];
    auto &k_source = runtime.bucket_tokens == 256U
                         ? layer.k_bo
                         : runtime.k_prefill_bo;
    const size_t q_bytes = runtime.prefill_q_elements * sizeof(uint16_t);
    const size_t kv_bytes = runtime.prefill_kv_elements * sizeof(uint16_t);
    const auto total_begin = Clock::now();
    runtime.io0_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, q_bytes, 0);
    k_source.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_bytes, 0);
    runtime.v_prefill_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_bytes, 0);
    const auto h2d_end = Clock::now();

    runtime.run.set_arg(0, 0U);
    runtime.run.set_arg(1, runtime.io0_bo);
    runtime.run.set_arg(2, k_source);
    runtime.run.set_arg(3, runtime.v_prefill_bo);
    runtime.run.set_arg(4, runtime.out_bo);
    runtime.run.set_arg(5, 0U);
    runtime.run.start();
    runtime.run.wait();
    const auto kernel_end = Clock::now();
    runtime.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, q_bytes, 0);
    const auto total_end = Clock::now();

    layer.k_synced_by_prefill = runtime.bucket_tokens == 256U;
    if (metrics != nullptr) {
      metrics->pack_ms = 0.0;
      metrics->host_to_device_ms = elapsed_ms(total_begin, h2d_end);
      metrics->kernel_ms = elapsed_ms(h2d_end, kernel_end);
      metrics->device_to_host_ms = elapsed_ms(kernel_end, total_end);
      metrics->unpack_ms = 0.0;
      metrics->total_ms = elapsed_ms(total_begin, total_end);
      metrics->host_to_device_sync_calls = 3U;
      metrics->device_to_host_sync_calls = 1U;
      metrics->host_to_device_bytes = q_bytes + 2U * kv_bytes;
      metrics->device_to_host_bytes = q_bytes;
    }
    session->last_error.clear();
    return FPT_ATTENTION_OK;
  } catch (const std::exception &error) {
    session->last_error = error.what();
    return FPT_ATTENTION_XRT_ERROR;
  }
}

extern "C" fpt_attention_status_t fpt_attention_commit_prompt_cache(
    fpt_attention_session_t *session, uint32_t layer_index,
    uint32_t prompt_tokens, fpt_attention_metrics_t *metrics) {
  clear_metrics(metrics);
  if (session == nullptr || session->runtime == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (layer_index >= session->cache_layers || prompt_tokens == 0U ||
      prompt_tokens > session->bucket_tokens) {
    session->last_error = "invalid layer or prompt length";
    return FPT_ATTENTION_UNSUPPORTED_SHAPE;
  }
  try {
    auto &layer = *session->runtime->layers[layer_index];
    const auto begin = Clock::now();
    SyncStats sync_stats{};
    if (!layer.k_synced_by_prefill) {
      const SyncStats k_stats = sync_prompt_cache_to_device(
          layer.k_bo, prompt_tokens, session->runtime->cache_sync_strategy);
      sync_stats.calls += k_stats.calls;
      sync_stats.bytes += k_stats.bytes;
    }
    const SyncStats v_stats = sync_prompt_cache_to_device(
        layer.v_bo, prompt_tokens, session->runtime->cache_sync_strategy);
    sync_stats.calls += v_stats.calls;
    sync_stats.bytes += v_stats.bytes;
    const auto end = Clock::now();
    layer.next_position = prompt_tokens;
    layer.prompt_committed = true;
    layer.k_synced_by_prefill = false;
    if (metrics != nullptr) {
      metrics->pack_ms = 0.0;
      metrics->host_to_device_ms = elapsed_ms(begin, end);
      metrics->kernel_ms = 0.0;
      metrics->device_to_host_ms = 0.0;
      metrics->unpack_ms = 0.0;
      metrics->total_ms = elapsed_ms(begin, end);
      metrics->host_to_device_sync_calls = sync_stats.calls;
      metrics->device_to_host_sync_calls = 0U;
      metrics->host_to_device_bytes = sync_stats.bytes;
      metrics->device_to_host_bytes = 0U;
    }
    session->last_error.clear();
    return FPT_ATTENTION_OK;
  } catch (const std::exception &error) {
    session->last_error = error.what();
    return FPT_ATTENTION_XRT_ERROR;
  }
}

extern "C" fpt_attention_status_t fpt_attention_get_decode_mapped_io(
    fpt_attention_session_t *session,
    fpt_attention_decode_io_t *out_view) {
  if (session == nullptr || session->runtime == nullptr ||
      out_view == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  auto &runtime = *session->runtime;
  *out_view = {runtime.io0_map,
               kDecodeQElements,
               runtime.io0_map + kDecodeKNewOffset,
               kDecodeKvElements,
               runtime.io0_map + kDecodeVNewOffset,
               kDecodeKvElements,
               runtime.out_map,
               kDecodeQElements};
  session->last_error.clear();
  return FPT_ATTENTION_OK;
}

extern "C" fpt_attention_status_t fpt_attention_run_decode_mapped(
    fpt_attention_session_t *session, uint32_t layer_index,
    uint32_t position, fpt_attention_metrics_t *metrics) {
  clear_metrics(metrics);
  if (session == nullptr || session->runtime == nullptr) {
    return FPT_ATTENTION_INVALID_ARGUMENT;
  }
  if (layer_index >= session->cache_layers ||
      position >= FPT_ATTENTION_MAX_KV_TOKENS) {
    session->last_error = "layer or position is out of range";
    return FPT_ATTENTION_UNSUPPORTED_SHAPE;
  }
  auto &runtime = *session->runtime;
  auto &layer = *runtime.layers[layer_index];
  if (!layer.prompt_committed || position != layer.next_position) {
    session->last_error = "prompt cache is not committed or position is not sequential";
    return FPT_ATTENTION_CACHE_STATE_ERROR;
  }
  try {
    const auto total_begin = Clock::now();
    runtime.io0_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                        kDecodeInputElements * sizeof(uint16_t), 0);
    const auto h2d_end = Clock::now();

    runtime.run.set_arg(0, 1U);
    runtime.run.set_arg(1, runtime.io0_bo);
    runtime.run.set_arg(2, layer.k_bo);
    runtime.run.set_arg(3, layer.v_bo);
    runtime.run.set_arg(4, runtime.out_bo);
    runtime.run.set_arg(5, position);
    runtime.run.start();
    runtime.run.wait();
    const auto kernel_end = Clock::now();
    runtime.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
                        kDecodeQElements * sizeof(uint16_t), 0);
    const auto total_end = Clock::now();

    ++layer.next_position;
    if (metrics != nullptr) {
      metrics->pack_ms = 0.0;
      metrics->host_to_device_ms = elapsed_ms(total_begin, h2d_end);
      metrics->kernel_ms = elapsed_ms(h2d_end, kernel_end);
      metrics->device_to_host_ms = elapsed_ms(kernel_end, total_end);
      metrics->unpack_ms = 0.0;
      metrics->total_ms = elapsed_ms(total_begin, total_end);
      metrics->host_to_device_sync_calls = 1U;
      metrics->device_to_host_sync_calls = 1U;
      metrics->host_to_device_bytes =
          kDecodeInputElements * sizeof(uint16_t);
      metrics->device_to_host_bytes =
          kDecodeQElements * sizeof(uint16_t);
    }
    session->last_error.clear();
    return FPT_ATTENTION_OK;
  } catch (const std::exception &error) {
    session->last_error = error.what();
    return FPT_ATTENTION_XRT_ERROR;
  }
}

extern "C" uint32_t fpt_attention_session_bucket_tokens(
    const fpt_attention_session_t *session) {
  return session == nullptr ? 0U : session->bucket_tokens;
}

extern "C" uint32_t fpt_attention_session_cache_layers(
    const fpt_attention_session_t *session) {
  return session == nullptr ? 0U : session->cache_layers;
}

extern "C" const char *fpt_attention_session_last_error(
    const fpt_attention_session_t *session) {
  return session == nullptr ? g_last_create_error.c_str()
                            : session->last_error.c_str();
}

extern "C" void fpt_attention_session_destroy(
    fpt_attention_session_t *session) {
  delete session;
}
