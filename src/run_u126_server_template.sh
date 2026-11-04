#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <llama-server> <model.gguf> <U126.xclbin> [runtime-lib-dir]" >&2
  exit 2
fi

llama_server=$1
model_path=$2
xclbin_path=$3
runtime_lib_dir=${4:-"$(cd "$(dirname "$0")/../board/lib" && pwd)"}
server_dir="$(cd "$(dirname "$llama_server")" && pwd)"

export LD_LIBRARY_PATH="${runtime_lib_dir}:${server_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FPT_LLAMA_PL_ENABLE=1
export FPT_LLAMA_PL_STRICT=1
export FPT_LLAMA_PL_PROFILE=U126_L512
export FPT_LLAMA_PL_XCLBIN="$xclbin_path"
export FPT_LLAMA_PL_CACHE_SYNC=delta
export FPT_LLAMA_PL_DIRECT_UNPACK=1
export FPT_LLAMA_PL_DISABLE_ROW_CONVERT=0
export FPT_LLAMA_PL_DISABLE_ROW_MEMCPY=0
export FPT_LLAMA_PL_SHADOW=0
export FPT_LLAMA_PL_REPORT="${FPT_LLAMA_PL_REPORT:-/tmp/u126_pl_report_${U126_PORT:-8083}.txt}"

exec "$llama_server" \
  -m "$model_path" \
  --host "${U126_HOST:-127.0.0.1}" \
  --port "${U126_PORT:-8083}" \
  -c 512 -b 512 -ub 512 -fa on \
  -ctk bf16 -ctv bf16 \
  -t 4 -tb 4 --no-warmup --no-context-shift --parallel 1 \
  --alias llama-u126-pl
