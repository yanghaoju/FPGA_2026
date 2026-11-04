#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${FPT_LLAMA_BUILD_DIR:-$ROOT/build/llama-u126}"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'Build the PL runtime on AArch64 KV260 with XRT development files installed.' >&2
  exit 2
fi
for tool in cmake g++; do command -v "$tool" >/dev/null; done
CMAKE_VERSION="$(cmake --version | sed -n '1s/^cmake version //p')"
if [[ "$(printf '%s\n' 3.18 "$CMAKE_VERSION" | sort -V | sed -n '1p')" != 3.18 ]]; then
  echo 'CMake 3.18 or newer is required by the supplied llama.cpp server sources.' >&2
  exit 2
fi
XRT_HEADER_ROOT="${FPT_XRT_INCLUDE_DIR:-}"
XRT_CORE_LIBRARY="${FPT_XRT_LIBRARY:-}"
if [[ -z "$XRT_HEADER_ROOT" ]]; then
  for directory in /usr/include/xrt /opt/xilinx/xrt/include /usr/include; do
    if [[ -f "$directory/xrt/xrt_bo.h" ]]; then XRT_HEADER_ROOT="$directory"; break; fi
  done
fi
if [[ -z "$XRT_CORE_LIBRARY" ]]; then
  for library in /usr/lib/aarch64-linux-gnu/libxrt_coreutil.so /opt/xilinx/xrt/lib/libxrt_coreutil.so /usr/lib/libxrt_coreutil.so; do
    if [[ -f "$library" ]]; then XRT_CORE_LIBRARY="$library"; break; fi
  done
fi
if [[ ! -f "$XRT_HEADER_ROOT/xrt/xrt_bo.h" || ! -f "$XRT_CORE_LIBRARY" ]]; then
  echo 'XRT development files missing. Set FPT_XRT_INCLUDE_DIR (parent of xrt/xrt_bo.h) and FPT_XRT_LIBRARY (libxrt_coreutil.so).' >&2
  exit 2
fi
cmake -S "$ROOT/llama.cpp" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DFPT_LLAMA_PL_U126=ON -DFPT_LLAMA_PL_U19=OFF \
  -DFPT_XRT_INCLUDE_DIR="$XRT_HEADER_ROOT" -DFPT_XRT_LIBRARY="$XRT_CORE_LIBRARY" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
  "$@"
cmake --build "$BUILD" --target llama-server llama-completion llama-tokenize \
  --parallel "${FPT_BUILD_JOBS:-2}"
printf 'U126_SOURCE_BUILD=PASS\nRuntime directory: %s/bin\n' "$BUILD"
printf 'Prebuilt /opt/u126 payload was not replaced. Validate rebuilt binaries separately.\n'
