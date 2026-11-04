#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCLBIN="/lib/firmware/xilinx/fpt-u126-dynamic-34fd414e/binary_container_1.bin"
TEST_BIN="$ROOT/validation/bin/test_u126_l512_board"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --build ) ]]; then
  echo "Usage: bash scripts/run_u126_numeric_gate.sh [--build]" >&2
  exit 2
fi
test "$(uname -m)" = aarch64
test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating
test "$(sha256sum "$XCLBIN" | awk '{print $1}')" = \
  34fd414e8694597c257c19b8a8db8e3c45e7f6ececbf8fc07837789ce63a8959
for proc in /proc/[0-9]*; do
  if tr '\0' '\n' 2>/dev/null <"${proc}/environ" | grep -qx 'FPT_LLAMA_PL_ENABLE=1'; then
    echo 'Stop the PL llama-server before running the hardware numeric gate.' >&2
    exit 3
  fi
done
if [[ "${1:-}" == --build ]]; then
  mkdir -p "$ROOT/validation/bin"
  XRT_INCLUDE="${XRT_INCLUDE:-/usr/include/xrt}"
  if [[ ! -f "$XRT_INCLUDE/xrt/xrt_bo.h" ]]; then
    echo 'XRT development headers missing; set XRT_INCLUDE to their parent directory.' >&2
    exit 4
  fi
  g++ -O2 -std=c++17 -I"$XRT_INCLUDE" -I"$ROOT/llama.cpp/fpt_u126" \
    "$ROOT/validation/src/test_u126_l512_board.cpp" \
    "$ROOT/llama.cpp/fpt_u126/fpt_attention_session.cpp" \
    -o "$TEST_BIN" -lxrt_coreutil -pthread
fi
test -f "$TEST_BIN"
# ZIP extractors may discard executable permission bits.
chmod u+x "$TEST_BIN"
exec "$TEST_BIN" "$XCLBIN" "$ROOT/validation/vectors" --runtime-length
