#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="fpt-u126-dynamic-34fd414e"
APP_DIR="/lib/firmware/xilinx/${APP_NAME}"
LIB_DIR="/opt/u126/lib"
BIN_DIR="/opt/u126/bin"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This runtime package targets AArch64 KV260/K26 Linux." >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_u126_payload.sh" >&2
  exit 2
fi

cd "$ROOT"
bash scripts/verify_u126_release.sh
install -D -m 0644 board/app/U126_L512.xclbin "${APP_DIR}/binary_container_1.bin"
install -D -m 0644 board/app/pl.dtbo "${APP_DIR}/pl.dtbo"
install -D -m 0644 board/app/shell.json "${APP_DIR}/shell.json"
install -D -m 0644 board/lib/libggml-cpu.so.0.17.0 \
  "${LIB_DIR}/libggml-cpu.so.0.17.0"
install -D -m 0755 board/bin/llama-server "${BIN_DIR}/llama-server"
for library in \
  libggml-base.so.0 libggml.so.0 libllama.so.0 libllama-common.so.0 \
  libllama-server-impl.so libmtmd.so.0; do
  install -D -m 0644 "board/lib/${library}" "${LIB_DIR}/${library}"
done
ln -sfn libggml-cpu.so.0.17.0 "${LIB_DIR}/libggml-cpu.so.0"
ln -sfn libggml-cpu.so.0.17.0 "${LIB_DIR}/libggml-cpu.so"

expected_xclbin="34fd414e8694597c257c19b8a8db8e3c45e7f6ececbf8fc07837789ce63a8959"
expected_dtbo="3929b83506e379e61d57588d4102806e9526beffde503313a6789f125d4ce2cd"
expected_shell="30ff653b82b73247d85df48355ae0c7bc76dae8132565f4e8d64e110f98ca562"
expected_runtime="489aa59502d3cd83381cdf4dd91768d9387b5d2511d20d924ac2082362bb6871"
expected_server="1417cd3f62b82bebb05172f99c47cc85a315f9eebf904ad89907fff3694fe8ec"
test "$(sha256sum "${APP_DIR}/binary_container_1.bin" | awk '{print $1}')" = "$expected_xclbin"
test "$(sha256sum "${APP_DIR}/pl.dtbo" | awk '{print $1}')" = "$expected_dtbo"
test "$(sha256sum "${APP_DIR}/shell.json" | awk '{print $1}')" = "$expected_shell"
test "$(sha256sum "${LIB_DIR}/libggml-cpu.so.0.17.0" | awk '{print $1}')" = "$expected_runtime"
test "$(sha256sum "${BIN_DIR}/llama-server" | awk '{print $1}')" = "$expected_server"
for library in \
  libggml-base.so.0 libggml.so.0 libllama.so.0 libllama-common.so.0 \
  libllama-server-impl.so libmtmd.so.0; do
  test -f "${LIB_DIR}/${library}"
done

printf 'U126 board payload installed and verified.\n'
printf 'Overlay app: %s\nRuntime binary: %s\nRuntime libraries: %s\n' "$APP_DIR" "$BIN_DIR" "$LIB_DIR"
printf 'Next: stop PL/XRT clients, then run sudo bash scripts/activate_u126_overlay.sh --reload\n'
