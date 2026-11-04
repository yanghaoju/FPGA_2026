#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-}"
PORT="${2:-8083}"
APP_DIR="/lib/firmware/xilinx/fpt-u126-dynamic-34fd414e"
SERVER="/opt/u126/bin/llama-server"
RUNTIME_LIB="/opt/u126/lib"

if [[ -z "$MODEL" || $# -gt 2 ]]; then
  echo "Usage: bash scripts/start_u126_demo.sh <model.gguf> [port]" >&2
  exit 2
fi
test -x "$SERVER"
test -f "$MODEL"
test -f "$APP_DIR/binary_container_1.bin"
test "$(sha256sum "$SERVER" | awk '{print $1}')" = \
  "1417cd3f62b82bebb05172f99c47cc85a315f9eebf904ad89907fff3694fe8ec"
test "$(sha256sum "$RUNTIME_LIB/libggml-cpu.so.0.17.0" | awk '{print $1}')" = \
  "489aa59502d3cd83381cdf4dd91768d9387b5d2511d20d924ac2082362bb6871"
test "$(sha256sum "$APP_DIR/binary_container_1.bin" | awk '{print $1}')" = \
  "34fd414e8694597c257c19b8a8db8e3c45e7f6ececbf8fc07837789ce63a8959"

U126_PORT="$PORT" exec bash "$ROOT/src/run_u126_server_template.sh" \
  "$SERVER" "$MODEL" \
  "$APP_DIR/binary_container_1.bin" "$RUNTIME_LIB"
