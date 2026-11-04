#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="fpt-u126-dynamic-34fd414e"
APP_DIR="/lib/firmware/xilinx/${APP_NAME}"
RELOAD=0
if [[ "${1:-}" == "--reload" && $# -eq 1 ]]; then
  RELOAD=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: sudo bash scripts/activate_u126_overlay.sh [--reload]" >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/activate_u126_overlay.sh [--reload]" >&2
  exit 2
fi
cd "$ROOT"
test -f "${APP_DIR}/binary_container_1.bin"
test -f "${APP_DIR}/pl.dtbo"
test -f "${APP_DIR}/shell.json"
test "$(sha256sum "${APP_DIR}/binary_container_1.bin" | awk '{print $1}')" = \
  "34fd414e8694597c257c19b8a8db8e3c45e7f6ececbf8fc07837789ce63a8959"

for proc in /proc/[0-9]*; do
  pid="${proc##*/}"
  if tr '\0' '\n' 2>/dev/null <"${proc}/environ" | grep -qx 'FPT_LLAMA_PL_ENABLE=1'; then
    echo "PL-enabled process ${pid} is running; stop it before loading the overlay." >&2
    exit 3
  fi
done

apps="$(xmutil listapps)"
active_app="$(awk '$NF ~ /^[0-9]+,?$/ {print $1; exit}' <<<"$apps")"
active_slot="$(awk -v app="$APP_NAME" '$1 == app && $NF ~ /^[0-9]+,?$/ {slot=$NF; sub(/,$/, "", slot); print slot; exit}' <<<"$apps")"
if [[ -n "$active_app" && "$active_app" != "$APP_NAME" ]]; then
  echo "Another PL app is active (${active_app}); unload it explicitly before continuing." >&2
  exit 4
fi

if [[ -n "$active_slot" && "$RELOAD" -eq 1 ]]; then
  xmutil unloadapp "$active_slot"
  active_slot=""
fi
if [[ -z "$active_slot" ]]; then
  xmutil loadapp "$APP_NAME"
fi

state="$(cat /sys/class/fpga_manager/fpga0/state)"
test "$state" = "operating"
apps="$(xmutil listapps)"
active_app="$(awk '$NF ~ /^[0-9]+,?$/ {print $1; exit}' <<<"$apps")"
test "$active_app" = "$APP_NAME"
sha256sum "${APP_DIR}/binary_container_1.bin" "${APP_DIR}/pl.dtbo" "${APP_DIR}/shell.json"
printf 'U126 overlay active: %s (FPGA manager: %s)\n' "$active_app" "$state"
