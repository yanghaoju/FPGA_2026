#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FPT_KV260_XPFM:?Set FPT_KV260_XPFM to a matching KV260 Vitis 2022.2 platform .xpfm}"
test -f "$FPT_KV260_XPFM"
for tool in vitis_hls v++; do
  command -v "$tool" >/dev/null || { echo 'Source Vitis and Vivado 2022.2 settings64.sh first.' >&2; exit 2; }
done
OUT="${FPT_HARDWARE_BUILD_ROOT:-$ROOT/build/hardware-u126}"
if [[ -e "$OUT" ]]; then
  echo "Use a new build output directory: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT/source"
OUT="$(cd "$OUT" && pwd)"
cp -a "$ROOT/hardware/U126/." "$OUT/source/"
export FPT_DATA_DIR="$ROOT/validation/vectors"
unset LD CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH
cd "$OUT/source"
for script in run_hls_pack_csim.tcl run_hls_prefill_csim.tcl run_hls_runtime_sweep_csim.tcl run_hls_decode_csim.tcl run_hls_csynth.tcl; do
  vitis_hls -f "$script"
done
XO="$(find "$OUT/source/fpt_u126_l512_runtime_length_no_q_zero_fill/solution1" -name '*.xo' -type f -print -quit)"
test -n "$XO" && test -f "$XO"
v++ --no_ip_cache --vivado.synth.jobs 1 --vivado.impl.jobs 1 \
  --link --target hw --platform "$FPT_KV260_XPFM" \
  --config "$OUT/source/connectivity.cfg" --save-temps \
  --temp_dir "$OUT/temp" --report_dir "$OUT/reports" --log_dir "$OUT/logs" \
  -o "$OUT/U126_L512.xclbin" "$XO"
sha256sum "$OUT/U126_L512.xclbin"
echo 'U126_HARDWARE_BUILD=PASS; validate the new platform-matched output before deployment.'
