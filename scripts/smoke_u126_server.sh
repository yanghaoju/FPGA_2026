#!/usr/bin/env bash
set -euo pipefail

port="${1:-8083}"
pid="$(pgrep -f "[l]lama-server.*--port ${port}" | head -1 || true)"
if [[ -z "$pid" ]]; then
  echo "No llama-server is listening on the requested port ${port}." >&2
  exit 2
fi
env_text="$(tr '\0' '\n' <"/proc/${pid}/environ")"
test "$(readlink -f "/proc/${pid}/exe")" = "/opt/u126/bin/llama-server"
tr '\0' '\n' <"/proc/${pid}/cmdline" | grep -qx -- '--no-context-shift'
test "$(sha256sum "/proc/${pid}/exe" | awk '{print $1}')" = \
  "1417cd3f62b82bebb05172f99c47cc85a315f9eebf904ad89907fff3694fe8ec"
for setting in \
  'FPT_LLAMA_PL_ENABLE=1' \
  'FPT_LLAMA_PL_STRICT=1' \
  'FPT_LLAMA_PL_PROFILE=U126_L512'; do
  grep -qx "$setting" <<<"$env_text"
done
xclbin="$(sed -n 's/^FPT_LLAMA_PL_XCLBIN=//p' <<<"$env_text")"
test -n "$xclbin"
test "$(sha256sum "$xclbin" | awk '{print $1}')" = \
  "34fd414e8694597c257c19b8a8db8e3c45e7f6ececbf8fc07837789ce63a8959"
runtime="$(awk '/libggml-cpu.so.0.17.0/ {print $NF; exit}' "/proc/${pid}/maps")"
test -n "$runtime"
test "$(sha256sum "$runtime" | awk '{print $1}')" = \
  "489aa59502d3cd83381cdf4dd91768d9387b5d2511d20d924ac2082362bb6871"
test "$(cat /sys/class/fpga_manager/fpga0/state)" = "operating"
curl -fsS "http://127.0.0.1:${port}/health"
printf '\n'

slots="$(curl -fsS "http://127.0.0.1:${port}/slots")"
python3 -c 'import json,sys; slots=json.loads(sys.argv[1]); sys.exit(1 if any(x.get("is_processing") for x in slots) else 0)' "$slots"
response="$(curl -fsS --max-time 180 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Reply with one short word: ready","n_predict":2,"temperature":0,"seed":42,"cache_prompt":false}' \
  "http://127.0.0.1:${port}/completion")"
printf '%s\n' "$response" | python3 -c 'import json,sys; r=json.load(sys.stdin); n=int(r.get("tokens_predicted",0)); print("tokens_predicted="+str(n)); sys.exit(0 if n>0 else 1)'
curl -fsS "http://127.0.0.1:${port}/health"
printf '\nU126 strict-PL server smoke test: PASS\n'
