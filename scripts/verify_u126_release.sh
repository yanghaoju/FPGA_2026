#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
test -f SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
python3 - "$ROOT" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'board/runtime_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    p = root / 'board' / relative
    data = p.read_bytes()
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise SystemExit('Runtime manifest mismatch: ' + relative)
    if data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 183:
        raise SystemExit('Expected little-endian AArch64 ELF: ' + relative)
print('Runtime manifest and AArch64 ELF metadata: PASS')
PY
printf 'U126 release package: PASS\n'
