#!/usr/bin/env python3
"""Check the submission payload and, optionally, the separate model file."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    count = 0
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        sha, rel = line.split('  ', 1)
        file = root / rel
        if not file.is_file() or digest(file) != sha:
            raise SystemExit('Checksum mismatch or missing file: ' + rel)
        count += 1
    manifest = json.loads((root / 'board/runtime_manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        file = root / 'board' / rel
        with file.open('rb') as f:
            header = f.read(20)
        if file.stat().st_size != expected['bytes'] or digest(file) != expected['sha256']:
            raise SystemExit('Runtime identity mismatch: ' + rel)
        if header[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', header, 18)[0] != 183:
            raise SystemExit('Expected little-endian AArch64 ELF: ' + rel)
    if args.model:
        expected = json.loads((root / 'model/fpt_model_manifest.json').read_text(encoding='utf-8'))['files'][0]
        if args.model.stat().st_size != expected['bytes'] or digest(args.model) != expected['sha256']:
            raise SystemExit('Model identity mismatch')
        print('U126_MODEL_VERIFY=PASS')
    print('U126_SUBMISSION_VERIFY=PASS; checked files=' + str(count))

if __name__ == '__main__':
    main()
