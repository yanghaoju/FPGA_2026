#!/usr/bin/env python3
"""Download the frozen GGUF weight file, then check size and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def valid(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes']:
        return False
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest() == record['sha256']

def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=root / 'model')
    parser.add_argument('--mirror', action='store_true', help='Use the manifest mirror instead of Hugging Face')
    args = parser.parse_args()
    manifest = json.loads((root / 'model/fpt_model_manifest.json').read_text(encoding='utf-8'))
    record = manifest['files'][0]
    target = args.directory / record['name']
    args.directory.mkdir(parents=True, exist_ok=True)
    if valid(target, record):
        print('Existing verified model: ' + str(target))
        return
    if target.exists():
        raise SystemExit('Existing model does not match; move it aside before downloading: ' + str(target))
    url = manifest['source']['mirror_url' if args.mirror else 'url']
    temporary = target.with_suffix(target.suffix + '.part')
    print('Built with Llama. Use is subject to licenses/Llama-3.2-LICENSE.txt and its Acceptable Use Policy.', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'U126-competition-release/20261104'})
    downloaded = 0
    last = 0
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as dest:
        while True:
            data = source.read(4 * 1024 * 1024)
            if not data:
                break
            dest.write(data)
            downloaded += len(data)
            if downloaded - last >= 64 * 1024 * 1024:
                print(str(downloaded) + ' / ' + str(record['bytes']) + ' bytes', flush=True)
                last = downloaded
    if not valid(temporary, record):
        raise SystemExit('Downloaded model failed verification; incomplete file retained as: ' + str(temporary))
    temporary.replace(target)
    print('U126_MODEL_DOWNLOAD=PASS: ' + str(target))

if __name__ == '__main__':
    main()
