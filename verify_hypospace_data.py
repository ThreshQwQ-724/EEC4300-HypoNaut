import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'HYPOSPACE_DATA_SHA256.json').read_text(encoding='utf-8'))
for filename, expected in manifest.items():
    path = root / filename
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise SystemExit(f'Missing file or incorrect size: {filename}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected['sha256']:
        raise SystemExit(f'SHA-256 mismatch: {filename}')
    print(f'OK  {filename}')
