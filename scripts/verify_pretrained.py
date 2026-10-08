#!/usr/bin/env python3
"""Verify the public upstream-only asset set against its release manifest."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory', default='pretrained')
a = p.parse_args()
root = Path(a.directory).resolve()
manifest = json.loads((root / 'manifest.json').read_text())
allowed = set()
for asset in manifest['assets']:
    path = (root / asset['path']).resolve()
    if not path.is_relative_to(root) or not asset['matches_official_asset']:
        raise ValueError('Invalid public asset provenance')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != asset['bytes'] or digest.hexdigest() != asset['sha256']:
        raise ValueError(f'Integrity check failed: {asset["path"]}')
    allowed.add(path)
    print('Verified', asset['path'], flush=True)
actual = {p.resolve() for p in (root / 'FlashVSR').rglob('*') if p.is_file()}
if actual != allowed:
    raise ValueError('Unexpected or missing file in public model asset directory')
if manifest.get('local_lora_included') is not False:
    raise ValueError('Private LoRA inclusion is forbidden')
print(f'Public upstream assets verified: {len(allowed)} files; private LoRA: 0')
