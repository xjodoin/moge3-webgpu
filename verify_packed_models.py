"""Stream-reconstruct every packaged file and verify its original SHA-256."""
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=ROOT / 'public/models/packed')
    parser.add_argument('--report', type=Path, default=ROOT / 'benchmarks/packed-file-verification.json')
    args = parser.parse_args()
    folder = args.package
    manifest_bytes = (folder / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['format'] >= 2:
        from lossless_codecs import inverse, decompress
        if manifest['format'] == 3:
            from fp24 import expand24,restore32
        for key, meta in manifest['bundles'].items():
            with (folder / f'bundles/{key}.pack').open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == key
    @lru_cache(maxsize=16)
    def chunk(key):
        meta = manifest['chunks'][key]
        if manifest['format'] >= 2:
            with (folder / f"bundles/{meta['bundle']}.pack").open('rb') as f:
                f.seek(meta['offset'])
                compressed = f.read(meta['bytes'])
        else:
            compressed = (folder / f'chunks/{key}.zlib').read_bytes()
        assert len(compressed) == meta['bytes'] and hashlib.sha256(compressed).hexdigest() == key
        raw = decompress(compressed, meta['codec'], meta.get('encodedBytes',meta['rawBytes'])) if manifest['format'] >= 2 else zlib.decompress(compressed)
        if manifest['format'] >= 2:
            raw = expand24(raw) if meta['encoding']=='fp24' else inverse(raw, meta['encoding'])
        elif meta['encoding'] == 'shuffle4':
            raw = np.frombuffer(raw, np.uint8).reshape(4, -1).T.copy().tobytes()
        assert len(raw) == meta['rawBytes'] and hashlib.sha256(raw).hexdigest() == meta['rawSha256']
        return raw
    rows = []
    for name, file in manifest['files'].items():
        if name.startswith('__corrections/'):continue
        digest, exact_digest, offset = hashlib.sha256(),hashlib.sha256(), 0
        for p in file['pieces']:
            assert p['offset'] == offset
            raw = chunk(p['chunk'])
            digest.update(raw[p['sourceOffset']:p['sourceOffset']+p['length']])
            meta=manifest['chunks'][p['chunk']]
            if meta['encoding']=='fp24':raw=restore32(raw,chunk(meta['correction']))
            exact_digest.update(raw[p['sourceOffset']:p['sourceOffset']+p['length']])
            offset += p['length']
        source = ROOT / 'public/models' / name
        with source.open('rb') as f:
            source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        assert offset == file['bytes'] == source.stat().st_size
        assert digest.hexdigest() == file['sha256']
        assert exact_digest.hexdigest() == file.get('exactSha256',file['sha256']) == source_hash
        rows.append({'path': name, 'bytes': offset, 'sha256': source_hash, 'compactSha256':digest.hexdigest(), 'exactUpgradeByteExact':True, 'byteExact':digest.hexdigest()==source_hash})
    output = args.report
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'packageSha256': hashlib.sha256(manifest_bytes).hexdigest(), 'files': rows, 'passed': True}, indent=2) + '\n')
    print('Verified all', len(rows), 'files against original bytes')


if __name__ == '__main__':
    main()
