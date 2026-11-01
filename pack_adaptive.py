"""Build adaptive byte-exact codecs and indexed bundles from a v1 package.

The original package is read-only. Old cache representations remain listed so
existing users can upgrade without downloading identical weights again.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from lossless_codecs import compress, decompress, transform, inverse

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(source, out, bundle_bytes=16*1024*1024):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pieces = out / 'temporary-chunks'
    pieces.mkdir()
    (out / 'bundles').mkdir()
    original_bytes = (source / 'manifest.json').read_bytes()
    old = json.loads(original_bytes)
    chunks, remap, rows = {}, {}, []
    for index, (key, meta) in enumerate(old['chunks'].items()):
        start = time.perf_counter()
        previous = (source / f'chunks/{key}.zlib').read_bytes()
        assert sha(previous) == key
        raw = inverse(decompress(previous, 'deflate', meta['rawBytes']), meta['encoding'])
        assert sha(raw) == meta['rawSha256']
        candidates = [(previous, 'deflate', meta['encoding'], 6)]
        encodings = ['identity'] if meta['encoding'] == 'identity' else ['shuffle4', 'exponent4'] + (['bitplane4'] if len(raw) % 32 == 0 else [])
        for encoding in encodings:
            encoded = transform(raw, encoding)
            level = 12 if encoding == 'bitplane4' else 19
            encoded_bytes = compress(encoded, 'zstd', level)
            restored = inverse(decompress(encoded_bytes, 'zstd', len(raw)), encoding)
            assert restored == raw
            candidates.append((encoded_bytes, 'zstd', encoding, level))
        selected, codec, encoding, level = min(candidates, key=lambda c: len(c[0]))
        new_key = sha(selected)
        remap[key] = new_key
        legacy = {'sha256': key, 'bytes': len(previous), 'codec': 'deflate', 'encoding': meta['encoding']}
        if new_key not in chunks:
            chunks[new_key] = {'bytes': len(selected), 'rawBytes': len(raw), 'rawSha256': sha(raw),
                               'codec': codec, 'encoding': encoding, 'level': level, 'legacy': [legacy]}
            (pieces / new_key).write_bytes(selected)
        else:
            assert chunks[new_key]['rawSha256'] == sha(raw)
            if legacy not in chunks[new_key]['legacy']:
                chunks[new_key]['legacy'].append(legacy)
        rows.append({'sourceChunk': key, 'targetChunk': new_key, 'rawBytes': len(raw),
                     'sourceBytes': len(previous), 'selectedBytes': len(selected), 'codec': codec,
                     'encoding': encoding, 'candidates': [{'codec': c, 'encoding': e, 'level': n, 'bytes': len(b)} for b,c,e,n in candidates],
                     'byteExact': True, 'packMs': (time.perf_counter()-start)*1000})
        if index % 100 == 0:
            print('Compressed', index+1, 'of', len(old['chunks']), flush=True)
    files = json.loads(json.dumps(old['files']))
    for file in files.values():
        for piece in file['pieces']:
            piece['chunk'] = remap[piece['chunk']]
    default_files = [name for name in files if name.startswith('refiner-wgsl/') or
                     (name.startswith('refiner/') and ('linear' in name or Path(name).stem in ['fuse','input_proj','out_proj']))]
    active = {p: {piece['chunk'] for name in [f'profiles/{p}/moge3-dense.onnx']+default_files for piece in files[name]['pieces']}
              for p in ['fast', 'detail']}
    groups = collections.defaultdict(list)
    for key in chunks:
        usage = (key in active['fast'], key in active['detail'])
        groups[{(True,True):'shared',(True,False):'fast',(False,True):'detail',(False,False):'comparison'}[usage]].append(key)
    bundles = {}
    def write_bundle(keys, group):
        data = b''.join((pieces / key).read_bytes() for key in keys)
        key = sha(data)
        (out / f'bundles/{key}.pack').write_bytes(data)
        bundles[key] = {'bytes': len(data), 'members': keys, 'usage': group}
        offset = 0
        for chunk in keys:
            chunks[chunk].update(bundle=key, offset=offset)
            offset += chunks[chunk]['bytes']
    for group, keys in groups.items():
        batch, size = [], 0
        for key in keys:
            if batch and size + chunks[key]['bytes'] > bundle_bytes:
                write_bundle(batch, group)
                batch, size = [], 0
            batch.append(key)
            size += chunks[key]['bytes']
        if batch:
            write_bundle(batch, group)
    manifest = {'format': 2, 'codec': 'adaptive-bundles-v2', 'checkpointRevision': old['checkpointRevision'],
                'previousPackageSha256': sha(original_bytes), 'bundleTargetBytes': bundle_bytes,
                'files': files, 'chunks': chunks, 'bundles': bundles}
    (out / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':'))+'\n')
    shutil.rmtree(pieces)
    report = {'sourcePackageSha256': sha(original_bytes), 'packageSha256': sha((out/'manifest.json').read_bytes()),
              'versions': {name: importlib.metadata.version(name) for name in ['numpy','zstandard']}, 'chunks': rows,
              'allByteExact': True, 'uniqueCompressedBytes': sum(c['bytes'] for c in chunks.values()), 'bundleCount': len(bundles)}
    return manifest, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'public/models/packed')
    p.add_argument('--out', type=Path, default=ROOT/'public/models/packed-v2')
    p.add_argument('--report', type=Path, default=ROOT/'benchmarks/adaptive-codec-full.json')
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest, report = build(args.source, args.out)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('All candidates round-trip exactly;',len(manifest['bundles']),'bundles;',report['uniqueCompressedBytes'],'bytes',flush=True)


if __name__ == '__main__':
    main()
