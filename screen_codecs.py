"""Compare real FP32 chunks before choosing the browser codec candidates."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zlib
import numpy as np
import bitshuffle
import zipnn
import zstandard as zstd
from lossless_codecs import transform, inverse, compress, decompress

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'benchmarks/adaptive-codec-screen.json'
    if output.exists():
        raise FileExistsError(output)
    folder = ROOT / 'public/models/packed'
    manifest = json.loads((folder / 'manifest.json').read_text())
    keys = sorted({p['chunk'] for p in manifest['files']['profiles/fast/moge3-dense.onnx']['pieces']
                   if manifest['chunks'][p['chunk']]['rawBytes'] == 4*1024*1024})
    keys = [keys[round(i * (len(keys)-1) / 7)] for i in range(8)]
    zn = zipnn.ZipNN(method='HUFFMAN', input_format='byte', bytearray_dtype='float32', threads=1)
    rows = []
    for key in keys:
        meta = manifest['chunks'][key]
        raw = inverse(zlib.decompress((folder / f'chunks/{key}.zlib').read_bytes()), meta['encoding'])
        assert hashlib.sha256(raw).hexdigest() == meta['rawSha256']
        results = []
        for codec, encoding, level in [('deflate', 'shuffle4', 6)] + [('zstd', e, n) for n in [12, 19] for e in ['identity', 'shuffle4', 'exponent4', 'bitplane4']]:
            start = time.perf_counter()
            encoded = transform(raw, encoding)
            packed = compress(encoded, codec, level)
            compress_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            restored = inverse(decompress(packed, codec, len(raw)), encoding)
            results.append({'codec': codec, 'encoding': encoding, 'level': level, 'bytes': len(packed),
                            'compressMs': compress_ms, 'decodeMs': (time.perf_counter()-start)*1000, 'byteExact': restored == raw})
        for name in ['zipnn-huffman', 'bitshuffle-c-zstd12']:
            try:
                start = time.perf_counter()
                if name == 'zipnn-huffman':
                    # The native extension may mutate its input buffer. Keep
                    # immutable source evidence isolated from that side effect.
                    isolated = bytes(bytearray(raw))
                    packed = zn.compress(isolated)
                    compress_ms = (time.perf_counter()-start)*1000
                    start = time.perf_counter()
                    restored = zn.decompress(packed)
                else:
                    array = np.frombuffer(raw, '<u4')
                    packed = zstd.ZstdCompressor(level=12).compress(bitshuffle.bitshuffle(array).tobytes())
                    compress_ms = (time.perf_counter()-start)*1000
                    start = time.perf_counter()
                    encoded = np.frombuffer(zstd.ZstdDecompressor().decompress(packed), '<u4').copy()
                    restored = bitshuffle.bitunshuffle(encoded).tobytes()
                results.append({'codec': name, 'bytes': len(packed), 'compressMs': compress_ms,
                                'decodeMs': (time.perf_counter()-start)*1000, 'byteExact': restored == raw})
            except Exception as error:
                results.append({'codec': name, 'error': str(error), 'byteExact': False})
        assert all(r['byteExact'] for r in results if 'error' not in r), results
        assert hashlib.sha256(raw).hexdigest() == meta['rawSha256']
        rows.append({'sourceChunk': key, 'rawSha256': meta['rawSha256'], 'rawBytes': len(raw), 'results': results})
        print('Screened', len(rows), 'of', len(keys), flush=True)
    totals = {}
    for row in rows:
        for r in row['results']:
            name = f"{r['codec']}/{r.get('encoding','native')}/{r.get('level','default')}"
            if 'error' in r:
                totals.setdefault(name, {'error': r['error']})
            else:
                t = totals.setdefault(name, {'bytes': 0, 'compressMs': 0, 'decodeMs': 0})
                for k in t:
                    t[k] += r[k]
    output.write_text(json.dumps({'method': 'Eight distinct real dense chunks, 32 MiB total; every successful candidate round-tripped byte-exactly. Native screening times are not browser decoding times.',
                                  'versions': {name: importlib.metadata.version(name) for name in ['numpy','zstandard','zipnn','bitshuffle']}, 'rows': rows, 'totals': totals}, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
