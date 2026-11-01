"""Lossless, content-addressed transport for byte-identical FP32 model files.

Extract protobuf initializer payload spans without reserializing ONNX. Split
weights at tensor-relative boundaries so different profiles share the chunks.
Byte-shuffle float32 payloads, zlib-compress, and retain the original file hash.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
CHUNK_BYTES = 4 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def varint(data, offset, end):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= end:
            raise ValueError('Truncated protobuf varint')
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError('Invalid protobuf varint')


def fields(data, start, end):
    """Yield field, wire type, payload start/end, scalar value (if wire=0)."""
    cursor = start
    while cursor < end:
        tag, cursor = varint(data, cursor, end)
        field, wire = tag >> 3, tag & 7
        if field < 1:
            raise ValueError('Invalid protobuf field')
        scalar = None
        if wire == 0:
            scalar, cursor = varint(data, cursor, end)
            yield field, wire, cursor, cursor, scalar
            continue
        if wire == 2:
            length, cursor = varint(data, cursor, end)
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
        else:
            raise ValueError(f'Unsupported protobuf wire type: {wire}')
        stop = cursor + length
        if stop > end:
            raise ValueError('Truncated protobuf field')
        yield field, wire, cursor, stop, scalar
        cursor = stop


def initializer_spans(data):
    spans = []
    for field, wire, a, b, _ in fields(data, 0, len(data)):
        if field != 7 or wire != 2:  # ModelProto.graph
            continue
        for field, wire, a, b, _ in fields(data, a, b):
            if field != 5 or wire != 2:  # GraphProto.initializer
                continue
            tensor_fields = list(fields(data, a, b))
            dtype = next((value for n, w, _, _, value in tensor_fields if n == 2 and w == 0), None)
            for n, w, a, b, _ in tensor_fields:
                if n == 9 and w == 2 and b > a:  # TensorProto.raw_data
                    spans.append((a, b, 'shuffle4' if dtype == 1 and (b-a) % 4 == 0 else 'identity'))
    return sorted(spans)


class Packer:
    def __init__(self, out, chunk_bytes=CHUNK_BYTES):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'chunks').mkdir()
        if chunk_bytes < 4 or chunk_bytes % 4:
            raise ValueError('Chunk size must be divisible by four')
        self.chunk_bytes = chunk_bytes
        self.chunks = {}
        self.raw_chunks = {}
        self.files = {}

    def chunk(self, raw, encoding):
        raw_hash = sha(raw)
        identity = (raw_hash, encoding)
        if identity in self.raw_chunks:
            return self.raw_chunks[identity]
        transformed = (np.frombuffer(raw, np.uint8).reshape(-1, 4).T.copy().tobytes()
                       if encoding == 'shuffle4' else raw)
        packed = zlib.compress(transformed, 6)
        key = sha(packed)
        meta = {'bytes': len(packed), 'rawBytes': len(raw), 'rawSha256': raw_hash, 'encoding': encoding}
        if key not in self.chunks:
            self.chunks[key] = meta
            (self.out / f'chunks/{key}.zlib').write_bytes(packed)
        elif self.chunks[key] != meta:
            raise ValueError('Chunk identity collision')
        self.raw_chunks[identity] = key
        return key

    def add(self, source, name):
        source = Path(source)
        with source.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            spans = initializer_spans(data) if source.suffix == '.onnx' else [(0, len(data), 'shuffle4')]
            pieces, metadata, cursor = [], bytearray(), 0
            for start, end, encoding in spans:
                if start < cursor:
                    raise ValueError('Overlapping tensor spans')
                if start > cursor:
                    pieces.append({'offset': cursor, 'length': start-cursor, 'sourceOffset': len(metadata), 'chunk': None})
                    metadata.extend(data[cursor:start])
                for offset in range(start, end, self.chunk_bytes):
                    raw = data[offset:min(end, offset+self.chunk_bytes)]
                    pieces.append({'offset': offset, 'length': len(raw), 'sourceOffset': 0, 'chunk': self.chunk(raw, encoding)})
                cursor = end
            if cursor < len(data):
                pieces.append({'offset': cursor, 'length': len(data)-cursor, 'sourceOffset': len(metadata), 'chunk': None})
                metadata.extend(data[cursor:])
            # Graph/protobuf headers share one small chunk, avoiding hundreds of
            # tiny network requests. Only weight chunks have the 4 MiB bound.
            if metadata:
                key = self.chunk(bytes(metadata), 'identity')
                for piece in pieces:
                    if piece['chunk'] is None:
                        piece['chunk'] = key
            self.files[name] = {'bytes': len(data), 'sha256': sha(data), 'pieces': pieces}
            print(name, len(data), 'bytes', len({p['chunk'] for p in pieces}), 'chunks', flush=True)

    def finish(self):
        manifest = {'format': 1, 'codec': 'zlib-byte-shuffle-v1', 'maxChunkBytes': self.chunk_bytes,
                    'checkpointRevision': '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be',
                    'files': self.files, 'chunks': self.chunks}
        (self.out / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')) + '\n')
        return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'public/models/packed')
    args = parser.parse_args()
    models = ROOT / 'public/models'
    packer = Packer(args.out)
    paths = [models / f'profiles/{profile}/moge3-dense.onnx' for profile in ['fast', 'detail']]
    # Include the optional ONNX comparison backend too; default WGSL inference
    # requests only its 11 linear/projection modules and 10 packed blocks.
    paths += sorted((models / 'refiner').glob('*.onnx'))
    paths += sorted((models / 'refiner-wgsl').glob('*.bin'))
    for path in paths:
        packer.add(path, str(path.relative_to(models)))
    manifest = packer.finish()
    print('Unique compressed bytes:', sum(c['bytes'] for c in manifest['chunks'].values()), flush=True)


if __name__ == '__main__':
    main()
