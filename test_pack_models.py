import hashlib
import tempfile
import unittest
from pathlib import Path
import zlib

import numpy as np
import onnx
from onnx import helper, numpy_helper
from pack_models import Packer, initializer_spans, fields


def recover(folder, manifest, name):
    file = manifest['files'][name]
    result = bytearray(file['bytes'])
    for piece in file['pieces']:
        key = piece['chunk']
        compressed = (folder / f'chunks/{key}.zlib').read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == key
        meta = manifest['chunks'][key]
        raw = zlib.decompress(compressed)
        if meta['encoding'] == 'shuffle4':
            raw = np.frombuffer(raw, np.uint8).reshape(4, -1).T.copy().tobytes()
        assert hashlib.sha256(raw).hexdigest() == meta['rawSha256']
        a, n = piece['sourceOffset'], piece['length']
        result[piece['offset']:piece['offset']+n] = raw[a:a+n]
    assert hashlib.sha256(result).hexdigest() == file['sha256']
    return bytes(result)


class PackingTests(unittest.TestCase):
    def test_onnx_byte_identity_and_shared_weights_with_different_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packer = Packer(root / 'packed', chunk_bytes=64)
            weights = np.arange(512, dtype=np.float32).reshape(32, 16) / 7
            for name in ['fast', 'detail']:
                graph = helper.make_graph([helper.make_node('Identity', ['weights'], ['output'])], name,
                                          [], [helper.make_tensor_value_info('output', 1, [32, 16])],
                                          [numpy_helper.from_array(weights, 'weights')])
                model = helper.make_model(graph)
                source = root / f'{name}.onnx'
                onnx.save(model, source)
                packer.add(source, name)
            manifest = packer.finish()
            for name in ['fast', 'detail']:
                self.assertEqual(recover(root / 'packed', manifest, name), (root / f'{name}.onnx').read_bytes())
            shared = {p['chunk'] for p in manifest['files']['fast']['pieces']} & {p['chunk'] for p in manifest['files']['detail']['pieces']}
            self.assertTrue(shared)
            self.assertEqual(sum(manifest['chunks'][key]['rawBytes'] for key in shared), weights.nbytes)

    def test_packed_float_bytes_include_nan_payloads_and_signed_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'block.bin'
            source.write_bytes(np.array([0, 0x80000000, 0x7fc00001, 0x12345678], '<u4').tobytes())
            packer = Packer(root / 'packed', chunk_bytes=8)
            packer.add(source, 'block')
            self.assertEqual(recover(root / 'packed', packer.finish(), 'block'), source.read_bytes())
            with self.assertRaises(FileExistsError):
                Packer(root / 'packed')

    def test_rejects_truncated_or_invalid_protobuf(self):
        for raw in [b'\x3a\x80', b'\x3a\xff\x7f', b'\x00', b'\x3f']:
            with self.assertRaises(ValueError):
                list(fields(raw, 0, len(raw)))
        self.assertEqual(initializer_spans(b'\x08\x01'), [])


if __name__ == '__main__':
    unittest.main()
