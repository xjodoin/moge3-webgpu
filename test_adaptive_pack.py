import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from lossless_codecs import transform, inverse, compress, decompress
from pack_models import Packer
from pack_adaptive import build


class AdaptivePackingTests(unittest.TestCase):
    def test_every_transform_preserves_nan_signed_zero_and_arbitrary_fp32_bits(self):
        words = np.array([0, 0x80000000, 0x7fc00001, 0x7f800000, 0xff800000] + [i*1234567 for i in range(27)], dtype='<u4')
        raw = words.tobytes()
        for encoding in ['identity', 'shuffle4', 'exponent4', 'bitplane4']:
            for codec in ['deflate', 'zstd']:
                packed = compress(transform(raw, encoding), codec)
                self.assertEqual(inverse(decompress(packed, codec, len(raw)), encoding), raw)

    def test_adaptive_bundles_are_exact_smallest_and_keep_legacy_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packer = Packer(root/'v1', chunk_bytes=128)
            sources = {}
            for profile in ['fast', 'detail']:
                raw = np.arange(1024, dtype='<f4').tobytes() + (b'fast' if profile == 'fast' else b'detl')
                path = root/f'{profile}.bin'
                path.write_bytes(raw)
                name = f'profiles/{profile}/moge3-dense.onnx'
                packer.add(path, name)
                sources[name] = raw
            old = packer.finish()
            manifest, report = build(root/'v1', root/'v2', bundle_bytes=512)
            self.assertTrue(report['allByteExact'])
            for row in report['chunks']:
                self.assertEqual(row['selectedBytes'], min(c['bytes'] for c in row['candidates']))
            for name, file in manifest['files'].items():
                assembled = bytearray(file['bytes'])
                for p in file['pieces']:
                    c = manifest['chunks'][p['chunk']]
                    blob = (root/f"v2/bundles/{c['bundle']}.pack").read_bytes()
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), c['bundle'])
                    packed = blob[c['offset']:c['offset']+c['bytes']]
                    raw = inverse(decompress(packed, c['codec'], c['rawBytes']), c['encoding'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), c['rawSha256'])
                    assembled[p['offset']:p['offset']+p['length']] = raw[p['sourceOffset']:p['sourceOffset']+p['length']]
                    self.assertTrue(all(v['sha256'] in old['chunks'] for v in c['legacy']))
                self.assertEqual(bytes(assembled), sources[name])
            self.assertFalse((root/'v2/temporary-chunks').exists())
            with self.assertRaises(FileExistsError):
                build(root/'v1', root/'v2')


if __name__ == '__main__':
    unittest.main()
