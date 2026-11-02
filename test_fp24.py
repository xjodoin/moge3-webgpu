import unittest
import tempfile,hashlib,json
from pathlib import Path
import numpy as np
from fp24 import split24,expand24,restore32


class WeightPrecisionTests(unittest.TestCase):
    def test_policy_keeps_nonencoder_and_shared_consumers_in_full_precision(self):
        from onnx import helper,numpy_helper,TensorProto
        from screen_fp24 import selected_spans
        arrays=[numpy_helper.from_array(np.ones((256,256),np.float32),name) for name in ['eligible','head','shared']]
        nodes=[helper.make_node('MatMul',['image',name],[name+'out'],name=node) for name,node in [('eligible','/model/encoder/blocks.0/fc1'),('head','/model/points_head/MatMul'),('shared','/model/encoder/blocks.1/fc2')]]
        nodes.append(helper.make_node('MatMul',['image','shared'],['extra'],name='/model/points_head/shared'))
        graph=helper.make_graph(nodes,'test',[helper.make_tensor_value_info('image',TensorProto.FLOAT,[256,256])],[],arrays)
        selected=selected_spans(helper.make_model(graph).SerializeToString())
        self.assertEqual([r['name'] for r in selected],['eligible'])

    def test_original_bits_restored_after_rounding_carries(self):
        words=np.array([0,0x80000000,1,127,128,255,256,0x3f80007f,
                        0x3f800080,0x3f8000ff,0x3f80ff80,0xbf80ff80,
                        0x7f7ffeff,0xff7ffeff],dtype='<u4')
        packed,low,rounded=split24(words.tobytes())
        self.assertEqual(len(packed),len(words)*3)
        self.assertEqual(expand24(packed),rounded)
        self.assertEqual(restore32(rounded,low),words.tobytes())
        self.assertEqual(np.frombuffer(rounded,'<u4')[8],0x3f800100)

    def test_random_finite_weights_restore_exactly(self):
        rng=np.random.default_rng(241032)
        words=rng.integers(0,2**32,size=100000,dtype=np.uint32)
        words=words[(words&0x7f800000)!=0x7f800000]
        words=words[(words&0x7fffff00)!=0x7f7fff00]
        packed,low,rounded=split24(words.astype('<u4').tobytes())
        self.assertEqual(expand24(packed),rounded)
        self.assertEqual(restore32(rounded,low),words.astype('<u4').tobytes())

    def test_rejects_nonfinite_overflow_and_invalid_sizes(self):
        for word in [0x7f800000,0x7fc00001,0xff800000,0x7f7fffff]:
            with self.assertRaises(ValueError):split24(np.array([word],dtype='<u4').tobytes())
        with self.assertRaises(ValueError):split24(b'a')
        with self.assertRaises(ValueError):expand24(b'a')
        with self.assertRaises(ValueError):restore32(b'1234',b'')

    def test_complete_package_separates_corrections_and_restores_original_hashes(self):
        from pack_models import Packer
        from pack_adaptive import build as adaptive
        from pack_progressive import build as progressive
        from lossless_codecs import inverse,decompress
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=np.linspace(-.91,.83,1024,dtype='<f4').tobytes()
            source=root/'weights.bin';source.write_bytes(raw)
            packer=Packer(root/'v1',chunk_bytes=128)
            for profile in ['fast','detail']:packer.add(source,f'profiles/{profile}/moge3-dense.onnx')
            packer.finish();old,_=adaptive(root/'v1',root/'v2',bundle_bytes=512)
            compact=split24(raw)[2]
            plan={'policy':'encoder-matmul-fp24-half-up-v1','files':{name:{'bytes':len(raw),'sourceSha256':hashlib.sha256(raw).hexdigest(),'compactSha256':hashlib.sha256(compact).hexdigest(),'spans':[{'offset':0,'bytes':len(raw)}]} for name in old['files']}}
            policy=root/'plan.json';policy.write_text(json.dumps(plan))
            manifest,report=progressive(root/'v2',policy,root/'v3',workers=2)
            self.assertTrue(report['allCorrectionsByteExact'])
            def chunk(key):
                c=manifest['chunks'][key];blob=(root/f"v3/bundles/{c['bundle']}.pack").read_bytes();packed=blob[c['offset']:c['offset']+c['bytes']]
                self.assertEqual(hashlib.sha256(packed).hexdigest(),key)
                raw=decompress(packed,c['codec'],c.get('encodedBytes',c['rawBytes']))
                return expand24(raw) if c['encoding']=='fp24' else inverse(raw,c['encoding'])
            file=manifest['files']['profiles/fast/moge3-dense.onnx'];base=bytearray();exact=bytearray()
            for p in file['pieces']:
                c=manifest['chunks'][p['chunk']];rounded=chunk(p['chunk']);base.extend(rounded)
                exact.extend(restore32(rounded,chunk(c['correction'])))
                self.assertEqual(manifest['bundles'][manifest['chunks'][c['correction']]['bundle']]['usage'],'correction')
            self.assertEqual(bytes(base),compact);self.assertEqual(bytes(exact),raw)
            with self.assertRaises(FileExistsError):progressive(root/'v2',policy,root/'v3')


if __name__=='__main__':unittest.main()
