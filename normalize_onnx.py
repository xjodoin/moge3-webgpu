"""Normalize negative Transpose axes emitted by the upstream PyTorch exporter."""
from pathlib import Path
import onnx,sys,json,hashlib
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'public/models/profiles/fast/moge3-dense.onnx'
m=onnx.load(p);count=0
for node in m.graph.node:
 if node.op_type=='Transpose':
  for a in node.attribute:
   if a.name=='perm' and any(v<0 for v in a.ints):
    rank=len(a.ints);a.ints[:]=[v%rank for v in a.ints];count+=1
onnx.checker.check_model(m)
onnx.save(m,p)
print('Normalized transpose nodes:',count)

manifest=p.parent/"manifest.json"
if manifest.exists():
 meta=json.loads(manifest.read_text());meta.update(modelBytes=p.stat().st_size,sha256=hashlib.file_digest(p.open("rb"),"sha256").hexdigest());manifest.write_text(json.dumps(meta,indent=2))
