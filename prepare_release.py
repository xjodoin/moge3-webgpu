"""Prepare a reviewable static Hugging Face Space; does not publish anything."""
from pathlib import Path
import json,os,shutil
ROOT=Path(__file__).resolve().parent
out=ROOT/'release/space'
if out.exists():
    raise SystemExit('release/space already exists; move it aside before preparing a new snapshot')
report=json.loads((ROOT/'benchmarks/release-validation.json').read_text())
assert len(report['results'])==6 and all(r['passed'] for r in report['results']), 'All six release comparisons must pass'
source=json.loads((ROOT/'dist/model-source.json').read_text())
for name in ['fast','detail']:
    assert (ROOT/f'dist/models/profiles/{name}/manifest.json').is_file(), 'Build exported model manifests first'
    if source['modelBaseUrl']=='/models':
        assert (ROOT/f'dist/models/profiles/{name}/moge3-dense.onnx').is_file(), 'Build exported models first'
shutil.copytree(ROOT/'dist',out,copy_function=os.link)
for folder in ['licenses','benchmarks']:
    shutil.copytree(ROOT/folder,out/folder)
shutil.copy(ROOT/'LICENSE',out/'LICENSE')
readme='''---
title: MoGe-3 WebGPU
emoji: 🏡
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: mit
models:
  - Ruicheng/moge-3-vitl
  - xjodoin/moge-3-vitl-webgpu
tags:
  - webgpu
  - onnx
  - depth-estimation
  - 3d-reconstruction
short_description: MoGe-3 with sparse refinement on WebGPU
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

'''+(ROOT/'README.md').read_text()
(out/'README.md').write_text(readme)
size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
if size>=1_000_000_000:raise SystemExit('Static Space exceeds 1 GB. Build with VITE_MODEL_BASE_URL pointing to a model repository.')
print(out, size, 'bytes')
