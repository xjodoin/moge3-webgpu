"""Prepare ONNX weights for a model repository, separate from the static Space."""
from pathlib import Path
import json,os,shutil
ROOT=Path(__file__).resolve().parent
out=ROOT/'release/model'
out.mkdir(parents=True,exist_ok=False)
for part in ['profiles/fast','profiles/detail','refiner','refiner-wgsl']:
 source=ROOT/'public/models'/part;target=out/part;target.mkdir(parents=True)
 for f in source.iterdir():
  if f.suffix in ('.onnx','.bin') or f.name=='manifest.json':os.link(f,target/f.name)
for folder in ['licenses','benchmarks']:shutil.copytree(ROOT/folder,out/folder)
shutil.copy(ROOT/'LICENSE',out/'LICENSE')
(out/'README.md').write_text('''---
license: mit
library_name: onnxruntime
pipeline_tag: depth-estimation
base_model: Ruicheng/moge-3-vitl
tags:
  - onnx
  - webgpu
  - 3d-reconstruction
---

# MoGe-3 ViT-L WebGPU

Unofficial FP32 ONNX and packed WGSL weight exports of Microsoft's MoGe-3 ViT-L, including the learned sparse 3D refiner. No retraining or quantization. These files require the browser-side sparse orchestration in the source project; they are not a single end-to-end ONNX graph.

- [Source, setup and complete validation](https://github.com/xjodoin/moge3-webgpu)
- [Browser demo](https://huggingface.co/spaces/xjodoin/moge3-webgpu)
- [Original model](https://huggingface.co/Ruicheng/moge-3-vitl)

Profiles: `profiles/fast` (512 x 344, 1200 tokens) and `profiles/detail` (768 x 512, 1733 tokens), both supporting one, two or three refinement passes. ONNX comparison blocks are in `refiner/`; dedicated kernel parameters and checksum manifests are in `refiner-wgsl/`. Each profile manifest records dimensions, upstream revisions, file size and SHA-256.

All 54 browser-versus-original CUDA FP32 comparisons passed on an Apple M4 Pro / Chrome 156, covering three images, two resolutions, one to three passes and three repeats. Maximum point-map NRMSE was 0.0943%, with matching masks. Pointwise outliers are larger. Earlier MLX comparisons remain diagnostic; original CUDA is the native reference. These synthetic-photo parity checks are not physical measurement validation. See the included benchmarks and the source README for exact gates, timing, memory accounting and limitations.

Requires approximately 1.5 GB of weights per selected profile and several GB of available memory. The browser uses WebGPU for learned operations and voxel indexing, with JavaScript orchestration and camera geometry. No photo upload or server inference is necessary.

Pinned original checkpoint: `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be`; MoGe source: `74fbce054ebed49800de42d0ad0e83495065719a`. Upstream code, weight and dependency licenses are included in `licenses/`. This is not an official Microsoft release.
''')
print(out)
