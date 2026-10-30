# MoGe-3 WebGPU

An **unofficial, MIT-licensed browser port of MoGe-3 ViT-L**, including the original learned sparse 3D refinement stage. Upload a photo, run inference on your GPU and explore the reconstructed point cloud. No inference server or photo upload is involved.

- Source: [xjodoin/moge3-webgpu](https://github.com/xjodoin/moge3-webgpu)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/xavierjodoin/moge3-webgpu)
- Original model: [Microsoft MoGe](https://github.com/microsoft/MoGe) / [Ruicheng/moge-3-vitl](https://huggingface.co/Ruicheng/moge-3-vitl)

This is an experimental port, not an official Microsoft release. The original FP32 learned weights are preserved; they are not quantized or retrained.

## Supported release profiles

| Profile | Input | Tokens | Refinement |
| --- | --- | --- | --- |
| Fast | 512 × 344 | 1200 | 1 pass |
| Detail | 768 × 512 | 1733 | 1 pass |

Images are center-cropped to the selected landscape aspect ratio. The model downloads about **1.5 GB per profile** and needs several GB of free memory. A compatible WebGPU browser/GPU is required. The sparse gather implementation needs individual GPU buffers of approximately **1.04 GB (Fast)** or **1.53 GB (Detail)**; the UI checks this adapter limit before loading.

The learned dense and sparse tensor operations run on WebGPU through ONNX Runtime. CPU execution-provider fallback is disabled. JavaScript performs voxel maps, mean pooling, skip additions, interpolation, camera recovery and projection. This is a hybrid implementation, not a claim that every operation runs on the GPU.

## Measured validation

Tested on **Apple M4 Pro, 24 GB, Chrome 156 / Metal-3**, October 30, 2026. All six release cases passed native comparison. Full evidence, including individual outputs and maximum errors, is in [`benchmarks/release-validation.json`](benchmarks/release-validation.json). These synthetic fixtures have no ground-truth measurements.

| Profile / photo | Browser inference + refinement | Native MLX inference | Final point-map NRMSE |
| --- | ---: | ---: | ---: |
| Fast / backyard | 5.06 s | 0.63 s | 0.00237% |
| Fast / brick ranch | 5.84 s | 0.65 s | 0.01067% |
| Fast / raised deck | 3.29 s | 0.71 s | 0.02224% |
| Detail / backyard | 4.32 s | 1.36 s | 0.02997% |
| Detail / brick ranch | 4.01 s | 2.02 s | 0.00997% |
| Detail / raised deck | 3.89 s | 1.45 s | 0.09747% |

Browser timings exclude dense session loading, network downloads and reference-file comparison, but include sparse block setup. They are single local runs with warm runtime caches, not a statistical speed benchmark. Native timings use the identical input and token/refinement settings; the browser remains slower than native MLX.

Peak explicitly tracked GPU buffers were approximately **2.4–2.65 GB**. This excludes browser/driver/shader overhead and includes retained ORT pools. MLX reported approximately 3.4–3.6 GB peak; these different memory counters are not directly comparable.

The gate requires NRMSE < 0.001 (0.1%). One mask pixel in the Fast brick-ranch case differs at a predicted confidence of 0.5000023. Only mask disagreements within 1e-5 of the 0.5 threshold, affecting at most 0.001% of scalars, are accepted. NaNs, unexplained mask disagreements and larger numerical deviations fail. See [`comparison.js`](comparison.js).

Maximum point-coordinate differences can be much larger than averages (up to about 0.96 model-estimated metres in the release tests), although the worst 99th-percentile absolute difference is about 0.011 m. Discrete voxel rounding amplifies small floating-point differences at some pixels. These tests establish implementation parity, **not physical measurement accuracy**. Use real measurements for scale calibration.

Two-pass refinement was also tested at 768 × 512. One case exceeded the numerical gate (0.1059% NRMSE), so it is **not included in the published release profiles**. Raw results remain in `benchmarks/initial-high-resolution.json`; they are not presented as a passing release.

## Run locally

After exporting/downloading models into `public/models`:

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5180/. Samples compare browser output against saved PyTorch/MLX references. Uploaded photos run inference without claiming reference validation.

### Reproduce the exports and references

Python 3.12, uv and Git are required for export; native comparison additionally requires Apple Silicon. No Python is needed by people using the hosted browser demo.

```sh
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python requirements-export.txt
.venv/bin/python setup_export.py --native
.venv/bin/python export_dense.py --profile fast --width 512 --height 344 --tokens 1200 --steps 1
.venv/bin/python normalize_onnx.py public/models/profiles/fast/moge3-dense.onnx
.venv/bin/python export_dense.py --profile detail --width 768 --height 512 --tokens 1733 --steps 1
.venv/bin/python normalize_onnx.py public/models/profiles/detail/moge3-dense.onnx
.venv/bin/python export_refiner.py
uv venv .venv-native --python 3.12
uv pip sync --python .venv-native/bin/python requirements-native.txt
.venv-native/bin/python reference_refiner.py --profile fast
.venv-native/bin/python reference_refiner.py --profile detail
npm ci
npm test
npm run dev
```

The committed fixtures are AI-generated test scenes, not customer properties; generation provenance is in `fixtures/*.json`. Package locks pin the tested environments. Model downloads are pinned to revisions below. `--references-only` regenerates dense reference fixtures without exporting the graph again.

Run the complete comparison from the browser console:

```js
const results = [];
for (const profile of ['fast', 'detail']) {
  for (const photo of ['sample-backyard', 'brick-ranch', 'raised-deck']) {
    results.push(await window.runTest(true, photo, null, profile));
  }
}
console.table(results.map(r => ({profile: r.profile, photo: r.caseName, passed: r.passed})));
```

`npm test` validates camera recovery and comparison gates. CI runs source tests/build only; it does not claim GPU validation. `npm run build` bundles the runtime and copies only the two release profiles plus shared refiner weights. No checkpoint, Python environment or private photo is included.

### Prepare Hugging Face

```sh
npm run build
.venv/bin/python prepare_release.py
python3 serve.py --directory release/space
```

The generated `release/space` contains a static Space, the exported models, references, licenses and benchmarks. Static hosting needs no server GPU. Upload it with the Hugging Face CLI/SDK to a Space you own. Large model weights belong in Hugging Face storage; they are ignored by this source repository. The Space uses the documented [static SDK](https://huggingface.co/docs/hub/spaces-sdks-static) and [cross-origin isolation headers](https://huggingface.co/docs/hub/spaces-config-reference).

## Implementation notes and remaining work

The original CUDA/FlexGEMM sparse convolutions are represented as padded gather + matrix multiplication, preserving all 27 neighbors and kernel order. Fixed-shape exports precompute native positional interpolation and antialiased resize filters; upstream ONNX compatibility mode otherwise changes these semantics. `native_export.py` preserves them using supported ONNX operations.

Current limitations: only the listed Mac/Chrome configuration is validated; no mobile/Safari compatibility claim, dynamic aspect ratios, Worker isolation, persistent cache management, download cancellation or custom memory-efficient WGSL sparse kernels. CPU orchestration can occupy the main thread. Do not present the two-pass profile as validated.

## Attribution and licenses

This project's code is MIT; third-party notices are in `licenses/`.

- Microsoft MoGe source: `74fbce054ebed49800de42d0ad0e83495065719a`.
- Original checkpoint `Ruicheng/moge-3-vitl`: `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be` (MIT model card).
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MIT): sparse layout and camera recovery adaptation/reference.
- MLX checkpoint `mlx-community/moge-3-vitl-mlx-fp32`: `f5ceca7512506731fb1c50307c313da76aac500f`.
- ONNX Runtime Web 1.24.3, Three.js 0.186.1, Vite 7.3.7.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and upstream contribution guidance.
