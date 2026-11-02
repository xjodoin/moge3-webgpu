# GPU reconstruction postprocessing

The default inference path now keeps dense raw coordinates, masks and metric scale on WebGPU. Depth-correction addition, align-corners-false bilinear interpolation, exponential remapping, camera recovery and final projection run in WGSL. Each refinement pass consumes the previous GPU coordinate tensor. Encoder features remain shared across passes.

Camera recovery uses the existing 64 × 64 sampling and damped Gauss–Newton algorithm, with 256 GPU invocations, compensated local sums, synchronized tree reductions, and the original iteration/convergence limits. The CPU uses JavaScript double precision; the GPU uses FP32. Outputs are therefore compared with the existing original-CUDA tolerances, not claimed bit-identical.

The CPU still decodes input photos, schedules commands, handles four voxel-size scalar readbacks per refinement pass, and formats final output for the existing Three.js/WebGL viewer. Final projection and four camera scalars are downloaded for display. Intermediate progressive previews request their own final-output downloads. Fixture validation separately downloads diagnostics after compute timing; uploaded inputs do not download reference data or claim validation. This is not a zero-readback renderer.

All GPU coordinate buffers are released on success or failure. Preview temporaries are released after rendering, and Unload clears the buffer pool. Shader compilation is checked before entering the GPU path. The CPU implementation remains available for comparisons:

```js
await runTest(true, 'sample-backyard', null, 'fast', 'wgsl', 3, true,
  {postprocess: 'cpu'});
```

GPU is the default; use `{postprocess:'gpu', progressive:true}` to enable previews in scripted runs. Stage timings on this asynchronous path measure enqueue time; `cameraMs` includes pending GPU work and final readback. Compare the summed `inferenceMs + refinementMs + cameraMs`, not individual stages, when assessing performance. Preview overhead is reported separately.

## Validation

Run in the local development browser:

```js
const {benchmarkGeometry}=await import('/benchmark-geometry.browser.js');
const report=await benchmarkGeometry();
```

The harness first tests known camera geometry, empty/single-sample masks, exact FP32 depth addition, and interpolation over odd-sized grids. It then alternates CPU/GPU postprocessing over the existing three synthetic images, two resolutions, three pass counts, and three repetitions: 54 checks per path. Reference files and tolerances remain unchanged. It rejects failures and checks released resources.

Saved evidence is in `benchmarks/gpu-geometry-browser.json` and `benchmarks/gpu-geometry-summary.json`. All 54 GPU and 54 CPU cases passed, with matching original-CUDA masks. Worst GPU point-map NRMSE is 0.0943%, below 0.1%. Every GPU case has zero refinement feature readbacks. The uploaded-input path produced two intermediate previews for three passes, remained `validated:false` / `passed:null`, and released the buffer pool completely. All 57 unit tests and the production build pass.

Development-build timing results are diagnostic and mixed: median Fast three-pass time improved about 5% and Detail three-pass about 6%, while some one/two-pass configurations regressed. These measurements do not establish a general speedup. AC power was verified before the run and in a log starting partway through it; this was not a fresh MLX benchmark. The real-world geometry accuracy limitations of the existing synthetic regression set still apply.

Firefox 158 passed all 18 production-build photo/resolution/pass combinations (`benchmarks/gpu-geometry-firefox.json`). The exact shipped 24-bit weight rounding also passed all 18 combinations through the development screening hook (`benchmarks/gpu-geometry-compact.json`); this tests weight accuracy, not a new package download. Both checks ended with an empty buffer pool.

## Output handling optimization

The current default uses contiguous GPU XYZ output and one combined camera/projection readback mapping. See [output performance](GEOMETRY-TUNING.md) for the newer production benchmark, paired accuracy checks, and limits. Earlier timing results above are retained as historical evidence.
