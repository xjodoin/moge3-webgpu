# Reconstruction output performance

Profiling identified output unpacking as the larger postprocessing cost. Isolated GPU camera passes took approximately 0.3–1.6 ms, while Detail output unpacking took about 11.6 ms. The camera solver, precision, refinement kernels and accuracy gate are unchanged.

The optimized projection writes contiguous XYZ values followed by mask values. A single staging buffer receives the four camera scalars and projection output, then maps once. The returned XYZ array is a view into a fresh CPU-owned frame buffer; the readback buffer can safely return to the pool. Depth and byte masks are materialized without creating temporary array views per pixel. Transfer bytes remain unchanged; mapping count decreases from two to one. The XYZ view retains the whole frame buffer, including the float mask region, for its lifetime.

The previous GPU layout is retained as `{geometryVariant:'baseline'}` for paired checks. The default is `optimized`. CPU postprocessing remains available through `{postprocess:'cpu'}`.

## Measured results

On this Mac, isolated median output handling fell from approximately **7.5 to 1.7 ms (Fast)** and **14.1 to 2.6 ms (Detail)**. Detail unpacking alone fell from approximately **11.6 to 0.5 ms**. These measurements do not represent the full model speedup. GPU timestamp traces were collected separately from latency runs.

A production-build timing matrix reused SHA-256-verified fixture inputs, disabled viewer rendering, and avoided reference/diagnostic downloads during timing. Each mode received two warmups; three images and three repetitions gave nine samples per configuration. CPU, previous GPU and optimized GPU order rotated across repetitions. AC power was logged throughout the run. Median total compute times:

| Profile / passes | CPU | Previous GPU | Optimized GPU | Reduction vs previous GPU |
| --- | ---: | ---: | ---: | ---: |
| Fast / 1 | 833 ms | 829 ms | 826 ms | 0.4% |
| Fast / 2 | 1,130 ms | 1,138 ms | 1,123 ms | 1.3% |
| Fast / 3 | 1,389 ms | 1,387 ms | 1,367 ms | 1.5% |
| Detail / 1 | 1,321 ms | 1,318 ms | 1,313 ms | 0.3% |
| Detail / 2 | 1,780 ms | 1,778 ms | 1,744 ms | 1.9% |
| Detail / 3 | 2,319 ms | 2,206 ms | 2,198 ms | 0.4% |

Optimized GPU medians were lower than CPU in all six configurations, with the largest difference 5.2% for Detail / 3. Small differences, especially below 1%, need broader repeated measurements before treating them as reliable device-independent gains. This is not a new MLX comparison.

The earlier accuracy matrix also recorded wall times; those results were mixed. It loads large validation arrays between runs, which can affect later allocation/GC behavior. Its raw data is preserved, not replaced. Use the separate validation-free timing matrix for the figures above.

## Accuracy and lifecycle evidence

- All 54 optimized-GPU original-CUDA comparisons passed, as did 54 previous-GPU and 54 CPU cases. Every paired GPU original-CUDA metric is identical. Thresholds and reference files are unchanged.
- All six isolated image/resolution output comparisons were bit-identical for points, depth, mask and intrinsics. Returned arrays remained unchanged after subsequent pooled readbacks.
- Known-camera, empty/single-sample mask, depth-add and odd-grid interpolation checks passed.
- An actual synthetic JPEG uploaded through the UI produced both intermediate previews for three passes, used one output mapping, retained `validated:false` / `passed:null`, and left an empty buffer pool after unload.
- Firefox 158 passed all 18 production-build reconstruction cases with an empty buffer pool after unloading (`benchmarks/geometry-tuning-firefox.json`).
- All 57 unit tests and the production build passed.

Evidence: `benchmarks/geometry-profile-before.json`, `geometry-profile-after.json`, `geometry-tuning-browser.json`, `geometry-tuning-timing.json`, `geometry-tuning-summary.json`, `geometry-tuning-upload.json` and the paired power logs. All images are existing synthetic fixtures; these tests establish regression parity, not general physical measurement accuracy.

## Reproduce

Build the explicit local benchmark entry:

```sh
VITE_GEOMETRY_BENCHMARK=1 npm run build
npm run preview
```

On a fresh preview page, run `await runGeometryTuning()` for the accuracy/parity matrix. Reload, then run `await runGeometryTiming()` for timing without validation downloads. Do not run other GPU benchmarks concurrently. Capture AC/thermal state with `benchmark_power.py`. Export `window.geometryTuning` and `window.geometryTiming` after completion.

For isolated development-server profiling, import `profileGeometry` from `/profile-geometry.browser.js`, then call `profileGeometry('baseline')` or `profileGeometry('optimized')`. Ordinary builds omit the production benchmark entry points.
