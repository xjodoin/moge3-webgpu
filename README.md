# MoGe-3 WebGPU

An **unofficial, MIT-licensed browser port of MoGe-3 ViT-L**, including the original learned sparse 3D refinement stage. Upload a photo, run inference on your GPU and explore the reconstructed point cloud. No inference server or photo upload is involved.

- Source: [xjodoin/moge3-webgpu](https://github.com/xjodoin/moge3-webgpu)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/xjodoin/moge3-webgpu)
- Exported weights: [Hugging Face model repository](https://huggingface.co/xjodoin/moge-3-vitl-webgpu)
- Original model: [Microsoft MoGe](https://github.com/microsoft/MoGe) / [Ruicheng/moge-3-vitl](https://huggingface.co/Ruicheng/moge-3-vitl)

This is an experimental port, not an official Microsoft release. The default release preserves the original FP32 learned weights without quantization or retraining. Optional [local FP16 experiments](FP16.md) halve the inference asset size but fail the existing accuracy gate, so FP32 remains the default.

## Supported release profiles

| Profile | Input | Tokens | Refinement |
| --- | --- | --- | --- |
| Fast | 512 × 344 | 1200 | 1–3 passes |
| Detail | 768 × 512 | 1733 | 1–3 passes |

Images are center-cropped to the selected landscape aspect ratio. Raw models contain about **1.5 GB per profile** and need several GB of free memory. The [adaptive lossless package](ADAPTIVE.md) reduces a cold profile to **about 1.24 GB in 89 model requests**, reuses saved chunks across visits, and downloads only **6.00 MB** when switching from cached Fast to Detail. Existing v1 caches upgrade without redownloading weights. All 54 CUDA comparisons still pass with unchanged error metrics. A compatible WebGPU browser/GPU is required. Dedicated sparse WGSL kernels avoid the large 27-neighbor gather buffers. The optional ONNX comparison backend still requires individual gather buffers of approximately **1.04 GB (Fast)** or **1.53 GB (Detail)** and checks that adapter limit.

Dense inference and linear layers run on WebGPU through ONNX Runtime. The default sparse blocks use dedicated FP32 WGSL kernels for layer normalization, SiLU, sparse convolution and residual addition. CPU execution-provider fallback is disabled. WGSL also keeps voxel indexing, mean pooling, encoder-feature gathering and skip additions on the GPU. JavaScript orchestrates scalar output-size readbacks and performs final depth updates, interpolation, camera recovery and projection. This is a hybrid implementation, not a claim that every operation runs on the GPU.

## Subgroup FP32 sparse kernels

Subgroup kernels are **enabled by default on supported devices**, with the previous GPU kernels retained as the capability fallback. All **54 subgroup original-CUDA accuracy comparisons pass**, plus 54 paired previous-kernel comparisons. Error metrics are identical across modes, all masks match, and worst point-map NRMSE remains **0.0943%**. No weights, precision or accuracy tolerance changed.

Warm medians on **Apple M4 Pro / Chrome 156**, with AC power verified throughout the separate native and browser runs (9 and 71 samples respectively):

| Profile | Passes | Previous WebGPU | Subgroup WebGPU | Fresh MLX | WebGPU reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 1 | 0.918 s | 0.826 s | 0.640 s | 10.1% |
| Fast | 2 | 1.221 s | 1.055 s | 0.989 s | 13.6% |
| Fast | 3 | 1.531 s | 1.272 s | 1.529 s | 16.9% |
| Detail | 1 | 1.390 s | 1.253 s | 0.980 s | 9.8% |
| Detail | 2 | 1.832 s | 1.628 s | 1.444 s | 11.1% |
| Detail | 3 | 2.263 s | 1.947 s | 1.925 s | 14.0% |

Refinement wall time falls **24–28%**. Fast three-pass inference is about 17% faster than MLX in this run; Detail three-pass is about 1.2% slower. One-pass WebGPU remains about 28–29% slower than MLX. These are grouped, counterbalanced warm measurements on one Mac, not a device-independent performance guarantee. Both implementations use the same exact FP32 photo inputs, token counts and pass counts. MLX inputs are prepared on GPU before timing; browser input upload is included. Loading, validation and rendering are excluded, and profiling is separate. Browser explicit GPU-buffer counters and MLX allocator peaks are not comparable process-memory measurements. Graph capture stays disabled by default.

The subgroup path uses an **8-row × 128-output-channel** convolution tile. Each of 32 lanes owns four output channels. Input features are broadcast from registers while each lane loads its own weight columns. This avoids shared matrix tiles and weight shuffles. A single workgroup-uniform occupancy flag skips empty neighborhoods outside the channel loop. Neighbor-major, input-channel-major FP32 FMA order is preserved; weights, SiLU, residual addition and the existing normalization remain unchanged. Narrow 32/64-channel layers retain their existing kernels.

The capability guard checks enabled `subgroups`, adapter minimum/maximum subgroup size of 32, and an actual 32-invocation workgroup with all 32 lanes active. It does not assume a relationship between subgroup lane IDs and local invocation indices. Other devices use the existing GPU kernels. See the [WGSL subgroup specification](https://www.w3.org/TR/WGSL/#subgroup-builtin-functions).

Screening covers four saved original-CUDA photo/pass maps at three channel widths: all **72 layout/map parity checks have zero FP32 numerical error**. The 8-row broadcast layout is the fastest tested candidate for every map, reducing isolated block GPU time by about 34–45%. Twenty additional ONNX comparisons pass on partial row tiles and empty neighborhoods. An earlier register-shuffle layout was slower and one large layout lost the GPU device; its incomplete report is retained for diagnosis and that implementation was removed from the runtime.

The fresh MLX baseline runs the pinned FP32 checkpoint without hooks, diagnostic dumps or per-step outputs, with explicit GPU evaluation. All **54 native runs exactly reproduce the saved MLX points, intrinsics and masks**. These are native-reference drift checks; the browser's independent CUDA references remain its accuracy gate. Input hashes, token counts, shapes, runtime versions and a model-source hash are recorded. Native work finishes before browser GPU benchmarking begins.

Use `await window.setSparseVariant('subgroup')` for subgroup kernels and `await window.setSparseVariant('optimized')` for the previous GPU kernels. Reproduce isolated screening with `tuneSubgroup()` from `/tune-subgroup.browser.js`. Run `.venv-native/bin/python benchmark_mlx_matrix.py` for a fresh native matrix; it refuses to overwrite existing evidence. Run `benchmarkSubgroup()` from `/benchmark-subgroup.browser.js` on a production build for the browser A/B matrix, save it to `benchmarks/subgroup-browser.json`, and run `node summarize-subgroup.js` with a matching `benchmark_power.py` power log. Profiling is separate from latency runs.

Evidence: [summary](benchmarks/subgroup-summary.json), [108 production runs and separate GPU traces](benchmarks/subgroup-browser.json), [MLX baseline](benchmarks/subgroup-mlx.json), [broadcast screening](benchmarks/subgroup-broadcast-screen.json), [ONNX block comparisons](benchmarks/subgroup-blocks.json), [AC/thermal log](benchmarks/subgroup-power.jsonl), [power configuration](benchmarks/subgroup-power-config.txt), [discarded register-layout experiment](benchmarks/subgroup-register-screen.json). Earlier buffer, wide-kernel, indexing and graph-capture helpers explicitly select the previous kernels for their historical comparisons and restore the current default afterwards.

## Dense graph capture

The static dense model supports native ONNX Runtime WebGPU graph capture. An object-form provider configuration forwards `enableGraphCapture` to the native EP; the string shorthand does not forward that option in the pinned 1.24.3 runtime. The GPU input buffer and ORT-owned output bindings stay fixed across photos. Each run updates the input buffer and returns separate output views, so CPU validation downloads cannot invalidate the captured tensors. Profile/mode changes, failures and Unload release the session and bindings. Sparse ONNX sessions keep capture disabled because voxel counts change between passes.

Independent instrumentation passes **16 dense checks** across both resolutions and alternating photos: warm uncaptured runs create **854 bind groups**, while captured runs create **zero**. This verifies native replay rather than just setting a configuration flag. Evidence: [capture checks](benchmarks/graph-capture-checks.json).

Capture remains **opt-in**. All **54 capture-enabled original CUDA comparisons pass**, plus 54 paired uncaptured comparisons. Error metrics are identical across modes, all masks match, and the unchanged worst point-map NRMSE is **0.0943%**. Warm medians on M4 Pro / Chrome 156, with AC power verified in 58 samples covering the full run:

| Profile | Passes | Dense uncaptured → captured | Full compute uncaptured → captured |
| --- | ---: | ---: | ---: |
| Fast | 1 | 567.0 → 565.9 ms | 0.907 → 0.904 s |
| Fast | 2 | 566.3 → 566.0 ms | 1.217 → 1.217 s |
| Fast | 3 | 568.9 → 567.0 ms | 1.526 → 1.523 s |
| Detail | 1 | 912.5 → 899.6 ms | 1.393 → 1.366 s |
| Detail | 2 | 909.8 → 899.0 ms | 1.808 → 1.800 s |
| Detail | 3 | 914.3 → 901.1 ms | 2.237 → 2.225 s |

These grouped, counterbalanced measurements show only a modest warm gain: dense latency changes by 0.06–0.33% for Fast and 1.19–1.44% for Detail. Fast two-pass full compute is 0.06% slower. This does not establish a general speedup on other hardware. Capture adds about **0.32 GB (Fast) / 0.49 GB (Detail)** to peak explicitly tracked GPU buffers; this is not process RSS or driver memory. The first dense invocation also does extra capture work: single observations were 0.80 → 1.15 s (Fast) and 1.04 → 1.84 s (Detail), excluding loading. Those are single first invocations, not repeated cold-start averages. Given the small warm gain and added startup/memory cost, the existing uncaptured path remains the default. Buffer reuse, optimized wide kernels and GPU voxel indexing remain enabled in both modes.

Evidence: [summary](benchmarks/graph-summary.json), [108 production runs](benchmarks/graph-browser.json), [AC/thermal log](benchmarks/graph-power.jsonl), [power configuration](benchmarks/graph-power-config.txt). Model loading, validation and rendering are excluded from compute. Dense timing includes upload and required CPU readbacks. Profiling/instrumentation runs are separate. FP32 weights and the independent original CUDA goldens are unchanged.

The raw coordinates, confidence mask and metric scale still return to the CPU, with concurrent readbacks included in dense timing. Dense encoder features stay on the GPU for refinement. Capture changes scheduling and buffer lifetime, without changing FP32 weights, kernels, photo preprocessing, pass counts or accuracy thresholds. See the [ONNX Runtime capture and IO-binding documentation](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).

Use `await window.setDenseGraphCapture(false)` for the uncaptured comparison and `await window.setDenseGraphCapture(true)` to select capture. Both unload existing model resources. `testGraphCapture()` from `/graph-capture.browser.js` instruments bind-group creation separately from latency runs; warmed captured dense runs must create zero bind groups while changing photos. `benchmarkGraph()` from `/benchmark-graph.browser.js` runs both 54-case matrices. Log AC power with `python3 -u benchmark_power.py results/graph-power.stop > benchmarks/graph-power.jsonl`, save browser results to `benchmarks/graph-browser.json` and capture instrumentation to `benchmarks/graph-capture-checks.json`, then run `node summarize-graph.js`. Create the stop file when finished. Earlier buffer, wide-kernel and indexing helpers explicitly disable capture to preserve their comparison scope.

## GPU voxel indexing

Voxel quantization, neighbor lookup, parent assignment, ordered pooling children and encoder sampling indexes now run on the GPU by default. Column CSR lists and four-way merges preserve lexicographic voxel order without a 64-bit packed-key sort or a dense 3D allocation. Hierarchical prefix scans allocate compact output rows. Pooling retains the original ascending child order and FP32 summation order.

Exact checks pass on **23 inputs / 115 levels**, including negative rounding ties, odd dimensions, all saved photo/pass inputs, every neighbor and parent index, encoder indexes, and actual pooled FP32 features. All **54 GPU-indexing original CUDA comparisons pass**, plus 54 paired CPU-baseline comparisons. Per-case error metrics are unchanged and masks match. The unchanged worst point-map NRMSE is 0.0943%.

Warm compute on M4 Pro / Chrome 156, AC power verified in 67 samples:

| Profile | Passes | CPU indexing | GPU indexing | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1 | 0.963 s | 0.937 s | 2.7% |
| Fast | 2 | 1.311 s | 1.252 s | 4.5% |
| Fast | 3 | 1.670 s | 1.588 s | 4.9% |
| Detail | 1 | 1.443 s | 1.409 s | 2.3% |
| Detail | 2 | 1.919 s | 1.831 s | 4.6% |
| Detail | 3 | 2.405 s | 2.264 s | 5.9% |

High-resolution three-pass index preparation changes from 336 ms of CPU coordinate work to 22 ms GPU-preparation wall time. Index-array uploads fall from 242 MB to zero. Four scalar output-size readbacks per pass remain for ONNX dynamic tensor sizing: **16 logical bytes per pass / 48 bytes across three passes**, plus 964 bytes of small indexing parameters uploaded per pass. These figures exclude driver transfer granularity. Final learned depth-correction readbacks, CPU depth updates, interpolation and camera recovery remain; this does not claim that all inference/postprocessing is GPU-only. Some previous CPU indexing overlapped GPU work, so the end-to-end gain is smaller than the indexing-stage reduction.

Both modes use the same optimized sparse kernels, FP32 weights and enabled buffer pool. Mode order is counterbalanced across resolutions; each configuration has a warmup and three repeats per photo. Loading, reference validation and rendering are excluded. Timestamp profiling runs are separate from timing runs. Performance on other devices is not established. The pooled-buffer downloader also now bounds its result to the logical tensor size instead of returning padding; the 1x1 parity case covers that regression.

Use `await window.setIndexingBackend('cpu')` for the comparison path and `await window.setIndexingBackend('gpu')` to restore the default. Both unload model resources on a mode change. Reproduce exact-map checks with `testGpuIndexing()` from `/gpu-indexing.browser.js`. Reproduce the production comparison with `benchmarkIndexing()` from `/benchmark-indexing.browser.js`, save to `benchmarks/index-browser.json`, then run `node summarize-indexing.js` with a matching AC power log. Earlier buffer/wide-kernel benchmark helpers explicitly select CPU indexing to preserve their comparison scope.

Evidence: [summary](benchmarks/index-summary.json), [108 production runs and separate GPU traces](benchmarks/index-browser.json), [115 exact map-level checks](benchmarks/gpu-indexing-parity.json), [AC power log](benchmarks/index-power.jsonl).

## Wide sparse-kernel optimization

Wide layers now skip neighborhoods that are empty for an entire workgroup. The 128-channel layers use a 32-row × 64-output-channel tile with a shared occupancy mask; 256/512-channel layers use 16 × 32 tiles with a uniform emptiness check per neighborhood. Narrow layers retain the original kernel. FP32 weights, normalization and the order of nonzero accumulation are preserved. Uniform decisions use [WGSL workgroupUniformLoad](https://www.w3.org/TR/WGSL/#workgroupUniformLoad-builtin), keeping barrier control flow uniform.

The selected kernels are **enabled by default**. Buffer reuse remains enabled in both comparison modes. All **54 optimized CUDA comparisons pass**, along with 54 same-build baseline comparisons. Worst point-map NRMSE remains **0.0943%**, masks match, and per-case error metrics are unchanged. Twenty isolated checks against ONNX also pass, including partial tiles and completely empty neighborhoods.

Warm compute on M4 Pro / Chrome 156, with AC power verified in 77 samples covering the complete run:

| Profile | Passes | Original kernel | Selected kernels | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1 | 1.000 s | 0.962 s | 3.8% |
| Fast | 2 | 1.498 s | 1.327 s | 11.4% |
| Fast | 3 | 1.809 s | 1.677 s | 7.3% |
| Detail | 1 | 1.545 s | 1.475 s | 4.6% |
| Detail | 2 | 2.103 s | 2.069 s | 1.6% |
| Detail | 3 | 2.782 s | 2.491 s | 10.5% |

Refinement improves 9.5–12.5%. These are same-build grouped A/B trials with mode order counterbalanced by resolution, warmup and three repeats per photo/configuration. Loading, reference validation and rendering are excluded. Profiling is measured separately. Performance on other GPUs is not established. Baseline comparisons remain available via `await window.setSparseVariant('baseline')`; restore the default with `await window.setSparseVariant('optimized')`. Changing variants unloads existing model resources.

Evidence: [full summary and per-layer GPU comparison](benchmarks/wide-summary.json), [108 production runs](benchmarks/wide-browser.json), [isolated ONNX checks](benchmarks/wide-blocks.json), [first screening](benchmarks/wide-screen.json), [mask/layout screening](benchmarks/wide-mask-screen.json), [AC power log](benchmarks/wide-power.jsonl). `tuneWide()` from `/tune-wide.browser.js` reproduces mask/layout screening using saved CUDA diagnostic tensors. `testWideBlocks()` from `/wide-blocks.browser.js` runs boundary/empty-neighborhood checks. `benchmarkWide()` from `/benchmark-wide.browser.js` runs the complete A/B comparison; save its result to `benchmarks/wide-browser.json`, then run `node summarize-wide.js` with a matching AC power log. The original kernel is preserved unchanged for comparisons.

## Buffer reuse and per-layer GPU profiling

Reusable custom GPU buffers are **enabled by default**, retaining the allocation-churn benefit by user preference. The pool retains at most 256 MiB of free buffers, uses queue ordering to reuse buffers after submission, and clears on unload, failure or profile changes. GPU devices use weak keys so old devices are not retained by the pool registry.

The definitive same-build comparison was rerun on **verified AC power**, with automatic power mode enabled and power/thermal status sampled every 10 seconds. All 75 samples covering the run show AC power. macOS reported no recorded thermal/performance warnings. The earlier unverified-power run is preserved separately.

The AC run includes **54 measurements with reuse disabled and 54 enabled, all passing original CUDA accuracy checks**. Point-error metrics are identical between modes and all masks match. Custom buffer allocations fell 64–75%, but this did **not** establish a consistent latency improvement. Warm medians:

| Profile | Passes | Reuse disabled | Reuse enabled |
| --- | ---: | ---: | ---: |
| Fast | 1 | 0.991 s | 1.000 s |
| Fast | 2 | 1.497 s | 1.384 s |
| Fast | 3 | 1.775 s | 1.783 s |
| Detail | 1 | 1.482 s | 1.501 s |
| Detail | 2 | 2.030 s | 2.045 s |
| Detail | 3 | 2.681 s | 2.687 s |

Five configurations changed by about 0.2–1.3% toward slower latency. Fast two-pass measured 7.6% faster, with part of the difference occurring in dense inference rather than refinement. This single grouped benchmark does not establish a general pooling speedup. Reuse added about 0.23–0.28 GB to peak explicitly tracked GPU buffers and is enabled by user preference for reduced allocation churn, rather than a demonstrated average-latency gain. These are grouped, counterbalanced measurements on M4 Pro / Chrome 156, not device-independent performance guarantees. Profiling runs are separate from timing runs.

Per-layer profiling uses WebGPU timestamp queries for custom kernels, including normalization, both sparse convolutions, pooling, gathering and upsampling. ONNX linear layers report separately labeled synchronized wall times, which include dispatch overhead. Profiling requires `timestamp-query` support on the ORT GPU device. It fails explicitly when unavailable; wall time is never labeled GPU time. The standard run has profiling disabled.

The high-resolution three-pass profile identifies the deepest encoder block (~181 ms), bottleneck (~181 ms), and adjacent wide blocks (~164–166 ms each) as the largest individual custom GPU stages. Sparse-kernel tuning is the next meaningful target.

```js
await window.setBufferReuse(true); // unloads existing sessions before changing mode
const result = await window.runTest(true, 'sample-backyard', null, 'detail', 'wgsl', 3, true, {profileGpu: true});
console.table(result.gpuProfile.gpu);
console.table(result.gpuProfile.linear); // synchronized wall time, not shader time
await window.releaseModels();
console.log(window.bufferPoolStats()); // empty after unload
await window.setBufferReuse(true); // restore the default
```

Reproduce the full same-build comparison from the development console using `benchmarkBuffers()` from `/benchmark-buffers.browser.js`; save it to `benchmarks/buffer-browser.json`, then run `node summarize-buffer-speed.js`. The AC rerun also records start/end timestamps and a `pmset` power log; `node summarize-buffer-speed.js -ac` checks AC coverage and summarizes that run. Evidence: [AC summary and layer ranking](benchmarks/buffer-summary-ac.json), [AC runs and GPU timestamps](benchmarks/buffer-browser-ac.json), [power samples](benchmarks/buffer-ac-power.jsonl), [power configuration](benchmarks/buffer-ac-power-config.txt). The earlier [unverified-power comparison](benchmarks/buffer-summary.json) remains historical evidence.

## Latest browser speed improvement

Compact column indexes replace per-neighbor 3D hash lookups, and typed-array sorting reduces pooling allocations. Voxel quantization, sorted parent coordinates and the original FP32 child summation order are preserved. All 90 tested map levels match the previous implementation exactly. All **54 original-CUDA comparisons pass**, with identical point-error metrics and matching masks.

Warm compute on Apple M4 Pro / Chrome 156:

| Profile | Passes | Before | After | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1 | 1.288 s | 1.013 s | 21.3% |
| Fast | 2 | 2.011 s | 1.417 s | 29.5% |
| Fast | 3 | 2.733 s | 1.816 s | 33.6% |
| Detail | 1 | 2.152 s | 1.544 s | 28.3% |
| Detail | 2 | 2.980 s | 2.049 s | 31.2% |
| Detail | 3 | 3.959 s | 2.752 s | 30.5% |

These compare the preceding benchmark and the optimized production build on the same hardware, using unchanged models, warmups and three repeats per photo. They are sequential benchmark batches, not interleaved A/B trials. Loading, reference comparisons and rendering are excluded from compute. No weights, pass counts or accuracy tolerances changed. Cold startup still downloads/prepares the same models.

Evidence: [summary](benchmarks/coordinate-summary.json), [54 production runs](benchmarks/coordinate-browser.json), [exact map comparisons and isolated timings](benchmarks/coordinate-performance.json). Run `node benchmark_coordinates.js` with saved original-CUDA diagnostic tensors to reproduce the CPU comparison. Run the full `benchmarkOriginal()` browser matrix and save it as `benchmarks/coordinate-browser.json`, then run `node summarize-coordinate-speed.js` for the full comparison.

The earlier measurements below describe the preceding implementations.

## Earlier sparse-kernel validation

Tested on **Apple M4 Pro, 24 GB, Chrome 156 / Metal-3**, October 30, 2026. All **36 production-browser runs** passed: 18 using the ONNX gather backend and 18 using the dedicated WGSL kernels (three photos × two resolutions × three repeats per backend). All ten isolated sparse blocks also passed direct comparisons against ONNX, including missing neighbors and partial tiles. Fresh native MLX runs reproduced the saved native points, masks and camera references exactly on all 18 runs.

| Profile | ONNX gather compute | Dedicated WGSL compute | Native MLX | Peak tracked buffers, ONNX → WGSL |
| --- | ---: | ---: | ---: | ---: |
| Fast · 512 × 344 | 1.37 s | **1.26 s** | 0.67 s | **4.05 → 2.76 GB** |
| Detail · 768 × 512 | 2.03 s | **2.04 s** | 0.94 s | **4.69 → 3.26 GB** |

These are medians after one warmup per profile, using the same production build with explicitly selected backends and local model files. Browser compute includes dense inference, refinement and camera recovery; it excludes session preparation, fixture/reference loading, numerical comparisons and rendering. First network download and cold startup are excluded. MLX ran separately to avoid GPU contention, using identical FP32 input, token count, one refinement pass and explicit GPU synchronization.

The main improvement is **30–32% lower peak tracked GPU buffers**. The largest individual allocation, including warmup, fell **1.04 GB → 134 MB** (Fast) and **1.53 GB → 193 MB** (Detail). Median refinement improved **0.788 → 0.681 s** (13.6%) and **1.116 → 1.054 s** (5.6%). Fast end-to-end compute improved about 8%; Detail's aggregate median was essentially unchanged. These results do not establish a broad latency improvement across devices. Browser compute remains approximately 1.9–2.2× slower than warm native MLX.

The numerical gates are unchanged. Worst final point-map NRMSE versus native MLX is **0.02224%** (Fast) and **0.09747%** (Detail). This establishes tolerance parity on these fixtures, not bitwise equality or real-world measurement accuracy. Full evidence: [`sparse-summary.json`](benchmarks/sparse-summary.json), [`sparse-validation.json`](benchmarks/sparse-validation.json), [`sparse-onnx-baseline.json`](benchmarks/sparse-onnx-baseline.json), [`sparse-native.json`](benchmarks/sparse-native.json), [`sparse-blocks.json`](benchmarks/sparse-blocks.json) and [`sparse-tiles.json`](benchmarks/sparse-tiles.json).

Buffer counters include ORT pools and exclude driver/shader/browser overhead; they are not process RSS or directly comparable with MLX memory counters. Only one profile's model resources are retained. The ONNX comparison path caches 22 sessions; the dedicated kernel path caches 12 ONNX sessions and 10 GPU weight bundles. Starting a different profile or changing backends evicts the previous resources. **Unload model** releases them explicitly.

The earlier session-reuse optimization and original release results remain in [`performance-summary.json`](benchmarks/performance-summary.json) and [`release-validation.json`](benchmarks/release-validation.json); they are historical measurements of earlier implementations.

The gate requires NRMSE < 0.001 (0.1%). In the historical MLX comparison, one mask pixel in the Fast brick-ranch case differs at a predicted confidence of 0.5000023; the current original-CUDA comparisons have matching masks. Only mask disagreements within 1e-5 of the 0.5 threshold, affecting at most 0.001% of scalars, are accepted. NaNs, unexplained mask disagreements and larger numerical deviations fail. See [`comparison.js`](comparison.js).

Maximum point-coordinate differences can be much larger than averages (up to about 0.96 model-estimated metres in the historical MLX release tests), although the worst 99th-percentile absolute difference is about 0.011 m. Discrete voxel rounding amplifies small floating-point differences at some pixels. These tests establish implementation parity, **not physical measurement accuracy**. Use real measurements for scale calibration.

The earlier multi-pass MLX comparisons exceeded the numerical gate on some cases. The independent original PyTorch/CUDA test below now validates all three pass counts. MLX comparisons remain in the diagnostic report; their thresholds and saved outputs were not changed. The browser defaults to one pass for latency; select three to match the original model's default refinement count.

### Original NVIDIA CUDA comparison

Tested the unmodified pinned MoGe-3 source and checkpoint on a rented RTX 4090, with PyTorch 2.11.0+cu128 and Triton 3.6.0. **All 54 browser comparisons pass** against original strict FP32 inference: three photos × two resolutions × three pass counts × three repetitions. Worst point-map NRMSE is **0.0943%**, below the unchanged 0.1% gate; all masks match. This is numerical agreement on six synthetic fixtures, not a general accuracy or physical measurement guarantee. The GPU rental was destroyed after results were downloaded.

| Profile | Passes | Original CUDA FP32 | Original CUDA sparse TF32 | Browser M4 Pro |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1 | 0.195 s | 0.076 s | 1.288 s |
| Fast | 2 | 0.338 s | 0.100 s | 2.011 s |
| Fast | 3 | 0.482 s | 0.126 s | 2.733 s |
| Detail | 1 | 0.277 s | 0.110 s | 2.152 s |
| Detail | 2 | 0.466 s | 0.146 s | 2.980 s |
| Detail | 3 | 0.654 s | 0.177 s | 3.959 s |

Warm medians use identical FP32 inputs, tokens and weights, excluding loading, validation and rendering. CUDA measurements synchronize the GPU and include camera recovery. **The RTX 4090 and Apple M4 Pro are different hardware**: these ratios do not isolate implementation overhead. Browser inference is still 5.7–7.8× slower than strict FP32 CUDA on this hardware. The original sparse kernels allow TF32 by default; a second, separately recorded run disables it for the FP32 accuracy reference. PyTorch TF32 is disabled in both; FlexGEMM autotuning is disabled as in the official demo to prevent tuning during timed repeats.

The UI uses original CUDA final points, intrinsics and masks as its authoritative native reference, together with the dense export checks. Existing hybrid PyTorch/MLX stage checks and end-to-end MLX checks remain visible as `outputs` / `diagnosticPassed`; a diagnostic failure does not override passing original CUDA checks. No geometry threshold was relaxed and no golden output was generated from the browser.

Evidence: [summary](benchmarks/original-summary.json), [54 browser runs](benchmarks/original-browser.json), [strict CUDA](benchmarks/original-cuda.json), [default sparse TF32 CUDA](benchmarks/original-cuda-default.json), [environment](benchmarks/original-environment.txt), [rental cleanup](benchmarks/original-rental.json).

To reproduce the original benchmark, run `python prepare_original_inputs.py`, copy the resulting archive to `/workspace` on an NVIDIA CUDA development host, extract it there, then run `bash setup_original_cuda.sh`. The script pins original source/checkpoint and the tested PyTorch/Triton stack. It writes independent `original-cuda` and `original-cuda-default` outputs. Copy those folders into `public/` locally and their `benchmark.json` files into `benchmarks/original-cuda.json` and `benchmarks/original-cuda-default.json`. Original per-pass raw tensors are saved separately for diagnosis. Run the browser matrix with `benchmarkOriginal()` from `/benchmark-original.browser.js` on the dev server, save its result to `benchmarks/original-browser.json`, then run `node summarize-original.js`. All input SHA-256 hashes are checked during summarization. Diagnostic tensors are excluded from release bundles.

## Run locally

After exporting/downloading models into `public/models` and independent CUDA sample references into `public/original-cuda`:

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5180/. Samples compare browser output against independent original CUDA references; PyTorch/MLX stage diagnostics remain available. Uploaded photos run inference without claiming reference validation.

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
.venv/bin/python export_sparse_weights.py
uv venv .venv-native --python 3.12
uv pip sync --python .venv-native/bin/python requirements-native.txt
.venv-native/bin/python reference_refiner.py --profile fast
.venv-native/bin/python reference_refiner.py --profile detail
.venv-native/bin/python reference_multipass.py
# Generate or download original CUDA goldens as described above before sample validation.
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

`npm test` validates camera recovery, comparison gates, session lifecycle and sparse coordinate maps. For repeated browser timing in the development console:

```js
const {benchmark} = await import('/benchmark-browser.js');
const results = await benchmark(3);
const {testGpuOps} = await import('/gpu-ops.browser.js');
await testGpuOps(); // Real WGSL pooling, gathering and chained skip-add checks.
const {testSparseBlocks} = await import('/sparse-block.browser.js');
await testSparseBlocks(); // All 10 blocks, actual weights vs ONNX, missing neighbors and partial tiles.
await window.releaseModels();
```

Use `benchmark(3, 'onnx')` for the previous gather backend and `benchmark(3, 'wgsl')` for the dedicated kernels. `window.runTest(true, photo, null, profile, 'onnx')` also explicitly selects the comparison path. There is no silent fallback between backends.

For production timing, build and serve `dist`, then use the same warmup/repeat loop with `window.runTest`; the source-only benchmark helpers are not bundled. Run native measurements separately after browser inference has stopped:

```sh
.venv-native/bin/python benchmark_native.py --repeats 3
```

This checks fresh native outputs against existing references without overwriting them. Browser fixture runs download diagnostic tensors **after** refinement for accuracy checks; uploaded photos skip those diagnostic downloads.

 CI runs source tests/build only; it does not claim GPU validation. `npm run build` bundles the runtime, profile metadata and shared compressed chunks when enabled; legacy builds retain the raw-model layout. No checkpoint, Python environment or private photo is included. See [compressed model setup and deployment](DOWNLOADS.md).

### Prepare Hugging Face

```sh
VITE_MODEL_BASE_URL=https://huggingface.co/xjodoin/moge-3-vitl-webgpu/resolve/64cf5edcfac57d146b57425f4fee541bf332a451 npm run build
.venv/bin/python prepare_release.py
python3 serve.py --directory release/space
```

The generated `release/space` contains the static app, references, licenses, benchmarks and the 151 MB of dedicated sparse weights. By default these new weights are bundled with the app; the existing pinned model repository continues to supply dense and ONNX linear weights. Set `VITE_SPARSE_MODEL_BASE_URL` only to a model revision containing `refiner-wgsl/` to host the new binaries externally. The two URLs are recorded separately in `model-source.json`. Large ONNX files are fetched from the pinned model repository because the static Space has a 1 GB limit. `model-source.json` records the chosen model URL. To publish your own weights, run `.venv/bin/python prepare_model.py`, upload `release/model` to a model repository, then build with its revision-pinned URL in `VITE_MODEL_BASE_URL`. Static hosting needs no server GPU. Upload it with the Hugging Face CLI/SDK to a Space you own. Large model weights belong in Hugging Face storage; they are ignored by this source repository. The Space uses the documented [static SDK](https://huggingface.co/docs/hub/spaces-sdks-static) and [cross-origin isolation headers](https://huggingface.co/docs/hub/spaces-config-reference).

## Implementation notes and remaining work

The dense encoder feature and all intermediate refinement features stay on the GPU. Refinement has one final depth-correction readback instead of 21 block readbacks; raw coordinates, mask and scale still cross to the CPU for depth updates and camera recovery. Voxel indexing runs on the GPU with four scalar sizing readbacks per pass. GPU pooling preserves the original FP32 child summation order. Sessions are cached by profile/block, failed loads can retry, and temporary tensors are disposed after use.

The dedicated convolutions compute 16 × 32 output tiles with 32-channel reduction tiles and 64 threads per workgroup. Neighbor lookup happens as features enter shared memory, so no `[voxels, 27, channels]` tensor is materialized. Each workgroup uses 6 KiB of shared storage. Partial tiles and missing-neighbor sentinels are masked; all threads reach the synchronization barriers. Bias/SiLU and bias/residual are fused into the respective convolution outputs. Layer normalization uses the original epsilon of 1e-6. The original 27-neighbor order and FP32 parameters are preserved.

`export_sparse_weights.py` packs unchanged checkpoint parameters in `[neighbor, input channel, output channel]` order. Ten binary files total 150,879,232 bytes. Their manifest records field offsets, lengths and SHA-256 checksums; the loader validates the layout, byte length and checksum before allocating GPU weights. `benchmark-sparse.browser.js` compares 32- and 64-element reduction tiles with synchronized, alternating runs. The 32-element tile was faster on the tested Mac; the 64-element candidate remains available to that research harness.

The original padded gather + matrix multiplication ONNX implementation remains available as the explicit `onnx` comparison backend. Fixed-shape exports precompute native positional interpolation and antialiased resize filters; upstream ONNX compatibility mode otherwise changes these semantics. `native_export.py` preserves them using supported ONNX operations.

Current limitations: only the listed Mac/Chrome configuration is validated; no mobile/Safari compatibility claim, dynamic aspect ratios, Worker isolation, download cancellation or automatic device-specific kernel tuning. Persistent model downloads are subject to browser quotas and eviction. CPU orchestration can occupy the main thread. Numerical validation covers the listed fixtures and one to three passes on this device; it does not establish general reconstruction or measurement accuracy.

## Attribution and licenses

This project's code is MIT; third-party notices are in `licenses/`.

- Microsoft MoGe source: `74fbce054ebed49800de42d0ad0e83495065719a`.
- Original checkpoint `Ruicheng/moge-3-vitl`: `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be` (MIT model card).
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MIT): sparse layout and camera recovery adaptation/reference.
- MLX checkpoint `mlx-community/moge-3-vitl-mlx-fp32`: `f5ceca7512506731fb1c50307c313da76aac500f`.
- ONNX Runtime Web 1.24.3, Three.js 0.186.1, Vite 7.3.7.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and upstream contribution guidance.
