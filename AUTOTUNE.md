# Device-specific FP32 sparse autotuning

The sparse refiner now selects a kernel separately for each layer, profile and
power-of-two voxel-count bucket. Model weights and FP32 accumulation are unchanged.
The new subgroup kernels support 32- and 64-channel layers as well as 128-channel
tiles; alternate pipelines share the existing GPU weight buffer.

The search is bounded to four local, allowlisted candidates. It compares each
candidate against the baseline on the actual input, using bit-exact GPU comparison
and a four-byte mismatch readback. Nonfinite results fail this check. Five rotated
GPU timestamp samples measure the normalization and both convolutions. A candidate
must improve median latency by at least 8%, 0.025 ms, and twice the combined median
absolute deviations. Otherwise the existing kernel remains selected.

Decisions are stored in localStorage (at most 128 entries, 30-day expiry). Keys
include layer, shape bucket, weight checksum, generated shader source, browser,
adapter information, device features and limits. No photos or feature tensors are
stored. Nonbaseline decisions loaded after a page refresh are checked again against
the baseline before use. Storage failure falls back to an in-memory cache.
Devices without timestamp queries, FP16 experiments and disabled autotuning use
the existing implementation. Missing 32-lane subgroup support limits the candidate
set to shared-memory kernels. Dispatch sizes are checked before compilation.

First use incurs compilation, exactness checks and tuning time. Warm performance
must be reported separately; this is not a cold-start latency optimization.

## Reproduction

Build with `VITE_AUTOTUNE_BENCHMARK=1 npm run build`, then `npm run preview`.
`runAutotuneBenchmark(false)` compares baseline and tuned inference against the
original CUDA references: 3 photos × 2 profiles × 3 refinement counts × 3 repeats
= 54 checks per mode. `runAutotuneBenchmark(true)` measures the same matrix using
preloaded, checksummed fixture inputs without allocating reference arrays between
timed calls. Calibration occurs before each measured group. Inspect
`autotuneReport` and `autotuneProgress` in the console. Ordinary production builds
omit this benchmark entry point.

Developer controls: `setSparseAutotune(false)`, `setSparseAutotune(true)`,
`sparseAutotuneStats()` and `clearSparseAutotuneCache()`. Toggling does not unload
model sessions. Clearing affects tuning metadata only, not downloaded weights.

## Research adaptation

[GPU Goldmines](https://arxiv.org/abs/2610.17321) motivates tuning GPU kernels by
shape and device rather than assuming one universal configuration.
[TorchSparse++](https://arxiv.org/abs/2311.12862) motivates layer-specific sparse
execution choices. This implementation adapts those ideas to a small WebGPU search;
it does not port their entire systems or assume their published speedups apply.

## Local validation (2026-11-02)

Apple M4 Pro, Chrome 156, AC power throughout the accuracy/timing run.
All 54 tuned and 54 baseline CUDA checks passed. Each paired case had identical
CUDA and dense-output error metrics. All 30 tuned layer/shape keys had zero
candidate parity failures. The full cold calibration across both profiles and
three photos accumulated 27.0 seconds; this is excluded from warm timings.

Median inference time, nine samples per cell, alternating baseline and tuned runs:

| Profile | Passes | Baseline | Tuned | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1 | 1139 ms | 1049 ms | 7.9% |
| Fast | 2 | 1321 ms | 1160 ms | 12.2% |
| Fast | 3 | 1578 ms | 1309 ms | 17.1% |
| Detail | 1 | 1731 ms | 1567 ms | 9.5% |
| Detail | 2 | 2083 ms | 1804 ms | 13.4% |
| Detail | 3 | 2405 ms | 2167 ms | 9.9% |

These are local warm measurements, not a cross-device guarantee or a fresh MLX
comparison. Reports: `benchmarks/autotune-cuda.json`, `autotune-timing.json`,
`autotune-power.json`, and `autotune-source.json`. GPU pools were empty after unload.


Firefox 158 also passed the 18-case existing matrix using the non-subgroup path.
Reload persistence checks passed four three-pass runs across both profiles: eight
nonbaseline selections were revalidated, no searches repeated, and no pooled
buffers remained after unload. See `autotune-firefox.json` and
`autotune-persistence.json` under `benchmarks/`.

Production format-3 compact weights were additionally tested on the current build:
all 54 synthetic Chrome checks passed (`autotune-compact-release.json`). The
exact-weight real-photo failures remain documented separately; compact validation
does not claim that every unseen photo matches CUDA.
