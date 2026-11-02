# Real-photo CUDA / WebGPU benchmark

Six real camera photographs complement the original three synthetic test scenes:
architecture and vegetation, a furnished interior, a tabletop object, a portrait,
an animal with fine fur/whiskers, and a distant outdoor launch scene. Sources and
licenses are recorded in [the fixture manifest](fixtures/real/manifest.json).

Open [the visual report](benchmarks/real-visual/index.html) and select one, two or
three refinement passes. Each row shows the exact cropped input, original CUDA
depth, WebGPU depth and an error heatmap with a shared range. The report explicitly
marks failed comparisons. These photos have no measured scene-depth ground truth.

## Results on 2026-11-02

- Original MoGe-3, pinned source/checkpoint, RTX 4090: 36 strict-FP32 cases and
  36 original-default cases. All 12 input hashes match the browser inputs and the
  checkpoint hash matches the prior baseline. Setup and warmup are excluded.
- Chrome on M4 Pro: **34/36 new cases pass** the unchanged strict CUDA thresholds
  in both baseline and autotuned modes. All 36 paired error-metric records are
  identical, with zero candidate parity failures. No pooled GPU buffers remain
  after unload.
- The two failures are `real-wood-deck`, Detail, two and three refinement passes.
  Point NRMSE is 0.0014773 and 0.0015900 respectively; the gate remains `<0.001`.
  The two-pass maximum absolute point difference is about 1.417, while the 99th
  percentile is about 0.00382. Intrinsics and masks pass. This is a baseline port
  discrepancy, not introduced by autotuning. It persists with both CPU/GPU
  indexing and CPU/GPU postprocessing. Its exact cause remains unresolved.
- The existing synthetic suite still has its recorded 54/54 tuned CUDA checks;
  Firefox 158 passed all 18 existing compatibility cases. The new real-photo
  matrix above was measured in Chrome, not Firefox.
- The RTX 4090 rental was destroyed and absence verified after the references
  were downloaded and checked. Estimated compute rental: $0.13, excluding
  transfer charges; this is not an invoice.

Warm inference medians, 18 samples per mode/configuration (six photos × three
repeats), alternating baseline and tuned runs on AC power:

| Profile | Passes | WebGPU baseline | Autotuned | Reduction | Native RTX 4090 strict FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 1 | 790 ms | 744 ms | 5.9% | 192 ms |
| Fast | 2 | 1010 ms | 914 ms | 9.5% | 323 ms |
| Fast | 3 | 1253 ms | 1105 ms | 11.8% | 453 ms |
| Detail | 1 | 1345 ms | 1271 ms | 5.5% | 264 ms |
| Detail | 2 | 1652 ms | 1467 ms | 11.2% | 440 ms |
| Detail | 3 | 2054 ms | 1778 ms | 13.5% | 612 ms |

The NVIDIA column is the median of six per-photo medians on different hardware;
it is contextual, not an isolated backend comparison. The WebGPU timing report
contains one missing-favicon console error after all 216 measurements completed.
The runner now serves that request locally; the subsequent accuracy run has no
browser errors. Model cold download and tuning are excluded from these warm times.

## Reproduce

1. Run `python prepare_real_photos.py` in the existing export environment. It
   verifies source fixture hashes and creates deterministic profile inputs.
2. On a disposable NVIDIA host, unpack `results/real-photo-inputs.tar.gz` and run
   the pinned `setup_original_cuda.sh`. Retain both benchmark JSON files,
   environment metadata, and strict references. Validate the input hashes before
   copying strict case directories into `public/original-cuda/{profile}/`.
3. Build `VITE_AUTOTUNE_BENCHMARK=1 npm run build`, then `npm run preview`.
4. With Playwright available, run `REAL=1 REPEATS=1 node benchmark-browser.cjs`
   for 36 unique cases per mode; add `TIMING=1 REPEATS=3` for timing. Set
   `REPORT_FILE` to separate output paths and `PLAYWRIGHT_MODULE` if necessary.
   Accuracy failures are collected without changing thresholds and cause a
   nonzero exit status after the full report is saved.
5. Run the dev server on port 5180, then `node benchmark-visual.cjs` to regenerate
   the visual report for all 18 Detail scene/pass combinations.

`original-only` validation explicitly checks original CUDA points, intrinsics and
masks; it does not claim the optional MLX/dense diagnostic fixtures exist for the
new photographs. The original fixture suite retains its full diagnostic gate.
Reports, power samples, source fingerprints and reference hashes are stored under
`benchmarks/real-photo-*`. Raw model reference arrays stay outside Git.
