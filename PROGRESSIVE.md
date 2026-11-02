# Selective 24-bit downloads and exact FP32 upgrades

The v3 package changes storage for the encoder's large MatMul initializers.
The compact representation is approximate; all learned computation remains FP32.
Normalization parameters, biases, output heads, geometry, and learned refinement
weights retain their original FP32 values. There is no retraining, architecture
change, FP16 execution, or dependency on `shader-f16`.

The UI offers **Smaller download** and **Original weights · exact upgrade**.
The latter restores the original model bytes using separately downloaded
correction data. Both choices retain saved model chunks across visits. Switching
releases the existing sessions before another inference can begin.

## Measured downloads

Production build, Apple M4 Pro / Chrome 156. MB and GB are decimal. Totals cover
the manifest and model payload; they exclude app JS/WASM, photos, numerical
references, HTTP headers and additional HTTP compression.

| Scenario | Lossless v2 | Progressive v3 |
| --- | ---: | ---: |
| Cold Fast | 1,235,672,117 bytes / 89 requests | **934,147,998 bytes / 65 requests** |
| Independent cold Detail | 1,237,593,397 bytes / 89 requests | **936,069,278 bytes / 65 requests** |
| Both resolutions | 1,241,675,064 bytes | **940,150,945 bytes** |
| Detail after Fast | 6,002,947 bytes | **6,002,947 bytes** |
| Exact upgrade after compact | Not applicable | **302,000,544 additional bytes / 20 requests** |
| Reload compact or exact | 0 model bytes | **0 model bytes** |

Cold Fast is **24.40% smaller than v2**. Downloading compact and all corrections
totals 1,242,151,489 bytes, about 0.04% more than the old combined lossless package.
The upgrade prioritizes a smaller initial download while retaining the original
weights; it does not reduce the information required for exact reconstruction.

An existing v2 cache serves both modes with **zero new weight bytes**: only the
new 1,170,852-byte manifest is downloaded. Deleting one saved correction recovers
with one **1,048,613-byte HTTP range request**, rather than its entire bundle.
All 33 original files are restored byte-exactly. There are **54 passing compact
CUDA comparisons**, **54 passing exact comparisons** and **18 passing compact
Firefox 158 comparisons**. Exact mode's original-CUDA and six dense-output
reports are identical to the previous release. All masks match in both modes.
Worst observed point-map NRMSE is 0.0807% for compact and 0.0943% for exact;
compact's lower error on this small set does not establish greater accuracy.
The 0.1% gate remains unchanged. All 54 JavaScript tests and ten Python tests pass.
The mobile selector, disabled selector during inference and uploaded-photo flow
also pass. Uploaded photos remain explicitly unvalidated.

Evidence: [summary](benchmarks/progressive-summary.json),
[compact matrix](benchmarks/progressive-compact-browser.json),
[exact matrix](benchmarks/progressive-exact-browser.json),
[Firefox matrix](benchmarks/progressive-firefox-browser.json),
[file verification](benchmarks/progressive-file-verification.json),
[cache reload](benchmarks/progressive-cache-reload.json),
[cache migration](benchmarks/progressive-cache-migration.json),
[cold Detail](benchmarks/progressive-detail-cold.json),
[correction recovery](benchmarks/progressive-correction-recovery.json).

## Encoding

The format stores one sign bit, eight exponent bits, and fifteen fraction bits
for selected weights. This is a custom transport format, not a native WebGPU
data type. The browser expands it to ordinary FP32 before creating ONNX sessions.

For each finite FP32 bit pattern `w`, the compact bits are
`(w + 128) & 0xffffff00`, rounding the discarded low byte half-up. Nonfinite
weights and rounding overflow are rejected. Exponent, middle mantissa byte and
sign/upper mantissa planes are Zstd-compressed. The original low byte `r` is
stored in a separate correction bundle. The inverse is exactly
`(compact - ((r >> 7) << 8)) | r`: the high bit of the correction byte also
recovers the rounding carry. No separate carry bitmap is required.

The exporter identifies initializers by actual graph consumers. Only sufficiently
large weights used exclusively by encoder-block MatMul nodes qualify. Shared
weights with a nonencoder consumer remain untouched. Tensor-relative chunking
preserves sharing between both resolutions. The current checkpoint has 288 unique
selected 4 MiB chunks, totaling 1,207,959,552 original bytes.

The package manifest records the compact file hash, original file hash, original
cache representations, compact-chunk hash and correction-chunk hash. Correction
bundles are separate from compact bundles and are never prefetched by compact
loads. Exact upgrades validate each correction and the final original file hash.
An existing v1/v2 full-precision cache can supply both modes without replacement
weight downloads. Cache corruption triggers verified recovery instead of silent
use. Unavailable browser storage falls back to verified network loading.

## Validation and limitations

The accuracy gate remains the original NVIDIA strict-FP32 reference comparison:
points, camera intrinsics, all six dense outputs and the existing mask-boundary
rule. The compact and exact modes are tested separately. Exact reconstruction
also undergoes independent streaming verification against all 33 original files.
Compact weights do not produce identical outputs; exact weights must reproduce
the original release's numerical reports. No tolerance or goldens are changed.

These tests cover three synthetic yard photos, two resolutions and 1/2/3 learned
refinement passes. They are a reconstruction regression set, not a general
real-world depth benchmark or evidence of physical measurement accuracy. Use
Original weights when comparing the original model or investigating another
scene. Timings in accuracy reports are diagnostic; these runs are not an AC-power
controlled performance comparison.

## Reproduce

Preserve the original ONNX files and v2 package under `public/models/`.
All exporters refuse to overwrite an existing output. Keep prior reports when
repeating an experiment.

```sh
uv venv --python 3.12 .venv-progressive
uv pip install --python .venv-progressive/bin/python -r requirements-progressive.txt
.venv-progressive/bin/python screen_fp24.py
.venv-progressive/bin/python pack_progressive.py
.venv-progressive/bin/python verify_packed_models.py \
  --package public/models/packed-v3 \
  --report benchmarks/progressive-file-verification.json
.venv-progressive/bin/python -m unittest test_fp24 test_adaptive_pack test_pack_models -q
npm test
VITE_MODEL_PACKAGE_FORMAT=3 npm run build
npm run preview
```

`benchmark-progressive.browser.js` runs the full 54-comparison matrix for one
mode. Run compact with a cold dedicated model cache, then exact with the compact
cache retained to measure only the upgrade. Reload with model URLs blocked to
verify persistence. Hosted builds additionally set version-pinned
`VITE_MODEL_BASE_URL` and `VITE_MODEL_PACKAGE_BASE_URL` URLs, with the latter
ending in `packed-v3`. Format 3 is opt-in; ordinary builds retain v2 selection.

The source checkpoint, v1/v2 packages and original raw exports remain unchanged.
