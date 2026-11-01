# Adaptive lossless model downloads

The new default package combines per-chunk codec selection, indexed bundles,
cache-compatible codec upgrades and partial-download recovery. All original
FP32 model files are reconstructed byte for byte before inference.

## Measured results

Production build, Apple M4 Pro 24 GB / Chrome 156, AC power throughout the
54-comparison matrix. MB and GB below are decimal.

| Scenario | Previous compressed package | Adaptive package |
| --- | ---: | ---: |
| Cold Fast model payload | 1,262,717,135 bytes / 797 requests | **1,235,672,117 bytes / 89 requests** |
| Cold Detail, independent empty cache | 1,264,692,561 bytes / 797 requests | **1,237,593,397 bytes / 89 requests** |
| Both resolutions combined | 1,268,897,894 bytes / 802 requests | **1,241,675,064 bytes / 90 requests** |
| Detail after cached Fast | 6,180,759 additional bytes / 5 requests | **6,002,947 additional bytes / 1 request** |
| Fresh reload of either profile | 0 model bytes | **0 model bytes / 0 model requests** |

The cold profile payload is **2.14% smaller than the previous compressed
package**, or **16.66–16.74% smaller than the original raw FP32 assets**.
Model request count falls **88.83%**. These are measured byte/request reductions;
they do not establish an Internet startup-time improvement.

Totals cover the package manifest and active model payloads. They exclude app
JS/WASM, photos, reference-validation files and HTTP headers. Browser Resource
Timing reports 1,235,231,462 encoded body bytes for cold Fast, 1,237,153,082 for
cold Detail and 1,241,234,602 for both. Small differences reflect the local
server's additional HTTP compression. The lazily loaded Zstd decoder adds a
73.3 KB JS/WASM-containing asset before HTTP compression.

## Two additional behaviors verified in Chrome

**An existing v1 cache upgrades without redownloading weights.** A fresh page
loaded 775 older compressed representations and reused the unchanged entries.
Only the new **842,438-byte manifest** was fetched; all 22 model files verified
and accurate inference completed. Older cache entries are decoded as-is rather
than re-encoded into the new codec.

**A missing chunk recovers with one range request.** After deliberately removing
one cached chunk, the loader fetched exactly **3,475,923 bytes** using HTTP 206
from a 13,719,511-byte bundle. It restored accurate inference without fetching
the rest of that bundle. Servers ignoring Range are supported through one
verified full-response fallback; that path is covered by unit tests.

## Codec study and implementation

An initial screen compares eight distinct 4 MiB dense-weight chunks: 32 MiB
total. Candidates include native ZipNN Huffman, C Bitshuffle + Zstd, and portable
transforms with Deflate/Zstd. Every successful screened candidate round-trips
byte-exactly. Native codec timings are CPU measurements, not browser decoding
times. Inputs passed to native codecs are isolated copies so native buffer
mutation cannot affect subsequent candidates or the source evidence.

The full pass then evaluates all **894 unique chunks** from the v1 package and
chooses the smallest tested browser-portable representation for each chunk:

- The existing Deflate representation, retained when smaller.
- Zstd level 19 with identity byte ordering, four byte planes, or separate
  exponent and sign/mantissa fields.
- Zstd level 12 with 32 bit planes when the float count supports eight-value
  grouping. The bit-plane representation wins mainly on small structured data;
  fewer than 1 MB of raw data selects it in this package.

The transforms rearrange bytes/bits without rounding numeric values. The
browser's Zstd decoding uses the pinned `zstddec@0.3.4` WebAssembly module;
transform reversal is implemented locally. Its MIT/BSD notices are included in
`licenses/Zstddec-LICENSE.txt`. Native ZipNN and C Bitshuffle are screening
references; they are not required in the browser.

The exporter writes **97 bundles including optional comparison-backend data**,
targeting 16 MiB each. Bundles are grouped by actual profile usage: shared,
Fast-only, Detail-only or optional comparison data. This preserves reuse between
resolutions. Cold loads prefetch complete bundles into independently verified
chunk cache entries. Partial caches request missing contiguous ranges, merging
gaps no larger than 64 KiB. Four workers bound concurrent network/decode work.

Integrity checks cover the build-pinned manifest, full bundle responses,
compressed chunks, decoded chunks and complete original model files. HTTP
partial responses are checked against expected byte counts, available range
headers and chunk hashes. Corrupt data cannot become an inference input.

Browser storage remains subject to quota and eviction. Verified inference still
works when storage is unavailable, though network reuse may be reduced. Clear
downloaded model removes only this application's model cache. Photos and output
geometry are never stored in it.

Inspiration: [ZipNN](https://github.com/zipnn/zipnn),
[Bitshuffle](https://github.com/kiyo-masui/bitshuffle),
[Zstd decoder](https://github.com/donmccurdy/zstddec-wasm).

## Accuracy and tests

All **54 original NVIDIA CUDA strict-FP32 comparisons pass**, covering three
photos, two resolutions, 1/2/3 refinement passes and three repetitions. All mask
pixels match. Every dense-output and original-CUDA error report is identical to
the previous transport; worst point-map NRMSE stays **0.0943%**. The original
graphs, parameters, FP32 execution and accuracy gates remain unchanged.

All **33 packaged original files** also pass an independent streaming
reconstruction/hash check. Both profiles pass fresh-page inference with every
model URL blocked, followed by zero retained sessions and pooled GPU buffers.
The clear-cache UI test passes. There are **42 passing JavaScript unit tests**
and **five Python packing/transform tests**, including legacy-cache reuse,
range recovery/fallback, concurrent partial-cache readers, unavailable storage,
invalid layouts, corrupt payloads and arbitrary FP32 bit patterns. Production
build passes with the existing chunk-size advisory.

## Reproduce

Keep the existing v1 package in `public/models/packed/` read-only. Create an
isolated codec environment, then build the v2 package:

```sh
uv venv --python 3.12 .venv-codecs
uv pip install --python .venv-codecs/bin/python -r requirements-codecs.txt
.venv-codecs/bin/python screen_codecs.py
.venv-codecs/bin/python pack_adaptive.py
.venv-codecs/bin/python verify_packed_models.py \
  --package public/models/packed-v2 \
  --report benchmarks/adaptive-file-verification.json
npm test
.venv/bin/python -m unittest test_pack_models -q
.venv-codecs/bin/python -m unittest test_adaptive_pack -q
npm run build
npm run preview
```

Exporters/reports refuse to overwrite existing evidence. Preserve existing
outputs or use the supported output arguments for another experiment. Large
packages and isolated environments remain ignored by Git.

Vite automatically selects v2 when its manifest exists. Set
`VITE_MODEL_PACKAGE_FORMAT=1` to retain the original transport. For hosted builds,
set `VITE_MODEL_PACKAGE_BASE_URL` to a version-pinned repository's `packed-v2`
directory. The small manifest is bundled locally and hash-pinned; weight bundles
remain in model storage. `prepare_model.py` includes available package formats;
`prepare_release.py` requires a passing summary for the exact selected manifest.
The hosted Space uses a version-pinned package; its `release.json` records the
source commit, model revision and validated package checksum.

Evidence: [summary](benchmarks/adaptive-summary.json),
[all 54 comparisons](benchmarks/adaptive-browser.json),
[codec screen](benchmarks/adaptive-codec-screen.json),
[full candidate pass](benchmarks/adaptive-codec-full.json),
[file reconstruction](benchmarks/adaptive-file-verification.json),
[cache upgrade](benchmarks/adaptive-cache-migration.json),
[range recovery](benchmarks/adaptive-range-recovery.json),
[Fast reload](benchmarks/adaptive-cache-reload.json),
[Detail reload](benchmarks/adaptive-detail-reload.json),
[cold Detail](benchmarks/adaptive-detail-cold.json).

`summarize_adaptive.py` validates exact matrix coverage, unchanged metrics,
payload/request totals, all hash checks, cache behavior, unloads and AC power.
