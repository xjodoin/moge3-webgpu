# Lossless shared downloads and persistent caching

This page records the v1 transport experiment. The current local default uses
the [adaptive v2 package](ADAPTIVE.md) when generated: about 1.24 GB and 89 model
requests for a cold profile. The v1 measurements and evidence below are preserved.

The FP32 transport now downloads compressed, content-addressed chunks, shares
identical weights between resolutions, and saves verified chunks in a dedicated
browser Cache Storage cache. Each original model file is reconstructed byte for
byte before ONNX Runtime or the WGSL weight loader receives it. No graph,
precision, learned parameter or accuracy threshold changes.

## Measured full model downloads

Production build, Chrome 156 on Apple M4 Pro, AC power throughout. Payloads
include the package manifest, selected dense model, 11 ONNX linear/projection
modules and ten packed WGSL blocks. GB and MB below are decimal.

| Scenario | Original model payload | New model payload |
| --- | ---: | ---: |
| Cold Fast profile | 1,482,651,855 bytes | **1,262,717,135 bytes — 14.83% less** |
| Cold Detail profile, separate empty cache | 1,486,366,529 bytes | **1,264,692,561 bytes — 14.91% less** |
| Detail after Fast has been cached | Another 1,328,831,957-byte dense file | **6,180,759 additional bytes**, five new chunks |
| Both profiles combined, shared refiner counted once | 2,811,483,812 bytes | **1,268,897,894 bytes — 54.87% less** |
| Fresh page reload, either cached profile | — | **0 model bytes / 0 model requests** |

These are full package payload measurements, not sample-based compression
estimates. They exclude app JavaScript/WASM, photos, validation references and
HTTP headers. Browser Resource Timing independently reports 1,262,731,997 encoded
body bytes for cold Fast and 1,264,703,828 for cold Detail. The small difference
from package byte totals reflects the preview server's additional HTTP
compression. Localhost load times do not predict Internet download speed.

Cold loads request 797 items including the manifest. Four concurrent chunk
loads bound temporary decoding memory. Request overhead and network latency
remain relevant on remote hosting; no Internet startup-speed claim is made.

## Accuracy and cache validation

All **54 original strict-FP32 CUDA comparisons pass**: three photos × two
resolutions × one/two/three passes × three repetitions. All mask pixels match.
Worst point-map NRMSE remains **0.0943%**, and every original-CUDA error report is
identical to the preceding FP32 control. There are also six passing warmups and
cold/cached browser checks. The default subgroup kernels, GPU indexing, buffer
reuse and disabled graph capture are unchanged.

All 33 packaged source files, including optional ONNX comparison blocks, were
independently stream-reconstructed and matched against their original SHA-256.
The package contains extra comparison-backend chunks, which default WGSL
inference does not request.

Fresh page reloads were tested at both resolutions with **all package URLs
blocked** and raw ONNX/bin URLs blocked. Both completed accurate inference using
797 cache hits and zero model network requests, then unloaded all sessions and
pooled GPU buffers. The Clear downloaded model UI button also removed its
dedicated cache successfully. The temporary second-origin test cache was cleared.

34 JavaScript unit tests and three Python packing tests pass. Cache tests cover
corruption recovery, truncation, interrupted downloads, invalid layouts,
decompression overflow, unavailable storage, quota failures, concurrent readers,
fresh loader instances and preserving unrelated caches. The production build
passes with the existing chunk-size advisory.

## Format and integrity

`pack_models.py` locates initializer payload spans in the original ONNX wire
format without reserializing the graph. It splits weights into tensor-relative
chunks of at most 4 MiB so profile-specific protobuf offsets do not prevent
sharing. Float32 payload bytes are rearranged into four byte planes and compressed
with zlib level 6. Other bytes use identity ordering plus zlib. Small protobuf
headers are combined per file to avoid a request for every header fragment.

The manifest lists original file sizes/checksums, ordered assembly pieces,
compressed chunk checksums, decoded checksums and encoding. The loader verifies
the build-pinned manifest, compressed and decoded chunks, then each complete
assembled file. Invalid downloaded data is never cached; invalid cached data is
removed and fetched again. A failed model load does not silently switch to a
different precision, raw model or CPU learned-operation backend.

Content hashes are cache identities, allowing reuse across profiles and package
revisions. Compressed bytes persist on disk across page reloads; decoded models
are not kept in a second application memory cache. Unload releases GPU sessions
while retaining downloads. Clear downloaded model removes only
`moge3-model-chunks-v1`, not photos or another application's cache.

Cache Storage is browser-managed and subject to quotas and eviction. Unavailable
storage or failed writes permit verified inference without persistent saving.
Already saved chunks remain reusable; missing chunks are fetched again. No
customer photos or output geometry enter this cache. See the browser's
[cache](https://developer.mozilla.org/en-US/docs/Web/API/Cache) and
[storage quota](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)
documentation.

## Build and reproduce

With the existing FP32 exports in `public/models/`:

```sh
.venv/bin/python pack_models.py
.venv/bin/python verify_packed_models.py
npm test
.venv/bin/python -m unittest test_pack_models -q
npm run build
npm run preview
```

Packaging refuses to overwrite `public/models/packed/`. Verification and report
tools also refuse to overwrite scientific evidence. Preserve old outputs before
preparing another version. Model binaries and compressed chunks remain ignored
by Git; source and JSON benchmark evidence are tracked separately.

Vite enables packaged downloads for local builds when a package exists. For a
hosted build, publish `packed/` through a version-pinned model repository and set
`VITE_MODEL_PACKAGE_BASE_URL` to that revision's `packed` directory. The small
manifest is bundled with the app and hash-pinned in its JavaScript; chunks stay
in model storage. `prepare_model.py` includes the package when present.

```sh
VITE_MODEL_BASE_URL='https://host/pinned-model-revision' \
VITE_MODEL_PACKAGE_BASE_URL='https://host/pinned-model-revision/packed' \
npm run build
```

An external legacy build without `VITE_MODEL_PACKAGE_BASE_URL` retains raw-model
loading for compatibility; it does not enable this new persistent model cache.
Raw files need roughly 1.5 GB per profile. These v1 measurements were collected
locally; the hosted Space's `release.json` identifies its deployed package.

`benchmark-downloads.browser.js` performs the cold-cache 54-comparison matrix.
It is a development helper, not part of the production bundle. For production
testing, inject its exported function into the page as done in this experiment.
`window.modelDownloadStats()` exposes network/cache counters and
`window.clearModelCache()` removes the app's model cache while inference is idle.
Use a fresh power-log filename with `benchmark_power.py` for another run.

Evidence:
[summary](benchmarks/download-summary.json),
[54 CUDA comparisons and browser resources](benchmarks/download-browser.json),
[independent cold Detail](benchmarks/download-detail-cold.json),
[Fast reload](benchmarks/download-cache-reload.json),
[Detail reload](benchmarks/download-detail-reload.json),
[file reconstruction](benchmarks/packed-file-verification.json),
[power samples](benchmarks/download-power.jsonl).

`summarize_downloads.py` validates exact matrix coverage, unchanged accuracy,
byte totals, package hashes, cache persistence, unload checks and AC power.
`prepare_release.py` requires a passing summary for the exact manifest before
preparing a compressed-package release.
