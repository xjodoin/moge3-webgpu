# Progressive previews and layer precision pilot

The Run button now displays each intermediate refinement result during a multi-pass run. One dense inference supplies the encoder features for every pass. Preview projection, rendering and yielding are reported separately as `previewMs`; existing benchmark callers remain nonprogressive unless they pass `{progressive:true}` as the final diagnostics argument. Intermediate previews are unvalidated. Final output retains the original validation logic. This does not yet implement resumable refinement across separate Run actions or Worker isolation.

Local storage experiments use `window.setLayerScreen([{name,bits}])`. They require a development build or `VITE_WEIGHT_SCREEN=1`, exact original weights and the verified FP24 screening plan. The probe retains IEEE sign/exponent bits, rounds mantissa bits and computes in FP32. A 20-bit probe is not native FP20 arithmetic. Only in-memory model copies change; package files, cached chunks and original goldens remain untouched. Clear with `setLayerScreen(null)`; changing weight mode also clears the override.

Run the isolated-layer matrix in a local development browser:

```js
const {benchmarkLayers}=await import('/benchmark-layers.browser.js');
const report=await benchmarkLayers({names:['onnx::MatMul_4390'],bits:[20]});
```

Omitting names tests all 96 eligible initializers and can take considerable time. Every candidate is tested independently against the exact model. Independent passing results do not establish combined accuracy. Test a combined configuration against all profiles, photos and pass counts before considering packaging. No automatic promotion or download-size claim is made.

## Evidence

`benchmarks/layer-preview-pilot.json` records:

- 54/54 unchanged CUDA regression checks with progressive previews enabled, with 54 intermediate preview events.
- Four isolated 20-bit candidates, 18 combinations each (72 checks). Only `onnx::MatMul_4390` passed all 18. The other three were rejected.
- The accepted initializer at 20 bits combined with all other eligible weights at the shipped 24-bit precision: 18/18 pass.
- These are the existing three synthetic images, two resolutions, and one to three passes. This is not a real-world holdout evaluation.

The accepted initializer contains 12,582,912 FP32 bytes. Dropping four storage bits from 24 to 20 would save 1,572,864 uncompressed bytes before packing, metadata and Zstd; actual transfer savings have not been measured. That alone does not justify replacing the published format. The current package and Hugging Face release remain unchanged.

Next research work is an expanded layer search, combined configuration validation, a separate diverse real-image holdout, and actual compressed package measurement. Worker isolation, continuation across Run actions, and preserving the orbit viewpoint across preview replacement remain UI follow-up work.
