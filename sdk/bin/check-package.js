import {access} from 'node:fs/promises';
const required=['dist/index.js','dist/worker.js','dist/assets/ort-wasm-simd-threaded.asyncify.mjs','dist/assets/ort-wasm-simd-threaded.asyncify.wasm','dist/models/profiles/fast/manifest.json','dist/models/profiles/detail/manifest.json','dist/models/refiner-wgsl/manifest.json','index.d.ts','LICENSE','licenses/ONNX-Runtime-LICENSE.txt','licenses/MoGe-LICENSE.txt'];
for(const path of required)await access(new URL('../'+path,import.meta.url));
