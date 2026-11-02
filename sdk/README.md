# @xjodoin/moge3-webgpu

Browser photo reconstruction, isolated in a dedicated worker. No DOM, viewer,
backend or inference server required. Photos remain on the device. Weights are
fetched from the pinned public model revision and cached in IndexedDB.

```js
import {MoGeClient} from '@xjodoin/moge3-webgpu';
const client = new MoGeClient({workerUrl:'/moge-sdk/worker.js'});
const result = await client.reconstruct(file, {
  profile:'fast', passes:1, weightMode:'exact', signal:controller.signal,
  onProgress:message => console.log(message),
});
// OpenCV coordinates: right, down, forward; normalized row-major intrinsics.
console.log(result.points, result.mask, result.imageTransform);
client.dispose();
```

Install the experimental release and copy its runtime assets:

```sh
npm install @xjodoin/moge3-webgpu@experimental
npx moge3-copy-assets public/moge-sdk
```

Add `moge3-copy-assets public/moge-sdk` to your app's predev/prebuild scripts so
clean installs and deployments include the worker, WASM and model manifests.
The destination should be a dedicated same-origin static directory. The command
overlays SDK files and preserves unrelated files. Keep worker.js, assets/ and
models/ together. Serve over HTTPS or
localhost; WASM needs application/wasm. Use a CSP allowing same-origin workers,
WASM compilation and HTTPS Hugging Face model downloads. No cross-origin isolation
is required: the WASM loader uses one thread and learned inference uses WebGPU.
Check `await MoGeClient.isSupported()` before offering reconstruction.

Each client accepts one request at a time and retains GPU sessions until disposal.
An AbortSignal terminates the worker and releases its GPU context; the next call
creates a new worker. dispose() is idempotent and permanently closes the client.
Results own transferred typed arrays. Progress callbacks run on the caller thread.
The SDK center-crops inputs to the fixed profile aspect ratio; imageTransform.crop
records the crop as fractions of the original image. Consumers must map annotations
and camera intrinsics back to original photo coordinates. No reference validation
is implied for uploaded photos, and physical dimensions require calibration.

Fast is 512x344; Detail is 768x512. One to three learned refinement passes are
supported. Exact original weights are the SDK default. Compact is optional and
uses approximate encoder storage with FP32 computation. Initial downloads are
large (about 1 GB); no model weights are included in the npm package.

Known port limitation: the real deck Detail fixture exceeds the CUDA point-error
gate at two/three passes in both baseline and autotuned execution. This SDK does
not represent construction measurements or structural approval.

Build from repository root: `npm run build:sdk`, then `npm pack ./sdk`.
Code is MIT; upstream model weights retain their original license.
