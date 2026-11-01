import * as ort from 'onnxruntime-web/webgpu';
import {SessionCache} from './session-cache.js';
import {downloadModel} from './download.js';
import {modelUrl,sparseModelUrl} from './model-location.js';
import {loadSparseBlock} from './sparse-block.js';
import {selectedSparseConfig} from './sparse-kernels.js';
import {releaseGpuBuffers,gpuOps} from './gpu-ops.js';
import {CapturedDenseSession} from './dense-capture.js';
import {subgroup32Support} from './subgroup-support.js';
import {packagedModel} from './model-package.js';
import {installIndexArrayCompatibility} from './gpu-shader-compat.js';
const shaderCompatibility=installIndexArrayCompatibility();
const cache = new SessionCache();
let activeProfile = null;
let sparseVariant='subgroup';
let subgroupSupport=null,sparseConfigs={};
let indexingBackend='gpu';
let denseGraphCapture=false,activeDense=null;
let precision='fp32';
export function setPrecision(value){if(!['fp32','fp16-storage','fp16-compute'].includes(value))throw Error('Unknown precision');precision=value;weightManifest=null;}
const precisionPath=path=>precision==='fp32'?path:`precision/${precision}/${path}`;
const packedFolder=()=>precision==='fp32'?'refiner-wgsl':'precision/fp16-weights/refiner-wgsl';
export function setDenseGraphCapture(value){if(typeof value!=='boolean')throw Error('Graph capture must be boolean');denseGraphCapture=value;}
export function setIndexingBackend(value){if(!['cpu','gpu'].includes(value))throw Error('Unknown indexing backend');indexingBackend=value;}
export const getIndexingBackend=()=>indexingBackend;
export function setSparseVariant(value){if(!['baseline','optimized','subgroup'].includes(value))throw Error('Unknown sparse variant');sparseVariant=value;}
let activeBackend = null,weightManifest;
// The native EP forwards enableGraphCapture only for object-form provider options.
const gpuOptions = {executionProviders:[{name:'webgpu'}],enableGraphCapture:false,extra:{session:{disable_cpu_ep_fallback:'1'}}};
// Only one dense profile is retained. Switching releases its sparse sessions too,
// allowing ORT to reclaim pools associated with the previous resolution.
export async function denseSession(profile, meta, log,backend='wgsl') {
  if (activeProfile !== profile || activeBackend !== backend) {
    await releaseSessions();
    activeProfile = profile;
    activeBackend = backend;
  }
  return cache.get(`dense:${profile}:${denseGraphCapture}`,async () => {
    const path=precisionPath(profile==='legacy'?'moge3-dense.onnx':`profiles/${profile}/moge3-dense.onnx`);
    if(precision!=='fp32'){
      if(backend!=='wgsl'||denseGraphCapture)throw Error('Experimental FP16 requires WGSL and graph capture disabled');
      const response=await fetch(modelUrl(precisionPath(`profiles/${profile}/manifest.json`)));
      if(!response.ok)throw Error(`FP16 manifest: HTTP ${response.status}`);
      const converted=await response.json();
      if(converted.precision!==precision||JSON.stringify(converted.inputShape)!==JSON.stringify(meta.inputShape))throw Error('FP16 manifest mismatch');
      meta=converted;
    }
    const data=await (precision==='fp32'?packagedModel(path,modelUrl(path),meta.modelBytes,log):downloadModel(modelUrl(path),meta.modelBytes,log));
    const session=await ort.InferenceSession.create(data,{
      ...gpuOptions,
      enableGraphCapture:denseGraphCapture,
      graphOptimizationLevel:'all',
      preferredOutputLocation:denseGraphCapture?'gpu-buffer':{encoder_feature:'gpu-buffer',points:'gpu-buffer',normal:'gpu-buffer'},
    });
    try {
      const device=await ort.env.webgpu.device,ops=gpuOps(device);
      activeDense=denseGraphCapture?new CapturedDenseSession(session,device,meta,ort.Tensor,(...args)=>ops.download(...args),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST):session;
      return activeDense;
    }catch(error){await session.release();throw error;}
  });
}
export const sparseSession = name => cache.get(`sparse:${name}`,async () => {
 const path=precisionPath(`refiner/${name}.onnx`);
 const source=precision==='fp32'?await packagedModel(path,modelUrl(path),0):modelUrl(path);
 return ort.InferenceSession.create(source,{...gpuOptions,preferredOutputLocation:'gpu-buffer'});
});
export const directSparseBlock = name => cache.get(`wgsl:${sparseVariant}:${name}`,async () => {
  if(!weightManifest)weightManifest=fetch(sparseModelUrl(`${packedFolder()}/manifest.json`)).then(async r=>{if(!r.ok)throw Error(`Sparse manifest: HTTP ${r.status}`);return r.json();}).catch(e=>{weightManifest=null;throw e;});
  const manifest=await weightManifest;
  const device=await ort.env.webgpu.device;
  if(sparseVariant==='subgroup')subgroupSupport=await subgroup32Support(device);
  const variant=sparseVariant==='subgroup'&&!subgroupSupport.supported?'optimized':sparseVariant;
  const block=await loadSparseBlock(device,name,manifest,selectedSparseConfig(manifest.blocks[name].channels,variant),packedFolder());
  sparseConfigs[name]=block.config;return block;
});
export async function releaseSessions() { activeProfile=null;activeBackend=null;activeDense=null;subgroupSupport=null;sparseConfigs={};try{await cache.clear();}finally{releaseGpuBuffers();} }
export const sessionStats = () => ({precision,profile:activeProfile,sparseBackend:activeBackend,sparseVariant,subgroupSupport,sparseConfigs,indexingBackend,denseGraphCapture,denseGraph:activeDense?.snapshot?.()??null,shaderCompatibility:shaderCompatibility.snapshot(),...cache.snapshot()});
