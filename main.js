import {setSparseAutotune,clearSparseAutotune,sparseAutotuneStats} from './sparse-autotune.js';
import {gpuGeometry} from './gpu-geometry.js';
import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {refine,remap} from './refiner.js';
import {project} from './geometry.js';
import {compare,passes} from './comparison.js';
import {denseSession,releaseSessions,sessionStats,setPrecision as selectPrecision,setSparseVariant as selectSparseVariant,getIndexingBackend,setIndexingBackend as selectIndexingBackend,setDenseGraphCapture as selectDenseGraphCapture} from './runtime.js';
import {trackGpuBuffers} from './gpu-memory.js';
import {gpuOps,gpuBufferStats,setBufferReuse} from './gpu-ops.js';
import {GpuProfiler} from './gpu-profiler.js';
import {modelDownloadStats,clearModelCache,setLayerScreen,setWeightScreen,setWeightMode} from './model-package.js';
const memory=globalThis.GPUDevice?trackGpuBuffers():null;
ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:mjsUrl};
ort.env.wasm.numThreads=1;
if(new URLSearchParams(location.search).has('verbose'))ort.env.logLevel='verbose';
const status=document.querySelector('#status');
const log=t=>{status.textContent=t;};
const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`${r.status} ${url}`);return r.arrayBuffer();};
let running=false,upload=null;
const showFixture=()=>{upload=null;document.querySelector('#upload').value='';document.querySelector('#photo').src=`/models/profiles/${document.querySelector('#profile').value}/cases/${document.querySelector('#case').value}/photo.jpg`;};
document.querySelector('#profile').onchange=showFixture;
window.runTest=async(withRefinement=true,caseName='sample-backyard',customInput=null,profile='fast',sparseBackend='wgsl',refineSteps=1,validateOriginal=true,diagnostics={})=>{
 if(running)throw Error('Inference is already running');
 running=true;for(const id of ['run','case','upload','profile','steps','weights','unload','clear-cache'])document.querySelector('#'+id).disabled=true;
 let inputTensor,out,profiler,ops,failed=true;const geometryOwned=new Set();
 try{
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('This browser has no WebGPU adapter. Try a current Chrome browser on a supported computer.');
  memory?.reset();
  const base=profile==='legacy'?'/models':`/models/profiles/${profile}`;
  const meta=await (await fetch(`${base}/manifest.json`)).json();
  if(!['wgsl','onnx'].includes(sparseBackend))throw Error('Unknown sparse backend');
  const [,,height,width]=meta.inputShape,n=width*height,options={grid:meta.grid??[272,272],width,height,sparseBackend,indexingBackend:getIndexingBackend()};
  if(!Number.isInteger(refineSteps)||refineSteps<1||refineSteps>3)throw Error('Refinement passes must be 1, 2 or 3');
  const steps=refineSteps;
  const originalOnly=diagnostics.referenceScope==='original-only';
  if(originalOnly&&(!validateOriginal||!withRefinement||customInput||profile==='legacy'))throw Error('Original-only validation requires a named CUDA fixture and refinement');
  document.querySelector('#steps').value=String(steps);
  const gatherBytes=options.grid[0]*options.grid[1]*27*32*4;
  if(sparseBackend==='onnx'&&adapter.limits.maxStorageBufferBindingSize<gatherBytes)throw Error('This profile requires larger GPU buffers than your device supports. Select a lower resolution.');
  const fixture=`${base}/cases/${caseName}`;
  document.querySelector('#profile').value=profile;
  const input=customInput??new Float32Array(await read(`${fixture}/input.f32`));
  if(!customInput)document.querySelector('#case').value=caseName;
  if(!customInput)document.querySelector('#photo').src=`${fixture}/photo.jpg`;
  log('Preparing model…');const start=performance.now(),before=sessionStats(),downloadsBefore=modelDownloadStats();
  const session=await denseSession(profile,meta,log,sparseBackend);
  const loadMs=performance.now()-start;
  inputTensor=new ort.Tensor('float32',input,meta.inputShape);
  const bufferBefore=gpuBufferStats();
  ops=gpuOps(await ort.env.webgpu.device);
  if(diagnostics.profileGpu){profiler=new GpuProfiler(await ort.env.webgpu.device);ops.profiler=profiler;options.profiler=profiler;}
  const t=performance.now();
  out=await session.run({image:inputTensor});
  const gpuPostprocess=diagnostics.postprocess!=='cpu';options.gpuPostprocess=gpuPostprocess;
  const geo=gpuGeometry(await ort.env.webgpu.device);if(gpuPostprocess)await geo.prepare();
  const keepGeometry=t=>{geometryOwned.add(t);return t;};
  const cpuOutputs=['raw_coord','mask','metric_scale'];
  const values=gpuPostprocess?[]:await Promise.all(cpuOutputs.map(name=>out[name].getData()));
  const data=Object.fromEntries(cpuOutputs.map((name,i)=>[name,values[i]]));
  // Include required CPU output readbacks in both paths for comparable timing.
  const inferenceMs=performance.now()-t;
  const outputs={};
  async function check(name,actual){
   const path=steps>1&&/^(refined-|projected-|native-)/.test(name)?`${fixture}/passes/${steps}`:fixture;
   if(!customInput&&!originalOnly)outputs[name]=compare(actual,new Float32Array(await read(`${path}/reference-${name}.f32`)),name==='native-points'?{boundaryMask:data.mask}:{});
  }
  let refinementMs=0,points,raw=gpuPostprocess?out.raw_coord:data.raw_coord;
  const transfers={},perPassTransfers=[];let previewMs=0;
  if(withRefinement){
   const t=performance.now();
   for(let step=0;step<steps;step++){
    options.pass=step+1;if(profiler)profiler.prefix=`pass${step+1}/`;options.telemetry={};const previousRaw=raw;raw=await refine(raw,out.encoder_feature,log,options);if(gpuPostprocess){keepGeometry(raw);if(geometryOwned.delete(previousRaw))previousRaw.dispose();}
    perPassTransfers.push(options.telemetry);
    for(const [key,value] of Object.entries(options.telemetry))transfers[key]=(transfers[key]??0)+value;
    if(diagnostics.progressive&&step<steps-1){
     const previewStart=performance.now();
     let preview;if(gpuPostprocess){const mapped=geo.remap(raw,options);try{preview=await geo.project(mapped,out.mask,out.metric_scale,width,height,{variant:diagnostics.geometryVariant??'optimized'});}finally{mapped.dispose();}}else preview=project(remap(raw,options),data.mask,data.metric_scale[0],width,height);
     render(preview.points,preview.mask,input,preview.intrinsics,n);
     log(`Preview: refinement ${step+1}/${steps}. Continuing…`);
     window.dispatchEvent(new CustomEvent('moge-progress',{detail:{pass:step+1,total:steps,validated:false}}));
     await new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,0)));
     previewMs+=performance.now()-previewStart;
    }
   }
   points=gpuPostprocess?keepGeometry(geo.remap(raw,options)):remap(raw,options);refinementMs=performance.now()-t-previewMs;
  }else points=gpuPostprocess?out.points:await out.points.getData();
  const cameraStart=performance.now(),geometry=gpuPostprocess?await geo.project(points,out.mask,out.metric_scale,width,height,{variant:diagnostics.geometryVariant??'optimized'}):project(points,data.mask,data.metric_scale[0],width,height);
  const cameraMs=performance.now()-cameraStart;
  // Validation downloads are outside compute timings and never run for uploads.
  const gpuProfile=profiler?await profiler.finish():undefined;
  const validationStart=performance.now();
  if(gpuPostprocess&&!customInput)data.mask=await out.mask.getData();
  if(!customInput&&!originalOnly){
   for(const [name,tensor] of Object.entries(out))await check(name,await tensor.getData());
   if(withRefinement){await check('refined-raw',gpuPostprocess?await raw.getData():raw);await check('refined-points',gpuPostprocess?await points.getData():points);}
  }
  if(withRefinement){
   await check('projected-points',geometry.points);
   if(profile!=='legacy'){
    await check('native-points',geometry.points);
    await check('native-intrinsics',new Float32Array(geometry.intrinsics));
   }
  }
  let original;
  if(validateOriginal&&withRefinement&&!customInput&&profile!=='legacy'){
   const originalBase=`/original-cuda/${profile}/${caseName}/${steps}`;
   const originalPoints=new Float32Array(await read(`${originalBase}/points.f32`));
   const originalIntrinsics=new Float32Array(await read(`${originalBase}/intrinsics.f32`));
   const originalMask=new Uint8Array(await read(`${originalBase}/mask.u8`));
   const points=compare(geometry.points,originalPoints,{boundaryMask:data.mask});
   const intrinsics=compare(new Float32Array(geometry.intrinsics),originalIntrinsics);
   let maskMismatches=0,unexplainedMaskMismatches=0;
   if(originalMask.length!==geometry.mask.length)throw Error('Original mask shape mismatch');
   for(let i=0;i<originalMask.length;i++)if(Boolean(originalMask[i])!==Boolean(geometry.mask[i])){
    maskMismatches++;if(Math.abs(data.mask[i]-.5)>1e-5)unexplainedMaskMismatches++;
   }
   original={points,intrinsics,maskMismatches,unexplainedMaskMismatches,
    passed:passes(points)&&passes(intrinsics)&&maskMismatches/originalMask.length<=.00001&&unexplainedMaskMismatches===0};
  }
  const denseNames=['points','normal','mask','metric_scale','raw_coord','encoder_feature'];
  const densePassed=denseNames.every(name=>outputs[name]&&passes(outputs[name]));
  const diagnosticPassed=!customInput&&Object.values(outputs).every(passes);
  const after=sessionStats();
  const result={profile,geometryVariant:diagnostics.geometryVariant??'optimized',geometryTimings:gpuPostprocess?{...geo.lastTimings}:null,postprocess:gpuPostprocess?'gpu':'cpu',timingSemantics:gpuPostprocess?'Stage enqueue times; camera includes queued GPU work and final readback':'CPU stages synchronized by readbacks',previewMs,downloads:{before:downloadsBefore,after:modelDownloadStats()},gpuProfile,buffers:{before:bufferBefore,after:gpuBufferStats()},sessions:{...after,createdThisRun:after.created-before.created},transfers,perPassTransfers,original,validationMs:performance.now()-validationStart,gpuMemory:memory?.report(),caseName:customInput?'uploaded-photo':caseName,loadMs,inferenceMs,refinementMs,cameraMs,outputs,
   refineSteps:withRefinement?steps:0,numTokens:meta.numTokens,inputShape:meta.inputShape,backend:'webgpu',sparseBackend,cpuLearnedFallback:false,
   camera:{focal:geometry.focal,shift:geometry.shift,intrinsics:geometry.intrinsics},
   validationReference:customInput?null:original?'original-cuda-strict-fp32':'pytorch-mlx-diagnostics',referenceScope:originalOnly?'original-only':'full-diagnostics',diagnosticPassed:originalOnly?null:diagnosticPassed,
   validated:!customInput,passed:customInput?null:original?(original.passed&&(originalOnly||densePassed)):diagnosticPassed};
  if(typeof diagnostics.onGeometry==='function')await diagnostics.onGeometry(geometry,{width,height,input});
  failed=false;window.lastResult=result;log(JSON.stringify(result,null,2));if(diagnostics.render!==false)render(geometry.points,geometry.mask,input,geometry.intrinsics,n);return result;
 }finally{
  if(ops)ops.profiler=null;profiler?.dispose();
  for(const tensor of geometryOwned)tensor.dispose();
  inputTensor?.dispose();if(out)for(const tensor of Object.values(out))tensor.dispose();
  try{if(failed)await releaseSessions();}finally{running=false;for(const id of ['run','case','upload','profile','steps','weights','unload','clear-cache'])document.querySelector('#'+id).disabled=false;}
 }
};
window.releaseModels=async()=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();log('Model unloaded. GPU sessions released.');return sessionStats();}finally{running=false;}};
window.setLayerScreen=async value=>{if(running)throw Error('Inference is already running');await releaseSessions();setLayerScreen(value);};
window.modelDownloadStats=modelDownloadStats;
window.setWeightScreen=async enabled=>{if(running)throw Error('Inference is already running');await releaseSessions();setWeightScreen(enabled);};
window.setWeightMode=async mode=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();setWeightMode(mode);document.querySelector('#weights').value=mode;return sessionStats();}finally{running=false;}};
document.querySelector('#weight-mode-control').hidden=!modelDownloadStats().progressive;
document.querySelector('#weights').value=modelDownloadStats().weightMode;
document.querySelector('#weights').onchange=event=>window.setWeightMode(event.target.value).then(()=>log(event.target.value==='exact'?'Original weights selected. The next run restores the original FP32 weights using correction data; saved compact weights are reused.':'Smaller download selected. Selected encoder weights use compact storage; computation remains FP32.')).catch(error=>{document.querySelector('#weights').value=modelDownloadStats().weightMode;log(String(error));});
window.clearModelCache=async()=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();return await clearModelCache();}finally{running=false;}};
document.querySelector('#clear-cache').onclick=()=>window.clearModelCache().then(()=>log('Downloaded model files removed from this device.')).catch(e=>log(String(e)));
// Developer-only experiments; the public UI and fresh page default stay FP32.
window.setPrecision=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectPrecision(value);return sessionStats();}finally{running=false;}};
window.setDenseGraphCapture=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectDenseGraphCapture(value);return sessionStats();}finally{running=false;}};
window.setIndexingBackend=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectIndexingBackend(value);return sessionStats();}finally{running=false;}};
window.setSparseVariant=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectSparseVariant(value);return sessionStats();}finally{running=false;}};
window.bufferPoolStats=gpuBufferStats;
window.setBufferReuse=async enabled=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();setBufferReuse(Boolean(enabled));return gpuBufferStats();}finally{running=false;}};
document.querySelector('#unload').onclick=()=>window.releaseModels().catch(e=>log(String(e)));
if(import.meta.hot)import.meta.hot.dispose(()=>{disposePreview();void releaseSessions();});
document.querySelector('#run').onclick=()=>window.runTest(true,document.querySelector('#case').value,upload,document.querySelector('#profile').value,'wgsl',Number(document.querySelector('#steps').value),true,{progressive:true}).catch(e=>{window.lastError=String(e);log(String(e));});
document.querySelector('#case').onchange=showFixture;
document.querySelector('#upload').onchange=async event=>{
 try{
  const file=event.target.files[0];if(!file)return;
  const profile=document.querySelector('#profile').value,base=profile==='legacy'?'/models':`/models/profiles/${profile}`;
  const meta=await (await fetch(`${base}/manifest.json`)).json(),[,,height,width]=meta.inputShape,n=width*height;
  const bitmap=await createImageBitmap(file),canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d');const scale=Math.max(width/bitmap.width,height/bitmap.height),sw=width/scale,sh=height/scale;
  ctx.drawImage(bitmap,(bitmap.width-sw)/2,(bitmap.height-sh)/2,sw,sh,0,0,width,height);bitmap.close();
  const rgba=ctx.getImageData(0,0,width,height).data;upload=new Float32Array(3*n);
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)upload[c*n+i]=rgba[4*i+c]/255;
  document.querySelector('#photo').src=canvas.toDataURL();log('Photo ready. Run inference to reconstruct it locally.');
 }catch(e){upload=null;log(`Cannot read this image: ${e.message}`);}
};
let disposePreview=()=>{};
function render(points,mask,rgb,intrinsics,n){
 disposePreview();
 const canvas=document.querySelector('#scene'),renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setSize(960,520,false);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#dfe6df');const p=[],c=[],depths=[];
 const linear=v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4;
 for(let i=0;i<mask.length;i++)if(mask[i]&&Number.isFinite(points[i*3+2])){p.push(points[i*3],-points[i*3+1],-points[i*3+2]);c.push(linear(rgb[i]),linear(rgb[i+n]),linear(rgb[i+2*n]));depths.push(points[i*3+2]);}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));g.setAttribute('color',new THREE.Float32BufferAttribute(c,3));
 const cloud=new THREE.Points(g,new THREE.PointsMaterial({size:2.5,sizeAttenuation:false,vertexColors:true}));scene.add(cloud);
 depths.sort((a,b)=>a-b);const distance=depths[Math.floor(depths.length/2)]??1;
 const fov=2*Math.atan(.5/intrinsics[4])*180/Math.PI;
 const camera=new THREE.PerspectiveCamera(fov,960/520,.01,10000);
 const controls=new OrbitControls(camera,canvas);controls.target.set(0,0,-distance);controls.update();renderer.setAnimationLoop(()=>renderer.render(scene,camera));
 disposePreview=()=>{renderer.setAnimationLoop(null);controls.dispose();g.dispose();cloud.material.dispose();renderer.dispose();};
}

// Explicit local benchmark build; excluded from ordinary production builds.
if(import.meta.env.VITE_GEOMETRY_BENCHMARK==='1')window.runGeometryTuning=async()=>{const {benchmarkGeometryTuning}=await import('./benchmark-geometry-tuning.browser.js');return benchmarkGeometryTuning();};
if(import.meta.env.VITE_GEOMETRY_BENCHMARK==='1')window.runGeometryTiming=async()=>{const {benchmarkGeometryTiming}=await import('./benchmark-geometry-tuning.browser.js');return benchmarkGeometryTiming();};

window.setSparseAutotune=value=>{if(running)throw Error('Inference is already running');setSparseAutotune(value);};
window.clearSparseAutotuneCache=()=>{if(running)throw Error('Inference is already running');clearSparseAutotune();};
window.sparseAutotuneStats=()=>sparseAutotuneStats(true);

if(import.meta.env.VITE_AUTOTUNE_BENCHMARK==='1')window.runAutotuneBenchmark=async (timing,options)=>{const {benchmarkAutotune}=await import('./benchmark-autotune.browser.js');return benchmarkAutotune(timing,options);};
