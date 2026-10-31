import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {refine,remap} from './refiner.js';
import {project} from './geometry.js';
import {compare,passes} from './comparison.js';
import {denseSession,releaseSessions,sessionStats,setSparseVariant as selectSparseVariant,getIndexingBackend,setIndexingBackend as selectIndexingBackend,setDenseGraphCapture as selectDenseGraphCapture} from './runtime.js';
import {trackGpuBuffers} from './gpu-memory.js';
import {gpuOps,gpuBufferStats,setBufferReuse} from './gpu-ops.js';
import {GpuProfiler} from './gpu-profiler.js';
const memory=globalThis.GPUDevice?trackGpuBuffers():null;
ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:mjsUrl};
ort.env.wasm.numThreads=1;
const status=document.querySelector('#status');
const log=t=>{status.textContent=t;};
const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`${r.status} ${url}`);return r.arrayBuffer();};
let running=false,upload=null;
const showFixture=()=>{upload=null;document.querySelector('#upload').value='';document.querySelector('#photo').src=`/models/profiles/${document.querySelector('#profile').value}/cases/${document.querySelector('#case').value}/photo.jpg`;};
document.querySelector('#profile').onchange=showFixture;
window.runTest=async(withRefinement=true,caseName='sample-backyard',customInput=null,profile='fast',sparseBackend='wgsl',refineSteps=1,validateOriginal=true,diagnostics={})=>{
 if(running)throw Error('Inference is already running');
 running=true;for(const id of ['run','case','upload','profile','steps','unload'])document.querySelector('#'+id).disabled=true;
 let inputTensor,out,profiler,ops,failed=true;
 try{
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('This browser has no WebGPU adapter. Try a current Chrome browser on a supported computer.');
  memory?.reset();
  const base=profile==='legacy'?'/models':`/models/profiles/${profile}`;
  const meta=await (await fetch(`${base}/manifest.json`)).json();
  if(!['wgsl','onnx'].includes(sparseBackend))throw Error('Unknown sparse backend');
  const [,,height,width]=meta.inputShape,n=width*height,options={grid:meta.grid??[272,272],width,height,sparseBackend,indexingBackend:getIndexingBackend()};
  if(!Number.isInteger(refineSteps)||refineSteps<1||refineSteps>3)throw Error('Refinement passes must be 1, 2 or 3');
  const steps=refineSteps;
  document.querySelector('#steps').value=String(steps);
  const gatherBytes=options.grid[0]*options.grid[1]*27*32*4;
  if(sparseBackend==='onnx'&&adapter.limits.maxStorageBufferBindingSize<gatherBytes)throw Error('This profile requires larger GPU buffers than your device supports. Select a lower resolution.');
  const fixture=`${base}/cases/${caseName}`;
  document.querySelector('#profile').value=profile;
  const input=customInput??new Float32Array(await read(`${fixture}/input.f32`));
  if(!customInput)document.querySelector('#case').value=caseName;
  if(!customInput)document.querySelector('#photo').src=`${fixture}/photo.jpg`;
  log('Preparing model…');const start=performance.now(),before=sessionStats();
  const session=await denseSession(profile,meta,log,sparseBackend);
  const loadMs=performance.now()-start;
  inputTensor=new ort.Tensor('float32',input,meta.inputShape);
  const bufferBefore=gpuBufferStats();
  ops=gpuOps(await ort.env.webgpu.device);
  if(diagnostics.profileGpu){profiler=new GpuProfiler(await ort.env.webgpu.device);ops.profiler=profiler;options.profiler=profiler;}
  const t=performance.now();
  out=await session.run({image:inputTensor});
  const cpuOutputs=['raw_coord','mask','metric_scale'];
  const values=await Promise.all(cpuOutputs.map(name=>out[name].getData()));
  const data=Object.fromEntries(cpuOutputs.map((name,i)=>[name,values[i]]));
  // Include required CPU output readbacks in both paths for comparable timing.
  const inferenceMs=performance.now()-t;
  const outputs={};
  async function check(name,actual){
   const path=steps>1&&/^(refined-|projected-|native-)/.test(name)?`${fixture}/passes/${steps}`:fixture;
   if(!customInput)outputs[name]=compare(actual,new Float32Array(await read(`${path}/reference-${name}.f32`)),name==='native-points'?{boundaryMask:data.mask}:{});
  }
  let refinementMs=0,points,raw=data.raw_coord;
  const transfers={},perPassTransfers=[];
  if(withRefinement){
   const t=performance.now();
   for(let step=0;step<steps;step++){
    options.pass=step+1;if(profiler)profiler.prefix=`pass${step+1}/`;options.telemetry={};raw=await refine(raw,out.encoder_feature,log,options);
    perPassTransfers.push(options.telemetry);
    for(const [key,value] of Object.entries(options.telemetry))transfers[key]=(transfers[key]??0)+value;
   }
   points=remap(raw,options);refinementMs=performance.now()-t;
  }else points=await out.points.getData();
  const cameraStart=performance.now(),geometry=project(points,data.mask,data.metric_scale[0],width,height);
  const cameraMs=performance.now()-cameraStart;
  // Validation downloads are outside compute timings and never run for uploads.
  const gpuProfile=profiler?await profiler.finish():undefined;
  const validationStart=performance.now();
  if(!customInput){
   for(const [name,tensor] of Object.entries(out))await check(name,await tensor.getData());
   if(withRefinement){await check('refined-raw',raw);await check('refined-points',points);}
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
  const result={profile,gpuProfile,buffers:{before:bufferBefore,after:gpuBufferStats()},sessions:{...after,createdThisRun:after.created-before.created},transfers,perPassTransfers,original,validationMs:performance.now()-validationStart,gpuMemory:memory?.report(),caseName:customInput?'uploaded-photo':caseName,loadMs,inferenceMs,refinementMs,cameraMs,outputs,
   refineSteps:withRefinement?steps:0,numTokens:meta.numTokens,inputShape:meta.inputShape,backend:'webgpu',sparseBackend,cpuLearnedFallback:false,
   camera:{focal:geometry.focal,shift:geometry.shift,intrinsics:geometry.intrinsics},
   validationReference:customInput?null:original?'original-cuda-strict-fp32':'pytorch-mlx-diagnostics',diagnosticPassed,
   validated:!customInput,passed:customInput?null:original?(original.passed&&densePassed):diagnosticPassed};
  failed=false;window.lastResult=result;log(JSON.stringify(result,null,2));render(geometry.points,geometry.mask,input,geometry.intrinsics,n);return result;
 }finally{
  if(ops)ops.profiler=null;profiler?.dispose();
  inputTensor?.dispose();if(out)for(const tensor of Object.values(out))tensor.dispose();
  try{if(failed)await releaseSessions();}finally{running=false;for(const id of ['run','case','upload','profile','steps','unload'])document.querySelector('#'+id).disabled=false;}
 }
};
window.releaseModels=async()=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();log('Model unloaded. GPU sessions released.');return sessionStats();}finally{running=false;}};
window.setDenseGraphCapture=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectDenseGraphCapture(value);return sessionStats();}finally{running=false;}};
window.setIndexingBackend=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectIndexingBackend(value);return sessionStats();}finally{running=false;}};
window.setSparseVariant=async value=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();selectSparseVariant(value);return sessionStats();}finally{running=false;}};
window.bufferPoolStats=gpuBufferStats;
window.setBufferReuse=async enabled=>{if(running)throw Error('Inference is already running');running=true;try{await releaseSessions();setBufferReuse(Boolean(enabled));return gpuBufferStats();}finally{running=false;}};
document.querySelector('#unload').onclick=()=>window.releaseModels().catch(e=>log(String(e)));
if(import.meta.hot)import.meta.hot.dispose(()=>{disposePreview();void releaseSessions();});
document.querySelector('#run').onclick=()=>window.runTest(true,document.querySelector('#case').value,upload,document.querySelector('#profile').value,'wgsl',Number(document.querySelector('#steps').value),true).catch(e=>{window.lastError=String(e);log(String(e));});
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
