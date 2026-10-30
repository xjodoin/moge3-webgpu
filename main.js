import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {refine,remap} from './refiner.js';
import {project} from './geometry.js';
import {compare,passes} from './comparison.js';
import {downloadModel} from './download.js';
import {modelUrl} from './model-location.js';
import {trackGpuBuffers} from './gpu-memory.js';
const memory=globalThis.GPUDevice?trackGpuBuffers():null;
ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:mjsUrl};
ort.env.wasm.numThreads=1;
const status=document.querySelector('#status');
const log=t=>{status.textContent=t;};
const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`${r.status} ${url}`);return r.arrayBuffer();};
let running=false,upload=null;
const showFixture=()=>{upload=null;document.querySelector('#upload').value='';document.querySelector('#photo').src=`/models/profiles/${document.querySelector('#profile').value}/cases/${document.querySelector('#case').value}/photo.jpg`;};
document.querySelector('#profile').onchange=showFixture;
window.runTest=async(withRefinement=true,caseName='sample-backyard',customInput=null,profile='fast')=>{
 if(running)throw Error('Inference is already running');
 running=true;for(const id of ['run','case','upload','profile'])document.querySelector('#'+id).disabled=true;
 let session;
 try{
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('This browser has no WebGPU adapter. Try a current Chrome browser on a supported computer.');
  memory?.reset();
  const base=profile==='legacy'?'/models':`/models/profiles/${profile}`;
  const meta=await (await fetch(`${base}/manifest.json`)).json();
  const [,,height,width]=meta.inputShape,n=width*height,options={grid:meta.grid??[272,272],width,height};
  const steps=meta.refineSteps||1;
  const gatherBytes=options.grid[0]*options.grid[1]*27*32*4;
  if(adapter.limits.maxStorageBufferBindingSize<gatherBytes)throw Error('This profile requires larger GPU buffers than your device supports. Select a lower resolution.');
  const fixture=`${base}/cases/${caseName}`;
  document.querySelector('#profile').value=profile;
  const input=customInput??new Float32Array(await read(`${fixture}/input.f32`));
  if(!customInput)document.querySelector('#case').value=caseName;
  if(!customInput)document.querySelector('#photo').src=`${fixture}/photo.jpg`;
  log('Loading MoGe-3 dense weights (about 1.3 GB)…');const start=performance.now();
  let modelData=await downloadModel(modelUrl(profile==='legacy'?'moge3-dense.onnx':`profiles/${profile}/moge3-dense.onnx`),meta.modelBytes,log);
  session=await ort.InferenceSession.create(modelData,{executionProviders:['webgpu'],enableGraphCapture:false,extra:{session:{disable_cpu_ep_fallback:'1'}}});
  modelData=null;
  const loadMs=performance.now()-start;
  const t=performance.now();const out=await session.run({image:new ort.Tensor('float32',input,meta.inputShape)});
  const inferenceMs=performance.now()-t;
  // Copy before releasing the dense session to lower peak GPU memory.
  const data=Object.fromEntries(Object.entries(out).map(([name,tensor])=>[name,new Float32Array(tensor.data)]));
  for(const tensor of Object.values(out))tensor.dispose();
  await session.release();session=null;
  const outputs={};
  async function check(name,actual){
   if(!customInput)outputs[name]=compare(actual,new Float32Array(await read(`${fixture}/reference-${name}.f32`)),name==='native-points'?{boundaryMask:data.mask}:{});
  }
  for(const [name,actual] of Object.entries(data))await check(name,actual);
  let refinementMs=0,points=data.points;
  if(withRefinement){
   const t=performance.now();let raw=data.raw_coord;
   for(let step=0;step<steps;step++)raw=await refine(raw,data.encoder_feature,log,options);
   points=remap(raw,options);refinementMs=performance.now()-t;
   await check('refined-raw',raw);await check('refined-points',points);
  }
  const cameraStart=performance.now(),geometry=project(points,data.mask,data.metric_scale[0],width,height);
  const cameraMs=performance.now()-cameraStart;
  if(withRefinement){
   await check('projected-points',geometry.points);
   if(profile!=='legacy'){
    await check('native-points',geometry.points);
    await check('native-intrinsics',new Float32Array(geometry.intrinsics));
   }
  }
  const result={profile,gpuMemory:memory?.report(),caseName:customInput?'uploaded-photo':caseName,loadMs,inferenceMs,refinementMs,cameraMs,outputs,
   refineSteps:withRefinement?steps:0,numTokens:meta.numTokens,inputShape:meta.inputShape,backend:'webgpu',cpuLearnedFallback:false,
   camera:{focal:geometry.focal,shift:geometry.shift,intrinsics:geometry.intrinsics},
   validated:!customInput,passed:customInput?null:Object.values(outputs).every(passes)};
  window.lastResult=result;log(JSON.stringify(result,null,2));render(geometry.points,geometry.mask,input,geometry.intrinsics,n);return result;
 }finally{if(session)await session.release();running=false;for(const id of ['run','case','upload','profile'])document.querySelector('#'+id).disabled=false;}
};
document.querySelector('#run').onclick=()=>window.runTest(true,document.querySelector('#case').value,upload,document.querySelector('#profile').value).catch(e=>{window.lastError=String(e);log(String(e));});
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
