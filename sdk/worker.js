import * as ort from 'onnxruntime-web/webgpu';
import wasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import mjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import {denseSession,releaseSessions} from '../runtime.js';
import {setWeightMode} from '../model-package.js';
import {configureModelLocations} from '../model-location.js';
import {refine} from '../refiner.js';
import {gpuGeometry} from '../gpu-geometry.js';
ort.env.wasm.wasmPaths={wasm,mjs};ort.env.wasm.numThreads=1;
configureModelLocations({sparseModelBaseUrl:new URL('./models',import.meta.url).href});
let busy=false,mode='exact';
self.onmessage=async({data:m})=>{
 if(m.type!=='reconstruct')return;
 const send=(type,value)=>self.postMessage({id:m.id,type,...value});
 if(busy){send('error',{message:'Worker is busy'});return;}busy=true;
 let input,out;const owned=new Set();
 try{
 if(!navigator.gpu||!await navigator.gpu.requestAdapter())throw Object.assign(new Error('WebGPU is unavailable on this device'),{code:'WEBGPU_UNAVAILABLE'});
 if(m.photo.size>32*1024*1024)throw Error('Choose a photo smaller than 32 MB');
 const metadataPath='./models/profiles/'+m.profile+'/manifest.json';const response=await fetch(new URL(metadataPath,import.meta.url));if(!response.ok)throw Error('Cannot load SDK profile metadata');const meta=await response.json();
 const [,,height,width]=meta.inputShape,n=width*height,bitmap=await createImageBitmap(m.photo);
 let pixels,transform;
 try{if(bitmap.width*bitmap.height>80_000_000)throw Error('Photo resolution is too large');const scale=Math.max(width/bitmap.width,height/bitmap.height),sw=width/scale,sh=height/scale,left=(bitmap.width-sw)/2,top=(bitmap.height-sh)/2;
 transform={originalWidth:bitmap.width,originalHeight:bitmap.height,crop:[left/bitmap.width,top/bitmap.height,sw/bitmap.width,sh/bitmap.height]};
 const canvas=new OffscreenCanvas(width,height),ctx=canvas.getContext('2d');ctx.drawImage(bitmap,left,top,sw,sh,0,0,width,height);pixels=ctx.getImageData(0,0,width,height).data;
 }finally{bitmap.close();}
 const rgb=new Uint8Array(n*3),values=new Float32Array(n*3);for(let i=0;i<n;i++)for(let c=0;c<3;c++){rgb[3*i+c]=pixels[4*i+c];values[c*n+i]=pixels[4*i+c]/255;}
 const log=message=>send('progress',{message});
 if(mode!==m.weightMode){await releaseSessions();mode=m.weightMode;}setWeightMode(mode);
 log('Preparing reconstruction model');const session=await denseSession(m.profile,meta,log,'wgsl');input=new ort.Tensor('float32',values,meta.inputShape);out=await session.run({image:input});
 const geo=gpuGeometry(await ort.env.webgpu.device);await geo.prepare();let raw=out.raw_coord;
 for(let pass=1;pass<=m.passes;pass++){const next=await refine(raw,out.encoder_feature,log,{grid:meta.grid,width,height,sparseBackend:'wgsl',indexingBackend:'gpu',gpuPostprocess:true,pass});owned.add(next);if(owned.delete(raw))raw.dispose();raw=next;send('progress',{message:`Refinement ${pass}/${m.passes}`});}
 const mapped=geo.remap(raw,{grid:meta.grid,width,height});owned.add(mapped);const geometry=await geo.project(mapped,out.mask,out.metric_scale,width,height);
 const result={width,height,points:geometry.points,depth:geometry.depth,mask:geometry.mask,colors:rgb,intrinsics:new Float32Array(geometry.intrinsics),imageTransform:transform,profile:m.profile,passes:m.passes,weightMode:mode,measurementStatus:'estimated'};
 self.postMessage({id:m.id,type:'result',result},[...new Set([result.points.buffer,result.depth.buffer,result.mask.buffer,result.colors.buffer,result.intrinsics.buffer])]);
 }catch(e){await releaseSessions().catch(()=>{});send('error',{message:e.message,code:e.code||'RECONSTRUCTION_FAILED'});}
 finally{for(const t of owned)t.dispose();input?.dispose();if(out)for(const t of Object.values(out))t.dispose();busy=false;}
};
