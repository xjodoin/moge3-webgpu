import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,releaseSessions} from './runtime.js';
import {loadSparseBlock} from './sparse-block.js';
import {gpuOps} from './gpu-ops.js';
import {GpuProfiler} from './gpu-profiler.js';
import {voxelize,poolMap,neighbors} from './sparse-maps.js';
export const candidates=[32,{rows:16,cols:32,skip:'group'},{rows:16,cols:32,skip:'mask'},{rows:32,cols:32,skip:'mask'},{rows:16,cols:64,skip:'mask'},{rows:32,cols:64,skip:'mask'}];
function parity(a,b){let diff=0,scale=0,maxAbs=0;for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i]))throw Error('Nonfinite candidate');const d=a[i]-b[i];diff+=d*d;scale+=b[i]*b[i];maxAbs=Math.max(maxAbs,Math.abs(d));}const nrmse=Math.sqrt(diff/Math.max(scale,1e-20));if(nrmse>=.00001||maxAbs>=.001)throw Error(`Candidate error ${nrmse} ${maxAbs}`);return {nrmse,maxAbs};}
export async function tuneWide(){
 const report={rows:[],done:false};window.wideTuning=report;
 await releaseSessions();await sparseSession('input_proj');
 const device=await ort.env.webgpu.device,ops=gpuOps(device),manifest=await(await fetch('/models/refiner-wgsl/manifest.json')).json();
 try{
  for(const [profile,photo] of [['fast','sample-backyard'],['detail','sample-backyard'],['detail','raised-deck']]){
   const meta=await(await fetch(`/models/profiles/${profile}/manifest.json`)).json();
   const raw=new Float32Array(await(await fetch(`/original-cuda/${profile}/${photo}/diagnostics/raw-0.f32`)).arrayBuffer());
   let map=voxelize(raw,...meta.grid);
   for(let level=0;level<5;level++){
    if(level>=2){
     const n=map.coords.length/3,c=[32,64,128,256,512][level],name=`down_stages.${level}.0`;
     const input=ops.tensor([n,c],'float32',Float32Array.from({length:n*c},(_,i)=>Math.sin(i*.019))),index=ops.tensor([n,27],'int32',neighbors(map.coords,map.shape));
     const variants=[];let reference;
     const row={profile,photo,name,channels:c,rows:n,candidates:[]};report.rows.push(row);
     try{
      for(const config of candidates){
       const block=await loadSparseBlock(device,name,manifest,config);variants.push(block);
       const output=block.run(input,index),values=new Float32Array(await output.getData());output.dispose();
       if(!reference)reference=values;
       row.candidates.push({config,parity:parity(values,reference),gpuMs:[]});
      }
      const profiler=new GpuProfiler(device);
      try{
       for(let repeat=0;repeat<5;repeat++)for(const v of Array.from(variants.keys()).sort((a,b)=>repeat%2?b-a:a-b)){
        profiler.prefix=`${v}/${repeat}/`;const output=variants[v].run(input,index,profiler);output.dispose();
       }
       const trace=await profiler.finish();
       for(let v=0;v<variants.length;v++)for(let repeat=0;repeat<5;repeat++)row.candidates[v].gpuMs.push(trace.gpu.filter(e=>e.name.startsWith(`${v}/${repeat}/`)).reduce((s,e)=>s+e.gpuMs,0));
      }finally{profiler.dispose();}
     }finally{for(const v of variants)v.release();input.dispose();index.dispose();}
    }
    if(level<4)map=poolMap(map.coords,map.shape);
   }
  }
  report.done=true;return report;
 }finally{await releaseSessions();}
}
