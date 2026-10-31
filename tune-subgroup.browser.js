import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,releaseSessions} from './runtime.js';
import {selectedSparseConfig} from './sparse-kernels.js';
import {loadSparseBlock} from './sparse-block.js';
import {subgroup32Support} from './subgroup-support.js';
import {gpuOps} from './gpu-ops.js';
import {GpuProfiler} from './gpu-profiler.js';
import {voxelize,poolMap,neighbors} from './sparse-maps.js';
export const subgroupCandidates=[2,4,8,16].map(rows=>({kind:'subgroup-broadcast',rows,cols:128,k:32}));
function parity(a,b){
  if(a.length!==b.length)throw Error('Candidate shape mismatch');
  let diff=0,scale=0,maxAbs=0;
  for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i]))throw Error('Nonfinite subgroup output');const d=a[i]-b[i];diff+=d*d;scale+=b[i]*b[i];maxAbs=Math.max(maxAbs,Math.abs(d));}
  const nrmse=Math.sqrt(diff/Math.max(scale,1e-20));
  if(nrmse>=.00001||maxAbs>=.001)throw Error(`Subgroup parity failure: ${nrmse}, ${maxAbs}`);
  return {nrmse,maxAbs};
}
export async function tuneSubgroup(){
  const report={rows:[],done:false};window.subgroupTuning=report;
  await releaseSessions();await sparseSession('input_proj');
  const device=await ort.env.webgpu.device,ops=gpuOps(device),manifest=await(await fetch('/models/refiner-wgsl/manifest.json')).json();
  report.support=await subgroup32Support(device);if(!report.support.supported)throw Error(report.support.reason);
  try{
    for(const [profile,photo,step] of [['fast','sample-backyard',0],['detail','sample-backyard',0],['detail','raised-deck',0],['detail','brick-ranch',2]]){
      const meta=await(await fetch(`/models/profiles/${profile}/manifest.json`)).json();
      const raw=new Float32Array(await(await fetch(`/original-cuda/${profile}/${photo}/diagnostics/raw-${step}.f32`)).arrayBuffer());
      let map=voxelize(raw,...meta.grid);
      for(let level=0;level<5;level++){
        if(level>=2){
          const n=map.coords.length/3,c=32*2**level,name=`down_stages.${level}.0`;
          const input=ops.tensor([n,c],'float32',Float32Array.from({length:n*c},(_,i)=>Math.sin(i*.019))),index=ops.tensor([n,27],'int32',neighbors(map.coords,map.shape));
          const variants=[],configs=[selectedSparseConfig(c),{kind:'subgroup-norm',convolution:selectedSparseConfig(c)},...subgroupCandidates];let reference;
          const row={profile,photo,step,name,channels:c,rows:n,candidates:[]};report.rows.push(row);
          try{
            for(const config of configs){
              const block=await loadSparseBlock(device,name,manifest,config);variants.push(block);
              const output=block.run(input,index),values=new Float32Array(await output.getData());output.dispose();
              reference??=values;row.candidates.push({config,parity:parity(values,reference),gpuMs:[]});
            }
            const profiler=new GpuProfiler(device);
            try{
              for(let repeat=0;repeat<5;repeat++)for(const v of Array.from(variants.keys()).sort((a,b)=>repeat%2?b-a:a-b)){
                profiler.prefix=`${v}/${repeat}/`;const output=variants[v].run(input,index,profiler);output.dispose();
                await device.queue.onSubmittedWorkDone();
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
