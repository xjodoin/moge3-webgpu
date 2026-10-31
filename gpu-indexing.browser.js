import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,releaseSessions} from './runtime.js';
import {gpuOps} from './gpu-ops.js';
import {GpuVoxelMaps} from './gpu-indexing.js';
import {voxelize,neighbors,poolMap} from './sparse-maps.js';
function equal(a,b,label){if(a.length!==b.length)throw Error(`${label}: size ${a.length} != ${b.length}`);for(let i=0;i<a.length;i++)if(a[i]!==b[i])throw Error(`${label}: at ${i}, ${a[i]} != ${b[i]}`);}
export async function testGpuIndexing(){
 const report={rows:[],startedAt:Date.now()};window.indexTests=report;
 await releaseSessions();await sparseSession('input_proj');const ops=gpuOps(await ort.env.webgpu.device);
 const read=t=>ops.download(t.gpuBuffer,t.dims.reduce((a,b)=>a*b,1)*4,t.type);
 const inputs=[];
 for(const [H,W] of [[1,1],[3,5],[17,19],[33,35],[65,65]]){
  const raw=new Float32Array(H*W*3);for(let i=0;i<H*W;i++)raw[i*3+2]=((i*17)%37-19+.5)/256;
  inputs.push({name:`odd-ties-${H}x${W}`,H,W,raw});
 }
 for(const profile of ['fast','detail'])for(const photo of ['sample-backyard','brick-ranch','raised-deck'])for(const step of [0,1,2]){
  const meta=await(await fetch(`/models/profiles/${profile}/manifest.json`)).json();
  const raw=new Float32Array(await(await fetch(`/original-cuda/${profile}/${photo}/diagnostics/raw-${step}.f32`)).arrayBuffer());
  inputs.push({name:`${profile}/${photo}/${step}`,H:meta.grid[0],W:meta.grid[1],raw});
 }
 try{
  for(const {name,H,W,raw} of inputs){
   const stats={},builder=new GpuVoxelMaps(ops,stats),input=ops.tensor([H*W,3],'float32',raw);
   try{
    let cpu=voxelize(raw,H,W),gpu=builder.initial(input,H,W);
    for(let level=0;level<5;level++){
     equal(await read(gpu.coords),cpu.coords,`${name} coords ${level}`);
     const map=builder.neighbors(gpu);equal(await read(map),neighbors(cpu.coords,cpu.shape),`${name} neighbors ${level}`);builder.drop(map);
     if(level<4){
      const expected=poolMap(cpu.coords,cpu.shape),actual=await builder.pool(gpu);
      equal(await read(actual.parent),expected.parent,`${name} parents ${level}`);
      const counts=await read(actual.childCounts),children=await read(actual.children);
      for(let row=0;row<counts.length;row++){
       if(counts[row]!==expected.offsets[row+1]-expected.offsets[row])throw Error('Child count differs');
       for(let i=0;i<counts[row];i++)if(children[row*8+i]!==expected.children[expected.offsets[row]+i])throw Error('Child order differs');
      }
      // Compare actual pooled FP32 features too, preserving sum order exactly.
      const values=Float32Array.from({length:gpu.n*2},(_,i)=>Math.sin(i*.171));
      const features=ops.tensor([gpu.n,2],'float32',values),pooled=ops.run('pool8',[features,actual.childCounts,actual.children],actual.level.n,2);
      const offsets=ops.tensor([expected.offsets.length],'int32',expected.offsets),oldChildren=ops.tensor([expected.children.length],'int32',expected.children);
      const reference=ops.run('pool',[features,offsets,oldChildren],actual.level.n,2);
      equal(await pooled.getData(),await reference.getData(),`${name} pooled FP32 ${level}`);features.dispose();pooled.dispose();reference.dispose();offsets.dispose();oldChildren.dispose();cpu=expected;gpu=actual.level;
     }
    }
    const indices=builder.encoderIndices(gpu),expected=Int32Array.from({length:cpu.coords.length/3},(_,i)=>cpu.coords[i*3]*gpu.W+cpu.coords[i*3+1]);equal(await read(indices),expected,'encoder indices');
    report.rows.push({name,levels:5,...stats,exact:true});
   }finally{builder.dispose();input.dispose();}
  }
  report.done=true;return report;
 }finally{await releaseSessions();}
}
