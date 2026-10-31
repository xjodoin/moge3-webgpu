import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,releaseSessions} from './runtime.js';
import {loadSparseBlock} from './sparse-block.js';
import {sparseModelUrl} from './model-location.js';
import {gpuOps} from './gpu-ops.js';
import {compare} from './comparison.js';

export async function benchmarkSparseTiles(){
  await releaseSessions();await sparseSession('input_proj');
  const device=await ort.env.webgpu.device,ops=gpuOps(device);
  const manifest=await(await fetch(sparseModelUrl('refiner-wgsl/manifest.json'))).json(),report=[];
  try{
    for(const [i,n] of [8003,4003,1501,503,251].entries()){
      const name=`down_stages.${i}.0`,c=manifest.blocks[name].channels;
      const input=ops.tensor([n,c],'float32',Float32Array.from({length:n*c},(_,j)=>Math.sin(j*.019)));
      const neighbors=ops.tensor([n,27],'int32',Int32Array.from({length:n*27},(_,j)=>j%3===0?(j*7)%n:n));
      const variants=await Promise.all([32,64].map(k=>loadSparseBlock(device,name,manifest,k)));
      const timings={32:[],64:[]},values=[];
      try{
        for(let v=0;v<2;v++){const result=variants[v].run(input,neighbors);values.push(new Float32Array(await result.getData()));result.dispose();}
        const parity=compare(values[1],values[0]);if(!parity.finite||parity.nrmse>=.00001)throw Error('Tile parity failed');
        for(let repeat=0;repeat<5;repeat++)for(const v of (repeat%2?[1,0]:[0,1])){
          const start=performance.now();const result=variants[v].run(input,neighbors);await device.queue.onSubmittedWorkDone();
          timings[v?64:32].push(performance.now()-start);result.dispose();
        }
        report.push({name,channels:c,rows:n,timings,parity});
      }finally{for(const v of variants)v.release();input.dispose();neighbors.dispose();}
    }
    return report;
  }finally{await releaseSessions();}
}
