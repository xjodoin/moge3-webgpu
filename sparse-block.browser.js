// Run in the app's browser console; exercises real weights and real WebGPU.
import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,directSparseBlock,releaseSessions} from './runtime.js';
import {gpuOps} from './gpu-ops.js';
import {compare} from './comparison.js';

export async function testSparseBlocks(){
  const reports=[];
  const names=[...Array.from({length:5},(_,i)=>`down_stages.${i}.0`),'bottleneck_stage.0',...Array.from({length:4},(_,i)=>`up_stages.${i}.0`)];
  try{
    for(const [index,name] of names.entries()){
      const session=await sparseSession(name),device=await ort.env.webgpu.device,ops=gpuOps(device);
      const block=await directSparseBlock(name),c=block.meta.channels,n=19;
      const values=Float32Array.from({length:n*c},(_,i)=>Math.sin(i*.173+index)*2+Math.cos(i*.071));
      const map=Int32Array.from({length:n*27},(_,i)=>i%7===0?n:(Math.floor(i/27)*13+(i%27)*5)%n);
      // One entirely empty neighborhood, plus an irregular, non-tile-aligned row count.
      map.fill(n,(n-1)*27);
      const features=ops.tensor([n,c],'float32',values),neighbors=ops.tensor([n,27],'int32',map);
      let expected,actual;
      device.pushErrorScope('validation');
      try{
        expected=(await session.run({features,neighbors})).output;
        actual=block.run(features,neighbors);
        const reference=await expected.getData(),result=await actual.getData();
        const metrics=compare(result,reference);
        const emptyRowError=Math.max(...result.slice((n-1)*c).map((v,i)=>Math.abs(v-reference[(n-1)*c+i])));
        const error=await device.popErrorScope();if(error)throw Error(error.message);
        if(!metrics.finite||metrics.nrmse>=.00001||metrics.maxAbs>=.001)throw Error(`${name}: ${JSON.stringify(metrics)}`);
        reports.push({name,channels:c,rows:n,emptyRowError,...metrics,passed:true});
      }finally{features.dispose();neighbors.dispose();expected?.dispose();actual?.dispose();}
    }
    return reports;
  }finally{await releaseSessions();}
}
