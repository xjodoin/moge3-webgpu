import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,directSparseBlock,releaseSessions} from './runtime.js';
import {gpuOps} from './gpu-ops.js';
import {compare} from './comparison.js';
export async function testWideBlocks(){
 const rows=[];window.wideBlocks=rows;
 const names=[...Array.from({length:5},(_,i)=>`down_stages.${i}.0`),'bottleneck_stage.0',...Array.from({length:4},(_,i)=>`up_stages.${i}.0`)];
 try{
  for(const name of names){
   const session=await sparseSession(name),device=await ort.env.webgpu.device,ops=gpuOps(device),block=await directSparseBlock(name),c=block.meta.channels,n=19;
   for(const empty of [false,true]){
    const input=ops.tensor([n,c],'float32',Float32Array.from({length:n*c},(_,i)=>Math.sin(i*.17)+Math.cos(i*.07)));
    const map=Int32Array.from({length:n*27},(_,i)=>empty||i%7===0?n:(Math.floor(i/27)*13+i%27*5)%n);map.fill(n,(n-1)*27);
    const indices=ops.tensor([n,27],'int32',map);let expected,actual;
    device.pushErrorScope('validation');
    try{
     expected=(await session.run({features:input,neighbors:indices})).output;actual=block.run(input,indices);
     const metrics=compare(await actual.getData(),await expected.getData());const error=await device.popErrorScope();
     if(error)throw Error(error.message);
     if(!metrics.finite||metrics.nrmse>=.00001||metrics.maxAbs>=.001)throw Error(`${name}: parity failure`);
     rows.push({name,channels:c,rows:n,empty,config:block.config,...metrics,passed:true});
    }finally{input.dispose();indices.dispose();actual?.dispose();expected?.dispose();}
   }
  }
  window.wideBlocksDone=true;return rows;
 }finally{await releaseSessions();}
}
