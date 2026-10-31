import test from 'node:test';
import assert from 'node:assert/strict';
import {CapturedDenseSession} from './dense-capture.js';
function setup(run,release=async()=>{}){
  const buffers=[],writes=[],calls=[],owners=[];
  const device={createBuffer:({size})=>{const b={size,destroyed:0,destroy(){this.destroyed++;}};buffers.push(b);return b;},queue:{writeBuffer:(...args)=>writes.push(args)}};
  const Tensor={fromGpuBuffer:(gpuBuffer,options)=>({gpuBuffer,options})};
  const shapes={raw_coord:[1,2,2,3],mask:[1,2,2]},outBuffers={raw_coord:{size:48},mask:{size:16}};
  const session={run:async(...args)=>{
    calls.push(args);if(run)await run(...args);
    const out=Object.fromEntries(Object.entries(shapes).map(([name,dims])=>[name,{type:'float32',dims,location:'gpu-buffer',gpuBuffer:outBuffers[name],disposed:0,dispose(){this.disposed++;}}]));owners.push(out);return out;
  },release};
  const graph=new CapturedDenseSession(session,device,{inputShape:[1,3,2,2],outputs:shapes},Tensor,async()=>new Float32Array(4),7);
  const image={type:'float32',dims:[1,3,2,2],data:new Float32Array(12)};
  return {graph,image,buffers,writes,calls,owners,outBuffers};
}
test('captured bindings remain fixed while photos and disposable output views change',async()=>{
  const {graph,image,buffers,writes,calls,owners}=setup();
  const a=await graph.run({image});image.data.fill(.5);const b=await graph.run({image});
  assert.equal(calls[0][0].image,calls[1][0].image);
  assert.equal(calls[0][1],undefined);assert.equal(calls[1][1],graph.bindings);
  assert.notEqual(a.raw_coord,b.raw_coord);assert.notEqual(a.raw_coord,graph.bindings.raw_coord);
  assert.equal(a.raw_coord.gpuBuffer,b.raw_coord.gpuBuffer);assert.equal(writes.length,2);
  assert.equal(owners[0].raw_coord.disposed,0);assert.equal(owners[1].raw_coord.disposed,1);
  assert.equal(graph.snapshot().stableBindingReplays,1);assert.ok(buffers.every(b=>b.destroyed===0));
  await graph.release();await graph.release();assert.equal(owners[0].raw_coord.disposed,1);assert.ok(buffers.every(b=>b.destroyed===1));
  await assert.rejects(graph.run({image}),/released/);
});
test('rejects shape changes and concurrent runs before modifying captured buffers',async()=>{
  let complete;const {graph,image,writes}=setup(()=>new Promise(r=>{complete=r;}));
  await assert.rejects(graph.run({image:{...image,dims:[1,3,1,4]}}),/shape/);assert.equal(writes.length,0);
  const pending=graph.run({image});await assert.rejects(graph.run({image}),/already running/);
  await assert.rejects(graph.release(),/running/);complete();await pending;await graph.release();
});
test('run failures clear the busy guard and release failures still destroy input bindings',async()=>{
  const {graph,image,buffers}=setup(async()=>{throw Error('GPU failure');},async()=>{throw Error('device lost');});
  await assert.rejects(graph.run({image}),/GPU failure/);assert.equal(graph.busy,false);assert.equal(graph.runs,0);
  await assert.rejects(graph.release(),/device lost/);assert.ok(buffers.every(b=>b.destroyed===1));
});
test('unexpected output replacement is rejected and duplicate output handles are released',async()=>{
  const {graph,image,outBuffers,owners}=setup();await graph.run({image});outBuffers.mask={size:16};
  await assert.rejects(graph.run({image}),/changed its bound output/);
  assert.equal(owners[1].raw_coord.disposed,1);assert.equal(owners[0].raw_coord.disposed,0);await graph.release();
});
