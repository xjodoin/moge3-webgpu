// Real WebGPU checks; Node's unit-test runner cannot execute WGSL.
import {gpuOps} from './gpu-ops.js';
export async function testGpuOps() {
  const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice();
  const ops=gpuOps(device),owned=[];
  const tensor=(dims,type,data)=>{const t=ops.tensor(dims,type,data);owned.push(t);return t;};
  const run=(name,inputs,rows,channels)=>{const t=ops.run(name,inputs,rows,channels);owned.push(t);return t;};
  const check=async(t,expected)=>{
    const actual=Array.from(await t.getData());
    if(JSON.stringify(actual)!==JSON.stringify(expected))throw Error(`GPU mismatch: ${actual} != ${expected}`);
    return actual;
  };
  device.pushErrorScope('validation');
  try {
    const features=tensor([4,2],'float32',new Float32Array([1e8,2,1,4,-1e8,6,9,8]));
    const offsets=tensor([3],'int32',new Int32Array([0,3,4]));
    const children=tensor([4],'int32',new Int32Array([0,1,2,3]));
    const pooled=run('pool',[features,offsets,children],2,2);
    // Keep pooled GPU-resident for the next kernel, then validate both outputs.
    const parents=tensor([3],'int32',new Int32Array([1,0,1]));
    const skip=tensor([3,2],'float32',new Float32Array([1,2,3,4,5,6]));
    const added=run('upsampleAdd',[pooled,parents,skip],3,2);
    const indices=tensor([2],'int32',new Int32Array([3,0]));
    const gathered=run('gather',[features,indices],2,2);
    const result={pool:await check(pooled,[0,4,9,8]),upsampleAdd:await check(added,[10,10,3,8,14,14]),gather:await check(gathered,[9,8,1e8,2])};
    const error=await device.popErrorScope();if(error)throw Error(error.message);
    return {passed:true,...result};
  } finally {for(const t of owned)t.dispose();device.destroy();}
}
