import * as ort from 'onnxruntime-web/webgpu';
import {GpuBufferPool} from './gpu-buffer-pool.js';
const kernels = {
  pool: `
    @group(0) @binding(0) var<storage,read> input:array<f32>;
    @group(0) @binding(1) var<storage,read> offsets:array<u32>;
    @group(0) @binding(2) var<storage,read> children:array<u32>;
    @group(0) @binding(3) var<storage,read_write> output:array<f32>;
    @group(0) @binding(4) var<uniform> p:vec4<u32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      let i=id.x;if(i>=p.x*p.y){return;}
      let row=i/p.y;let channel=i%p.y;let start=offsets[row];let end=offsets[row+1u];
      var sum=0.0;
      for(var j=start;j<end;j++){sum=sum+input[children[j]*p.y+channel];}
      output[i]=sum/f32(end-start);
    }`,
  pool8: `
    @group(0) @binding(0) var<storage,read> input:array<f32>;
    @group(0) @binding(1) var<storage,read> counts:array<u32>;
    @group(0) @binding(2) var<storage,read> children:array<u32>;
    @group(0) @binding(3) var<storage,read_write> output:array<f32>;
    @group(0) @binding(4) var<uniform> p:vec4<u32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      let i=id.x;if(i>=p.x*p.y){return;}let row=i/p.y;let channel=i%p.y;let count=counts[row];var sum=0.0;
      for(var j=0u;j<count;j++){sum+=input[children[row*8u+j]*p.y+channel];}
      output[i]=sum/f32(count);
    }`,
  gather: `
    @group(0) @binding(0) var<storage,read> input:array<f32>;
    @group(0) @binding(1) var<storage,read> indices:array<u32>;
    @group(0) @binding(2) var<storage,read_write> output:array<f32>;
    @group(0) @binding(3) var<uniform> p:vec4<u32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      let i=id.x;if(i>=p.x*p.y){return;}
      output[i]=input[indices[i/p.y]*p.y+i%p.y];
    }`,
  upsampleAdd: `
    @group(0) @binding(0) var<storage,read> input:array<f32>;
    @group(0) @binding(1) var<storage,read> parents:array<u32>;
    @group(0) @binding(2) var<storage,read> skip:array<f32>;
    @group(0) @binding(3) var<storage,read_write> output:array<f32>;
    @group(0) @binding(4) var<uniform> p:vec4<u32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      let i=id.x;if(i>=p.x*p.y){return;}
      output[i]=input[parents[i/p.y]*p.y+i%p.y]+skip[i];
    }`,
};
const instances = new WeakMap();
const active = new Set();
// Bounded reuse reduces allocation churn; measured latency remains device-dependent.
let reuse=true;
export function setBufferReuse(enabled){reuse=enabled;for(const ops of active){ops.pool.clear();ops.pool.enabled=enabled;}}
export function releaseGpuBuffers(){for(const ops of active)ops.pool.clear();active.clear();}
export function gpuBufferStats(){return [...active].map(ops=>ops.pool.snapshot());}
export function gpuOps(device) {
  if (!instances.has(device)) instances.set(device,new GpuOps(device));
  const ops=instances.get(device);ops.pool.enabled=reuse;active.add(ops);return ops;
}
class GpuOps {
  constructor(device) { this.device=device;this.pipelines=new Map();this.pool=new GpuBufferPool(device,{enabled:reuse});this.profiler=null; }
  tensor(dims,type='float32',data) {
    const bytes=dims.reduce((a,b)=>a*b,1)*4;
    const lease=this.pool.acquire(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),buffer=lease.buffer;
    try {
      if (data) this.device.queue.writeBuffer(buffer,0,data);
      return ort.Tensor.fromGpuBuffer(buffer,{dataType:type,dims,dispose:()=>lease.release(),download:()=>this.download(buffer,bytes,type)});
    }catch(error){lease.release();throw error;}
  }
  async download(buffer,bytes,type) {
    const lease=this.pool.acquire(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),copy=lease.buffer;
    try {
      const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,copy,0,bytes);this.device.queue.submit([encoder.finish()]);
      await copy.mapAsync(GPUMapMode.READ);
      const result=copy.getMappedRange().slice(0,bytes);copy.unmap();
      return type==='float32'?new Float32Array(result):new Int32Array(result);
    } finally { lease.release(); }
  }
  run(name,inputs,rows,channels) {
    let pipeline=this.pipelines.get(name);
    if (!pipeline) {
      pipeline=this.device.createComputePipeline({layout:'auto',compute:{module:this.device.createShaderModule({code:kernels[name]}),entryPoint:'main'}});
      this.pipelines.set(name,pipeline);
    }
    const groups=Math.ceil(rows*channels/256);
    if(groups>this.device.limits.maxComputeWorkgroupsPerDimension)throw Error('GPU operation exceeds supported dimensions');
    const output=this.tensor([rows,channels]);
    const lease=this.pool.acquire(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),uniform=lease.buffer;
    try {
      this.device.queue.writeBuffer(uniform,0,new Uint32Array([rows,channels,0,0]));
      const buffers=[...inputs.map(t=>t.gpuBuffer),output.gpuBuffer,uniform];
      const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass(this.profiler?.pass(name));pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(groups);pass.end();
      this.device.queue.submit([encoder.finish()]);return output;
    } catch(error) { output.dispose();throw error; }
    finally { lease.release(); }
  }
}
