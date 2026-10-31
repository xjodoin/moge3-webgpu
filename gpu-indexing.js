import {indexingKernels} from './gpu-indexing-kernels.js';
// Only four scalar shape readbacks per refinement pass. All index arrays and
// ordered children remain on GPU. Each pooled voxel has at most 2x2x2 children.
const pipelineCaches=new WeakMap();
export class GpuVoxelMaps {
 constructor(ops,stats={}){this.ops=ops;this.device=ops.device;this.stats=stats;this.owned=new Set();if(!pipelineCaches.has(this.device))pipelineCaches.set(this.device,new Map());this.pipelines=pipelineCaches.get(this.device);}
 tensor(dims,type='int32',data){const t=this.ops.tensor(dims,type,data);this.owned.add(t);return t;}
 drop(t){if(this.owned.delete(t))t.dispose();}
 dispatch(name,buffers,params,count){
  let pipeline=this.pipelines.get(name);
  if(!pipeline){pipeline=this.device.createComputePipeline({layout:'auto',compute:{module:this.device.createShaderModule({code:indexingKernels[name]}),entryPoint:'main'}});this.pipelines.set(name,pipeline);}
  const groups=Math.ceil(count/256);if(groups>this.device.limits.maxComputeWorkgroupsPerDimension)throw Error('Voxel indexing dispatch exceeds device limits');
  const lease=this.ops.pool.acquire(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  try{
   this.device.queue.writeBuffer(lease.buffer,0,new Uint32Array([...params,...Array(8-params.length).fill(0)]));
   this.stats.indexParameterUploadBytes=(this.stats.indexParameterUploadBytes??0)+32;
   const entries=[...buffers.map(t=>t.gpuBuffer),lease.buffer].map((buffer,binding)=>({binding,resource:{buffer}}));
   const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass(this.ops.profiler?.pass(`index/${name}`));
   pass.setPipeline(pipeline);pass.setBindGroup(0,this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries}));pass.dispatchWorkgroups(groups);pass.end();this.device.queue.submit([encoder.finish()]);
  }finally{lease.release();}
 }
 initial(raw,H,W){
  if(!Number.isInteger(H)||!Number.isInteger(W)||H<1||W<1||raw.type!=='float32'||raw.dims.reduce((a,b)=>a*b,1)!==H*W*3)throw Error('Invalid GPU coordinate grid');
  const n=H*W,z=this.tensor([n]),minimum=this.tensor([1],'int32',new Int32Array([2147483647]));
  this.stats.indexParameterUploadBytes=(this.stats.indexParameterUploadBytes??0)+4;
  const coords=this.tensor([n,3]),columns=this.tensor([n+1]);
  this.dispatch('quantize',[raw,z,minimum],[n,H,W],n);
  this.dispatch('initial',[z,minimum,coords,columns],[n,H,W],n);
  this.drop(z);this.drop(minimum);return {coords,columns,H,W,n};
 }
 neighbors(level){const output=this.tensor([level.n,27]);this.dispatch('neighbors',[level.coords,level.columns,output],[level.n,level.H,level.W],level.n*27);return output;}
 scan(input,n){
  const blocks=Math.ceil(n/256),output=this.tensor([n+1]),sums=this.tensor([blocks]);
  this.dispatch('scan',[input,output,sums],[n,blocks],n);
  if(blocks>1){const offsets=this.scan(sums,blocks);this.dispatch('addScan',[output,offsets],[n,blocks],n+1);this.drop(offsets);}
  this.drop(sums);return output;
 }
 async count(columns,n){
  const lease=this.ops.pool.acquire(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  try{
   const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(columns.gpuBuffer,n*4,lease.buffer,0,4);this.device.queue.submit([encoder.finish()]);
   await lease.buffer.mapAsync(GPUMapMode.READ);const value=new Uint32Array(lease.buffer.getMappedRange())[0];lease.buffer.unmap();
   this.stats.indexShapeReadbacks=(this.stats.indexShapeReadbacks??0)+1;this.stats.indexShapeReadbackBytes=(this.stats.indexShapeReadbackBytes??0)+4;return value;
  }finally{lease.release();}
 }
 async pool(level){
  const H=Math.ceil(level.H/2),W=Math.ceil(level.W/2),params=[H*W,level.H,level.W,W],counts=this.tensor([H*W]);
  this.dispatch('count',[level.coords,level.columns,counts],params,H*W);
  const columns=this.scan(counts,H*W),n=await this.count(columns,H*W);this.drop(counts);
  if(n<1||n>level.n)throw Error('Invalid GPU voxel count');
  const coords=this.tensor([n,3]),parent=this.tensor([level.n,1]),childCounts=this.tensor([n,1]),children=this.tensor([n,8]);
  this.dispatch('fill',[level.coords,level.columns,columns,coords,parent,childCounts,children],params,H*W);
  return {level:{coords,columns,H,W,n},parent,childCounts,children};
 }
 async build(raw,H,W){
  let level=this.initial(raw,H,W);const maps=[],pools=[],sizes=[];
  for(let i=0;i<5;i++){
   sizes.push(level.n);maps.push(this.neighbors(level));
   if(i<4){const p=await this.pool(level);this.drop(level.coords);this.drop(level.columns);pools.push(p);level=p.level;}
  }
  const indices=this.encoderIndices(level);this.drop(level.coords);this.drop(level.columns);return {maps,pools,sizes,indices};
 }
 encoderIndices(level){const output=this.tensor([level.n,1]);this.dispatch('encoderIndex',[level.coords,output],[level.n,level.H,level.W],level.n);return output;}
 dispose(){for(const t of this.owned)t.dispose();this.owned.clear();}
}
