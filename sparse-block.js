import {gpuOps} from './gpu-ops.js';
import {normSilu,sparseConvolution,tunedSparseConvolution,sparseConfig} from './sparse-kernels.js';
import {validateSparseWeights} from './sparse-layout.js';
import {sparseModelUrl} from './model-location.js';
import {subgroup32Support} from './subgroup-support.js';
import {subgroupBroadcastConvolution,subgroupNormSilu} from './subgroup-broadcast.js';

const pipelines=new WeakMap();
async function compile(device,config) {
  const key=JSON.stringify(config);
  const convolution=residual=>config.kind==='subgroup-broadcast'?subgroupBroadcastConvolution(residual,config):config.baseline?sparseConvolution(residual,config.k):tunedSparseConvolution(residual,config);
  if(!pipelines.has(device))pipelines.set(device,new Map());
  const cache=pipelines.get(device);
  if(!cache.has(key))cache.set(key,Promise.all([config.subgroupNorm?subgroupNormSilu:normSilu,convolution(false),convolution(true)].map(async (code,index)=>{
    const module=device.createShaderModule({code});
    const info=await module.getCompilationInfo();
    const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const layout=index===0?'auto':device.createPipelineLayout({bindGroupLayouts:[device.createBindGroupLayout({entries:Array.from({length:6},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===5?'uniform':binding===4?'storage':'read-only-storage'}}))})]});
    return device.createComputePipelineAsync({layout,compute:{module,entryPoint:'main'}});
  })).catch(error=>{cache.delete(key);throw error;}));
  return cache.get(key);
}

export async function loadSparseBlock(device,name,manifest,tileK=32) {
  const config=sparseConfig(tileK);
  if((config.kind?.startsWith('subgroup')||config.subgroupNorm)&&!(await subgroup32Support(device)).supported)throw Error('This GPU does not support complete 32-lane subgroups');
  const meta=manifest.blocks[name];if(!meta||manifest.format!==1||manifest.dtype!=='float32-le')throw Error('Unsupported sparse weights');
  validateSparseWeights(meta);
  const response=await fetch(sparseModelUrl(`refiner-wgsl/${name}.bin`));if(!response.ok)throw Error(`Sparse weights ${name}: HTTP ${response.status}`);
  const bytes=await response.arrayBuffer();
  if(bytes.byteLength!==meta.bytes)throw Error(`Truncated sparse weights: ${name}`);
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(hash!==meta.sha256)throw Error(`Sparse weights checksum mismatch: ${name}`);
  if(meta.channels%config.cols||(!config.baseline&&meta.channels%config.k))throw Error('Kernel tile does not divide channels');
  const compiled=await compile(device,config),ops=gpuOps(device);
  const weights=ops.tensor([bytes.byteLength/4],'float32',new Float32Array(bytes));
  return new SparseBlock(device,ops,compiled,weights,meta,config);
}

class SparseBlock {
  constructor(device,ops,compiled,weights,meta,config){Object.assign(this,{device,ops,compiled,weights,meta,config});}
  run(features,neighbors,profiler){
    const [rows,c]=features.dims;
    if(c!==this.meta.channels||c%32||rows<1||neighbors.dims[0]!==rows||neighbors.dims[1]!==27)throw Error('Invalid sparse block dimensions');
    if(Math.ceil(rows/this.config.rows)>this.device.limits.maxComputeWorkgroupsPerDimension)throw Error('Too many sparse rows');
    const fields=this.meta.fields,temporary=[],uniforms=[];
    const tensor=()=>{const t=this.ops.tensor([rows,c]);temporary.push(t);return t;};
    let output;
    try{
      const norm=tensor(),hidden=tensor();output=this.ops.tensor([rows,c]);
      const encoder=this.device.createCommandEncoder();
      const dispatch=(pipeline,buffers,params,groups,label)=>{
        const lease=this.ops.pool.acquire(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),uniform=lease.buffer;uniforms.push(lease);
        this.device.queue.writeBuffer(uniform,0,new Uint32Array(params));
        const entries=[...buffers,uniform].map((buffer,binding)=>({binding,resource:{buffer}}));
        const pass=encoder.beginComputePass(profiler?.pass(label));pass.setPipeline(pipeline);pass.setBindGroup(0,this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries}));pass.dispatchWorkgroups(...groups);pass.end();
      };
      dispatch(this.compiled[0],[features.gpuBuffer,this.weights.gpuBuffer,norm.gpuBuffer],[rows,c,fields['norm1.weight'].offset,fields['norm1.bias'].offset],[Math.min(rows,65535),Math.ceil(rows/65535)],'normSilu');
      // Explicit layouts share bindings, although conv1 does not read skip.
      dispatch(this.compiled[1],[norm.gpuBuffer,neighbors.gpuBuffer,this.weights.gpuBuffer,features.gpuBuffer,hidden.gpuBuffer],[rows,c,fields['conv1.weight'].offset/4,fields['conv1.bias'].offset/4],[Math.ceil(rows/this.config.rows),c/this.config.cols],'conv1');
      dispatch(this.compiled[2],[hidden.gpuBuffer,neighbors.gpuBuffer,this.weights.gpuBuffer,features.gpuBuffer,output.gpuBuffer],[rows,c,fields['conv2.weight'].offset/4,fields['conv2.bias'].offset/4],[Math.ceil(rows/this.config.rows),c/this.config.cols],'conv2');
      this.device.queue.submit([encoder.finish()]);return output;
    }catch(error){output?.dispose();throw error;}
    finally{for(const t of temporary)t.dispose();for(const u of uniforms)u.release();}
  }
  release(){this.weights.dispose();}
}
