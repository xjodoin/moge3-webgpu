import {AUTOTUNE_VERSION,DecisionCache,candidatesFor,rowBucket,selectWinner} from './autotune-policy.js';
import {normSilu,sparseConvolution,tunedSparseConvolution} from './sparse-kernels.js';
import {subgroupBroadcastConvolution} from './subgroup-broadcast.js';
import {GpuProfiler} from './gpu-profiler.js';
let storage;try{storage=globalThis.localStorage;}catch{}
const decisions=new DecisionCache(storage),trusted=new Set(),records=new Map(),selected={};
let enabled=true;
const trust=key=>{trusted.add(key);while(trusted.size>128)trusted.delete(trusted.values().next().value);};
const record=(key,value)=>{records.set(key,value);while(records.size>128)records.delete(records.keys().next().value);};
const totals={tuned:0,cacheHits:0,revalidated:0,parityFailures:0,fallbacks:0,coldMs:0};
export function setSparseAutotune(value){if(typeof value!=='boolean')throw Error('Autotuning flag must be boolean');enabled=value;}
export function clearSparseAutotune(){decisions.clear();trusted.clear();records.clear();for(const k of Object.keys(selected))delete selected[k];for(const k of Object.keys(totals))totals[k]=0;}
export function sparseAutotuneStats(details=false){return {enabled,...totals,cacheEntries:decisions.entries.size,storageUnavailable:decisions.unavailable,selected:{...selected},...(details?{records:[...records.values()]}:{})};}
const digest=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join('');
const comparisonPipelines=new WeakMap();
async function comparisonPipeline(device){
 if(!comparisonPipelines.has(device))comparisonPipelines.set(device,(async()=>{
  const code=`@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> result:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> count:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=count[0]){return;}let x=bitcast<u32>(a[i]);let y=bitcast<u32>(b[i]);if(x!=y||(x&0x7f800000u)==0x7f800000u||(y&0x7f800000u)==0x7f800000u){atomicOr(&result[0],1u);}}`;
  return device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code}),entryPoint:'main'}});
 })().catch(e=>{comparisonPipelines.delete(device);throw e;}));
 return comparisonPipelines.get(device);
}
export class AutotunedSparseBlock{
 constructor(base,{name,profile,subgroups,onSelect=()=>{}}){Object.assign(this,{base,device:base.device,ops:base.ops,name,profile,onSelect});this.config=base.config;this.meta=base.meta;this.candidates=candidatesFor(base.meta.channels,subgroups,base.config);this.variants=new Map([['baseline',Promise.resolve(base)]]);this.keys=new Map();this.pending=new Map();}
 async key(rows){const bucket=rowBucket(rows);if(!this.keys.has(bucket))this.keys.set(bucket,(async()=>{
  const kernels=this.candidates.map(({config})=>[false,true].map(residual=>config.kind==='subgroup-broadcast'?subgroupBroadcastConvolution(residual,config):config.baseline?sparseConvolution(residual,config.k):tunedSparseConvolution(residual,config)).join('\n')).join('\n');
  const info=this.device.adapterInfo??{},identity={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,subgroupMinSize:info.subgroupMinSize,subgroupMaxSize:info.subgroupMaxSize};
  return digest(JSON.stringify({version:AUTOTUNE_VERSION,name:this.name,profile:this.profile,bucket,channels:this.meta.channels,weights:this.meta.sha256,browser:globalThis.navigator?.userAgent??'',device:identity,features:[...this.device.features].sort(),limits:{workgroup:this.device.limits.maxComputeInvocationsPerWorkgroup,shared:this.device.limits.maxComputeWorkgroupStorageSize,storage:this.device.limits.maxStorageBufferBindingSize},kernels:normSilu+kernels}));
 })());return this.keys.get(bucket);}
 variant(id){if(!this.variants.has(id)){const candidate=this.candidates.find(c=>c.id===id);if(!candidate)throw Error('Unknown tuning candidate');this.variants.set(id,this.base.withConfig(candidate.config).catch(e=>{this.variants.delete(id);throw e;}));}return this.variants.get(id);}
 async same(a,b){
 const n=a.dims.reduce((x,y)=>x*y,1),groups=Math.ceil(n/256);if(groups>this.device.limits.maxComputeWorkgroupsPerDimension)return false;
 const pipeline=await comparisonPipeline(this.device),flag=this.ops.tensor([1],'int32',new Int32Array([0])),count=this.ops.tensor([1],'int32',new Int32Array([n]));
 try{const buffers=[a.gpuBuffer,b.gpuBuffer,flag.gpuBuffer,count.gpuBuffer],encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))}));pass.dispatchWorkgroups(groups);pass.end();this.device.queue.submit([encoder.finish()]);return (await flag.getData())[0]===0;}finally{flag.dispose();count.dispose();}
 }
 async choose(key,features,neighbors){
 const eligible=this.candidates.filter(c=>Math.ceil(features.dims[0]/c.config.rows)<=this.device.limits.maxComputeWorkgroupsPerDimension);
 const allowed=eligible.map(c=>c.id),cached=decisions.get(key,allowed);
 if(cached&&trusted.has(key)){totals.cacheHits++;return cached;}
 if(cached==='baseline'){trust(key);totals.cacheHits++;return cached;}
 if(!this.device.features.has('timestamp-query')){totals.fallbacks++;return 'baseline';}
 const start=performance.now();let reference;
 try{
  reference=this.base.run(features,neighbors);
  if(cached){const block=await this.variant(cached),output=block.run(features,neighbors);let exact;try{exact=await this.same(reference,output);}finally{output.dispose();}
   if(exact){trust(key);totals.revalidated++;return cached;}totals.parityFailures++;decisions.delete(key);
  }
  const rows=[{id:'baseline',config:this.base.config,exact:true,samples:[]}];
  for(const candidate of eligible.slice(1)){
   const row={...candidate,exact:false,samples:[]};rows.push(row);
   try{const block=await this.variant(candidate.id),output=block.run(features,neighbors);try{row.exact=await this.same(reference,output);}finally{output.dispose();}if(!row.exact)totals.parityFailures++;}catch(error){row.error=String(error);}
  }
  const valid=rows.filter(r=>r.exact),profiler=new GpuProfiler(this.device);
  try{for(let repeat=0;repeat<5;repeat++)for(let offset=0;offset<valid.length;offset++){
   const row=valid[(repeat+offset)%valid.length];profiler.prefix=`${row.id}/${repeat}/`;const output=(await this.variant(row.id)).run(features,neighbors,profiler);output.dispose();
  }
  const trace=await profiler.finish();for(const row of valid)for(let repeat=0;repeat<5;repeat++)row.samples.push(trace.gpu.filter(e=>e.name.startsWith(`${row.id}/${repeat}/`)).reduce((s,e)=>s+e.gpuMs,0));
  }finally{profiler.dispose();}
  const winner=selectWinner(rows);decisions.set(key,winner);trust(key);totals.tuned++;
  record(key,{key,layer:this.name,profile:this.profile,rows:features.dims[0],bucket:rowBucket(features.dims[0]),channels:this.meta.channels,winner,candidates:rows});
  return winner;
 }catch(error){totals.fallbacks++;record(key,{key,layer:this.name,error:String(error),winner:'baseline'});decisions.set(key,'baseline');trust(key);return 'baseline';}
 finally{reference?.dispose();totals.coldMs+=performance.now()-start;}
 }
 async run(features,neighbors,profiler){
 if(!enabled||this.base.half||!this.device.features.has('timestamp-query')){this.onSelect(this.base.config);selected[this.name]={profile:this.profile,bucket:rowBucket(features.dims[0]),id:'baseline'};if(enabled&&!this.device.features.has('timestamp-query'))totals.fallbacks++;return this.base.run(features,neighbors,profiler);}
 const key=await this.key(features.dims[0]);let id;
 if(profiler&&!trusted.has(key)){id='baseline';}else{
  if(!this.pending.has(key))this.pending.set(key,this.choose(key,features,neighbors).finally(()=>this.pending.delete(key)));
  id=await this.pending.get(key);
 }
 const block=await this.variant(id);this.onSelect(block.config);selected[this.name]={profile:this.profile,bucket:rowBucket(features.dims[0]),id};return block.run(features,neighbors,profiler);
 }
 async release(){await Promise.allSettled([...this.pending.values()]);const blocks=await Promise.allSettled([...this.variants.values()]);for(const row of blocks)if(row.status==='fulfilled'&&row.value!==this.base)row.value.release();this.variants.clear();this.base.release();}
}
