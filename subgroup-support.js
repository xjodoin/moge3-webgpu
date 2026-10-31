// One 32-invocation workgroup must be a complete subgroup. Probe actual lanes;
// subgroup IDs have no portable relationship to local invocation indices.
const probes=new WeakMap();
export function subgroup32Support(device){
  if(!probes.has(device))probes.set(device,probe(device).catch(error=>({supported:false,reason:String(error)})));
  return probes.get(device);
}
async function probe(device){
  if(!device.features.has('subgroups'))return {supported:false,reason:'ORT device does not enable subgroups'};
  const adapterInfo=device.adapterInfo;
  if(adapterInfo?.subgroupMinSize!==32||adapterInfo?.subgroupMaxSize!==32)return {supported:false,reason:'Adapter must guarantee 32-lane subgroups for every pipeline'};
  const code=`enable subgroups;
    @group(0) @binding(0) var<storage,read_write> output:array<u32>;
    @compute @workgroup_size(32) fn main(@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) size:u32){
      output[local]=size;output[32u+local]=subgroupAdd(1u);output[64u+local]=lane;
    }`;
  const module=device.createShaderModule({code}),info=await module.getCompilationInfo();
  if(info.messages.some(m=>m.type==='error'))return {supported:false,reason:info.messages.filter(m=>m.type==='error').map(m=>m.message).join('\n')};
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  const output=device.createBuffer({size:384,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:384,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);
    pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}));
    pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,384);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    const sizes=[...new Set(data.slice(0,32))],active=[...new Set(data.slice(32,64))],lanes=new Set(data.slice(64));
    const supported=sizes.length===1&&sizes[0]===32&&active.length===1&&active[0]===32&&lanes.size===32&&[...lanes].every(n=>n<32);
    return {supported,sizes,active,subgroupMinSize:adapterInfo.subgroupMinSize,subgroupMaxSize:adapterInfo.subgroupMaxSize,reason:supported?'One complete 32-lane subgroup':'Requires a complete 32-lane subgroup'};
  }finally{output.destroy();read.destroy();}
}
