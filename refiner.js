// MoGe-3 sparse UNet: CPU coordinate maps, GPU-resident learned feature tensors.
import * as ort from 'onnxruntime-web/webgpu';
import {sparseSession,directSparseBlock} from './runtime.js';
import {gpuOps} from './gpu-ops.js';
import {voxelize,neighbors,poolMap} from './sparse-maps.js';
import {GpuVoxelMaps} from './gpu-indexing.js';
const channels=[32,64,128,256,512];

export async function refine(raw,encoder,log=()=>{},options={}) {
  const device=await ort.env.webgpu.device,ops=gpuOps(device),owned=new Set();
  const stats=options.telemetry??{};
  stats.featureReadbacks=0;stats.featureReadbackBytes=0;stats.indexUploadBytes=0;stats.coordinateMs=0;
  stats.sparseKernelBlocks=0;
  const useGpu=options.indexingBackend!=='cpu',builder=useGpu?new GpuVoxelMaps(ops,stats):null;
  const keep=t=>{owned.add(t);return t;};
  const drop=t=>{if(owned.delete(t))t.dispose();else builder?.drop(t);};
  const upload=(array,dims,type='int32')=>{
    if(type==='int32')stats.indexUploadBytes+=array.byteLength;
    return keep(ops.tensor(dims,type,array));
  };
  const mapTensor=(array,width=1)=>upload(array,[array.length/width,width]);
  async function run(name,feeds) {
    const session=await sparseSession(name);
    let result;
    try{result=options.profiler?await options.profiler.measure(name,()=>session.run(feeds)):await session.run(feeds);}
    catch(error){throw new Error(`Sparse ${name} failed: ${error.message}`,{cause:error});}
    const output=keep(result.output);
    if(output.location!=='gpu-buffer')throw Error(`${name} returned a non-GPU feature tensor`);
    return output;
  }
  const linear=(name,features)=>run(name,{features});
  const block=async(name,features,indices)=>{
    if(options.sparseBackend==='onnx')return run(name,{features,neighbors:indices});
    if(options.profiler)options.profiler.prefix=`pass${options.pass}/${name}/`;
    const result=keep((await directSparseBlock(name)).run(features,indices,options.profiler));
    if(options.profiler)options.profiler.prefix=`pass${options.pass}/`;
    stats.sparseKernelBlocks++;return result;
  };
  const [H,W]=options.grid??[272,272],n=H*W,encoderW=W/16;
  let t=performance.now(),coords,shape;
  if(!useGpu){({coords,shape}=voxelize(raw,H,W));stats.coordinateMs+=performance.now()-t;}
  const skips=[],maps=[],parents=[];
  try {
    const rawGpu=upload(raw,[n,3],'float32');
    let plan;
    if(useGpu){t=performance.now();plan=await builder.build(rawGpu,H,W);stats.gpuIndexingPreparationMs=performance.now()-t;}
    let data=await linear('input_proj',rawGpu);drop(rawGpu);
    for(let level=0;level<5;level++) {
      log(`Refinement: encoder level ${level+1}/5 (${useGpu?plan.sizes[level]:coords.length/3} voxels)`);
      let map;
      if(useGpu)map=plan.maps[level];else{t=performance.now();map=mapTensor(neighbors(coords,shape),27);stats.coordinateMs+=performance.now()-t;}
      maps.push(map);
      const next=await block(`down_stages.${level}.0`,data,map);drop(data);data=next;
      if(level<4) {
        skips.push(data);
        let pooled;
        if(useGpu){
          const p=plan.pools[level];parents.push(p.parent);
          pooled=keep(ops.run('pool8',[data,p.childCounts,p.children],p.level.n,channels[level]));
          drop(p.childCounts);drop(p.children);
        }else{
          t=performance.now();const p=poolMap(coords,shape);stats.coordinateMs+=performance.now()-t;
          parents.push(mapTensor(p.parent));
          const offsets=mapTensor(p.offsets),children=mapTensor(p.children);
          pooled=keep(ops.run('pool',[data,offsets,children],p.coords.length/3,channels[level]));
          drop(offsets);drop(children);coords=p.coords;shape=p.shape;
        }
        data=await linear(`downsample_blocks.${level}.linear`,pooled);drop(pooled);
      }
    }
    let indexTensor,indexCount;
    if(useGpu){indexTensor=plan.indices;indexCount=plan.sizes[4];}
    else{
      const indices=new Int32Array(coords.length/3);
      for(let i=0;i<indices.length;i++)indices[i]=coords[i*3]*encoderW+coords[i*3+1];
      indexTensor=mapTensor(indices);indexCount=indices.length;
    }
    const encoderTensor=ArrayBuffer.isView(encoder)
      ?keep(ops.tensor([encoder.length/1026,1026],'float32',encoder))
      :encoder;
    const sampled=keep(ops.run('gather',[encoderTensor,indexTensor],indexCount,1026));
    drop(indexTensor);drop(encoderTensor);
    let next=await run('fuse',{features:data,encoder:sampled});drop(data);drop(sampled);data=next;
    next=await block('bottleneck_stage.0',data,maps[4]);drop(data);drop(maps[4]);data=next;
    for(let i=0;i<4;i++) {
      const level=3-i,c=channels[level],skip=skips[level],parent=parents[level];
      log(`Refinement: decoder level ${i+1}/4`);
      const projected=await linear(`upsample_blocks.${i}.linear`,data);drop(data);
      data=keep(ops.run('upsampleAdd',[projected,parent,skip],skip.dims[0],c));
      drop(projected);drop(parent);drop(skip);
      next=await block(`up_stages.${i}.0`,data,maps[level]);drop(data);drop(maps[level]);data=next;
    }
    const deltaTensor=await linear('out_proj',data);drop(data);
    // The only feature readback: final depth correction needed by CPU geometry.
    const delta=await deltaTensor.getData();stats.featureReadbacks++;stats.featureReadbackBytes+=delta.byteLength;
    const result=new Float32Array(raw);
    for(let i=0;i<n;i++)result[i*3+2]+=delta[i];
    return result;
  } finally {for(const tensor of owned)tensor.dispose();builder?.dispose();}
}

// PyTorch align_corners=False bilinear interpolation, followed by exp remap.
export function remap(raw,options={}){
 const [H,W]=options.grid??[272,272],height=options.height??256,width=options.width??256;
 const out=new Float32Array(height*width*3),sy=H/height,sx=W/width;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const fy=Math.max(0,Math.min(H-1,(y+.5)*sy-.5)),fx=Math.max(0,Math.min(W-1,(x+.5)*sx-.5));
  const y0=Math.floor(fy),x0=Math.floor(fx),y1=Math.min(H-1,y0+1),x1=Math.min(W-1,x0+1),a=fy-y0,b=fx-x0,v=[];
  for(let c=0;c<3;c++)v[c]=(raw[(y0*W+x0)*3+c]*(1-b)+raw[(y0*W+x1)*3+c]*b)*(1-a)+(raw[(y1*W+x0)*3+c]*(1-b)+raw[(y1*W+x1)*3+c]*b)*a;
  const z=Math.exp(v[2]),at=(y*width+x)*3;out[at]=v[0]*z;out[at+1]=v[1]*z;out[at+2]=z;
 }return out;
}
