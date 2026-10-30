// MoGe-3 sparse UNet: deterministic coordinate maps on CPU, learned operations
// on WebGPU. Coordinates and kernel order follow Microsoft MoGe / MLX-VLM.
import * as ort from 'onnxruntime-web/webgpu';
const channels=[32,64,128,256,512];
const evenRound=x=>{const f=Math.floor(x),d=x-f;return d===.5?(f%2===0?f:f+1):Math.round(x);};
const key=(i,j,z,shape)=>(i*shape[1]+j)*shape[2]+z;
function neighbors(coords,shape){
 const n=coords.length/3,map=new Map();for(let i=0;i<n;i++)map.set(key(...coords.subarray(i*3,i*3+3),shape),i);
 const result=new Int32Array(n*27);result.fill(n);
 for(let i=0;i<n;i++){const [y,x,z]=coords.subarray(i*3,i*3+3);let k=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)for(let dz=-1;dz<=1;dz++,k++){
   const a=y+dy,b=x+dx,c=z+dz;if(a>=0&&a<shape[0]&&b>=0&&b<shape[1]&&c>=0&&c<shape[2])result[i*27+k]=map.get(key(a,b,c,shape))??n;
  }
 }return result;
}
function pool(feats,coords,shape,c){
 const shape2=shape.map(v=>Math.ceil(v/2)),keys=new Float64Array(coords.length/3),map=new Map();
 for(let i=0;i<keys.length;i++){const v=Array.from(coords.subarray(i*3,i*3+3),v=>Math.floor(v/2)),k=key(...v,shape2);keys[i]=k;map.set(k,v);}
 const sorted=[...map.keys()].sort((a,b)=>a-b),lookup=new Map(sorted.map((k,i)=>[k,i]));
 const coords2=new Int32Array(sorted.length*3),counts=new Int32Array(sorted.length),parent=new Int32Array(keys.length),data=new Float32Array(sorted.length*c);
 sorted.forEach((k,i)=>coords2.set(map.get(k),i*3));
 for(let i=0;i<keys.length;i++){const p=lookup.get(keys[i]);parent[i]=p;counts[p]++;for(let j=0;j<c;j++)data[p*c+j]+=feats[i*c+j];}
 for(let i=0;i<counts.length;i++)for(let j=0;j<c;j++)data[i*c+j]/=counts[i];
 return {data,coords:coords2,shape:shape2,parent};
}
export async function refine(raw,encoder,log=()=>{},options={}){
 const sessions=new Map();
 async function run(name,feeds){
  log(`Loading refiner block ${name}`);
  if(!sessions.has(name))sessions.set(name,await ort.InferenceSession.create(`/models/refiner/${name}.onnx`,{executionProviders:['webgpu'],extra:{session:{disable_cpu_ep_fallback:'1'}}}).catch(e=>{throw Error(`${name}: ${e.message}`)}));
  try{const result=await sessions.get(name).run(feeds);
   try{return new Float32Array(result.output.data);}finally{for(const tensor of Object.values(result))tensor.dispose();}
  }finally{for(const tensor of Object.values(feeds))tensor.dispose();}
 }
 const tensor=(data,c)=>new ort.Tensor('float32',data,[data.length/c,c]);
 const linear=(name,data,c)=>run(name,{features:tensor(data,c)});
 async function block(name,data,c,map){return run(name,{features:tensor(data,c),neighbors:new ort.Tensor('int32',map,[map.length/27,27])});}
 const [H,W]=options.grid??[272,272],encoderW=W/16,n=H*W,z=new Int32Array(n);let min=Infinity,max=-Infinity;
 for(let i=0;i<n;i++){z[i]=evenRound(Math.fround(raw[i*3+2]*256));min=Math.min(min,z[i]);max=Math.max(max,z[i]);}
 let shape=[H,W,max-min+1],coords=new Int32Array(n*3);
 for(let i=0;i<n;i++)coords.set([Math.floor(i/W),i%W,z[i]-min],i*3);
 const skips=[],maps=[],parents=[];
 try{
  let data=await linear('input_proj',raw,3);
  for(let level=0;level<5;level++){
   log(`Refinement: encoder level ${level+1}/5 (${coords.length/3} voxels)`);
   const map=neighbors(coords,shape);maps.push(map);
   data=await block(`down_stages.${level}.0`,data,channels[level],map);
   if(level<4){skips.push({data,coords,shape});const p=pool(data,coords,shape,channels[level]);parents.push(p.parent);coords=p.coords;shape=p.shape;data=await linear(`downsample_blocks.${level}.linear`,p.data,channels[level]);}
  }
  const enc=new Float32Array(coords.length/3*1026);
  for(let i=0;i<coords.length/3;i++){const at=(coords[i*3]*encoderW+coords[i*3+1])*1026;enc.set(encoder.subarray(at,at+1026),i*1026);}
  data=await run('fuse',{features:tensor(data,512),encoder:tensor(enc,1026)});
  data=await block('bottleneck_stage.0',data,512,maps[4]);
  for(let i=0;i<4;i++){
   const level=3-i,c=channels[level],skip=skips[level],parent=parents[level];log(`Refinement: decoder level ${i+1}/4`);
   const projected=await linear(`upsample_blocks.${i}.linear`,data,channels[level+1]);data=new Float32Array(skip.data.length);
   for(let row=0;row<parent.length;row++)for(let j=0;j<c;j++)data[row*c+j]=projected[parent[row]*c+j]+skip.data[row*c+j];
   data=await block(`up_stages.${i}.0`,data,c,maps[level]);
  }
  const delta=await linear('out_proj',data,32),result=new Float32Array(raw);
  for(let i=0;i<n;i++)result[i*3+2]+=delta[i];return result;
 }finally{for(const s of sessions.values())await s.release();}
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
