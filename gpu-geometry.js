import {gpuOps} from './gpu-ops.js';
const common=`@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> c:array<f32>;
@group(0) @binding(3) var<storage,read> d:array<f32>;
@group(0) @binding(4) var<storage,read_write> o:array<f32>;
@group(0) @binding(5) var<uniform> p:vec4<u32>;`;
const shaders={
add:common+`@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=p.x*3u){return;}o[i]=a[i];if(i%3u==2u){o[i]=a[i]+b[i/3u];}}`,
remap:common+`@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=p.z*p.w){return;}let y=i/p.w;let x=i%p.w;
let fy=clamp((f32(y)+.5)*f32(p.x)/f32(p.z)-.5,0.,f32(p.x-1u));let fx=clamp((f32(x)+.5)*f32(p.y)/f32(p.w)-.5,0.,f32(p.y-1u));
let y0=u32(fy);let x0=u32(fx);let y1=min(p.x-1u,y0+1u);let x1=min(p.y-1u,x0+1u);let ay=fy-f32(y0);let ax=fx-f32(x0);var v:vec3<f32>;
for(var k=0u;k<3u;k++){v[k]=(a[(y0*p.y+x0)*3u+k]*(1.-ax)+a[(y0*p.y+x1)*3u+k]*ax)*(1.-ay)+(a[(y1*p.y+x0)*3u+k]*(1.-ax)+a[(y1*p.y+x1)*3u+k]*ax)*ay;}
let z=exp(v.z);o[i*3u]=v.x*z;o[i*3u+1u]=v.y*z;o[i*3u+2u]=z;}`,
camera:common+`
var<private> lane:u32;
var<workgroup> sums:array<vec4<f32>,256>;
var<workgroup> state:vec4<f32>;
var<workgroup> fit:vec4<f32>;
fn reduce(v:vec4<f32>)->vec4<f32>{sums[lane]=v;workgroupBarrier();for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];}workgroupBarrier();}let result=sums[0];workgroupBarrier();return result;}

fn sample(j:u32)->vec4<f32>{let iy=(j/64u)*p.y/64u;let ix=(j%64u)*p.x/64u;let i=iy*p.x+ix;return vec4<f32>(a[i*3u],a[i*3u+1u],a[i*3u+2u],b[i]);}
fn uv(j:u32)->vec2<f32>{let aspect=f32(p.x)/f32(p.y);let sx=aspect/sqrt(1.+aspect*aspect);return (vec2<f32>(f32((j%64u)*p.x/64u)+.5,f32((j/64u)*p.y/64u)+.5)/vec2<f32>(f32(p.x),f32(p.y))*2.-1.)*vec2<f32>(sx,sx/aspect);}
fn evaluate(shift:f32)->vec4<f32>{var sum=vec4<f32>(0.);var corr=vec4<f32>(0.);
for(var j=lane;j<4096u;j+=256u){let s=sample(j);if(s.w<=.5){continue;}let inv=1./(s.z+shift);let xy=s.xy*inv;let q=-xy*inv;let t=uv(j);let term=vec4<f32>(dot(xy,t),dot(xy,xy),dot(q,t),2.*dot(xy,q));let y=term-corr;let next=sum+y;corr=(next-sum)-y;sum=next;}
sum=reduce(sum);let den=max(sum.y,1e-12);let f=sum.x/den;let df=(sum.z*den-sum.x*sum.w)/(den*den);var total=vec3<f32>(0.);var comp=vec3<f32>(0.);
for(var j=lane;j<4096u;j+=256u){let s=sample(j);if(s.w<=.5){continue;}let inv=1./(s.z+shift);let xy=s.xy*inv;let r=f*xy-uv(j);let jac=df*xy-f*xy*inv;let term=vec3<f32>(dot(r,r),dot(jac,r),dot(jac,jac));let y=term-comp;let next=total+y;comp=(next-total)-y;total=next;}return vec4<f32>(reduce(vec4<f32>(total,0.)).xyz,f);}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) local:u32){lane=local;var count=0.;for(var j=lane;j<4096u;j+=256u){if(sample(j).w>.5){count+=1.;}}
let total=reduce(vec4<f32>(count,0.,0.,0.)).x;
let initial=evaluate(0.);
if(lane==0u){state=vec4<f32>(0.,.001,0.,0.);fit=vec4<f32>(0.,0.,0.,1.);if(total>=2.){fit=initial;state.w=1.;}}workgroupBarrier();
for(var iter=0u;iter<30u;iter++){
 if(lane==0u){state.z=-fit.y/max(fit.z*(1.+state.y),1e-30);}workgroupBarrier();
 let trial=evaluate(state.x+state.z);
 if(lane==0u&&state.w>0.){let tiny=abs(state.z)<1e-8*max(abs(state.x),1.);if(trial.x<fit.x){let converged=fit.x-trial.x<=.001*trial.x;state.x+=state.z;fit=trial;state.y/=3.;if(converged||tiny){state.w=0.;}}else{state.y*=5.;if(tiny){state.w=0.;}}}
 workgroupBarrier();
 if(workgroupUniformLoad(&state).w==0.){break;}
}
if(lane==0u){let aspect=f32(p.x)/f32(p.y);let sx=aspect/sqrt(1.+aspect*aspect);o[0]=fit.w/(2.*sx);o[1]=fit.w/(2.*sx/aspect);o[2]=state.x;o[3]=fit.w;}
}
`,
project:common+`@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=p.x*p.y){return;}let z=(a[i*3u+2u]+d[2])*c[0];let valid=b[i]>.5&&z>0.&&(bitcast<u32>(z)&0x7f800000u)!=0x7f800000u;let inf=bitcast<f32>(0x7f800000u|id.y);o[i*4u]=inf;o[i*4u+1u]=inf;o[i*4u+2u]=inf;o[i*4u+3u]=0.;if(valid){o[i*4u]=((f32(i%p.x)+.5)/f32(p.x)-.5)/d[0]*z;o[i*4u+1u]=((f32(i/p.x)+.5)/f32(p.y)-.5)/d[1]*z;o[i*4u+2u]=z;o[i*4u+3u]=1.;}}`
};
// Keep the previous output layout for paired, same-session performance checks.
shaders.projectPlanar=shaders.project.replaceAll('i*4u+3u','p.x*p.y*3u+i').replaceAll('i*4u','i*3u');
const instances=new WeakMap();
export function gpuGeometry(device){if(!instances.has(device))instances.set(device,new Geometry(device));return instances.get(device);}
class Geometry{
 constructor(device){this.device=device;this.pipelines=new Map();}
 async prepare(){
 if(!this.ready)this.ready=Promise.all(Object.entries(shaders).map(async([name,code])=>{const info=await this.device.createShaderModule({code}).getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(`Geometry ${name}: ${errors.map(m=>m.message).join('; ')}`);})).catch(error=>{this.ready=null;throw error;});
 return this.ready;
 }
 run(name,inputs,params,size,groups){
 const ops=gpuOps(this.device);let pipeline=this.pipelines.get(name);
 if(!pipeline){const layout=this.device.createBindGroupLayout({entries:[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===5?'uniform':binding===4?'storage':'read-only-storage'}}))});pipeline=this.device.createComputePipeline({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:this.device.createShaderModule({code:shaders[name]}),entryPoint:'main'}});this.pipelines.set(name,pipeline);}
 const output=ops.tensor([size]);const lease=ops.pool.acquire(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 try{this.device.queue.writeBuffer(lease.buffer,0,new Uint32Array(params));const buffers=[0,1,2,3].map(i=>(inputs[i]??inputs[0]).gpuBuffer);buffers.push(output.gpuBuffer,lease.buffer);const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});const encoder=this.device.createCommandEncoder();const pass=encoder.beginComputePass(ops.profiler?.pass(`geometry/${name}`));pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(groups);pass.end();this.device.queue.submit([encoder.finish()]);return output;}catch(e){output.dispose();throw e;}finally{lease.release();}
 }
 add(raw,delta,n){return this.run('add',[raw,delta],[n,0,0,0],n*3,Math.ceil(n*3/256));}
 remap(raw,{grid:[h,w],height,width}){return this.run('remap',[raw],[h,w,height,width],height*width*3,Math.ceil(height*width/256));}
 async readback(camera,projected,n){
 const ops=gpuOps(this.device),bytes=16+n*16;
 const lease=ops.pool.acquire(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),buffer=lease.buffer;
 let mapped=false;
 try{
  const encoder=this.device.createCommandEncoder();
  encoder.copyBufferToBuffer(camera.gpuBuffer,0,buffer,0,16);
  encoder.copyBufferToBuffer(projected.gpuBuffer,0,buffer,16,n*16);
  this.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);mapped=true;
  return new Float32Array(buffer.getMappedRange().slice(0,bytes));
 }finally{if(mapped)buffer.unmap();lease.release();}
 }
 async project(points,mask,scale,width,height,{variant='optimized'}={}){
 if(!['baseline','optimized'].includes(variant))throw Error('Unknown geometry output variant');
 const started=performance.now(),n=width*height,camera=this.run('camera',[points,mask],[width,height,0,0],4,1);let projected;
 try{
  projected=this.run(variant==='baseline'?'project':'projectPlanar',[points,mask,scale,camera],[width,height,0,0],n*4,Math.ceil(n/256));
  const submitted=performance.now();let cam,packed;
  if(variant==='baseline')[cam,packed]=await Promise.all([camera.getData(),projected.getData()]);
  else{packed=await this.readback(camera,projected,n);cam=packed.subarray(0,4);}
  const downloaded=performance.now();
  if(!Number.isFinite(cam[0])||cam[0]<=0||!Number.isFinite(cam[1])||cam[1]<=0||!Number.isFinite(cam[2]))throw Error('GPU camera recovery failed');
  const depth=new Float32Array(n),valid=new Uint8Array(n);let xyz;
  if(variant==='baseline'){
   xyz=new Float32Array(n*3);
   for(let i=0;i<n;i++){xyz.set(packed.subarray(i*4,i*4+3),i*3);depth[i]=packed[i*4+2];valid[i]=packed[i*4+3];}
  }else{
   xyz=packed.subarray(4,4+n*3);
   for(let i=0;i<n;i++){depth[i]=xyz[i*3+2];valid[i]=packed[4+n*3+i];}
  }
  this.lastTimings={variant,enqueueMs:submitted-started,readbackWaitMs:downloaded-submitted,unpackMs:performance.now()-downloaded,totalMs:performance.now()-started,readbackMaps:variant==='baseline'?2:1,readbackBytes:16+n*16};
  return {points:xyz,depth,mask:valid,focal:cam[3],shift:cam[2],intrinsics:[cam[0],0,.5,0,cam[1],.5,0,0,1]};
 }finally{projected?.dispose();camera.dispose();}
 }
}
