// Direct, tiled implicit GEMM. No [voxel, 27, channel] gather allocation.
import {subgroupBroadcastConvolution} from './subgroup-broadcast.js';
// All invocations reach both tile barriers, including partial row tiles.
export const normSilu = `
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<uniform> p:vec4<u32>;
var<workgroup> sums:array<f32,64>;
var<workgroup> average:f32;
var<workgroup> inverseStd:f32;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let row=group.x+group.y*65535u;
  if(row>=p.x){return;}
  var sum=0.0;
  for(var c=lane;c<p.y;c+=64u){sum+=input[row*p.y+c];}
  sums[lane]=sum;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];}workgroupBarrier();}
  if(lane==0u){average=sums[0]/f32(p.y);}workgroupBarrier();
  var variance=0.0;
  for(var c=lane;c<p.y;c+=64u){let d=input[row*p.y+c]-average;variance+=d*d;}
  sums[lane]=variance;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];}workgroupBarrier();}
  if(lane==0u){inverseStd=inverseSqrt(sums[0]/f32(p.y)+0.000001);}workgroupBarrier();
  for(var c=lane;c<p.y;c+=64u){
    let value=((input[row*p.y+c]-average)*inverseStd)*weights[p.z+c]+weights[p.w+c];
    output[row*p.y+c]=value/(1.0+exp(-value));
  }
}`;

export function sparseConvolution(residual,tileK=32) {
  if(![32,64].includes(tileK))throw Error('Unsupported sparse tile size');
  return `
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> neighbors:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> skip:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p:vec4<u32>;
var<workgroup> tileA:array<f32,${16*tileK}>;
var<workgroup> tileB:array<vec4<f32>,${tileK*8}>;
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
  let lane=local.y*8u+local.x;let startRow=group.x*16u;let column4=group.y*8u+local.x;
  let c4=p.y/4u;let channelShift=firstLeadingBit(p.y);let channelMask=p.y-1u;
  var acc0=vec4<f32>(0.0);var acc1=vec4<f32>(0.0);
  for(var base=0u;base<27u*p.y;base+=${tileK}u){
    for(var t=lane;t<${16*tileK}u;t+=64u){
      let row=startRow+t/${tileK}u;let k=base+t%${tileK}u;var value=0.0;
      if(row<p.x && k<27u*p.y){let neighbor=neighbors[row*27u+(k>>channelShift)];if(neighbor<p.x){value=input[neighbor*p.y+(k&channelMask)];}}
      tileA[t]=value;
    }
    for(var t=lane;t<${tileK*8}u;t+=64u){
      let k=base+t/8u;var value=vec4<f32>(0.0);
      if(k<27u*p.y){value=weights[p.z+k*c4+group.y*8u+t%8u];}
      tileB[t]=value;
    }
    workgroupBarrier();
    for(var k=0u;k<${tileK}u;k++){
      let b=tileB[k*8u+local.x];
      acc0=fma(vec4<f32>(tileA[local.y*${tileK}u+k]),b,acc0);
      acc1=fma(vec4<f32>(tileA[(local.y+8u)*${tileK}u+k]),b,acc1);
    }
    workgroupBarrier();
  }
  let bias=weights[p.w+column4];let row0=startRow+local.y;let row1=row0+8u;
  acc0+=bias;acc1+=bias;
  if(row0<p.x){output[row0*c4+column4]=${residual?'acc0+skip[row0*c4+column4]':'acc0/(vec4<f32>(1.0)+exp(-acc0))'};}
  if(row1<p.x){output[row1*c4+column4]=${residual?'acc1+skip[row1*c4+column4]':'acc1/(vec4<f32>(1.0)+exp(-acc1))'};}
}`;
}

// Every lane follows the same barriers. Optional neighborhood skipping only
// removes FMAs whose input is the missing-neighbor zero sentinel.
export function tunedSparseConvolution(residual,{rows=16,cols=32,k=32,skip='none'}={}){
 if(![8,16,32].includes(rows)||![32,64].includes(cols)||![32,64].includes(k)||!['none','group','row','mask'].includes(skip))throw Error('Invalid sparse kernel configuration');
 const nr=rows/8,nc=cols/32,c4=cols/4;
 const declarations=[],accumulate=[],stores=[];
 for(let r=0;r<nr;r++)for(let c=0;c<nc;c++){
  const a=`acc${r}_${c}`,row=`(startRow+local.y+${r*8}u)`,column=`(group.y*${c4}u+local.x+${c*8}u)`;
  declarations.push(`var ${a}=vec4<f32>(0.0);`);
  accumulate.push(`${skip==='row'?`if(valid${r}){`:''}${a}=fma(vec4<f32>(tileA[(local.y+${r*8}u)*${k}u+t]),tileB[t*${c4}u+local.x+${c*8}u],${a});${skip==='row'?'}':''}`);
  stores.push(`if(${row}<p.x){let index=${row}*channels4+${column};let value=${a}+weights[p.w+${column}];output[index]=${residual?'value+skipInput[index]':'value/(vec4<f32>(1.0)+exp(-value))'};}`);
 }
 return `
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> neighbors:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> skipInput:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p:vec4<u32>;
var<workgroup> tileA:array<f32,${rows*k}>;
var<workgroup> tileB:array<vec4<f32>,${k*c4}>;
${['group','mask'].includes(skip)?'var<workgroup> occupied:atomic<u32>;var<workgroup> shouldRun:u32;':''}
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let lane=local.y*8u+local.x;let startRow=group.x*${rows}u;let channels4=p.y/4u;
 ${declarations.join('\n')}
 ${skip==='mask'?`if(lane==0u){atomicStore(&occupied,0u);}workgroupBarrier();
 var bits=0u;
 for(var t=lane;t<${rows*27}u;t+=64u){let row=startRow+t/27u;if(row<p.x){if(neighbors[row*27u+t%27u]<p.x){bits|=1u<<(t%27u);}}}
 atomicOr(&occupied,bits);workgroupBarrier();if(lane==0u){shouldRun=atomicLoad(&occupied);}
 let groupMask=workgroupUniformLoad(&shouldRun);`:''}
 for(var neighborIndex=0u;neighborIndex<27u;neighborIndex++){
 ${skip==='mask'?'if((groupMask&(1u<<neighborIndex))==0u){continue;}':''}
 ${skip==='group'?`if(lane==0u){atomicStore(&occupied,0u);}workgroupBarrier();
 if(lane<${rows}u&&startRow+lane<p.x){if(neighbors[(startRow+lane)*27u+neighborIndex]<p.x){atomicOr(&occupied,1u);}}
 workgroupBarrier();if(lane==0u){shouldRun=atomicLoad(&occupied);}if(workgroupUniformLoad(&shouldRun)==0u){continue;}`:''}
 ${skip==='row'?Array.from({length:nr},(_,r)=>`var valid${r}=false;if(startRow+local.y+${r*8}u<p.x){valid${r}=neighbors[(startRow+local.y+${r*8}u)*27u+neighborIndex]<p.x;}`).join('\n'):''}
 for(var base=0u;base<p.y;base+=${k}u){
  for(var t=lane;t<${rows*k}u;t+=64u){
   let row=startRow+t/${k}u;var value=0.0;
   if(row<p.x){let neighbor=neighbors[row*27u+neighborIndex];if(neighbor<p.x){value=input[neighbor*p.y+base+t%${k}u];}}
   tileA[t]=value;
  }
  for(var t=lane;t<${k*c4}u;t+=64u){tileB[t]=weights[p.z+(neighborIndex*p.y+base+t/${c4}u)*channels4+group.y*${c4}u+t%${c4}u];}
  workgroupBarrier();
  for(var t=0u;t<${k}u;t++){${accumulate.join('\n')}}
  workgroupBarrier();
 }
 }
 ${stores.join('\n')}
}`;
}
export function sparseConfig(value=32){
 if(typeof value==='number'){
  if(![32,64].includes(value))throw Error('Invalid baseline tile size');
  return {baseline:true,rows:16,cols:32,k:value,skip:'none'};
 }
 if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid sparse configuration');
 if(value.kind&&!['subgroup-norm','subgroup-broadcast'].includes(value.kind))throw Error('Unknown sparse kernel kind');
 if(value.kind==='subgroup-norm')return {...sparseConfig(value.convolution),subgroupNorm:true};
 if(value.kind==='subgroup-broadcast'){
  const config={kind:'subgroup-broadcast',rows:8,cols:128,k:32,...value,baseline:false};
  if(config.k!==32)throw Error('Broadcast kernels require 32-channel chunks');
  subgroupBroadcastConvolution(false,config);return config;
 }
 const config={rows:16,cols:32,k:32,skip:'none',...value,baseline:false};
 tunedSparseConvolution(false,config);return config;
}

// Selected using real photo neighbor maps and alternating timestamp trials.
export function selectedSparseConfig(channels,variant='optimized'){
 if(!['baseline','optimized','subgroup'].includes(variant))throw Error('Unknown sparse variant');
 if(variant==='baseline'||channels<128)return 32;
 if(variant==='subgroup')return {kind:'subgroup-broadcast',rows:8,cols:128,k:32};
 return channels===128?{rows:32,cols:64,k:32,skip:'mask'}:{rows:16,cols:32,k:32,skip:'group'};
}
