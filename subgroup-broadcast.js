// A subgroup spans 128 output channels. Features are broadcast from registers;
// each thread owns its weight columns, so weights never need cross-lane shuffles.
export function subgroupBroadcastConvolution(residual,{rows=8,cols=128}={}){
  if(![2,4,8,16].includes(rows)||cols!==128)throw Error('Invalid broadcast subgroup tile');
  const declarations=[],neighbors=[],loads=[],accumulate=[],stores=[];
  for(let r=0;r<rows;r++){
    declarations.push(`var acc${r}=vec4<f32>(0.0);`);
    neighbors.push(`let row${r}=startRow+${r}u;var neighbor${r}=p.x;if(row${r}<p.x){neighbor${r}=indices[row${r}*27u+neighborIndex];}occupied=occupied|(neighbor${r}<p.x);`);
    loads.push(`var a${r}=0.0;if(neighbor${r}<p.x){a${r}=input[neighbor${r}*p.y+base+lane];}`);
    stores.push(`if(startRow+${r}u<p.x){let index=(startRow+${r}u)*channels4+column;let value=acc${r}+weights[p.w+column];output[index]=${residual?'value+skipInput[index]':'value/(vec4<f32>(1.0)+exp(-value))'};}`);
  }
  for(let k=0;k<32;k++){
    accumulate.push(`let w${k}=weights[p.z+(neighborIndex*p.y+base+${k}u)*channels4+column];`);
    for(let r=0;r<rows;r++)accumulate.push(`acc${r}=fma(vec4<f32>(subgroupBroadcast(a${r},${k}u)),w${k},acc${r});`);
  }
  return `enable subgroups;
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> skipInput:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p:vec4<u32>;
var<workgroup> occupiedFlag:u32;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_invocation_id) lane:u32){
 let startRow=group.x*${rows}u;let channels4=p.y/4u;let column=group.y*32u+lane;
 ${declarations.join('\n')}
 for(var neighborIndex=0u;neighborIndex<27u;neighborIndex++){
  var occupied=false;${neighbors.join('\n')}
  if(lane==0u){occupiedFlag=select(0u,1u,occupied);}
  if(workgroupUniformLoad(&occupiedFlag)==0u){continue;}
  for(var base=0u;base<p.y;base+=32u){
   ${loads.join('\n')}
   ${accumulate.join('\n')}
  }
 }
 ${stores.join('\n')}
}`;
}

// Reproduce the existing 64-lane tree, including its initial stride-32 pair.
// Broadcast lane zero after each tree so all channels use identical rounding.
export const subgroupNormSilu=`enable subgroups;
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<uniform> p:vec4<u32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(subgroup_invocation_id) lane:u32){
 let row=group.x+group.y*65535u;
 if(row>=p.x){return;}
 var lo=0.0;var hi=0.0;
 for(var c=lane;c<p.y;c+=64u){lo+=input[row*p.y+c];if(c+32u<p.y){hi+=input[row*p.y+c+32u];}}
 var sum=lo+hi;
 sum+=subgroupShuffleXor(sum,16u);sum+=subgroupShuffleXor(sum,8u);sum+=subgroupShuffleXor(sum,4u);sum+=subgroupShuffleXor(sum,2u);sum+=subgroupShuffleXor(sum,1u);
 let average=subgroupBroadcast(sum,0u)/f32(p.y);
 lo=0.0;hi=0.0;
 for(var c=lane;c<p.y;c+=64u){let d=input[row*p.y+c]-average;lo+=d*d;if(c+32u<p.y){let e=input[row*p.y+c+32u]-average;hi+=e*e;}}
 var variance=lo+hi;
 variance+=subgroupShuffleXor(variance,16u);variance+=subgroupShuffleXor(variance,8u);variance+=subgroupShuffleXor(variance,4u);variance+=subgroupShuffleXor(variance,2u);variance+=subgroupShuffleXor(variance,1u);
 let inverseStd=inverseSqrt(subgroupBroadcast(variance,0u)/f32(p.y)+0.000001);
 for(var c=lane;c<p.y;c+=32u){let value=((input[row*p.y+c]-average)*inverseStd)*weights[p.z+c]+weights[p.w+c];output[row*p.y+c]=value/(1.0+exp(-value));}
}`;
