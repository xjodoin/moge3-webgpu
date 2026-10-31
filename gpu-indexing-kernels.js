// Column CSR keeps voxels ordered by (y,x,z), avoiding packed 64-bit keys.
const params='@group(0) @binding(PARAM) var<uniform> p:array<vec4<u32>,2>;';
export const indexingKernels={
 quantize:`
@group(0) @binding(0) var<storage,read> raw:array<f32>;
@group(0) @binding(1) var<storage,read_write> z:array<i32>;
@group(0) @binding(2) var<storage,read_write> minimum:atomic<i32>;
${params.replace('PARAM','3')}
var<workgroup> mins:array<i32,256>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 var q=2147483647;
 if(id.x<p[0].x){let v=raw[id.x*3u+2u]*256.0;let f=i32(floor(v));let d=v-f32(f);
 q=f;if(d>0.5||(d==0.5&&(f&1)!=0)){q=f+1;}z[id.x]=q;}
 mins[lane]=q;workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){mins[lane]=min(mins[lane],mins[lane+stride]);}workgroupBarrier();}
 if(lane==0u){atomicMin(&minimum,mins[0]);}
}`,
 initial:`
@group(0) @binding(0) var<storage,read> z:array<i32>;
@group(0) @binding(1) var<storage,read> minimum:array<i32>;
@group(0) @binding(2) var<storage,read_write> coords:array<i32>;
@group(0) @binding(3) var<storage,read_write> columns:array<u32>;
${params.replace('PARAM','4')}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=p[0].x){return;}
 coords[i*3u]=i32(i/p[0].z);coords[i*3u+1u]=i32(i%p[0].z);coords[i*3u+2u]=z[i]-minimum[0];columns[i]=i;
 if(i==p[0].x-1u){columns[p[0].x]=p[0].x;}
}`,
 neighbors:`
@group(0) @binding(0) var<storage,read> coords:array<i32>;
@group(0) @binding(1) var<storage,read> columns:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
${params.replace('PARAM','3')}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p[0].x*27u){return;}let row=i/27u;let k=i%27u;
 let y=coords[row*3u]+i32(k/9u)-1;let x=coords[row*3u+1u]+i32(k/3u%3u)-1;let z=coords[row*3u+2u]+i32(k%3u)-1;
 var found=p[0].x;
 if(y>=0&&y<i32(p[0].y)&&x>=0&&x<i32(p[0].z)){
 let col=u32(y)*p[0].z+u32(x);var lo=columns[col];let end=columns[col+1u];var hi=end;
 while(lo<hi){let mid=(lo+hi)/2u;if(coords[mid*3u+2u]<z){lo=mid+1u;}else{hi=mid;}}
 if(lo<end){if(coords[lo*3u+2u]==z){found=lo;}}
 }output[i]=found;
}`,
 scan:`
@group(0) @binding(0) var<storage,read> input:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@group(0) @binding(2) var<storage,read_write> sums:array<u32>;
${params.replace('PARAM','3')}
var<workgroup> values:array<u32,256>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 var v=0u;if(id.x<p[0].x){v=input[id.x];}values[lane]=v;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){var extra=0u;if(lane>=stride){extra=values[lane-stride];}workgroupBarrier();values[lane]+=extra;workgroupBarrier();}
 if(id.x<p[0].x){output[id.x]=values[lane]-v;}
 if(lane==255u){sums[group.x]=values[255];if(group.x==p[0].y-1u){output[p[0].x]=values[255];}}
}`,
 addScan:`
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
@group(0) @binding(1) var<storage,read> blocks:array<u32>;
${params.replace('PARAM','2')}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<p[0].x){output[i]+=blocks[i/256u];}else if(i==p[0].x){output[i]=blocks[p[0].y];}}
`,
 encoderIndex:`
@group(0) @binding(0) var<storage,read> coords:array<i32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
${params.replace('PARAM','2')}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<p[0].x){output[i]=u32(coords[i*3u])*p[0].z+u32(coords[i*3u+1u]);}}
`,
};
const mergeInput=`
@group(0) @binding(0) var<storage,read> coords:array<i32>;
@group(0) @binding(1) var<storage,read> columns:array<u32>;
fn range(column:u32,k:u32)->vec2<u32>{
 let y=(column/p[0].w)*2u+k/2u;let x=(column%p[0].w)*2u+k%2u;
 if(y>=p[0].y||x>=p[0].z){return vec2<u32>(0u);}
 let c=y*p[0].z+x;return vec2<u32>(columns[c],columns[c+1u]);
}`;
function merge(fill){return mergeInput+(fill?`
@group(0) @binding(2) var<storage,read> outColumns:array<u32>;
@group(0) @binding(3) var<storage,read_write> outCoords:array<i32>;
@group(0) @binding(4) var<storage,read_write> parent:array<u32>;
@group(0) @binding(5) var<storage,read_write> counts:array<u32>;
@group(0) @binding(6) var<storage,read_write> children:array<u32>;
${params.replace('PARAM','7')}`:`
@group(0) @binding(2) var<storage,read_write> counts:array<u32>;
${params.replace('PARAM','3')}`)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let column=id.x;if(column>=p[0].x){return;}var cursor:array<u32,4>;var ends:array<u32,4>;
 for(var k=0u;k<4u;k++){let r=range(column,k);cursor[k]=r.x;ends[k]=r.y;}
 var unique=0u;
 loop{
  var z=2147483647;
  for(var k=0u;k<4u;k++){if(cursor[k]<ends[k]){z=min(z,coords[cursor[k]*3u+2u]/2);}}
  if(z==2147483647){break;}
  ${fill?`let row=outColumns[column]+unique;outCoords[row*3u]=i32(column/p[0].w);outCoords[row*3u+1u]=i32(column%p[0].w);outCoords[row*3u+2u]=z;var count=0u;`:''}
  for(var k=0u;k<4u;k++){
   loop{if(cursor[k]>=ends[k]){break;}if(coords[cursor[k]*3u+2u]/2!=z){break;}
    ${fill?`parent[cursor[k]]=row;children[row*8u+count]=cursor[k];count++;`:''}
    cursor[k]++;
   }
  }
  ${fill?'counts[row]=count;':''}unique++;
 }
 ${fill?'':'counts[column]=unique;'}
}`;}
indexingKernels.count=merge(false);indexingKernels.fill=merge(true);
