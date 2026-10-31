const evenRound=x=>{const f=Math.floor(x),d=x-f;return d===.5?(f%2===0?f:f+1):Math.round(x);};
const key=(i,j,z,shape)=>(i*shape[1]+j)*shape[2]+z;
export function voxelize(raw,H,W) {
  if(raw.length!==H*W*3)throw Error('Unexpected coordinate shape');
  const n=H*W,z=new Int32Array(n);let min=Infinity,max=-Infinity;
  for(let i=0;i<n;i++){z[i]=evenRound(Math.fround(raw[i*3+2]*256));min=Math.min(min,z[i]);max=Math.max(max,z[i]);}
  const shape=[H,W,max-min+1],coords=new Int32Array(n*3);
  for(let i=0;i<n;i++){coords[i*3]=Math.floor(i/W);coords[i*3+1]=i%W;coords[i*3+2]=z[i]-min;}
  return {coords,shape};
}
export function neighbors(coords,shape) {
  // Index by image column, then visit its few depth voxels. This avoids 27
  // boxed-number hash lookups per voxel and needs no dense 3D allocation.
  const n=coords.length/3,width=shape[1],heads=new Int32Array(shape[0]*width),next=new Int32Array(n);
  heads.fill(-1);
  for(let i=0;i<n;i++){
    const column=coords[i*3]*width+coords[i*3+1];next[i]=heads[column];heads[column]=i;
  }
  const result=new Int32Array(n*27);result.fill(n);
  for(let i=0;i<n;i++){
    const y=coords[i*3],x=coords[i*3+1],z=coords[i*3+2];
    for(let dy=-1;dy<=1;dy++){
      const row=y+dy;if(row<0||row>=shape[0])continue;
      for(let dx=-1;dx<=1;dx++){
        const column=x+dx;if(column<0||column>=width)continue;
        const at=i*27+(dy+1)*9+(dx+1)*3;
        for(let j=heads[row*width+column];j!==-1;j=next[j]){
          const dz=coords[j*3+2]-z;
          if(dz>=-1&&dz<=1)result[at+dz+1]=j;
        }
      }
    }
  }return result;
}
// Group children in original row order, preserving the existing FP32 sum order.
export function poolMap(coords,shape) {
  const shape2=shape.map(v=>Math.ceil(v/2)),keys=new Float64Array(coords.length/3);
  for(let i=0;i<keys.length;i++)keys[i]=key(Math.floor(coords[i*3]/2),Math.floor(coords[i*3+1]/2),Math.floor(coords[i*3+2]/2),shape2);
  const sorted=keys.slice().sort();let unique=0;
  for(let i=0;i<sorted.length;i++)if(i===0||sorted[i]!==sorted[i-1])unique++;
  const coords2=new Int32Array(unique*3),counts=new Int32Array(unique),parent=new Int32Array(keys.length),lookup=new Map();
  let group=-1,previous=-1;
  for(let i=0;i<sorted.length;i++){
    const k=sorted[i];
    if(i===0||k!==previous){
      group++;previous=k;lookup.set(k,group);
      const column=Math.floor(k/shape2[2]);
      coords2[group*3]=Math.floor(column/shape2[1]);coords2[group*3+1]=column%shape2[1];coords2[group*3+2]=k%shape2[2];
    }
    counts[group]++;
  }
  for(let i=0;i<keys.length;i++)parent[i]=lookup.get(keys[i]);
  const offsets=new Int32Array(unique+1);
  for(let i=0;i<counts.length;i++)offsets[i+1]=offsets[i]+counts[i];
  const cursor=offsets.slice(),children=new Int32Array(keys.length);
  for(let i=0;i<parent.length;i++)children[cursor[parent[i]]++]=i;
  return {coords:coords2,shape:shape2,parent,offsets,children};
}
