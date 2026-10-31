// Historical hash-map implementation retained only as a benchmark oracle.
const evenRound=x=>{const f=Math.floor(x),d=x-f;return d===.5?(f%2===0?f:f+1):Math.round(x);};
const key=(i,j,z,shape)=>(i*shape[1]+j)*shape[2]+z;
export function voxelize(raw,H,W) {
  if(raw.length!==H*W*3)throw Error('Unexpected coordinate shape');
  const n=H*W,z=new Int32Array(n);let min=Infinity,max=-Infinity;
  for(let i=0;i<n;i++){z[i]=evenRound(Math.fround(raw[i*3+2]*256));min=Math.min(min,z[i]);max=Math.max(max,z[i]);}
  const shape=[H,W,max-min+1],coords=new Int32Array(n*3);
  for(let i=0;i<n;i++)coords.set([Math.floor(i/W),i%W,z[i]-min],i*3);
  return {coords,shape};
}
export function neighbors(coords,shape) {
  const n=coords.length/3,map=new Map();
  for(let i=0;i<n;i++)map.set(key(coords[i*3],coords[i*3+1],coords[i*3+2],shape),i);
  const result=new Int32Array(n*27);result.fill(n);
  for(let i=0;i<n;i++){const y=coords[i*3],x=coords[i*3+1],z=coords[i*3+2];let k=0;
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)for(let dz=-1;dz<=1;dz++,k++){
      const a=y+dy,b=x+dx,c=z+dz;
      if(a>=0&&a<shape[0]&&b>=0&&b<shape[1]&&c>=0&&c<shape[2])result[i*27+k]=map.get(key(a,b,c,shape))??n;
    }
  }return result;
}
// Group children in original row order, preserving the existing FP32 sum order.
export function poolMap(coords,shape) {
  const shape2=shape.map(v=>Math.ceil(v/2)),keys=new Float64Array(coords.length/3),map=new Map();
  for(let i=0;i<keys.length;i++){
    const v=[Math.floor(coords[i*3]/2),Math.floor(coords[i*3+1]/2),Math.floor(coords[i*3+2]/2)],k=key(v[0],v[1],v[2],shape2);keys[i]=k;map.set(k,v);
  }
  const sorted=[...map.keys()].sort((a,b)=>a-b),lookup=new Map(sorted.map((k,i)=>[k,i]));
  const coords2=new Int32Array(sorted.length*3),counts=new Int32Array(sorted.length),parent=new Int32Array(keys.length);
  sorted.forEach((k,i)=>coords2.set(map.get(k),i*3));
  for(let i=0;i<keys.length;i++){const p=lookup.get(keys[i]);parent[i]=p;counts[p]++;}
  const offsets=new Int32Array(sorted.length+1);
  for(let i=0;i<counts.length;i++)offsets[i+1]=offsets[i]+counts[i];
  const cursor=offsets.slice(),children=new Int32Array(keys.length);
  for(let i=0;i<parent.length;i++)children[cursor[parent[i]]++]=i;
  return {coords:coords2,shape:shape2,parent,offsets,children};
}
