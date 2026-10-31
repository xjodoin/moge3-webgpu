import test from 'node:test';
import assert from 'node:assert/strict';
import {voxelize,neighbors,poolMap} from './sparse-maps.js';
test('quantization uses round-to-even including negative ties',()=>{
  const raw=new Float32Array([0,0,-.5/256,0,0,.5/256,0,0,1.5/256,0,0,-1.5/256]);
  const {coords}=voxelize(raw,1,4);assert.deepEqual([coords[2],coords[5],coords[8],coords[11]],[2,2,4,0]);
});
test('sparse neighbor maps respect boundaries and missing-voxel sentinel',()=>{
  const coords=new Int32Array([0,0,0,0,1,0]),map=neighbors(coords,[1,2,1]);
  assert.equal(map[13],0);assert.equal(map[16],1);assert.equal(map[0],2);
  assert.equal(map[27+10],0);assert.equal(map[27+13],1);
});
test('pool children retain input summation order with consistent parent mapping',()=>{
  const coords=new Int32Array([0,0,0,0,1,2,0,1,1,1,0,0,1,1,2]);
  const p=poolMap(coords,[2,2,3]);
  assert.deepEqual([...p.parent],[0,1,0,0,1]);
  assert.deepEqual([...p.offsets],[0,3,5]);assert.deepEqual([...p.children],[0,2,3,1,4]);
  assert.deepEqual([...p.coords],[0,0,0,0,0,1]);
});
test('column neighbor indexing agrees with a spatial oracle through odd-sized pooling levels',()=>{
 let state=17;
 const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/2**32);
 let shape=[7,9,13],rows=[];
 for(let y=0;y<shape[0];y++)for(let x=0;x<shape[1];x++)for(let z=0;z<shape[2];z++)if(random()<.18)rows.push(y,x,z);
 let coords=Int32Array.from(rows);
 for(let level=0;level<4;level++){
  const n=coords.length/3,actual=neighbors(coords,shape),expected=new Int32Array(n*27);expected.fill(n);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const dy=coords[j*3]-coords[i*3],dx=coords[j*3+1]-coords[i*3+1],dz=coords[j*3+2]-coords[i*3+2];
   if(Math.abs(dy)<=1&&Math.abs(dx)<=1&&Math.abs(dz)<=1)expected[i*27+(dy+1)*9+(dx+1)*3+dz+1]=j;
  }
  assert.deepEqual(actual,expected);
  const pooled=poolMap(coords,shape);
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)assert.equal(pooled.coords[pooled.parent[i]*3+c],Math.floor(coords[i*3+c]/2));
  for(let p=0;p<pooled.coords.length/3;p++){
   const children=Array.from(pooled.parent.keys()).filter(i=>pooled.parent[i]===p);
   assert.deepEqual([...pooled.children.slice(pooled.offsets[p],pooled.offsets[p+1])],children);
  }
  coords=pooled.coords;shape=pooled.shape;
 }
});
