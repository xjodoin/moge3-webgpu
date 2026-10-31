import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as before from './benchmarks/coordinate-reference.js';
import * as after from './sparse-maps.js';
const rows=[];
for(const profile of ['fast','detail'])for(const photo of ['sample-backyard','brick-ranch','raised-deck'])for(const step of [0,1,2]){
 const m=JSON.parse(fs.readFileSync(`public/models/profiles/${profile}/manifest.json`));const b=fs.readFileSync(`public/original-cuda/${profile}/${photo}/diagnostics/raw-${step}.f32`);const raw=new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);
 const totals={before:0,after:0};let map=before.voxelize(raw,...m.grid);assert.deepEqual(after.voxelize(raw,...m.grid),map);
 for(let level=0;level<5;level++){
  let outputs=[];
  for(const [name,impl] of [['before',before],['after',after]]){const t=performance.now();const neighbors=impl.neighbors(map.coords,map.shape);const pool=level<4?impl.poolMap(map.coords,map.shape):null;totals[name]+=performance.now()-t;outputs.push({neighbors,pool});}
  assert.deepEqual(outputs[1],outputs[0]);if(level<4)map=outputs[0].pool;
 }
 rows.push({profile,photo,step,...totals,exact:true});
}
fs.writeFileSync('benchmarks/coordinate-performance.json',JSON.stringify(rows,null,2)+'\n');
for(const p of ['fast','detail']){const r=rows.filter(x=>x.profile===p);console.log(p,Object.fromEntries(['before','after'].map(k=>[k,r.reduce((s,x)=>s+x[k],0)/r.length])));}
