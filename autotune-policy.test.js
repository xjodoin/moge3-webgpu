import test from 'node:test';import assert from 'node:assert/strict';
import {rowBucket,candidatesFor,selectWinner,DecisionCache,AUTOTUNE_VERSION} from './autotune-policy.js';
import {subgroupBroadcastConvolution} from './subgroup-broadcast.js';
test('candidate sets are bounded and retain a valid baseline for every width',()=>{for(const c of [32,64,128,256,512])for(const groups of [false,true]){const rows=candidatesFor(c,groups,32);assert.equal(rows[0].id,'baseline');assert.ok(rows.length<=4);assert.ok(rows.every(r=>c%r.config.cols===0));if(!groups)assert.ok(rows.every(r=>r.config.kind!=='subgroup-broadcast'));}assert.equal(rowBucket(301056),524288);assert.throws(()=>rowBucket(0));});
test('selection rejects noisy, tiny, incomplete and inaccurate improvements',()=>{
 const baseline={id:'baseline',exact:true,samples:[1,1,1,1,1]};
 assert.equal(selectWinner([baseline,{id:'fast',exact:true,samples:[.7,.71,.7,.71,.7]}]),'fast');
 for(const row of [{id:'tiny',exact:true,samples:[.98,.98,.98,.98,.98]},{id:'bad',exact:false,samples:[.5,.5,.5,.5,.5]},{id:'short',exact:true,samples:[.5]},{id:'noisy',exact:true,samples:[.3,.5,.7,.9,1.1]},{id:'zero',exact:true,samples:[0,0,0,0,0]}])assert.equal(selectWinner([baseline,row]),'baseline');
});
test('persistent choices are bounded, allowlisted and expire; storage failure is optional',()=>{
 let raw=null,now=100000000;const storage={getItem:()=>raw,setItem:(_,v)=>raw=v,removeItem:()=>raw=null};const a='a'.repeat(64),b='b'.repeat(64);
 const cache=new DecisionCache(storage,{limit:1,now:()=>now});cache.set(a,'one');cache.set(b,'two');assert.equal(cache.get(a,['one']),null);
 assert.equal(new DecisionCache(storage,{now:()=>now}).get(b,['two']),'two');assert.equal(cache.get(b,['other']),null);
 now+=31*86400000;assert.equal(new DecisionCache(storage,{now:()=>now}).get(b,['two']),null);
 const failed=new DecisionCache({getItem(){throw Error('blocked')},setItem(){throw Error('quota')}},{now:()=>now});failed.set(a,'baseline');assert.equal(failed.get(a,['baseline']),'baseline');assert.equal(failed.unavailable,true);
 raw=JSON.stringify([{key:a,id:'unknown',time:now}]);assert.equal(new DecisionCache(storage,{now:()=>now}).get(a,['baseline']),null);assert.ok(AUTOTUNE_VERSION);
});
test('narrow subgroup shaders use matching scalar/vector weight layouts',()=>{for(const cols of [32,64,128]){const code=subgroupBroadcastConvolution(false,{rows:8,cols});const type=cols===32?'f32':`vec${cols/32}<f32>`;assert.ok(code.includes(`weights:array<${type}>`));assert.ok(!code.includes('${'));}assert.throws(()=>subgroupBroadcastConvolution(false,{cols:96}));});
