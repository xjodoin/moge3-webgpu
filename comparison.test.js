import test from 'node:test';
import assert from 'node:assert/strict';
import {compare,passes} from './comparison.js';
test('masked infinity is permitted, NaN and changed geometry are rejected',()=>{
 assert.ok(passes(compare([1,Infinity],[1,Infinity])));
 assert.equal(passes(compare([NaN],[1])),false);
 assert.equal(passes(compare([2],[1])),false);
 assert.throws(()=>compare([1],[]),/shape/);
});
test('a single confidence boundary is reported, without hiding invalid geometry',()=>{
 const a=new Float32Array(300000).fill(1),b=new Float32Array(a),mask=new Float32Array(100000).fill(1);
 b.fill(Infinity,0,3);mask[0]=.500002;
 const r=compare(a,b,{boundaryMask:mask});assert.equal(r.maskMismatches,3);assert.ok(passes(r));
 mask[0]=.6;assert.equal(passes(compare(a,b,{boundaryMask:mask})),false);
 mask[0]=.500002;b.fill(Infinity,3,6);mask[1]=.500002;assert.equal(passes(compare(a,b,{boundaryMask:mask})),false);
});
