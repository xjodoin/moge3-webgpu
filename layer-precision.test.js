import test from 'node:test';
import assert from 'node:assert/strict';
import {roundLayerWeights} from './layer-precision.js';
import {roundWeight24} from './weight-precision.js';
test('24-bit probe matches shipped codec including unaligned spans',()=>{
 const raw=new Uint8Array(21),v=new DataView(raw.buffer);
 [1.234,-3.456,0,1e-30,1e30].forEach((n,i)=>v.setFloat32(1+4*i,n,true));
 const spans=[{offset:1,bytes:20}];
 assert.deepEqual(roundLayerWeights(raw.slice(),spans,24),roundWeight24(raw.slice(),spans));
 assert.deepEqual(roundLayerWeights(raw.slice(),spans,32),raw);
});
test('only selected layer changes and low bits clear',()=>{
 const values=new Float32Array([1.234,2.345,3.456]);const raw=new Uint8Array(values.buffer),original=raw.slice();
 roundLayerWeights(raw,[{offset:4,bytes:4}],20);
 assert.deepEqual(raw.slice(0,4),original.slice(0,4));assert.deepEqual(raw.slice(8),original.slice(8));
 assert.equal(new DataView(raw.buffer).getUint32(4,true)&4095,0);
});
test('reject invalid precision, spans and nonfinite weights',()=>{
 const raw=new Uint8Array(new Float32Array([Infinity]).buffer);
 assert.throws(()=>roundLayerWeights(raw,[{offset:0,bytes:4}],20));
 assert.throws(()=>roundLayerWeights(raw,[],15));assert.throws(()=>roundLayerWeights(raw,[{offset:-1,bytes:4}],24));
});
