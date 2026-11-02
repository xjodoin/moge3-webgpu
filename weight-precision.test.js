import test from 'node:test';
import assert from 'node:assert/strict';
import {roundWeight24,expandWeight24,restoreWeight32} from './weight-precision.js';

test('rounds selected unaligned FP32 spans and restores signs/carries exactly',()=>{
 const words=[0,0x80000000,1,127,128,255,0x3f80007f,0x3f800080,0x3f80ff80,0xbf80ff80,0x7f7ffeff];
 const raw=new Uint8Array(3+words.length*4+2).fill(91),v=new DataView(raw.buffer);
 words.forEach((word,i)=>v.setUint32(3+i*4,word,true));
 const low=new Uint8Array(words.map(w=>w&255));
 roundWeight24(raw,[{offset:3,bytes:words.length*4}]);
 assert.deepEqual([...raw.slice(0,3)],[91,91,91]);assert.deepEqual([...raw.slice(-2)],[91,91]);
 assert.equal(v.getUint32(3+7*4,true),0x3f800100);
 restoreWeight32(raw.subarray(3,-2),low);
 assert.deepEqual(words.map((_,i)=>v.getUint32(3+i*4,true)),words);
});

test('expands independently constructed exponent/mantissa planes',()=>{
 const rounded=[0x3f800000,0xbf810000,0x00000100,0x80000000],n=rounded.length;
 const planes=new Uint8Array(n*3);
 rounded.forEach((w,i)=>{planes[i]=(w>>>23)&255;planes[n+i]=(w>>>8)&255;planes[n*2+i]=((w>>>24)&128)|((w>>>16)&127);});
 const raw=expandWeight24(planes),view=new DataView(raw.buffer);
 assert.deepEqual(rounded.map((_,i)=>view.getUint32(i*4,true)),rounded);
});

test('rejects malformed lengths, spans and nonfinite rounding',()=>{
 assert.throws(()=>expandWeight24(new Uint8Array(2)),/length/);
 assert.throws(()=>restoreWeight32(new Uint8Array(4),new Uint8Array(2)),/length/);
 assert.throws(()=>roundWeight24(new Uint8Array(4),[{offset:0,bytes:8}]),/span/);
 for(const word of [0x7f800000,0x7fc00001,0x7f7fffff]){
  const raw=new Uint8Array(4);new DataView(raw.buffer).setUint32(0,word,true);
  assert.throws(()=>roundWeight24(raw,[{offset:0,bytes:4}]),/Nonfinite/);
 }
});
