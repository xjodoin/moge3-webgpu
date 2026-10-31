import test from 'node:test';
import assert from 'node:assert/strict';
import {validateSparseWeights} from './sparse-layout.js';
function metadata(c=32){
  const fields={};let offset=0;
  for(const [name,length] of [['norm1.weight',c],['norm1.bias',c],['conv1.weight',27*c*c],['conv1.bias',c],['conv2.weight',27*c*c],['conv2.bias',c]]){fields[name]={offset,length};offset+=length;}
  return {channels:c,fields,bytes:offset*4,sha256:'a'.repeat(64)};
}
test('accepts all exported channel widths and rejects incompatible packed weights',()=>{
  for(const c of [32,64,128,256,512])assert.doesNotThrow(()=>validateSparseWeights(metadata(c)));
  assert.throws(()=>validateSparseWeights(metadata(48)),/channel/);
  const badOffset=metadata();badOffset.fields['conv2.weight'].offset+=4;
  assert.throws(()=>validateSparseWeights(badOffset),/layout/);
  assert.throws(()=>validateSparseWeights({...metadata(),bytes:4}),/length/);
  assert.throws(()=>validateSparseWeights({...metadata(),sha256:'invalid'}),/checksum/);
});
