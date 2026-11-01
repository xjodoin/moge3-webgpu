import test from 'node:test';
import assert from 'node:assert/strict';
import {halfWeightShader} from './fp16-weights.js';
import {normSilu,sparseConvolution,tunedSparseConvolution} from './sparse-kernels.js';
import {subgroupBroadcastConvolution} from './subgroup-broadcast.js';
import {validateSparseWeights} from './sparse-layout.js';

test('half storage widens every weight load and leaves FMA/feature math FP32',()=>{
 for(const source of [normSilu,sparseConvolution(false),tunedSparseConvolution(true),subgroupBroadcastConvolution(false)]){
  const result=halfWeightShader(source);
  assert.match(result,/enable f16;/);
  assert.match(result,/weights:array<(vec4<)?f16/);
  const stripped=result.replace(/(?:vec4<f32>|f32)\(weights\[[^\]]+\]\)/g,'wideLoad');
  assert.doesNotMatch(stripped,/weights\[/);
  assert.match(result,/input:array<f32>/);
 }
 assert.throws(()=>halfWeightShader('invalid'),/declaration/);
});
test('half manifests require half-sized payload and retain exact layout validation',()=>{
 const c=32,fields={};let offset=0;
 for(const [name,length] of [['norm1.weight',c],['norm1.bias',c],['conv1.weight',27*c*c],['conv1.bias',c],['conv2.weight',27*c*c],['conv2.bias',c]]){fields[name]={offset,length};offset+=length;}
 const meta={channels:c,fields,bytes:offset*2,sha256:'a'.repeat(64)};
 assert.doesNotThrow(()=>validateSparseWeights(meta,'float16-le'));
 assert.throws(()=>validateSparseWeights(meta,'float32-le'),/length/);
 assert.throws(()=>validateSparseWeights({...meta,bytes:meta.bytes*2},'float16-le'),/length/);
 assert.throws(()=>validateSparseWeights(meta,'int8'),/dtype/);
});
