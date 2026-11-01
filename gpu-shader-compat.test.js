import test from 'node:test';
import assert from 'node:assert/strict';
import {unifyIndexArrayAliases,installIndexArrayCompatibility} from './gpu-shader-compat.js';

test('unifies equivalent ORT high-rank input/output aliases without changing operations',()=>{
 const code=`alias input_indices_t = array<u32, 5>;
alias output_0_indices_t = array<u32, 5>;
alias output_1_indices_t = array<u32, 5>;
alias output_2_indices_t = array<u32, 5>;
fn write_buffer_data(indices: output_0_indices_t) {
 let copy = output_1_indices_t(1u,2u,3u,4u,5u);
 set_output_2_by_indices(indices);
}`;
 const actual=unifyIndexArrayAliases(code);
 assert.equal((actual.match(/alias /g)??[]).length,1);
 assert.ok(actual.includes('indices: input_indices_t'));
 assert.ok(actual.includes('let copy = input_indices_t(1u,2u,3u,4u,5u)'));
 assert.ok(actual.includes('set_output_2_by_indices(indices)'));
 assert.equal(unifyIndexArrayAliases(actual),actual);
});

test('keeps different ranks, vector aliases and unrelated declarations intact',()=>{
 const code=`alias input_indices_t=array<u32,5>;
alias output_0_indices_t=array<u32,6>;
alias output_1_indices_t = array< u32 , 6 >;
alias other_indices_t=array<u32,6>;
alias output_2_indices_t=vec4<u32>;
fn use(a:output_1_indices_t,b:other_indices_t,c:output_2_indices_t) {}`;
 const actual=unifyIndexArrayAliases(code);
 assert.ok(actual.includes('alias output_0_indices_t=array<u32,6>;'));
 assert.ok(actual.includes('alias other_indices_t=array<u32,6>;'));
 assert.ok(actual.includes('alias output_2_indices_t=vec4<u32>;'));
 assert.ok(actual.includes('a:output_0_indices_t,b:other_indices_t,c:output_2_indices_t'));
 const unrelated='@compute @workgroup_size(1) fn main() {}';
 assert.equal(unifyIndexArrayAliases(unrelated),unrelated);
});

test('also unifies Concat multiple-input/single-output aliases',()=>{
 const code=`alias input_0_indices_t=array<u32,5>;
alias input_1_indices_t=array<u32,5>;
alias output_indices_t=array<u32,5>;
fn copy(indices:input_1_indices_t)->output_indices_t {return indices;}`;
 assert.equal(unifyIndexArrayAliases(code),`alias input_0_indices_t=array<u32,5>;


fn copy(indices:input_0_indices_t)->input_0_indices_t {return indices;}`);
});

test('installs once, forwards native receiver/options and leaves descriptor untouched',()=>{
 const calls=[];
 const prototype={createShaderModule(descriptor){calls.push({receiver:this,descriptor});return descriptor;}};
 const device=Object.create(prototype),state=installIndexArrayCompatibility(prototype);
 assert.equal(installIndexArrayCompatibility(prototype),state);
 const code='alias input_indices_t=array<u32,5>; alias output_0_indices_t=array<u32,5>;';
 const original={code,label:'Split',compilationHints:[]};
 device.createShaderModule(original);
 assert.equal(original.code,code);
 assert.equal(calls[0].receiver,device);
 assert.equal(calls[0].descriptor.label,'Split');
 assert.equal(calls[0].descriptor.compilationHints,original.compilationHints);
 assert.equal(state.snapshot().shaderModulesPatched,1);
 const unchanged={code:'fn helper() {}'};
 device.createShaderModule(unchanged);
 assert.equal(calls[1].descriptor,unchanged);
 assert.equal(state.snapshot().shaderModulesPatched,1);
});
