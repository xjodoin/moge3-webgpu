import test from 'node:test';
import assert from 'node:assert/strict';
import {sparseConfig,selectedSparseConfig} from './sparse-kernels.js';
import {subgroupBroadcastConvolution} from './subgroup-broadcast.js';
import {subgroup32Support} from './subgroup-support.js';
test('invalid subgroup layouts cannot reach GPU dispatch',()=>{
  for(const config of [{kind:'subgroup-broadcast',rows:3},{kind:'subgroup-broadcast',cols:64},{kind:'subgroup-broadcast',k:8}])assert.throws(()=>sparseConfig(config));
  assert.throws(()=>subgroupBroadcastConvolution(false,{rows:7}));
  assert.throws(()=>sparseConfig({kind:'unknown'}));
  assert.equal(sparseConfig(selectedSparseConfig(512,'subgroup')).kind,'subgroup-broadcast');
  assert.equal(selectedSparseConfig(32,'subgroup'),32);
});
test('devices without enabled subgroups use a cached unsupported result without allocating',async()=>{
  const device={features:new Set(),createBuffer(){throw Error('Unexpected allocation');}};
  const a=subgroup32Support(device),b=subgroup32Support(device);
  assert.equal(a,b);assert.equal((await a).supported,false);
});
test('variable subgroup sizes use the existing kernel before allocating',async()=>{
  const device={features:new Set(['subgroups']),adapterInfo:{subgroupMinSize:16,subgroupMaxSize:32},createBuffer(){throw Error('Unexpected allocation');}};
  assert.equal((await subgroup32Support(device)).supported,false);
});
