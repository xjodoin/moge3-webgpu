import test from 'node:test';
import assert from 'node:assert/strict';
import {sparseConfig,selectedSparseConfig,tunedSparseConvolution} from './sparse-kernels.js';
test('baseline stays available and narrow layers retain the validated original kernel',()=>{
 for(const c of [32,64])assert.equal(selectedSparseConfig(c),32);
 for(const c of [128,256,512])assert.equal(selectedSparseConfig(c,'baseline'),32);
 assert.equal(sparseConfig(64).baseline,true);
 assert.equal(selectedSparseConfig(128).cols,64);
 assert.equal(selectedSparseConfig(512).skip,'group');
});
test('unsupported tile shapes and skip modes fail before GPU allocation',()=>{
 for(const config of [null,'bad',[],{rows:7},{cols:48},{k:17},{skip:'unsafe'}])assert.throws(()=>sparseConfig(config));
 assert.throws(()=>selectedSparseConfig(512,'unknown'));
 assert.throws(()=>tunedSparseConvolution(false,{rows:7}));
});
