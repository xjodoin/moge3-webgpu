import test from 'node:test';
import assert from 'node:assert/strict';
import {GpuVoxelMaps} from './gpu-indexing.js';
test('invalid coordinate grids fail before any GPU allocation or dispatch',()=>{
 const builder=new GpuVoxelMaps({device:{}});
 for(const [H,W,type,dims] of [[0,1,'float32',[0,3]],[1,-1,'float32',[1,3]],[1.5,2,'float32',[3,3]],[2,2,'float32',[3,3]],[1,1,'int32',[1,3]]])assert.throws(()=>builder.initial({type,dims},H,W),/Invalid GPU coordinate grid/);
 builder.dispose();
});
