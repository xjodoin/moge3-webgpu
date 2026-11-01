// Separate control: FP32 constant folding alone must not explain FP16 drift.
import * as ort from 'onnxruntime-web/webgpu';
import {compare,passes} from './comparison.js';
const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`HTTP ${r.status}: ${url}`);return r.arrayBuffer();};
export async function validateFp16Folding(){
 await window.releaseModels();
 const runs=[];
 for(const profile of ['fast','detail']){
  const base=`/models/profiles/${profile}`,meta=await (await fetch(`${base}/manifest.json`)).json();
  const session=await ort.InferenceSession.create(`/models/precision/fold-control/${profile}/model.onnx`,{executionProviders:[{name:'webgpu'}],graphOptimizationLevel:'all',extra:{session:{disable_cpu_ep_fallback:'1'}}});
  try{
   for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
    const fixture=`${base}/cases/${photo}`,input=new ort.Tensor('float32',new Float32Array(await read(`${fixture}/input.f32`)),meta.inputShape);
    let outputs;
    try{
     outputs=await session.run({image:input});const comparisons={};
     for(const [name,tensor] of Object.entries(outputs))comparisons[name]=compare(await tensor.getData(),new Float32Array(await read(`${fixture}/reference-${name}.f32`)));
     runs.push({profile,photo,comparisons,passed:Object.values(comparisons).every(r=>passes(r)&&r.nrmse<1e-5)});
    }finally{input.dispose();if(outputs)for(const t of Object.values(outputs))t.dispose();}
   }
  }finally{await session.release();}
 }
 return {cpuLearnedFallback:false,precision:'float32',runs,passed:runs.every(r=>r.passed)};
}
