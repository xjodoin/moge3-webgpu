import {testGeometry} from './benchmark-geometry.browser.js';
import {gpuGeometry} from './gpu-geometry.js';
import {gpuOps} from './gpu-ops.js';
export async function geometryLayoutParity(){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice(),ops=gpuOps(device),geo=gpuGeometry(device),report=[];
 try{await geo.prepare();for(const profile of ['fast','detail'])for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
 const base=`/models/profiles/${profile}`,meta=await(await fetch(`${base}/manifest.json`)).json(),[,,height,width]=meta.inputShape;
 const arrays=await Promise.all(['refined-points','mask','metric_scale'].map(async n=>new Float32Array(await(await fetch(`${base}/cases/${photo}/reference-${n}.f32`)).arrayBuffer()))),tensors=arrays.map(a=>ops.tensor([a.length],'float32',a));
 try{const before=await geo.project(...tensors,width,height,{variant:'baseline'}),after=await geo.project(...tensors,width,height,{variant:'optimized'});
 const keys=['points','depth','mask','intrinsics'];const equal=keys.every(key=>before[key].length===after[key].length&&before[key].every((value,i)=>Object.is(value,after[key][i])));
 // A subsequent pooled readback must never mutate a previously returned result.
 const snapshot=after.points.slice();await geo.project(...tensors,width,height);const stable=snapshot.every((v,i)=>Object.is(v,after.points[i]));
 report.push({profile,photo,equal,stable,readbackMaps:geo.lastTimings.readbackMaps});if(!equal||!stable)throw Error('Geometry output layout parity failed');
 }finally{tensors.forEach(t=>t.dispose());}
 }return report;}finally{ops.pool.clear();device.destroy();}
}
export async function benchmarkGeometryTuning(){
 const report={started:Date.now(),userAgent:navigator.userAgent,runs:[],stages:await testGeometry(),layout:await geometryLayoutParity(),complete:false};window.geometryTuning=report;
 const modes=[{mode:'cpu',postprocess:'cpu'},{mode:'previous-gpu',postprocess:'gpu',geometryVariant:'baseline'},{mode:'optimized-gpu',postprocess:'gpu',geometryVariant:'optimized'}];
 try{for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
 for(const config of modes)await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true,{...config,render:false});
 for(let repeat=0;repeat<3;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck'])for(let offset=0;offset<3;offset++){
 const config={...modes[(repeat+offset)%3],render:false};window.geometryTuningProgress={profile,steps,repeat,photo,mode:config.mode};
 const result=await window.runTest(true,photo,null,profile,'wgsl',steps,true,config);
 report.runs.push({repeat,mode:config.mode,...result,computeMs:result.inferenceMs+result.refinementMs+result.cameraMs});if(!result.passed)throw Error(`Accuracy failure: ${profile}/${steps}/${photo}/${config.mode}`);
 }
 }report.complete=true;return report;}finally{report.finished=Date.now();await window.releaseModels();report.pool=window.bufferPoolStats();}
}

// Accuracy is established separately. Reuse exact fixture inputs without loading
// diagnostic/reference arrays between timed runs, to avoid their allocation load.
export async function benchmarkGeometryTiming(){
 const report={started:Date.now(),userAgent:navigator.userAgent,runs:[],inputs:[],complete:false,validationDuringTiming:false};window.geometryTiming=report;
 const modes=[{mode:'cpu',postprocess:'cpu',render:false},{mode:'previous-gpu',postprocess:'gpu',geometryVariant:'baseline',render:false},{mode:'optimized-gpu',postprocess:'gpu',geometryVariant:'optimized',render:false}];
 try{for(const profile of ['fast','detail']){
 const inputs={};for(const photo of ['sample-backyard','brick-ranch','raised-deck']){const bytes=await(await fetch(`/models/profiles/${profile}/cases/${photo}/input.f32`)).arrayBuffer();inputs[photo]=new Float32Array(bytes);report.inputs.push({profile,photo,sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('')});}
 for(const steps of [1,2,3]){
 for(const config of modes)for(let warmup=0;warmup<2;warmup++)await window.runTest(true,'sample-backyard',inputs['sample-backyard'],profile,'wgsl',steps,false,config);
 for(let repeat=0;repeat<3;repeat++)for(const photo of Object.keys(inputs))for(let offset=0;offset<3;offset++){
 const config=modes[(repeat+offset)%3];window.geometryTimingProgress={profile,steps,repeat,photo,mode:config.mode};
 const r=await window.runTest(true,photo,inputs[photo],profile,'wgsl',steps,false,config);
 if(r.validated||r.original||Object.keys(r.outputs).length)throw Error('Validation unexpectedly entered timing loop');
 report.runs.push({profile,steps,repeat,photo,mode:config.mode,computeMs:r.inferenceMs+r.refinementMs+r.cameraMs,geometryTimings:r.geometryTimings,transfers:r.transfers});
 }
 }
 }report.complete=true;return report;}finally{await window.releaseModels();report.pool=window.bufferPoolStats();report.finished=Date.now();}
}
