import {gpuGeometry} from './gpu-geometry.js';
import {gpuOps} from './gpu-ops.js';
import {GpuProfiler} from './gpu-profiler.js';
import {project} from './geometry.js';
import {compare,passes} from './comparison.js';
export async function profileGeometry(variant='optimized'){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice({requiredFeatures:['timestamp-query']}),ops=gpuOps(device),geo=gpuGeometry(device),report={started:Date.now(),runs:[],traces:[],userAgent:navigator.userAgent};window.geometryProfile=report;
 try{await geo.prepare();for(const profile of ['fast','detail'])for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
 const base=`/models/profiles/${profile}`,meta=await(await fetch(`${base}/manifest.json`)).json(),[,,height,width]=meta.inputShape;
 const arrays=await Promise.all(['refined-points','mask','metric_scale'].map(async n=>new Float32Array(await(await fetch(`${base}/cases/${photo}/reference-${n}.f32`)).arrayBuffer())));
 const tensors=arrays.map(a=>ops.tensor([a.length],'float32',a));
 try{const expected=project(...arrays.slice(0,2),arrays[2][0],width,height);await geo.project(...tensors,width,height,{variant});
 for(let repeat=0;repeat<10;repeat++){const result=await geo.project(...tensors,width,height,{variant});const metric=compare(result.points,expected.points);if(!passes(metric))throw Error('Geometry profile accuracy failed');report.runs.push({profile,photo,repeat,...geo.lastTimings,metric});}
 const profiler=new GpuProfiler(device);ops.profiler=profiler;try{await geo.project(...tensors,width,height,{variant});report.traces.push({profile,photo,...await profiler.finish()});}finally{ops.profiler=null;profiler.dispose();}
 }finally{for(const t of tensors)t.dispose();}
 }report.complete=true;return report;}finally{ops.pool.clear();device.destroy();report.finished=Date.now();}
}
