import {gpuGeometry} from './gpu-geometry.js';
import {gpuOps} from './gpu-ops.js';
import {project} from './geometry.js';
import {remap} from './refiner.js';
import {compare,passes} from './comparison.js';
export async function testGeometry(){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice(),ops=gpuOps(device),geo=gpuGeometry(device);const tensors=[];const keep=t=>{tensors.push(t);return t;};
 const report={checks:[]};
 try{
 await geo.prepare();const width=65,height=65,n=width*height,points=new Float32Array(n*3),mask=new Float32Array(n).fill(1),aspect=width/height,sx=aspect/Math.sqrt(1+aspect*aspect),sy=sx/aspect;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x,z=1+2*(x/width)**2+Math.sin(y/height*Math.PI);points.set([((x+.5)/width*2-1)*sx/1.2*(z+.3),((y+.5)/height*2-1)*sy/1.2*(z+.3),z],i*3);}
 mask[0]=0;points.set([999,999,999],0);
 const pointTensor=keep(ops.tensor([n,3],'float32',points)),scale=keep(ops.tensor([1],'float32',new Float32Array([5])));
 for(const kind of ['known-camera','empty-mask','single-sample']){
 const m=kind==='known-camera'?mask:new Float32Array(n);if(kind==='single-sample')m[0]=1;
 const maskTensor=keep(ops.tensor([n],'float32',m)),actual=await geo.project(pointTensor,maskTensor,scale,width,height),expected=project(points,m,5,width,height);
 const metric=compare(actual.points,expected.points);const camera=compare(new Float32Array(actual.intrinsics),new Float32Array(expected.intrinsics));
 report.checks.push({kind,metric,camera,passed:passes(metric)&&passes(camera)&&actual.mask.every((v,i)=>v===expected.mask[i])});
 }
 const options={grid:[7,9],width:17,height:11},raw=Float32Array.from({length:7*9*3},(_,i)=>Math.sin(i*.1)*.2),delta=Float32Array.from({length:63},(_,i)=>Math.cos(i)*.03),rt=keep(ops.tensor([63,3],'float32',raw)),dt=keep(ops.tensor([63,1],'float32',delta));
 const added=keep(geo.add(rt,dt,63)),cpu=new Float32Array(raw);for(let i=0;i<63;i++)cpu[3*i+2]+=delta[i];
 report.checks.push({kind:'depth-add',passed:(await added.getData()).every((v,i)=>v===cpu[i])});
 const mapped=keep(geo.remap(added,options)),metric=compare(await mapped.getData(),remap(cpu,options));report.checks.push({kind:'odd-grid-remap',metric,passed:passes(metric)});
 if(report.checks.some(r=>!r.passed))throw Error('Geometry stage comparison failed');return report;
 }finally{for(const t of tensors)t.dispose();ops.pool.clear();device.destroy();}
}
export async function benchmarkGeometry(){
 const report={started:Date.now(),userAgent:navigator.userAgent,runs:[],stages:await testGeometry(),complete:false};window.geometryReport=report;
 try{
 for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
 for(const postprocess of ['cpu','gpu'])await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true,{postprocess});
 for(let repeat=0;repeat<3;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck'])for(const postprocess of repeat%2?['gpu','cpu']:['cpu','gpu']){
 window.geometryProgress={profile,steps,repeat,photo,postprocess};const r=await window.runTest(true,photo,null,profile,'wgsl',steps,true,{postprocess});report.runs.push({repeat,...r,computeMs:r.inferenceMs+r.refinementMs+r.cameraMs});if(!r.passed)throw Error(`Accuracy failed: ${profile}/${photo}/${steps}/${postprocess}`);
 }
 }
 report.complete=true;return report;
 }finally{report.finished=Date.now();await window.releaseModels();report.pool=window.bufferPoolStats();}
}
