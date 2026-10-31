import {readFileSync,writeFileSync} from 'node:fs';
import {selectedSparseConfig} from './sparse-kernels.js';
const report=JSON.parse(readFileSync('benchmarks/wide-browser.json','utf8'));
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
for(const variant of ['baseline','optimized']){
 const rows=report.runs.filter(r=>r.variant===variant);
 if(rows.length!==54||new Set(rows.map(signature)).size!==54||!rows.every(r=>r.passed&&r.original.passed&&!r.gpuProfile&&r.sessions.sparseVariant===variant))throw Error('All 54 unprofiled CUDA comparisons per variant must pass');
}
const power=readFileSync('benchmarks/wide-power.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.time*1000>=report.startedAt&&r.time*1000<=report.finishedAt);
if(!power.length||!power.every(r=>r.ac)||power[0].time*1000-report.startedAt>20000||report.finishedAt-power.at(-1).time*1000>20000||power.some((r,i)=>i&&(r.time-power[i-1].time)>30))throw Error('AC power coverage is incomplete');
const profiles=[],gpuLayers=[];
for(const profile of ['fast','detail']){
 for(const steps of [1,2,3]){
  const groups=['baseline','optimized'].map(variant=>report.runs.filter(r=>r.profile===profile&&r.refineSteps===steps&&r.variant===variant));
  const stats=groups.map(rows=>({computeMs:median(rows.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs)),refinementMs:median(rows.map(r=>r.refinementMs)),peakTrackedGpuBytes:Math.max(...rows.map(r=>r.gpuMemory.peakExplicitBufferBytes)),worstPointsNrmse:Math.max(...rows.map(r=>r.original.points.nrmse)),maskMismatches:Math.max(...rows.map(r=>r.original.maskMismatches))}));
  const [baseline,optimized]=stats;
  const maxNrmseChange=Math.max(...groups[1].map(r=>Math.abs(r.original.points.nrmse-groups[0].find(b=>signature(b)===signature(r)).original.points.nrmse)));
  profiles.push({profile,refineSteps:steps,baseline,optimized,maxNrmseChange,latencyReductionPercent:100*(1-optimized.computeMs/baseline.computeMs),refinementReductionPercent:100*(1-optimized.refinementMs/baseline.refinementMs)});
 }
 const group=name=>name.replace(/^pass\d+\//,'').replace(/\/(normSilu|conv1|conv2)$/,'');
 const keys=[...new Set(report.profiles.find(r=>r.profile===profile).gpuProfile.gpu.map(e=>group(e.name)))];
 for(const layer of keys){
  const times=['baseline','optimized'].map(variant=>{
   const traces=report.profiles.filter(r=>r.profile===profile&&r.variant===variant);
   if(traces.length!==3||!traces.every(r=>r.passed&&r.gpuProfile.gpu.length===117))throw Error('Incomplete layer profiling');
   return median(traces.map(r=>r.gpuProfile.gpu.filter(e=>group(e.name)===layer).reduce((s,e)=>s+e.gpuMs,0)));
  });
  gpuLayers.push({profile,layer,baselineGpuMs:times[0],optimizedGpuMs:times[1],reductionPercent:100*(1-times[1]/times[0])});
 }
}
if(report.unloaded.retained!==0||report.poolAfterUnload.length)throw Error('GPU resources must be unloaded');
const result={date:new Date().toISOString(),device:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',powerSource:'AC verified throughout run',powerSamples:power.length,method:'Same production build, original 16x32 baseline versus selected wide kernels. Both use FP32 and buffer reuse. Mode order counterbalanced across resolutions; warmup plus three repeats per photo/configuration. Profiling runs are separate. GPU times are kernel timestamps, compute excludes loading, validation and rendering. Grouped trials do not establish performance on other GPUs.',selectedConfigs:Object.fromEntries([32,64,128,256,512].map(c=>[c,selectedSparseConfig(c)])),profiles,gpuLayers,passed:true};
writeFileSync('benchmarks/wide-summary.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({profiles,powerSamples:power.length,passed:true},null,2));
