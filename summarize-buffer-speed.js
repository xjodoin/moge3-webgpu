import {readFileSync,writeFileSync} from 'node:fs';
const suffix=process.argv[2]??'';
if(!['','-ac'].includes(suffix))throw Error('Unknown benchmark suffix');
const report=JSON.parse(readFileSync(`benchmarks/buffer-browser${suffix}.json`,'utf8'));
let power;
if(suffix==='-ac'){
 power=readFileSync('benchmarks/buffer-ac-power.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.time*1000>=report.startedAt&&r.time*1000<=report.finishedAt);
 if(!power.length||!power.every(r=>r.ac))throw Error('AC power must be verified throughout the measurement');
 if(power[0].time*1000-report.startedAt>20000||report.finishedAt-power.at(-1).time*1000>20000||power.some((r,i)=>i>0&&(r.time-power[i-1].time)>30))throw Error('Power log has a coverage gap');
}
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
for(const reuse of [false,true]){
 const rows=report.runs.filter(r=>r.reuse===reuse);
 if(rows.length!==54||new Set(rows.map(signature)).size!==54||!rows.every(r=>r.passed&&r.original.passed&&!r.gpuProfile))throw Error('All 54 unprofiled CUDA comparisons per mode must pass');
}
const profiles=[],gpuLayers=[];
for(const profile of ['fast','detail']){
 for(const steps of [1,2,3]){
  const groups=[false,true].map(reuse=>report.runs.filter(r=>r.profile===profile&&r.refineSteps===steps&&r.reuse===reuse));
  const stats=groups.map(rows=>({computeMs:median(rows.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs)),refinementMs:median(rows.map(r=>r.refinementMs)),
   createdBuffers:median(rows.map(r=>r.buffers.after.reduce((s,x)=>s+x.created,0)-r.buffers.before.reduce((s,x)=>s+x.created,0))),
   reusedBuffers:median(rows.map(r=>r.buffers.after.reduce((s,x)=>s+x.hits,0)-r.buffers.before.reduce((s,x)=>s+x.hits,0))),
   peakTrackedGpuBytes:Math.max(...rows.map(r=>r.gpuMemory.peakExplicitBufferBytes)),
   worstPointsNrmse:Math.max(...rows.map(r=>r.original.points.nrmse)),maskMismatches:Math.max(...rows.map(r=>r.original.maskMismatches))}));
  const [disabled,enabled]=stats;
  const maxNrmseChangeAcrossReuseModes=Math.max(...groups[1].map(r=>Math.abs(r.original.points.nrmse-groups[0].find(x=>signature(x)===signature(r)).original.points.nrmse)));
  profiles.push({profile,refineSteps:steps,disabled,enabled,maxNrmseChangeAcrossReuseModes,latencyReductionPercent:100*(1-enabled.computeMs/disabled.computeMs)});
 }
 const traces=report.profiles.filter(r=>r.profile===profile);
 if(traces.length!==3||!traces.every(r=>r.passed&&r.gpuProfile.gpu.length===117&&r.gpuProfile.linear.length===33))throw Error('Incomplete per-layer profiling');
 const keys=[...new Set(traces[0].gpuProfile.gpu.map(e=>e.name.replace(/^pass\d+\//,'').replace(/\/(normSilu|conv1|conv2)$/,'')))];
 for(const layer of keys){
  const totals=traces.map(r=>r.gpuProfile.gpu.filter(e=>e.name.replace(/^pass\d+\//,'').replace(/\/(normSilu|conv1|conv2)$/,'')===layer).reduce((s,e)=>s+e.gpuMs,0));
  gpuLayers.push({profile,layer,medianGpuMsAcrossThreePasses:median(totals)});
 }
}
const result={powerSource:suffix==='-ac'?'AC verified throughout run':'Unverified; user reported battery operation',powerSamples:power?.length,date:new Date().toISOString(),device:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',method:'Same production build; reuse disabled and enabled; group order counterbalanced by resolution. One warmup per configuration, three repeats per photo. Compute excludes loading, validation and rendering. Timestamp-profiling runs are separate and deliberately synchronized; GPU timestamps measure shader execution, ONNX linear timings include CPU dispatch and GPU completion. Baseline disabled retains the same capacity rounding to isolate reuse.',performanceDecision:'Reuse enabled by user preference for reduced allocation churn; measured latency does not improve consistently and retained buffers increase memory.',defaultReuse:true,profiles,gpuLayers:gpuLayers.sort((a,b)=>b.medianGpuMsAcrossThreePasses-a.medianGpuMsAcrossThreePasses),passed:true};
writeFileSync(`benchmarks/buffer-summary${suffix}.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
