import {readFileSync,writeFileSync} from 'node:fs';
const report=JSON.parse(readFileSync('benchmarks/graph-browser.json','utf8'));
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
const expected=new Set(['fast','detail'].flatMap(p=>['sample-backyard','brick-ranch','raised-deck'].flatMap(c=>[1,2,3].flatMap(s=>[0,1,2].map(r=>`${p}/${c}/${s}/${r}`)))));
for(const capture of [false,true]){
  const rows=report.runs.filter(r=>r.capture===capture);
  if(rows.length!==54||new Set(rows.map(signature)).size!==54||rows.some(r=>!expected.has(signature(r)))||!rows.every(r=>r.passed&&r.original.passed&&!r.gpuProfile&&r.sessions.denseGraphCapture===capture&&r.sessions.indexingBackend==='gpu'&&r.sessions.sparseVariant==='optimized'))throw Error('All 54 unprofiled CUDA comparisons per capture mode must pass');
  if(capture&&!rows.every(r=>r.sessions.denseGraph?.completedRuns>=4&&r.sessions.denseGraph.bindingBytes>0))throw Error('All capture measurements must use warmed persistent bindings');
}
if(report.warmups.length!==12||!report.warmups.every(r=>r.passed&&r.original.passed))throw Error('All 12 warmup comparisons must pass');
const power=readFileSync('benchmarks/graph-power.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.time*1000>=report.startedAt&&r.time*1000<=report.finishedAt);
if(!power.length||!power.every(r=>r.ac)||power[0].time*1000-report.startedAt>20000||report.finishedAt-power.at(-1).time*1000>20000||power.some((r,i)=>i&&(r.time-power[i-1].time)>30))throw Error('AC coverage is incomplete');
if(report.unloaded.retained!==0||report.poolAfterUnload.length)throw Error('Resources must unload');
const proof=JSON.parse(readFileSync('benchmarks/graph-capture-checks.json','utf8'));
if(!proof.done||proof.rows.length!==16||!proof.rows.every(r=>r.passed)||!proof.rows.filter(r=>r.capture&&r.iteration>=2).every(r=>r.bindGroups===0)||!proof.rows.filter(r=>!r.capture).every(r=>r.bindGroups>0))throw Error('Native graph replay evidence is required');
const profiles=[];
for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
  const groups=[false,true].map(capture=>report.runs.filter(r=>r.profile===profile&&r.refineSteps===steps&&r.capture===capture));
  const stats=groups.map(rows=>({computeMs:median(rows.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs)),denseMs:median(rows.map(r=>r.inferenceMs)),refinementMs:median(rows.map(r=>r.refinementMs)),cameraMs:median(rows.map(r=>r.cameraMs)),peakTrackedGpuBytes:Math.max(...rows.map(r=>r.gpuMemory.peakExplicitBufferBytes)),worstPointsNrmse:Math.max(...rows.map(r=>r.original.points.nrmse)),maskMismatches:Math.max(...rows.map(r=>r.original.maskMismatches))}));
  const [baseline,captured]=stats;
  profiles.push({profile,refineSteps:steps,baseline,captured,latencyReductionPercent:100*(1-captured.computeMs/baseline.computeMs),denseReductionPercent:100*(1-captured.denseMs/baseline.denseMs),maxNrmseChange:Math.max(...groups[1].map(r=>Math.abs(r.original.points.nrmse-groups[0].find(b=>signature(b)===signature(r)).original.points.nrmse)))});
}
const firstRuns=['fast','detail'].flatMap(profile=>[false,true].map(capture=>{
  const r=report.warmups.find(r=>r.profile===profile&&r.capture===capture);
  return {profile,capture,loadMs:r.loadMs,denseMs:r.inferenceMs,refinementMs:r.refinementMs,note:'Single first invocation; includes preparation/capture, not a repeated cold-start study'};
}));
const accuracy={baselineComparisons:54,capturedComparisons:54,total:108,worstPointsNrmse:Math.max(...report.runs.map(r=>r.original.points.nrmse)),maxNrmseChange:Math.max(...profiles.map(r=>r.maxNrmseChange)),allMasksMatch:report.runs.every(r=>r.original.maskMismatches===0)};
const result={date:new Date().toISOString(),device:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',runtime:JSON.parse(readFileSync('package.json')).dependencies['onnxruntime-web'],defaultCapture:false,defaultDecision:'Opt-in: small measured warm gains, extra first-run capture work and approximately 0.32/0.49 GB additional tracked GPU buffers for Fast/Detail.',powerSource:'AC verified throughout run',powerSamples:power.length,method:'Same production build; dense graph capture disabled vs enabled. GPU indexing, optimized wide kernels and buffer reuse enabled in both modes. Three warmup runs per mode/resolution, three repeats per photo/pass configuration, order counterbalanced by resolution. Compute includes input upload and required raw/mask/scale readbacks, excludes loading, validation and rendering. Original independent CUDA FP32 goldens and unchanged tolerance. GPU memory counts explicit buffers, not RSS/driver memory.',accuracy,firstRuns,profiles,passed:true};
writeFileSync('benchmarks/graph-summary.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
