import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const read=file=>JSON.parse(readFileSync(`benchmarks/${file}.json`,'utf8'));
const report=read('subgroup-browser'),native=read('subgroup-mlx'),screen=read('subgroup-broadcast-screen'),blocks=read('subgroup-blocks');
const ui=read('subgroup-ui');
if(!ui.done||!ui.result.passed||!ui.result.original.passed||ui.result.sessions.sparseVariant!=='subgroup'||!ui.result.sessions.subgroupSupport.supported||ui.unloaded.retained!==0||ui.poolAfterUnload.length)throw Error('Fresh default UI run and cleanup must pass');
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
const expected=new Set(['fast','detail'].flatMap(p=>['sample-backyard','brick-ranch','raised-deck'].flatMap(c=>[1,2,3].flatMap(s=>[0,1,2].map(r=>`${p}/${c}/${s}/${r}`)))));
function matrix(rows){
  if(rows.length!==54||new Set(rows.map(signature)).size!==54||rows.some(r=>!expected.has(signature(r)))||!rows.every(r=>r.passed))throw Error('All 54 distinct accuracy comparisons must pass');
}
for(const variant of ['optimized','subgroup']){
  const rows=report.runs.filter(r=>r.variant===variant);matrix(rows);
  if(!rows.every(r=>r.original.passed&&!r.gpuProfile&&r.sessions.sparseVariant===variant&&!r.sessions.denseGraphCapture&&r.sessions.indexingBackend==='gpu'&&r.transfers.indexUploadBytes===0&&r.transfers.indexShapeReadbackBytes===16*r.refineSteps&&r.buffers.after.length&&r.buffers.after.every(p=>p.enabled)))throw Error('Comparison settings must be identical outside sparse kernels');
  if(variant==='subgroup'&&!rows.every(r=>r.sessions.subgroupSupport?.supported&&r.sessions.subgroupSupport.subgroupMinSize===32&&r.sessions.subgroupSupport.subgroupMaxSize===32&&Object.values(r.sessions.sparseConfigs).filter(c=>c.kind==='subgroup-broadcast'&&c.rows===8&&c.cols===128&&c.k===32).length===6))throw Error('All 54 candidate runs must execute the selected subgroup kernels');
}
matrix(native.runs);
if(!native.passed||native.diagnosticTracing||native.returnPerStep||native.revision!=='f5ceca7512506731fb1c50307c313da76aac500f'||native.parameterDtypes.join()!=='mlx.core.float32')throw Error('Native baseline must use pinned FP32 weights without tracing');
for(const row of native.runs){
  const base=`public/models/profiles/${row.profile}`,meta=JSON.parse(readFileSync(`${base}/manifest.json`));
  const hash=createHash('sha256').update(readFileSync(`${base}/cases/${row.caseName}/input.f32`)).digest('hex');
  if(hash!==row.inputSha256||row.numTokens!==meta.numTokens||JSON.stringify(row.inputShape)!==JSON.stringify(meta.inputShape))throw Error('Native/browser input or token mismatch');
  const browser=report.runs.find(r=>signature(r)===signature(row));
  if(browser.numTokens!==row.numTokens||JSON.stringify(browser.inputShape)!==JSON.stringify(row.inputShape))throw Error('Native/browser configuration mismatch');
}
if(!screen.done||screen.rows.length!==12||!screen.rows.every(r=>r.candidates.length===6&&r.candidates.every(c=>c.parity.maxAbs===0&&c.gpuMs.length===5)))throw Error('All 72 isolated candidate/map checks must have zero FP32 numerical error');
if(!blocks.done||blocks.rows.length!==20||!blocks.rows.every(r=>r.passed)||blocks.rows.filter(r=>r.channels>=128).some(r=>r.config.kind!=='subgroup-broadcast'))throw Error('All 20 ONNX partial/empty block checks must pass');
if(report.warmups.length!==36||!report.warmups.every(r=>r.passed&&r.original.passed))throw Error('All warmups must pass');
const power=readFileSync('benchmarks/subgroup-power.jsonl','utf8').trim().split('\n').map(JSON.parse);
function acCoverage(r){
  const samples=power.filter(p=>p.time*1000>=r.startedAt&&p.time*1000<=r.finishedAt);
  if(!samples.length||!samples.every(p=>p.ac)||samples[0].time*1000-r.startedAt>20000||r.finishedAt-samples.at(-1).time*1000>20000||samples.some((p,i)=>i&&(p.time-samples[i-1].time)>30))throw Error('AC coverage is incomplete');
  return samples.length;
}
const browserPowerSamples=acCoverage(report),nativePowerSamples=acCoverage(native),profiles=[],gpuLayers=[];
for(const profile of ['fast','detail']){
  for(const steps of [1,2,3]){
    const groups=['optimized','subgroup'].map(variant=>report.runs.filter(r=>r.profile===profile&&r.refineSteps===steps&&r.variant===variant));
    const stats=groups.map(rows=>({computeMs:median(rows.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs)),denseMs:median(rows.map(r=>r.inferenceMs)),refinementMs:median(rows.map(r=>r.refinementMs)),peakTrackedGpuBytes:Math.max(...rows.map(r=>r.gpuMemory.peakExplicitBufferBytes)),worstPointsNrmse:Math.max(...rows.map(r=>r.original.points.nrmse)),maskMismatches:Math.max(...rows.map(r=>r.original.maskMismatches))}));
    const mlx=native.runs.filter(r=>r.profile===profile&&r.refineSteps===steps);
    const nativeMs=median(mlx.map(r=>r.inferenceMs)),[optimized,subgroup]=stats;
    profiles.push({profile,refineSteps:steps,optimized,subgroup,mlx:{computeMs:nativeMs,peakBytes:Math.max(...mlx.map(r=>r.mlxPeakBytes))},latencyReductionPercent:100*(1-subgroup.computeMs/optimized.computeMs),refinementReductionPercent:100*(1-subgroup.refinementMs/optimized.refinementMs),browserToMlxRatio:subgroup.computeMs/nativeMs,maxNrmseChange:Math.max(...groups[1].map(r=>Math.abs(r.original.points.nrmse-groups[0].find(b=>signature(b)===signature(r)).original.points.nrmse)))});
  }
  for(const variant of ['optimized','subgroup']){
    const traces=report.profiles.filter(r=>r.profile===profile&&r.variant===variant);
    if(traces.length!==3||!traces.every(r=>r.passed&&r.original.passed&&r.gpuProfile.gpu.length===207))throw Error('Separate GPU traces must pass');
    for(const stage of [...new Set(traces[0].gpuProfile.gpu.map(e=>e.name.replace(/^pass\d+\//,'').replace(/\/(normSilu|conv1|conv2)$/,'')))]){
      gpuLayers.push({profile,variant,stage,gpuMs:median(traces.map(r=>r.gpuProfile.gpu.filter(e=>e.name.replace(/^pass\d+\//,'').replace(/\/(normSilu|conv1|conv2)$/,'')===stage).reduce((n,e)=>n+e.gpuMs,0)))});
    }
  }
}
if(report.unloaded.retained!==0||report.poolAfterUnload.length)throw Error('Model resources must unload');
const result={date:new Date().toISOString(),device:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',defaultVariant:'subgroup',fallbackVariant:'optimized',defaultUiCheckPassed:true,powerSource:'AC verified throughout both runs',browserPowerSamples,nativePowerSamples,method:'Native MLX runs completed before browser GPU work, with exact fixture hashes, token counts and 1/2/3 passes; no native tracing or per-step output. Browser A/B uses the same production build, buffer reuse and GPU indexing enabled, graph capture disabled. Three warmups per configuration and three repeats per photo. Mode order counterbalanced by resolution. Profiling separate from latency. Browser timing includes input upload and CPU camera recovery; MLX inputs are preloaded on GPU. Loading, validation and rendering excluded. Memory counters are explicit browser buffers versus MLX allocator peaks, not comparable RSS measurements. Grouped trials on one device.',accuracy:{subgroupComparisons:54,browserComparisons:108,nativeReferenceChecks:54,allMasksMatch:report.runs.every(r=>r.original.maskMismatches===0),worstCudaPointsNrmse:Math.max(...report.runs.map(r=>r.original.points.nrmse)),maxNrmseChange:Math.max(...profiles.map(r=>r.maxNrmseChange))},selectedConfig:{kind:'subgroup-broadcast',rows:8,cols:128,k:32,channels:[128,256,512],narrowLayers:'existing kernels',normalization:'existing 64-lane FP32 tree',requires:'Enabled subgroups, adapter min/max subgroup size 32, verified complete 32-lane workgroup; otherwise existing GPU kernels'},profiles,gpuLayers,passed:true};
writeFileSync('benchmarks/subgroup-summary.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({accuracy:result.accuracy,profiles,browserPowerSamples,nativePowerSamples,passed:true},null,2));
