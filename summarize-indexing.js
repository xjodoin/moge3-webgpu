import {readFileSync,writeFileSync} from 'node:fs';
const report=JSON.parse(readFileSync('benchmarks/index-browser.json','utf8'));
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
const expected=new Set(['fast','detail'].flatMap(p=>['sample-backyard','brick-ranch','raised-deck'].flatMap(c=>[1,2,3].flatMap(s=>[0,1,2].map(r=>`${p}/${c}/${s}/${r}`)))));
for(const indexing of ['cpu','gpu']){
 const rows=report.runs.filter(r=>r.indexing===indexing);
 if(rows.length!==54||new Set(rows.map(signature)).size!==54||rows.some(r=>!expected.has(signature(r)))||!rows.every(r=>r.passed&&r.original.passed&&!r.gpuProfile&&r.sessions.indexingBackend===indexing&&r.sessions.sparseVariant==='optimized'))throw Error('All 54 unprofiled CUDA comparisons per backend must pass');
 if(indexing==='gpu'&&!rows.every(r=>r.transfers.indexUploadBytes===0&&r.transfers.indexShapeReadbackBytes===16*r.refineSteps&&r.transfers.indexShapeReadbacks===4*r.refineSteps))throw Error('GPU index arrays must stay resident; only scalar sizes may return');
}
const parity=JSON.parse(readFileSync('benchmarks/gpu-indexing-parity.json','utf8'));
if(!parity.done||parity.rows.length!==23||!parity.rows.every(r=>r.exact&&r.levels===5))throw Error('Exact map and ordered pooling parity must pass');
const power=readFileSync('benchmarks/index-power.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.time*1000>=report.startedAt&&r.time*1000<=report.finishedAt);
if(!power.length||!power.every(r=>r.ac)||power[0].time*1000-report.startedAt>20000||report.finishedAt-power.at(-1).time*1000>20000||power.some((r,i)=>i&&(r.time-power[i-1].time)>30))throw Error('AC coverage is incomplete');
const profiles=[];
for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
 const groups=['cpu','gpu'].map(indexing=>report.runs.filter(r=>r.profile===profile&&r.refineSteps===steps&&r.indexing===indexing));
 const stats=groups.map(rows=>({computeMs:median(rows.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs)),refinementMs:median(rows.map(r=>r.refinementMs)),cpuCoordinateMs:median(rows.map(r=>r.transfers.coordinateMs)),gpuIndexPreparationWallMs:median(rows.map(r=>r.transfers.gpuIndexingPreparationMs??0)),indexUploadBytes:median(rows.map(r=>r.transfers.indexUploadBytes)),indexParameterUploadBytes:median(rows.map(r=>r.transfers.indexParameterUploadBytes??0)),shapeReadbackBytes:median(rows.map(r=>r.transfers.indexShapeReadbackBytes??0)),peakTrackedGpuBytes:Math.max(...rows.map(r=>r.gpuMemory.peakExplicitBufferBytes)),worstPointsNrmse:Math.max(...rows.map(r=>r.original.points.nrmse)),maskMismatches:Math.max(...rows.map(r=>r.original.maskMismatches))}));
 const [cpu,gpu]=stats;profiles.push({profile,refineSteps:steps,cpu,gpu,latencyReductionPercent:100*(1-gpu.computeMs/cpu.computeMs),refinementReductionPercent:100*(1-gpu.refinementMs/cpu.refinementMs),maxNrmseChange:Math.max(...groups[1].map(r=>Math.abs(r.original.points.nrmse-groups[0].find(b=>signature(b)===signature(r)).original.points.nrmse)))});
}
for(const profile of ['fast','detail'])for(const indexing of ['cpu','gpu']){
 const traces=report.profiles.filter(r=>r.profile===profile&&r.indexing===indexing);
 if(traces.length!==3||!traces.every(r=>r.passed&&r.gpuProfile.gpu.length===(indexing==='cpu'?117:207)))throw Error('Incomplete GPU profiling');
}
if(report.unloaded.retained!==0||report.poolAfterUnload.length)throw Error('GPU resources must unload');
const result={date:new Date().toISOString(),device:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',powerSource:'AC verified throughout run',powerSamples:power.length,method:'Same production build; CPU vs GPU voxel indexing with optimized wide kernels and buffer reuse enabled. Order counterbalanced by resolution, warmup plus three repeats per photo/configuration. Compute excludes loading, validation and rendering; profiling runs separate. Index byte counts are logical payloads, excluding GPU allocation padding and driver granularity. GPU index preparation wall time includes CPU dispatch and scalar sizing fences. CPU final depth update, interpolation and camera geometry remain.',exactMapLevels:115,profiles,passed:true};
writeFileSync('benchmarks/index-summary.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
