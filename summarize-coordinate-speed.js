import {readFileSync,writeFileSync} from 'node:fs';
const read=name=>JSON.parse(readFileSync(`benchmarks/${name}.json`,'utf8'));
const median=a=>{const s=[...a].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const before=read('original-browser'),after=read('coordinate-browser'),maps=read('coordinate-performance');
if(after.runs.length!==54||!after.runs.every(r=>r.passed&&r.original.passed))throw Error('Complete CUDA accuracy matrix must pass');
const signature=r=>`${r.profile}/${r.caseName}/${r.refineSteps}/${r.repeat}`;
if(new Set(after.runs.map(signature)).size!==54||before.runs.some(r=>!after.runs.some(a=>signature(a)===signature(r))))throw Error('Before/after cases must match');
if(maps.length!==18||!maps.every(r=>r.exact))throw Error('Coordinate map equality failed');
const profiles=[];
for(const profile of ['fast','detail'])for(const refineSteps of [1,2,3]){
 const select=r=>r.profile===profile&&r.refineSteps===refineSteps;
 const b=before.runs.filter(select),a=after.runs.filter(select);
 const compute=r=>r.inferenceMs+r.refinementMs+r.cameraMs;
 const beforeMs=median(b.map(compute)),afterMs=median(a.map(compute));
 const metricDrift=Math.max(...a.map(r=>Math.abs(r.original.points.nrmse-b.find(x=>signature(x)===signature(r)).original.points.nrmse)));
 profiles.push({profile,refineSteps,beforeMs,afterMs,speedup:beforeMs/afterMs,reductionPercent:100*(1-afterMs/beforeMs),
  beforeCoordinateMs:median(b.map(r=>r.transfers.coordinateMs)),afterCoordinateMs:median(a.map(r=>r.transfers.coordinateMs)),
  worstOriginalPointsNrmse:Math.max(...a.map(r=>r.original.points.nrmse)),maxPointsNrmseChangeFromBefore:metricDrift,maskMismatches:Math.max(...a.map(r=>r.original.maskMismatches))});
}
const result={date:new Date().toISOString(),hardware:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',
 method:'Production build, local unchanged model files, warmup and three repetitions per photo, original CUDA strict FP32 goldens. Compute includes dense, refinement and camera; excludes loading, validation and rendering. Baseline is the preceding original-browser benchmark on the same hardware, not interleaved A/B trials.',
 accuracy:'All 54 runs pass the original CUDA gate. All 90 map levels across 18 saved refinement inputs match the historical implementation exactly, including pooling child order.',
 profiles,passed:true};
writeFileSync('benchmarks/coordinate-summary.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
