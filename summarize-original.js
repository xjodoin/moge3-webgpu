import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {compare,passes} from './comparison.js';
const json=path=>JSON.parse(readFileSync(path,'utf8'));
const f32=path=>{const b=readFileSync(path);return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);};
const median=values=>{const s=[...values].sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2;};
const cuda=json('benchmarks/original-cuda.json');
const cudaDefault=json('benchmarks/original-cuda-default.json');
const browser=json('benchmarks/original-browser.json');
const mlxComparisons=[],sparseTf32Comparisons=[];
for(const row of cuda.runs){
 const {profile,caseName,refineSteps:steps}=row;
 const fixture=`public/models/profiles/${profile}/cases/${caseName}`;
 if(createHash('sha256').update(readFileSync(`${fixture}/input.f32`)).digest('hex')!==row.inputSha256)throw Error('CUDA input differs from browser fixture');
 const native=steps===1?fixture:`${fixture}/passes/${steps}`;
 const original=`public/original-cuda/${profile}/${caseName}/${steps}`;
 const points=compare(f32(`${native}/reference-native-points.f32`),f32(`${original}/points.f32`),{boundaryMask:f32(`${fixture}/reference-mask.f32`)});
 const intrinsics=compare(f32(`${native}/reference-native-intrinsics.f32`),f32(`${original}/intrinsics.f32`));
 mlxComparisons.push({profile,caseName,refineSteps:steps,points,intrinsics,passed:passes(points)&&passes(intrinsics)});
 const tf32=compare(f32(`public/original-cuda-default/${profile}/${caseName}/${steps}/points.f32`),f32(`${original}/points.f32`),{boundaryMask:f32(`${fixture}/reference-mask.f32`)});
 sparseTf32Comparisons.push({profile,caseName,refineSteps:steps,points:tf32,passed:passes(tf32)});
}
const profiles=[];
for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
 const select=r=>r.profile===profile&&r.refineSteps===steps;
 const native=cuda.runs.filter(select),web=browser.runs.filter(select);
 const defaultMs=median(cudaDefault.runs.filter(select).flatMap(r=>r.seconds.map(t=>t*1000)));
 const cudaMs=median(native.flatMap(r=>r.seconds.map(t=>t*1000)));
 const browserMs=median(web.map(r=>r.inferenceMs+r.refinementMs+r.cameraMs));
 profiles.push({profile,refineSteps:steps,cudaMedianMs:cudaMs,cudaDefaultSparseTf32MedianMs:defaultMs,browserMedianMs:browserMs,
  browserToCudaRatio:browserMs/cudaMs,passedBrowserRuns:web.filter(r=>r.original.passed).length,
  browserRuns:web.length,worstPointsNrmse:Math.max(...web.map(r=>r.original.points.nrmse)),
  worstIntrinsicsNrmse:Math.max(...web.map(r=>r.original.intrinsics.nrmse)),
  maxMaskMismatches:Math.max(...web.map(r=>r.original.maskMismatches)),
  cudaPeakAllocatedBytes:Math.max(...native.map(r=>r.peakAllocatedBytes))});
}
const result={date:new Date().toISOString(),cudaHardware:cuda.gpu,browserHardware:'Apple M4 Pro 24 GB, Chrome 156 Metal-3',
 methodology:'Same FP32 inputs, weights, tokens and pass counts. One warmup per configuration; three measured repeats per photo. CUDA synchronized wall time includes original infer camera recovery. Browser compute includes dense, refinement and camera recovery; excludes loading, reference validation and rendering. Different hardware: speed ratios are not implementation-only comparisons. MLX references are preserved separately.',
 profiles,mlxComparisons,sparseTf32Comparisons,passed:browser.runs.length===54&&browser.runs.every(r=>r.original.passed)};
writeFileSync('benchmarks/original-summary.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({profiles,passed:result.passed},null,2));
