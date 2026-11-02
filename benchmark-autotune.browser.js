const photos=['sample-backyard','brick-ranch','raised-deck'];
export async function benchmarkAutotune(timing=false,{photos:cases=photos,referenceScope,repeats=3}={}){
 if(!Number.isInteger(repeats)||repeats<1||repeats>10)throw Error('Invalid repeat count');
 const report={cases,referenceScope,repeats,started:Date.now(),userAgent:navigator.userAgent,timing,runs:[],inputs:[],failures:[],complete:false};window.autotuneReport=report;
 try{for(const profile of ['fast','detail']){
 const inputs={};if(timing)for(const photo of cases){const bytes=await(await fetch(`/models/profiles/${profile}/cases/${photo}/input.f32`)).arrayBuffer();inputs[photo]=new Float32Array(bytes);report.inputs.push({profile,photo,sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('')});}
 for(const steps of [1,2,3]){
 window.setSparseAutotune(true);
 for(const photo of cases)await window.runTest(true,photo,inputs[photo]??null,profile,'wgsl',steps,!timing,{render:false,...(!timing&&referenceScope?{referenceScope}:{})});
 const calibrated=window.sparseAutotuneStats().tuned;
 for(let repeat=0;repeat<repeats;repeat++)for(const photo of cases)for(const enabled of repeat%2?[true,false]:[false,true]){
 window.autotuneProgress={timing,profile,steps,repeat,photo,enabled};window.setSparseAutotune(enabled);
 const r=await window.runTest(true,photo,inputs[photo]??null,profile,'wgsl',steps,!timing,{render:false,...(!timing&&referenceScope?{referenceScope}:{})});
 report.runs.push(timing?{profile,steps,repeat,photo,enabled,computeMs:r.inferenceMs+r.refinementMs+r.cameraMs}:{repeat,enabled,...r,computeMs:r.inferenceMs+r.refinementMs+r.cameraMs});
 if(!timing&&!r.passed)report.failures.push({profile,steps,photo,enabled,repeat,original:r.original});
 if(timing&&(r.validated||r.original||Object.keys(r.outputs).length))throw Error('Validation entered timing loop');
 if(window.sparseAutotuneStats().tuned!==calibrated)throw Error('Unexpected tuning during measured loop');
 }
 }
 }report.complete=true;report.passed=report.failures.length===0;return report;
 }finally{report.stats=window.sparseAutotuneStats();report.finished=Date.now();window.setSparseAutotune(true);await window.releaseModels();report.pool=window.bufferPoolStats();}
}
