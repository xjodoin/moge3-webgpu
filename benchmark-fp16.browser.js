// All precision modes use the same immutable FP32 input and original CUDA goldens.
export async function benchmarkFp16(repeats=3){
 const report={startedAt:Date.now(),userAgent:navigator.userAgent,runs:[],warmups:[],unloaded:[],errors:[]};
 window.fp16Report=report;
 for(const profile of ['fast','detail']){
  const modes=profile==='fast'?['fp32','fp16-storage','fp16-compute']:['fp16-compute','fp16-storage','fp32'];
  for(const precision of modes){
   await window.setPrecision(precision);
   for(const steps of [1,2,3]){
    for(let warmup=0;warmup<3;warmup++){
     window.fp16Progress={profile,precision,steps,warmup,runs:report.runs.length};
     report.warmups.push({precision,warmup,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true)});
    }
    for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
     window.fp16Progress={profile,precision,steps,repeat,photo,runs:report.runs.length};
     report.runs.push({precision,repeat,startedAt:Date.now(),...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
    }
   }
   report.unloaded.push({profile,precision,sessions:await window.releaseModels(),buffers:window.bufferPoolStats()});
  }
 }
 await window.setPrecision('fp32');report.finishedAt=Date.now();return report;
}
