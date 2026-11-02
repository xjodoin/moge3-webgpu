// Run on a production build explicitly selecting package format 3.
// Clear only the dedicated, regenerable model cache when cold=true.
export async function benchmarkProgressive(mode='compact',repeats=3,cold=false){
 if(!window.modelDownloadStats().progressive)throw Error('Progressive package required');
 if(cold)await window.clearModelCache();
 await window.setWeightMode(mode);
 const report={mode,userAgent:navigator.userAgent,started:Date.now(),before:window.modelDownloadStats(),runs:[],warmups:[]};
 window.progressiveReport=report;
 for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
  window.progressiveProgress={mode,profile,steps,warmup:true,runs:report.runs.length};
  report.warmups.push(await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true));
  for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
   window.progressiveProgress={mode,profile,steps,repeat,photo,runs:report.runs.length};
   const row=await window.runTest(true,photo,null,profile,'wgsl',steps,true);
   report.runs.push({repeat,...row});
   if(!row.passed)throw Error(`${mode} accuracy failed: ${profile}/${steps}/${photo}`);
  }
 }
 report.after=window.modelDownloadStats();report.unloaded=await window.releaseModels();report.pool=window.bufferPoolStats();report.finished=Date.now();return report;
}
