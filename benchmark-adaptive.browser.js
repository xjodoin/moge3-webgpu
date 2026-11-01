export async function benchmarkAdaptive(repeats=3){
 if(!window.modelDownloadStats().packaged)throw Error('Adaptive package must be enabled');
 await window.clearModelCache();await window.setPrecision('fp32');
 performance.setResourceTimingBufferSize(10000);performance.clearResourceTimings();
 const report={startedAt:Date.now(),userAgent:navigator.userAgent,packageSha256:window.modelDownloadStats().packageSha256,
  cacheEmptyAtStart:!await caches.has('moge3-model-chunks-v1'),runs:[],warmups:[]};
 window.adaptiveReport=report;
 for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
  window.adaptiveProgress={profile,steps,warmup:true,runs:report.runs.length};
  report.warmups.push(await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true));
  for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
   window.adaptiveProgress={profile,steps,repeat,photo,runs:report.runs.length};
   report.runs.push({repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
  }
 }
 report.unloaded={sessions:await window.releaseModels(),buffers:window.bufferPoolStats()};
 report.downloads=window.modelDownloadStats();
 report.resources=performance.getEntriesByType('resource').filter(r=>r.name.includes('/models/packed')).map(r=>({name:new URL(r.name).pathname,encodedBodySize:r.encodedBodySize,transferSize:r.transferSize,responseStatus:r.responseStatus,durationMs:r.duration}));
 report.finishedAt=Date.now();return report;
}
