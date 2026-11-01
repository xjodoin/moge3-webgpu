export async function benchmarkDownloads(repeats=3){
 if(!window.modelDownloadStats().packaged)throw Error('Compressed model package must be enabled');
 performance.setResourceTimingBufferSize(10000);performance.clearResourceTimings();
 await window.clearModelCache();await window.setPrecision('fp32');
 const report={startedAt:Date.now(),userAgent:navigator.userAgent,packageSha256:window.modelDownloadStats().packageSha256,runs:[],warmups:[],unloaded:[]};
 window.downloadReport=report;
 for(const profile of ['fast','detail'])for(const steps of [1,2,3]){
  window.downloadProgress={profile,steps,warmup:true,runs:report.runs.length};
  report.warmups.push(await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true));
  for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
   window.downloadProgress={profile,steps,repeat,photo,runs:report.runs.length};
   report.runs.push({repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
  }
 }
 report.unloaded.push({sessions:await window.releaseModels(),buffers:window.bufferPoolStats()});
 report.downloads=window.modelDownloadStats();
 report.resources=performance.getEntriesByType('resource').filter(r=>r.name.includes('/models/packed/')).map(r=>({name:new URL(r.name).pathname,encodedBodySize:r.encodedBodySize,transferSize:r.transferSize,durationMs:r.duration}));
 report.finishedAt=Date.now();return report;
}
