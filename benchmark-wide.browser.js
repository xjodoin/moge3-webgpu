// Run on AC power. Do not run other GPU benchmarks at the same time.
export async function benchmarkWide(repeats=3){
 const report={runs:[],warmups:[],profiles:[],startedAt:Date.now()};
 await window.setDenseGraphCapture(false);
 await window.setIndexingBackend('cpu');
 await window.setBufferReuse(true);
 for(const profile of ['fast','detail'])for(const variant of(profile==='fast'?['baseline','optimized']:['optimized','baseline'])){
  await window.setSparseVariant(variant);
  await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true});
  for(let repeat=0;repeat<repeats;repeat++)report.profiles.push({variant,repeat,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true})});
  for(const steps of [1,2,3]){
   report.warmups.push({variant,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true)});
   for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck'])report.runs.push({variant,repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
  }
 }
 await window.setSparseVariant('subgroup');report.unloaded=await window.releaseModels();report.poolAfterUnload=window.bufferPoolStats();report.finishedAt=Date.now();await window.setIndexingBackend('gpu');await window.setDenseGraphCapture(false);report.done=true;return report;
}
