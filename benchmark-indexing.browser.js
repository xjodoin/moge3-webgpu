// AC power; original CUDA goldens; wide kernels and buffer reuse held enabled.
export async function benchmarkIndexing(repeats=3){
 const report={runs:[],warmups:[],profiles:[],startedAt:Date.now()};
 await window.setDenseGraphCapture(false);
 await window.setBufferReuse(true);await window.setSparseVariant('optimized');
 for(const profile of ['fast','detail'])for(const indexing of(profile==='fast'?['cpu','gpu']:['gpu','cpu'])){
  await window.setIndexingBackend(indexing);
  await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true});
  for(let repeat=0;repeat<repeats;repeat++)report.profiles.push({indexing,repeat,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true})});
  for(const steps of [1,2,3]){
   report.warmups.push({indexing,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true)});
   for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck'])report.runs.push({indexing,repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
  }
 }
 await window.setIndexingBackend('gpu');report.unloaded=await window.releaseModels();report.poolAfterUnload=window.bufferPoolStats();report.finishedAt=Date.now();await window.setDenseGraphCapture(false);await window.setSparseVariant('subgroup');report.done=true;return report;
}
