// Profiling adds synchronization overhead and is excluded from latency runs.
export async function benchmarkBuffers(repeats=3){
 const report={runs:[],warmups:[],profiles:[]};
 await window.setDenseGraphCapture(false);
 await window.setSparseVariant('optimized');
 await window.setIndexingBackend('cpu');
 await window.setBufferReuse(true);
 for(const profile of ['fast','detail']){
  await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true});
  for(let repeat=0;repeat<repeats;repeat++)report.profiles.push({repeat,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true})});
 }
 for(const profile of ['fast','detail'])for(const reuse of (profile==='fast'?[false,true]:[true,false])){
  await window.setBufferReuse(reuse);
  for(const steps of [1,2,3]){
   report.warmups.push({reuse,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true)});
   for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck'])report.runs.push({reuse,repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
  }
 }
 await window.setBufferReuse(true);report.unloaded=await window.releaseModels();await window.setIndexingBackend('gpu');await window.setDenseGraphCapture(false);await window.setSparseVariant('subgroup');report.done=true;return report;
}
