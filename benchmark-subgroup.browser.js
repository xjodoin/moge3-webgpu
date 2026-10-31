// Same production build, original CUDA reference gates, no graph capture.
export async function benchmarkSubgroup(repeats=3){
  const report={runs:[],warmups:[],profiles:[],startedAt:Date.now()};window.subgroupBenchmarkReport=report;
  await window.setDenseGraphCapture(false);await window.setBufferReuse(true);await window.setIndexingBackend('gpu');
  for(const profile of ['fast','detail'])for(const variant of(profile==='fast'?['optimized','subgroup']:['subgroup','optimized'])){
    window.subgroupBenchmarkProgress={profile,variant,phase:'warmup',completed:report.runs.length};
    await window.setSparseVariant(variant);
    for(const steps of [1,2,3]){
      for(let i=0;i<3;i++)report.warmups.push({variant,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true)});
      for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
        window.subgroupBenchmarkProgress={profile,variant,steps,repeat,photo,completed:report.runs.length};
        report.runs.push({variant,repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
      }
    }
    // Timestamp profiling is separate from latency measurements.
    await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true});
    for(let repeat=0;repeat<3;repeat++)report.profiles.push({variant,repeat,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',3,true,{profileGpu:true})});
  }
  await window.setSparseVariant('subgroup');
  report.unloaded=await window.releaseModels();report.poolAfterUnload=window.bufferPoolStats();report.finishedAt=Date.now();report.done=true;return report;
}
