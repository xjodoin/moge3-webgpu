// Production build, AC power. Other optimized paths are identical in both modes.
export async function benchmarkGraph(repeats=3){
  const report={runs:[],warmups:[],startedAt:Date.now()};
  window.graphBenchmarkReport=report;
  await window.setBufferReuse(true);await window.setSparseVariant('optimized');await window.setIndexingBackend('gpu');
  for(const profile of ['fast','detail'])for(const capture of(profile==='fast'?[false,true]:[true,false])){
    window.graphBenchmarkProgress={profile,capture,phase:'warmup',completed:report.runs.length};
    await window.setDenseGraphCapture(capture);
    // Cover ORT preparation/capture before recording warm replay latency.
    for(let i=0;i<3;i++)report.warmups.push({capture,...await window.runTest(true,'sample-backyard',null,profile,'wgsl',1,true)});
    for(const steps of [1,2,3]){
      for(let repeat=0;repeat<repeats;repeat++)for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
        window.graphBenchmarkProgress={profile,capture,steps,repeat,photo,phase:'measurement',completed:report.runs.length};
        report.runs.push({capture,repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
      }
    }
  }
  await window.setDenseGraphCapture(false);
  await window.setSparseVariant('subgroup');
  report.unloaded=await window.releaseModels();report.poolAfterUnload=window.bufferPoolStats();report.finishedAt=Date.now();report.done=true;return report;
}
