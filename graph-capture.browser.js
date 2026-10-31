// Separate instrumentation proves native replay, outside latency measurements.
export async function testGraphCapture(){
  const report={rows:[],startedAt:Date.now()},original=GPUDevice.prototype.createBindGroup;
  let groups=0;
  GPUDevice.prototype.createBindGroup=function(...args){groups++;return original.apply(this,args);};
  try{
    for(const profile of ['fast','detail'])for(const capture of [false,true]){
      await window.setDenseGraphCapture(capture);
      for(const [i,photo] of ['sample-backyard','brick-ranch','raised-deck','sample-backyard'].entries()){
        groups=0;const result=await window.runTest(false,photo,null,profile,'wgsl',1,false);
        report.rows.push({capture,iteration:i,bindGroups:groups,...result});
        if(!result.passed)throw Error('Dense accuracy failed');
        if(capture&&i>=2&&groups!==0)throw Error('Warm dense runs must replay without creating bind groups');
        if(!capture&&groups===0)throw Error('Uncaptured baseline must dispatch kernels');
      }
    }
    report.unloaded=await window.releaseModels();report.poolAfterUnload=window.bufferPoolStats();
    report.done=true;report.finishedAt=Date.now();return report;
  }finally{GPUDevice.prototype.createBindGroup=original;}
}
