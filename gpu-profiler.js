// GPU timestamps measure shader execution. Synchronized linear wall timings
// are separate and include ORT dispatch overhead. Never enable during timing runs.
export class GpuProfiler {
  constructor(device){
    this.device=device;this.events=[];this.linear=[];this.prefix='';
    if(!device.features.has('timestamp-query'))throw Error('Per-layer GPU profiling requires timestamp-query on the ORT device');
    this.queries=device.createQuerySet({type:'timestamp',count:512});
  }
  pass(name){
    const i=this.events.length;if(i*2+1>=512)throw Error('GPU profiler query capacity exceeded');
    this.events.push({name:this.prefix+name});
    return {timestampWrites:{querySet:this.queries,beginningOfPassWriteIndex:i*2,endOfPassWriteIndex:i*2+1}};
  }
  async measure(name,fn){
    await this.device.queue.onSubmittedWorkDone();const t=performance.now();
    const result=await fn();await this.device.queue.onSubmittedWorkDone();
    this.linear.push({name:this.prefix+name,synchronizedWallMs:performance.now()-t});return result;
  }
  async finish(){
    const bytes=this.events.length*16;if(!bytes)return {gpu:[],linear:this.linear};
    const resolved=this.device.createBuffer({size:bytes,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
    const read=this.device.createBuffer({size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    try{
      const cmd=this.device.createCommandEncoder();cmd.resolveQuerySet(this.queries,0,this.events.length*2,resolved,0);cmd.copyBufferToBuffer(resolved,0,read,0,bytes);this.device.queue.submit([cmd.finish()]);
      await read.mapAsync(GPUMapMode.READ);const values=new BigUint64Array(read.getMappedRange());
      const gpu=this.events.map((e,i)=>({...e,gpuMs:Number(values[i*2+1]-values[i*2])/1e6}));read.unmap();
      return {gpu,linear:this.linear,instrumented:true};
    }finally{resolved.destroy();read.destroy();}
  }
  dispose(){this.queries.destroy();}
}
