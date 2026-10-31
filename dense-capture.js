// The session owns a stable GPU input and ORT-owned output bindings. Reading a
// tensor changes its location to CPU, so each call gets disposable output views.
// Views are borrowed until the next run or unload; the session owns GPU storage.
export class CapturedDenseSession {
  constructor(session,device,meta,Tensor,download,usage) {
    this.session=session;this.device=device;this.Tensor=Tensor;this.download=download;
    this.runs=0;this.released=false;this.busy=false;this.bindings=null;this.stableReplays=0;
    this.inputDims=[...meta.inputShape];this.outputShapes=meta.outputs;
    for(const dims of [this.inputDims,...Object.values(this.outputShapes)]){
      if(!Array.isArray(dims)||!dims.length||!dims.every(n=>Number.isSafeInteger(n)&&n>0))throw Error('Graph capture requires static positive shapes');
    }
    this.inputBytes=this.inputDims.reduce((a,b)=>a*b,1)*4;
    this.inputBuffer=device.createBuffer({size:Math.ceil(this.inputBytes/16)*16,usage});
    try{this.inputTensor=Tensor.fromGpuBuffer(this.inputBuffer,{dataType:'float32',dims:this.inputDims});}
    catch(error){this.inputBuffer.destroy();throw error;}
  }
  async run({image}) {
    if(this.released)throw Error('Dense graph has been released');
    if(this.busy)throw Error('Dense graph is already running');
    if(image.type!=='float32'||image.dims.length!==this.inputDims.length||image.dims.some((n,i)=>n!==this.inputDims[i])||image.data.byteLength!==this.inputBytes)throw Error('Dense graph input shape/type mismatch');
    this.busy=true;
    let results;
    try {
      this.device.queue.writeBuffer(this.inputBuffer,0,image.data);
      // Let ORT allocate and bind its outputs once, then supply the same owners
      // as fetches. Keep their GPU location intact across validation readbacks.
      try{results=await this.session.run({image:this.inputTensor},this.bindings??undefined);}
      catch(error){throw new Error(`Dense graph execution failed: ${error.message}`,{cause:error});}
      for(const [name,dims] of Object.entries(this.outputShapes)){
        const t=results[name];
        if(!t||t.location!=='gpu-buffer'||t.type!=='float32'||t.dims.length!==dims.length||t.dims.some((n,i)=>n!==dims[i]))throw Error(`Dense graph output shape/type mismatch: ${name}`);
        if(this.bindings&&t.gpuBuffer!==this.bindings[name].gpuBuffer)throw Error(`Dense graph changed its bound output: ${name}`);
      }
      if(!this.bindings){this.bindings=results;results=null;}else this.stableReplays++;
      this.runs++;
      return Object.fromEntries(Object.entries(this.bindings).map(([name,t])=>{
        const buffer=t.gpuBuffer,bytes=t.dims.reduce((a,b)=>a*b,1)*4;
        return [name,this.Tensor.fromGpuBuffer(buffer,{dataType:'float32',dims:t.dims,download:()=>this.download(buffer,bytes,'float32')})];
      }));
    }finally{
      try{if(results)for(const [name,t] of Object.entries(results))if(t!==this.bindings?.[name])t.dispose();}finally{this.busy=false;}
    }
  }
  snapshot(){return {enabled:true,completedRuns:this.runs,stableBindingReplays:this.stableReplays,bindingBytes:this.inputBuffer.size+Object.values(this.bindings??{}).reduce((n,t)=>n+t.gpuBuffer.size,0)};}
  async release(){
    if(this.released)return;
    if(this.busy)throw Error('Cannot release a running dense graph');
    this.released=true;
    try{if(this.bindings)for(const t of Object.values(this.bindings))t.dispose();}
    finally{try{await this.session.release();}finally{this.inputBuffer.destroy();this.bindings=null;}}
  }
}
