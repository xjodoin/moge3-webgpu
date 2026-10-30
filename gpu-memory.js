// Diagnostic accounting of explicit WebGPU buffer allocations, not process RSS.
// Includes ORT's retained buffer pools; excludes driver/shader/browser overhead.
export function trackGpuBuffers(){
 const sizes=new WeakMap();let live=0,peak=0,largest=0;
 const create=GPUDevice.prototype.createBuffer,destroy=GPUBuffer.prototype.destroy;
 GPUDevice.prototype.createBuffer=function(descriptor){
  const buffer=create.call(this,descriptor);sizes.set(buffer,descriptor.size);
  live+=descriptor.size;peak=Math.max(peak,live);largest=Math.max(largest,descriptor.size);return buffer;
 };
 GPUBuffer.prototype.destroy=function(){
  const size=sizes.get(this);if(size!==undefined){live-=size;sizes.delete(this);}return destroy.call(this);
 };
 return {reset(){peak=live;largest=0;},report(){return {peakExplicitBufferBytes:peak,liveExplicitBufferBytes:live,largestBufferBytes:largest};}};
}
