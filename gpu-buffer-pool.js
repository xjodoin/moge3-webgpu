// Leases are returned only after their commands have been submitted. WebGPU's
// queue ordering then protects reuse without a CPU/GPU synchronization fence.
export class GpuBufferPool {
  constructor(device,{enabled=true,maxBytes=256*1024*1024}={}){
    Object.assign(this,{device,enabled,maxBytes});this.free=[];this.bytes=0;this.epoch=0;this.created=0;this.hits=0;
  }
  acquire(size,usage){
    const capacity=size>65536?Math.ceil(size/1048576)*1048576:Math.max(16,Math.ceil(size/16)*16);
    const i=this.enabled?this.free.findIndex(x=>x.size===capacity&&x.usage===usage):-1;
    let entry;
    if(i>=0){entry=this.free.splice(i,1)[0];this.bytes-=entry.size;this.hits++;}
    else {entry={buffer:this.device.createBuffer({size:capacity,usage}),size:capacity,usage};this.created++;}
    const epoch=this.epoch;let released=false;
    return {buffer:entry.buffer,release:()=>{
      if(released)return;released=true;
      if(this.enabled&&epoch===this.epoch&&entry.size+this.bytes<=this.maxBytes&&entry.buffer.mapState!=='mapped'&&entry.buffer.mapState!=='pending'){
        this.free.push(entry);this.bytes+=entry.size;
      }else entry.buffer.destroy();
    }};
  }
  clear(){this.epoch++;for(const e of this.free)e.buffer.destroy();this.free=[];this.bytes=0;}
  snapshot(){return {created:this.created,hits:this.hits,retainedBytes:this.bytes,retainedBuffers:this.free.length,enabled:this.enabled};}
}
