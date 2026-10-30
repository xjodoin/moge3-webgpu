export async function downloadModel(url,bytes,progress){
 const response=await fetch(url);if(!response.ok)throw Error(`Model download failed (${response.status})`);
 if(!bytes||!response.body)return new Uint8Array(await response.arrayBuffer());
 const data=new Uint8Array(bytes),reader=response.body.getReader();let offset=0,last=0;
 try{
  while(true){
   const {done,value}=await reader.read();if(done)break;
   if(offset+value.length>bytes)throw Error('Model size exceeds its manifest');
   data.set(value,offset);offset+=value.length;
   if(performance.now()-last>200){progress(`Loading model: ${Math.round(offset/bytes*100)}% (${Math.round(offset/1e6)} / ${Math.round(bytes/1e6)} MB)`);last=performance.now();}
  }
 }catch(error){await reader.cancel();throw error;}finally{reader.releaseLock();}
 if(offset!==bytes)throw Error(`Incomplete model download: ${offset} of ${bytes} bytes`);
 progress('Preparing WebGPU model…');return data;
}
