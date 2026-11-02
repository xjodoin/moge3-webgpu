// Custom transport only: all expanded weights and every operation remain FP32.
export function expandWeight24(bytes){
 if(bytes.length%3)throw Error('Invalid FP24 plane length');
 const n=bytes.length/3,raw=new Uint8Array(n*4);
 for(let i=0;i<n;i++){
  const e=bytes[i],s=bytes[2*n+i],offset=i*4;
  raw[offset+1]=bytes[n+i];raw[offset+2]=(s&127)|((e&1)<<7);raw[offset+3]=(e>>>1)|(s&128);
 }
 return raw;
}
export function restoreWeight32(raw,low){
 if(raw.length!==low.length*4)throw Error('Invalid FP24 correction length');
 const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
 for(let i=0;i<low.length;i++){const offset=i*4;view.setUint32(offset,((view.getUint32(offset,true)-((low[i]>>>7)<<8))|low[i])>>>0,true);}
 return raw;
}
export function roundWeight24(raw,spans){
 const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
 for(const span of spans){
  if(!Number.isSafeInteger(span.offset)||!Number.isSafeInteger(span.bytes)||span.offset<0||span.bytes<4||span.bytes%4||span.offset+span.bytes>raw.length)throw Error('Invalid FP24 weight span');
  for(let i=span.offset;i<span.offset+span.bytes;i+=4){
   const word=view.getUint32(i,true),rounded=((word+128)&0xffffff00)>>>0;
   if((word&0x7f800000)===0x7f800000||(rounded&0x7f800000)===0x7f800000)throw Error('Nonfinite FP24 weight');
   view.setUint32(i,rounded,true);
  }
 }
 return raw;
}
