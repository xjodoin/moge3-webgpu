let zstd;
export async function inflate(bytes,meta,boundedBody){
 if((meta.codec??'deflate')==='deflate')return boundedBody(new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))),meta.rawBytes);
 if(meta.codec!=='zstd')throw Error('Unsupported model codec');
 if(!zstd)zstd=import('zstddec').then(async({ZSTDDecoder})=>{const decoder=new ZSTDDecoder();await decoder.init();return decoder;}).catch(error=>{zstd=null;throw error;});
 const result=(await zstd).decode(bytes,meta.rawBytes);
 if(result.byteLength!==meta.rawBytes)throw Error('Invalid Zstd decoded length');
 return result;
}

export function untransform(bytes,encoding){
 if(encoding==='identity')return bytes;
 if(bytes.length%4)throw Error('Invalid FP32 transform length');
 const raw=new Uint8Array(bytes.length),count=bytes.length/4;
 if(encoding==='shuffle4'){
  for(let plane=0;plane<4;plane++){const start=plane*count;for(let i=0;i<count;i++)raw[i*4+plane]=bytes[start+i];}
 }else if(encoding==='exponent4'){
  for(let i=0;i<count;i++){
   const e=bytes[i],signMantissa=bytes[3*count+i],offset=i*4;
   raw[offset]=bytes[count+i];raw[offset+1]=bytes[2*count+i];
   raw[offset+2]=(signMantissa&127)|((e&1)<<7);raw[offset+3]=(e>>>1)|(signMantissa&128);
  }
 }else if(encoding==='bitplane4'){
  if(count%8)throw Error('Invalid bit-plane length');
  const groups=count/8;
  for(let plane=0;plane<32;plane++){
   const column=plane>>>3,shift=7-(plane&7),start=plane*groups;
   for(let i=0;i<groups;i++){
    const value=bytes[start+i],offset=i*32+column;
    for(let bit=0;bit<8;bit++)raw[offset+bit*4]|=((value>>>(7-bit))&1)<<shift;
   }
  }
 }else throw Error('Unsupported model transform');
 return raw;
}
