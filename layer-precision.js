// Storage-only experiments: preserve sign/exponent, expand to FP32 for inference.
export function roundLayerWeights(raw, spans, bits) {
 if(!Number.isInteger(bits)||bits<16||bits>32)throw Error('Storage bits must be 16–32');
 const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength),step=2**(32-bits);
 for(const {offset,bytes} of spans){
  if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(bytes)||offset<0||bytes<4||bytes%4||offset+bytes>raw.length)throw Error('Invalid weight span');
  for(let i=offset;i<offset+bytes;i+=4){
   const word=view.getUint32(i,true),rounded=bits===32?word:Math.floor((word+step/2)/step)*step;
   if((word&0x7f800000)===0x7f800000||((rounded>>>0)&0x7f800000)===0x7f800000)throw Error('Nonfinite rounded weight');
   view.setUint32(i,rounded>>>0,true);
  }
 }
 return raw;
}
