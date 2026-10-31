export function validateSparseWeights(meta) {
  const c=meta?.channels;
  if(![32,64,128,256,512].includes(c))throw Error('Unsupported sparse channel count');
  let offset=0;
  for(const [name,length] of [['norm1.weight',c],['norm1.bias',c],['conv1.weight',27*c*c],['conv1.bias',c],['conv2.weight',27*c*c],['conv2.bias',c]]){
    const field=meta.fields?.[name];
    if(field?.offset!==offset||field?.length!==length)throw Error(`Invalid sparse field layout: ${name}`);
    offset+=length;
  }
  if(meta.bytes!==offset*4||!/^[a-f0-9]{64}$/.test(meta.sha256))throw Error('Invalid sparse weight length or checksum');
}
