// FP16 storage only. Every load explicitly widens before FP32 normalization/FMA.
// Features, accumulation, camera fitting and voxel indexing stay FP32.
export function halfWeightShader(code) {
  const vector=code.includes('weights:array<vec4<f32>>');
  const declaration=vector?'weights:array<vec4<f32>>':'weights:array<f32>';
  if(!code.includes(declaration))throw Error('Missing sparse weight declaration');
  let result=code.replace(declaration,vector?'weights:array<vec4<f16>>':'weights:array<f16>');
  // Weight indices in these generated kernels contain no nested brackets.
  result=result.replace(/weights\[([^\[\]]+)\]/g,(_,index)=>`${vector?'vec4<f32>':'f32'}(weights[${index}])`);
  return `enable f16;\n${result}`;
}
