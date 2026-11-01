// ORT emits separate aliases for identical high-rank index arrays. Firefox's
// Naga/Metal translation can give those aliases distinct MSL struct types, so
// passing input/output indices between helpers fails pipeline compilation.
// WGSL aliases are synonyms: unifying identical arrays preserves every index,
// buffer access, binding, and FP32 operation. Keep different ranks separate.
export function unifyIndexArrayAliases(code){
 const canonical=new Map(),replacements=new Map();
 const declarations=/\balias\s+((?:input|output)(?:_\d+)?_indices_t)\s*=\s*array\s*<\s*u32\s*,\s*(\d+)\s*>\s*;/g;
 const stripped=code.replace(declarations,(declaration,name,rank)=>{
  const key=Number(rank);
  if(!canonical.has(key)){canonical.set(key,name);return declaration;}
  replacements.set(name,canonical.get(key));return '';
 });
 if(!replacements.size)return code;
 return stripped.replace(/\b(?:input|output)(?:_\d+)?_indices_t\b/g,name=>replacements.get(name)??name);
}

const installed=new WeakMap();
export function installIndexArrayCompatibility(prototype=globalThis.GPUDevice?.prototype){
 if(!prototype)return {snapshot:()=>({shaderModulesPatched:0})};
 if(installed.has(prototype))return installed.get(prototype);
 const create=prototype.createShaderModule;
 let shaderModulesPatched=0;
 const state={snapshot:()=>({shaderModulesPatched})};
 prototype.createShaderModule=function(descriptor){
  const code=unifyIndexArrayAliases(descriptor.code);
  if(code===descriptor.code)return create.call(this,descriptor);
  shaderModulesPatched++;
  return create.call(this,{...descriptor,code});
 };
 installed.set(prototype,state);
 return state;
}
