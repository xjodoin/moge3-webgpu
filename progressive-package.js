import {BundledModelPackage} from './model-bundles.js';
import {decodeChunk,sha256} from './model-cache.js';
import {roundWeight24,restoreWeight32} from './weight-precision.js';

export class ProgressiveModelPackage extends BundledModelPackage {
 constructor(...args){super(...args);if(this.manifest.format!==3)throw Error('Expected progressive weights');}
 async originalCached(key){
  const original=this.manifest.chunks[key].original;if(!original)return {hit:false};
  for(const meta of [original,...original.legacy.map(old=>({...original,...old}))]){
   const saved=await this.cache.readCached(meta,bytes=>decodeChunk(bytes,meta));
   if(saved.hit){this.cache.stats.originalWeightCacheHits=(this.cache.stats.originalWeightCacheHits??0)+1;return saved;}
  }
  return {hit:false};
 }
 async saved(key){
  const saved=await super.saved(key);if(saved.hit)return saved;
  const old=await this.originalCached(key);if(!old.hit)return old;
  const c=this.manifest.chunks[key],raw=roundWeight24(old.value,[{offset:0,bytes:old.value.length}]);
  if(await sha256(raw)!==c.rawSha256)throw Error('Compact cache conversion mismatch');
  return {hit:true,value:raw};
 }
 async load(path,progress=()=>{},mode='compact'){
  if(!['compact','exact'].includes(mode))throw Error('Unknown weight download mode');
  this.cache.activeTransfers=(this.cache.activeTransfers??0)+1;
  try{
   const data=await super.load(path,progress),file=this.manifest.files[path];
   if(mode==='compact'||file.exactSha256===file.sha256)return data;
   const groups=new Map();
   for(const p of file.pieces)if(this.manifest.chunks[p.chunk].encoding==='fp24'){
    if(!groups.has(p.chunk))groups.set(p.chunk,[]);groups.get(p.chunk).push(p);
   }
   progress('Restoring original FP32 weights from correction data…');
   await this.parallel([...groups],async([key,pieces])=>{
    const original=await this.originalCached(key);
    if(original.hit){for(const p of pieces)data.set(original.value,p.offset);return;}
    const low=await super.load(`__corrections/${key}`);
    for(const p of pieces)restoreWeight32(data.subarray(p.offset,p.offset+p.length),low);
   });
   if(await sha256(data)!==file.exactSha256)throw Error('Restored original model checksum mismatch');
   this.cache.stats.restoredExactFiles=(this.cache.stats.restoredExactFiles??0)+1;
   return data;
  }finally{this.cache.activeTransfers--;}
 }
}
