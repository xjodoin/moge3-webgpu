// Only immutable, verified model data enters this dedicated cache. Photos,
// tensors, authentication and other application caches never enter this path.
import {inflate,untransform} from './model-codecs.js';
export const MODEL_CACHE='moge3-model-chunks-v1';
const hashPattern=/^[a-f0-9]{64}$/;
const integer=n=>Number.isSafeInteger(n)&&n>=0;
export const newDownloadStats=()=>({networkBytes:0,networkRequests:0,rangeRequests:0,fullBundleFallbacks:0,legacyCacheHits:0,prefetchedChunks:0,cacheHits:0,cacheReadBytes:0,cacheWrites:0,cacheReadFailures:0,cacheWriteFailures:0,cacheUnavailable:0,corruptCacheEntries:0,decodedBytes:0,assembledBytes:0,verifiedFiles:0});
export async function sha256(data){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',data))].map(v=>v.toString(16).padStart(2,'0')).join('');}

export async function boundedBody(response,bytes){
 if(!response.body)throw Error('Model response has no readable body');
 const result=new Uint8Array(bytes),reader=response.body.getReader();let offset=0;
 try{
  while(true){const {done,value}=await reader.read();if(done)break;
   if(offset+value.length>bytes)throw Error('Model response exceeds expected length');
   result.set(value,offset);offset+=value.length;
  }
 }catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
 if(offset!==bytes)throw Error(`Truncated model data: ${offset} of ${bytes} bytes`);
 return result;
}

export class VerifiedModelCache {
 constructor({cacheStorage=globalThis.caches,fetcher=(...args)=>globalThis.fetch(...args),origin=globalThis.location?.origin??'http://localhost',stats=newDownloadStats()}={}){
  Object.assign(this,{cacheStorage,fetcher,origin,stats});this.pending=new Map();
 }
 async cache(){
  if(!this.opened)this.opened=(async()=>{try{if(!this.cacheStorage){this.stats.cacheUnavailable++;return null;}return await this.cacheStorage.open(MODEL_CACHE);}catch{this.stats.cacheReadFailures++;this.stats.cacheUnavailable++;return null;}})();
  return this.opened;
 }
 key(hash){return `${this.origin}/__moge3_model_cache_v1/${hash}`;}
 async read(url,meta,validate=async bytes=>bytes){
  const hash=meta.sha256;
  if(!hashPattern.test(hash)||!integer(meta.bytes)||meta.bytes<1)throw Error('Invalid model checksum or length');
  if(this.pending.has(hash))return this.pending.get(hash);
  const promise=this.readOnce(url,meta,validate);this.pending.set(hash,promise);
  try{return await promise;}finally{this.pending.delete(hash);}
 }
 async readOnce(url,meta,validate){
  const saved=await this.readCached(meta,validate);if(saved.hit)return saved.value;
  this.stats.networkRequests++;
  const response=await this.fetcher(url,{cache:'no-store'});
  if(!response.ok)throw Error(`Model download failed (${response.status}): ${url}`);
  const bytes=await boundedBody(response,meta.bytes);this.stats.networkBytes+=bytes.byteLength;
  return this.adopt(bytes,meta,validate);
 }
 async readCached(meta,validate=async bytes=>bytes){
  const cache=await this.cache(),key=this.key(meta.sha256);
  let cached;
  try{cached=await cache?.match(key);}catch{this.stats.cacheReadFailures++;}
  if(cached){
   try{
    const bytes=await boundedBody(cached,meta.bytes);
    if(await sha256(bytes)!==meta.sha256)throw Error('Cached model checksum mismatch');
    const value=await validate(bytes);this.stats.cacheHits++;this.stats.cacheReadBytes+=bytes.byteLength;return {hit:true,value};
   }catch{
    this.stats.corruptCacheEntries++;
    try{await cache.delete(key);}catch{this.stats.cacheWriteFailures++;}
   }
  }
  return {hit:false};
 }
 async adopt(bytes,meta,validate=async bytes=>bytes){
  if(bytes.byteLength!==meta.bytes)throw Error('Invalid downloaded chunk length');
  if(await sha256(bytes)!==meta.sha256)throw Error('Downloaded model checksum mismatch');
  const value=await validate(bytes);
  const cache=await this.cache(),key=this.key(meta.sha256);
  if(cache&&!this.writeDisabled){try{await cache.put(key,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));this.stats.cacheWrites++;}catch{this.stats.cacheWriteFailures++;this.writeDisabled=true;}}
  return value;
 }
 async clear(){
  if(this.pending.size||this.activeTransfers)throw Error('Cannot clear model cache while downloads are active');
  await this.cacheStorage?.delete(MODEL_CACHE);this.opened=null;this.writeDisabled=false;
 }
}

export function validatePackage(manifest){
 if(!((manifest?.format===1&&manifest.codec==='zlib-byte-shuffle-v1')||(manifest?.format===2&&manifest.codec==='adaptive-bundles-v2'))||!manifest.files||!manifest.chunks)throw Error('Unsupported model package');
 for(const [hash,c] of Object.entries(manifest.chunks)){
  if(!hashPattern.test(hash)||!hashPattern.test(c.rawSha256)||!integer(c.bytes)||c.bytes<1||c.bytes>64*1024*1024||!integer(c.rawBytes)||c.rawBytes<1||c.rawBytes>64*1024*1024||!['identity','shuffle4','exponent4','bitplane4'].includes(c.encoding)||(c.encoding!=='identity'&&c.rawBytes%4)||(c.encoding==='bitplane4'&&c.rawBytes%32)||!['deflate','zstd'].includes(c.codec??'deflate'))throw Error('Invalid model chunk');
  if(manifest.format===2){
   if(!hashPattern.test(c.bundle)||!integer(c.offset)||!Array.isArray(c.legacy))throw Error('Invalid model bundle reference');
   for(const old of c.legacy)if(!hashPattern.test(old.sha256)||!integer(old.bytes)||old.bytes<1||old.bytes>64*1024*1024||old.codec!=='deflate'||!['identity','shuffle4'].includes(old.encoding))throw Error('Invalid legacy chunk');
  }
 }
 if(manifest.format===2){
  const seen=new Set();
  for(const [hash,b] of Object.entries(manifest.bundles??{})){
   if(!hashPattern.test(hash)||!integer(b.bytes)||b.bytes<1||b.bytes>64*1024*1024||!Array.isArray(b.members)||b.members.length===0)throw Error('Invalid model bundle');
   let offset=0;for(const key of b.members){const c=manifest.chunks[key];if(!c||seen.has(key)||c.bundle!==hash||c.offset!==offset)throw Error('Invalid bundle member layout');seen.add(key);offset+=c.bytes;}
   if(offset!==b.bytes)throw Error('Incomplete model bundle');
  }
  if(seen.size!==Object.keys(manifest.chunks).length)throw Error('Incomplete bundle index');
 }
 for(const f of Object.values(manifest.files)){
  if(!hashPattern.test(f.sha256)||!integer(f.bytes)||f.bytes<1||f.bytes>2_000_000_000||!Array.isArray(f.pieces))throw Error('Invalid model file');
  let offset=0;
  for(const p of f.pieces){const c=manifest.chunks[p.chunk];
   if(!c||p.offset!==offset||!integer(p.length)||p.length<1||!integer(p.sourceOffset)||p.sourceOffset+p.length>c.rawBytes)throw Error('Invalid model piece layout');
   offset+=p.length;
  }
  if(offset!==f.bytes)throw Error('Incomplete model piece layout');
 }
 return manifest;
}

export async function decodeChunk(bytes,meta){
 const raw=untransform(await inflate(bytes,meta,boundedBody),meta.encoding);
 if(await sha256(raw)!==meta.rawSha256)throw Error('Decoded model checksum mismatch');
 return raw;
}

export class ModelPackage {
 constructor(manifest,baseUrl,cache,concurrency=4){
  this.manifest=validatePackage(manifest);this.baseUrl=baseUrl.replace(/\/$/,'');this.cache=cache;this.concurrency=concurrency;
  if(!Number.isInteger(concurrency)||concurrency<1||concurrency>8)throw Error('Invalid download concurrency');
 }
 async load(path,progress=()=>{}){
  const file=this.manifest.files[path];if(!file)throw Error(`Model is absent from package: ${path}`);
  const data=new Uint8Array(file.bytes),groups=new Map();
  for(const piece of file.pieces){if(!groups.has(piece.chunk))groups.set(piece.chunk,[]);groups.get(piece.chunk).push(piece);}
  const entries=[...groups],stats=this.cache.stats;let next=0,completed=0,last=0,error;
  const worker=async()=>{
   while(next<entries.length&&!error){
    const [hash,pieces]=entries[next++],meta=this.manifest.chunks[hash];
    try{
     const raw=await this.cache.read(`${this.baseUrl}/chunks/${hash}.zlib`,{bytes:meta.bytes,sha256:hash},bytes=>decodeChunk(bytes,meta));
     stats.decodedBytes+=raw.byteLength;
     for(const p of pieces){data.set(raw.subarray(p.sourceOffset,p.sourceOffset+p.length),p.offset);completed+=p.length;}
     if(performance.now()-last>200){progress(`Preparing model: ${Math.round(completed/file.bytes*100)}% · ${Math.round(stats.networkBytes/1e6)} MB downloaded · ${Math.round(stats.cacheReadBytes/1e6)} MB from device`);last=performance.now();}
    }catch(e){error=e;}
   }
  };
  // Drain all workers on failure before the caller may clear/retry the cache.
  await Promise.all(Array.from({length:this.concurrency},worker));if(error)throw error;
  if(await sha256(data)!==file.sha256)throw Error(`Assembled model checksum mismatch: ${path}`);
  stats.assembledBytes+=data.byteLength;stats.verifiedFiles++;
  return data;
 }
}
