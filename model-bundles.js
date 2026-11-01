import {validatePackage,decodeChunk,boundedBody,sha256} from './model-cache.js';

// Bundles reduce cold round trips. Chunks remain independent cache entries:
// partial caches, interrupted loads and old codec versions do not force a full
// model download. All returned files are still checked against original SHA-256.
export class BundledModelPackage {
 constructor(manifest,baseUrl,cache,concurrency=4){
  this.manifest=validatePackage(manifest);
  if(manifest.format!==2)throw Error('Expected a bundled model package');
  if(!Number.isInteger(concurrency)||concurrency<1||concurrency>8)throw Error('Invalid download concurrency');
  Object.assign(this,{baseUrl:baseUrl.replace(/\/$/,''),cache,concurrency});this.bundleJobs=new Map();
 }
 async saved(key){
  const c=this.manifest.chunks[key];
  let saved=await this.cache.readCached({sha256:key,bytes:c.bytes},bytes=>decodeChunk(bytes,c));
  if(saved.hit)return saved;
  for(const old of c.legacy){
   if(old.sha256===key)continue;
   saved=await this.cache.readCached(old,bytes=>decodeChunk(bytes,{...c,...old}));
   if(saved.hit){this.cache.stats.legacyCacheHits++;return saved;}
  }
  return {hit:false};
 }
 async parallel(jobs,fn){
  let next=0,error;
  const worker=async()=>{while(next<jobs.length&&!error){const item=jobs[next++];try{await fn(item);}catch(e){error=e;}}};
  await Promise.all(Array.from({length:this.concurrency},worker));if(error)throw error;
 }
 async bundle(hash,missing){
  // Loading different model files concurrently can need the same indexed pack.
  // Waiting callers read the verified cache or the returned decoded values.
  if(this.bundleJobs.has(hash)){
   const values=await this.bundleJobs.get(hash),remaining=new Set();
   for(const key of missing)if(!values.has(key)){const saved=await this.saved(key);if(saved.hit)values.set(key,saved.value);else remaining.add(key);}
   if(remaining.size){const extra=await this.bundle(hash,remaining);for(const [key,raw] of extra)values.set(key,raw);}
   return values;
  }
  const pending=this.fetchBundle(hash,missing);this.bundleJobs.set(hash,pending);
  try{return await pending;}finally{this.bundleJobs.delete(hash);}
 }
 async fetchBundle(hash,missing){
  const bundle=this.manifest.bundles[hash],chunks=this.manifest.chunks,stats=this.cache.stats;
  const missingBytes=[...missing].reduce((n,key)=>n+chunks[key].bytes,0);
  const selected=bundle.members.filter(key=>missing.has(key));
  const ranges=[];
  // Cold loads read one full bundle. For a partial cache, coalesce adjacent
  // missing chunks, accepting at most 64 KiB of intervening already-cached data.
  let stored=false;const disk=await this.cache.cache();
  if(disk){
   for(const key of bundle.members){
    try{if(await disk.match(this.cache.key(key))){stored=true;break;}
     for(const old of chunks[key].legacy)if(await disk.match(this.cache.key(old.sha256))){stored=true;break;}
    }catch{this.cache.stats.cacheReadFailures++;}
    if(stored)break;
   }
  }
  if(missingBytes>=bundle.bytes*.65||(disk&&!stored))ranges.push({start:0,end:bundle.bytes-1,full:true});
  else for(const key of selected){
   const c=chunks[key],last=ranges.at(-1);
   if(last&&c.offset-last.end-1<=65536)last.end=c.offset+c.bytes-1;
   else ranges.push({start:c.offset,end:c.offset+c.bytes-1,full:false});
  }
  const decoded=new Map();
  this.cache.activeTransfers=(this.cache.activeTransfers??0)+1;
  try{
   for(const range of ranges){
    stats.networkRequests++;
    const headers=range.full?{}:{Range:`bytes=${range.start}-${range.end}`};if(!range.full)stats.rangeRequests++;
    const response=await this.cache.fetcher(`${this.baseUrl}/bundles/${hash}.pack`,{cache:'no-store',headers});
    if(response.status!==200&&response.status!==206)throw Error(`Model bundle download failed (${response.status})`);
    const full=response.status===200;
    if(range.full&&!full)throw Error('Unexpected partial bundle response');
    if(!full){const header=response.headers.get('Content-Range');if(header&&header!==`bytes ${range.start}-${range.end}/${bundle.bytes}`)throw Error('Invalid bundle Content-Range');}
    const bytes=await boundedBody(response,full?bundle.bytes:range.end-range.start+1);stats.networkBytes+=bytes.byteLength;
    if(full){
     if(await sha256(bytes)!==hash)throw Error('Model bundle checksum mismatch');
     if(!range.full)stats.fullBundleFallbacks++;
    }
    const start=full?0:range.start,end=full?bundle.bytes:range.end+1;
    for(const key of bundle.members){
     const c=chunks[key];if(c.offset<start||c.offset+c.bytes>end||decoded.has(key))continue;
     const raw=await this.cache.adopt(bytes.subarray(c.offset-start,c.offset-start+c.bytes),{sha256:key,bytes:c.bytes},packed=>decodeChunk(packed,c));
     decoded.set(key,raw);stats.decodedBytes+=raw.byteLength;
     if(!missing.has(key))stats.prefetchedChunks++;
    }
    if(full)break; // A server ignoring Range must not trigger repeated full fetches.
   }
  }finally{this.cache.activeTransfers--;}
  return decoded;
 }
 async load(path,progress=()=>{}){
  const file=this.manifest.files[path];if(!file)throw Error(`Model is absent from package: ${path}`);
  const data=new Uint8Array(file.bytes),groups=new Map(),missing=new Map(),stats=this.cache.stats;
  const before={network:stats.networkBytes,cache:stats.cacheReadBytes};let completed=0,last=0;
  for(const p of file.pieces){if(!groups.has(p.chunk))groups.set(p.chunk,[]);groups.get(p.chunk).push(p);}
  const apply=(key,raw)=>{
   for(const p of groups.get(key)){data.set(raw.subarray(p.sourceOffset,p.sourceOffset+p.length),p.offset);completed+=p.length;}
   if(performance.now()-last>150){
    progress(`Preparing model: ${Math.round(completed/file.bytes*100)}% · ${Math.round((stats.networkBytes-before.network)/1e6)} MB downloaded · ${Math.round((stats.cacheReadBytes-before.cache)/1e6)} MB reused`);last=performance.now();
   }
  };
  this.cache.activeTransfers=(this.cache.activeTransfers??0)+1;
  try{
   await this.parallel([...groups.keys()],async key=>{
    const saved=await this.saved(key);
    if(saved.hit){stats.decodedBytes+=saved.value.byteLength;apply(key,saved.value);}
    else{const bundle=this.manifest.chunks[key].bundle;if(!missing.has(bundle))missing.set(bundle,new Set());missing.get(bundle).add(key);}
   });
   await this.parallel([...missing],async([hash,keys])=>{
    const decoded=await this.bundle(hash,keys);
    for(const key of keys){
     const raw=decoded.get(key)??(await this.saved(key)).value;
     if(!raw)throw Error('Model bundle did not provide required chunk');
     apply(key,raw);
    }
   });
   if(await sha256(data)!==file.sha256)throw Error(`Assembled model checksum mismatch: ${path}`);
   stats.assembledBytes+=data.byteLength;stats.verifiedFiles++;return data;
  }finally{this.cache.activeTransfers--;}
 }
}
