import test from 'node:test';
import assert from 'node:assert/strict';
import {deflateSync} from 'node:zlib';
import {VerifiedModelCache,ModelPackage,sha256,newDownloadStats,MODEL_CACHE,validatePackage} from './model-cache.js';

function storage({writeFails=false}={}){
 const entries=new Map(),other=new Map([['unrelated',true]]);
 const cache={match:async key=>entries.get(key)?.clone(),put:async(key,value)=>{if(writeFails)throw Error('QuotaExceededError');entries.set(key,value.clone());},delete:async key=>entries.delete(key)};
 return {entries,other,open:async name=>{assert.equal(name,MODEL_CACHE);return cache;},delete:async name=>{assert.equal(name,MODEL_CACHE);entries.clear();return true;}};
}
async function fixture(){
 // Include signed zero, infinities, a NaN payload and arbitrary mantissa bits:
 // transport must preserve bytes, not reinterpret or round the float values.
 const raw=new Uint8Array(new Uint32Array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0x12345678]).buffer);
 const shuffle=new Uint8Array(raw.length),n=raw.length/4;
 for(let p=0;p<4;p++)for(let i=0;i<n;i++)shuffle[p*n+i]=raw[i*4+p];
 const packed=new Uint8Array(deflateSync(shuffle)),hash=await sha256(packed);
 const meta={bytes:packed.length,rawBytes:raw.length,rawSha256:await sha256(raw),encoding:'shuffle4'};
 const file={bytes:raw.length,sha256:await sha256(raw),pieces:[{chunk:hash,offset:0,sourceOffset:0,length:raw.length}]};
 return {raw,packed,hash,manifest:{format:1,codec:'zlib-byte-shuffle-v1',chunks:{[hash]:meta},files:{fast:file,detail:{...file}}}};
}

test('lossless chunks survive a new loader/offline visit and share across profiles',async()=>{
 const f=await fixture(),disk=storage();let requests=0;
 const cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>{requests++;return new Response(f.packed);}});
 assert.deepEqual(await new ModelPackage(f.manifest,'/weights',cache).load('fast'),f.raw);
 assert.equal(requests,1);
 const next=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>{throw Error('offline');}});
 assert.deepEqual(await new ModelPackage(f.manifest,'/other-revision',next).load('detail'),f.raw);
 assert.equal(next.stats.networkBytes,0);assert.equal(next.stats.cacheHits,1);
 await next.clear();assert.equal(disk.entries.size,0);assert.equal(disk.other.size,1);
});
test('corrupted and truncated cached chunks are evicted and fetched once',async()=>{
 for(const bad of [new Uint8Array([1]),new Uint8Array((await fixture()).packed.length)]){
  const f=await fixture(),disk=storage();let calls=0;
  const cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>{calls++;return new Response(f.packed);}});
  disk.entries.set(cache.key(f.hash),new Response(bad));
  assert.deepEqual(await new ModelPackage(f.manifest,'/weights',cache).load('fast'),f.raw);
  assert.equal(calls,1);assert.equal(cache.stats.corruptCacheEntries,1);
 }
});
test('network corruption and decompression overflow cannot enter persistent storage',async()=>{
 const f=await fixture();
 for(const packed of [new Uint8Array([1]),new Uint8Array(f.packed.length)]){
  const disk=storage(),cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>new Response(packed)});
  await assert.rejects(new ModelPackage(f.manifest,'/weights',cache).load('fast'),/Truncated|checksum/);
  assert.equal(disk.entries.size,0);
 }
 const disk=storage(),cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>new Response(f.packed)});
 const bad=structuredClone(f.manifest);bad.chunks[f.hash].rawBytes-=4;bad.files={};
 await assert.rejects(new ModelPackage({...bad,files:{short:{bytes:20,sha256:'0'.repeat(64),pieces:[{chunk:f.hash,offset:0,sourceOffset:0,length:20}]}}},'/weights',cache).load('short'),/exceeds/);
 assert.equal(disk.entries.size,0);
});
test('unavailable or full storage still permits verified inference',async()=>{
 const f=await fixture();
 for(const disk of [storage({writeFails:true}),{open:async()=>{throw Error('denied');}}]){
  const cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>new Response(f.packed)});
  assert.deepEqual(await new ModelPackage(f.manifest,'/weights',cache).load('fast'),f.raw);
  assert.ok(cache.stats.cacheWriteFailures+cache.stats.cacheReadFailures>0);
 }
});
test('concurrent users share requests; interrupted responses retry without partial cache entries',async()=>{
 const f=await fixture(),disk=storage();let calls=0;
 const cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>{calls++;await new Promise(r=>setTimeout(r,5));return new Response(f.packed);}});
 const pack=new ModelPackage(f.manifest,'/weights',cache);
 assert.deepEqual(await Promise.all([pack.load('fast'),pack.load('detail')]),[f.raw,f.raw]);assert.equal(calls,1);
 await cache.clear();
 cache.fetcher=async()=>new Response(new ReadableStream({start(c){c.enqueue(f.packed.subarray(0,2));c.error(Error('interrupted'));}}));
 await assert.rejects(pack.load('fast'),/interrupted/);assert.equal(disk.entries.size,0);
 cache.fetcher=async()=>new Response(f.packed);
 assert.deepEqual(await pack.load('fast'),f.raw);
});
test('invalid layouts, changed chunks and assembled hashes fail before model execution',async()=>{
 const f=await fixture(),bad=structuredClone(f.manifest);bad.files.fast.pieces[0].offset=1;
 assert.throws(()=>validatePackage(bad),/layout/);
 const disk=storage(),cache=new VerifiedModelCache({cacheStorage:disk,fetcher:async()=>new Response(f.packed),stats:newDownloadStats()});
 const wrong=structuredClone(f.manifest);wrong.files.fast.sha256='0'.repeat(64);
 await assert.rejects(new ModelPackage(wrong,'/weights',cache).load('fast'),/Assembled/);
 assert.equal(cache.stats.verifiedFiles,0);
 const altered=new Uint8Array(f.packed);altered[0]^=1;const hash=await sha256(altered);
 assert.notEqual(hash,f.hash); // New bytes have a distinct persistent cache key.
});
test('default fetch preserves the browser global receiver',async()=>{
 const original=globalThis.fetch,f=await fixture();
 try{
  globalThis.fetch=async function(){assert.equal(this,globalThis);return new Response(f.packed);};
  const cache=new VerifiedModelCache({cacheStorage:storage()});
  assert.deepEqual(await new ModelPackage(f.manifest,'/weights',cache).load('fast'),f.raw);
 }finally{globalThis.fetch=original;}
});
