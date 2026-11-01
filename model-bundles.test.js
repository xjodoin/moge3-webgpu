import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {VerifiedModelCache,sha256,decodeChunk,validatePackage,MODEL_CACHE} from './model-cache.js';
import {BundledModelPackage} from './model-bundles.js';
const fixture=JSON.parse(await readFile(new URL('./testdata/model-codecs.json',import.meta.url),'utf8'));
const bytes=hex=>new Uint8Array(Buffer.from(hex,'hex'));
const raw=bytes(fixture.rawHex);

function disk(){
 const entries=new Map();return {entries,open:async name=>{assert.equal(name,MODEL_CACHE);return {match:async key=>entries.get(key)?.clone(),put:async(key,value)=>entries.set(key,value.clone()),delete:async key=>entries.delete(key)};},delete:async()=>{entries.clear();return true;}};
}
async function setup({ignoreRange=false,badRange=false,corrupt=false,noStorage=false}={}){
 const rows=fixture.rows.filter(r=>r.codec==='zstd'&&['shuffle4','exponent4','identity'].includes(r.encoding));
 const bundle=new Uint8Array(rows.reduce((n,r)=>n+bytes(r.hex).length,0));let offset=0;
 const chunks={};for(const r of rows){const p=bytes(r.hex);bundle.set(p,offset);chunks[r.sha256]={bytes:p.length,rawBytes:raw.length,rawSha256:fixture.rawSha256,codec:r.codec,encoding:r.encoding,offset,legacy:[]};offset+=p.length;}
 const hash=await sha256(bundle),members=rows.map(r=>r.sha256);for(const c of Object.values(chunks))c.bundle=hash;
 const makeFile=keys=>({bytes:keys.length*raw.length,sha256:'',pieces:keys.map((key,i)=>({offset:i*raw.length,length:raw.length,sourceOffset:0,chunk:key}))});
 const a=makeFile([members[0]]),b=makeFile([members[1],members[2]]);
 a.sha256=await sha256(raw);b.sha256=await sha256(new Uint8Array([...raw,...raw]));
 const manifest={format:2,codec:'adaptive-bundles-v2',chunks,bundles:{[hash]:{bytes:bundle.length,members}},files:{fast:a,detail:b}};
 const storage=disk(),requests=[];
 const cache=new VerifiedModelCache({cacheStorage:noStorage?null:storage,fetcher:async(url,options)=>{
  const range=options.headers?.Range;requests.push({url,range});
  if(range&&!ignoreRange){const [,start,end]=range.match(/bytes=(\d+)-(\d+)/).map(Number);return new Response(bundle.slice(start,end+1),{status:206,headers:{'Content-Range':badRange?'bytes 0-0/1':`bytes ${start}-${end}/${bundle.length}`}});}
  const body=bundle.slice();if(corrupt)body[0]^=1;return new Response(body);
 }});
 return {manifest,cache,storage,requests,members,bundle,pack:new BundledModelPackage(manifest,'/models/packed-v2',cache)};
}

test('all portable transforms and both codecs preserve every original bit',async()=>{
 for(const row of fixture.rows){
  const result=await decodeChunk(bytes(row.hex),{...row,rawBytes:raw.length,rawSha256:fixture.rawSha256});
  assert.deepEqual(result,raw);
 }
});
test('one indexed cold request prefetches later model files into verified cache',async()=>{
 const s=await setup();assert.deepEqual(await s.pack.load('fast'),raw);
 assert.equal(s.requests.length,1);assert.equal(s.requests[0].range,undefined);
 assert.deepEqual(await s.pack.load('detail'),new Uint8Array([...raw,...raw]));
 assert.equal(s.requests.length,1);assert.ok(s.cache.stats.prefetchedChunks>0);
 const next=new VerifiedModelCache({cacheStorage:s.storage,fetcher:async()=>{throw Error('network disabled');}});
 assert.deepEqual(await new BundledModelPackage(s.manifest,'/new-revision',next).load('fast'),raw);
 assert.equal(next.stats.networkBytes,0);
});
test('old verified codec entries upgrade without any new weight requests',async()=>{
 const s=await setup(),legacy=fixture.rows.find(r=>r.codec==='deflate'&&r.encoding==='shuffle4');
 s.manifest.chunks[s.members[0]].legacy=[{sha256:legacy.sha256,bytes:bytes(legacy.hex).length,codec:'deflate',encoding:legacy.encoding}];
 s.storage.entries.set(s.cache.key(legacy.sha256),new Response(bytes(legacy.hex)));
 assert.deepEqual(await s.pack.load('fast'),raw);
 assert.equal(s.requests.length,0);assert.equal(s.cache.stats.legacyCacheHits,1);
});
test('partially saved bundles fetch only missing ranges and validate range headers',async()=>{
 for(const badRange of [false,true]){
  const s=await setup({badRange});
  for(const key of [s.members[0],s.members[1]]){const c=s.manifest.chunks[key];s.storage.entries.set(s.cache.key(key),new Response(s.bundle.slice(c.offset,c.offset+c.bytes)));}
  if(badRange)await assert.rejects(s.pack.load('detail'),/Content-Range/);
  else{
   assert.deepEqual(await s.pack.load('detail'),new Uint8Array([...raw,...raw]));
   assert.equal(s.requests.length,1);assert.ok(s.requests[0].range);assert.equal(s.cache.stats.networkBytes,s.manifest.chunks[s.members[2]].bytes);
  }
 }
});
test('servers ignoring Range cause one verified full fetch, and unavailable storage still works',async()=>{
 const s=await setup({ignoreRange:true});
 for(const key of [s.members[0],s.members[1]]){const c=s.manifest.chunks[key];s.storage.entries.set(s.cache.key(key),new Response(s.bundle.slice(c.offset,c.offset+c.bytes)));}
 assert.deepEqual(await s.pack.load('detail'),new Uint8Array([...raw,...raw]));assert.equal(s.requests.length,1);assert.ok(s.requests[0].range);assert.equal(s.cache.stats.fullBundleFallbacks,1);
 const no=await setup({noStorage:true});assert.deepEqual(await no.pack.load('fast'),raw);assert.equal(no.cache.stats.verifiedFiles,1);
});
test('bad bundles cannot populate storage; invalid indexed layouts fail early',async()=>{
 const s=await setup({corrupt:true});await assert.rejects(s.pack.load('fast'),/bundle checksum/);assert.equal(s.storage.entries.size,0);
 const bad=structuredClone(s.manifest);bad.chunks[s.members[0]].offset=1;assert.throws(()=>validatePackage(bad),/layout/);
});
test('parallel models using the same bundle finish correctly without duplicated cold fetches',async()=>{
 const s=await setup();const result=await Promise.all([s.pack.load('fast'),s.pack.load('detail')]);
 assert.deepEqual(result,[raw,new Uint8Array([...raw,...raw])]);assert.equal(s.requests.length,1);assert.equal(s.cache.activeTransfers,0);
});
test('concurrent partial-cache consumers can request different missing ranges safely',async()=>{
 const s=await setup(),key=s.members[2],c=s.manifest.chunks[key];
 s.storage.entries.set(s.cache.key(key),new Response(s.bundle.slice(c.offset,c.offset+c.bytes)));
 assert.deepEqual(await Promise.all([s.pack.load('fast'),s.pack.load('detail')]),[raw,new Uint8Array([...raw,...raw])]);
 assert.equal(s.cache.activeTransfers,0);assert.equal(s.pack.bundleJobs.size,0);
});
