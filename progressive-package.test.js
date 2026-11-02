import test from 'node:test';
import assert from 'node:assert/strict';
import {deflateSync} from 'node:zlib';
import {VerifiedModelCache,validatePackage,sha256} from './model-cache.js';
import {ProgressiveModelPackage} from './progressive-package.js';
import {roundWeight24} from './weight-precision.js';

async function setup({noStorage=false}={}){
 const words=[0x3f80007f,0x3f800080,0xbf80ff80,0x80000000],raw=new Uint8Array(words.length*4);
 words.forEach((w,i)=>new DataView(raw.buffer).setUint32(i*4,w,true));
 const rounded=roundWeight24(raw.slice(),[{offset:0,bytes:raw.length}]),n=words.length,planes=new Uint8Array(n*3),low=new Uint8Array(n);
 for(let i=0;i<n;i++){
  const w=new DataView(rounded.buffer).getUint32(i*4,true);
  planes[i]=(w>>>23)&255;planes[n+i]=(w>>>8)&255;planes[2*n+i]=((w>>>24)&128)|((w>>>16)&127);low[i]=words[i]&255;
 }
 const metadata=new Uint8Array([9,8,7,6,5]),payloads=new Map(),chunks={},bundles={};
 async function add(data,decoded,encoding='identity',extra={}){
  const packed=new Uint8Array(deflateSync(data)),key=await sha256(packed);payloads.set(key,packed);
  chunks[key]={bytes:packed.length,rawBytes:decoded.length,rawSha256:await sha256(decoded),codec:'deflate',encoding,legacy:[],bundle:key,offset:0,...extra};
  bundles[key]={bytes:packed.length,members:[key]};return key;
 }
 const originalPacked=new Uint8Array(deflateSync(raw)),old={sha256:await sha256(originalPacked),bytes:originalPacked.length,rawBytes:raw.length,rawSha256:await sha256(raw),codec:'deflate',encoding:'identity',legacy:[]};
 const correction=await add(low,low),base=await add(planes,rounded,'fp24',{encodedBytes:planes.length,correction,original:old}),header=await add(metadata,metadata);
 const assemble=weights=>new Uint8Array([...metadata.slice(0,3),...weights,...metadata.slice(3)]);
 const compact=assemble(rounded),exact=assemble(raw);
 const manifest={format:3,codec:'progressive-fp24-v1',policy:'encoder-matmul-fp24-half-up-v1',chunks,bundles,files:{model:{bytes:compact.length,sha256:await sha256(compact),exactSha256:await sha256(exact),pieces:[{offset:0,length:3,sourceOffset:0,chunk:header},{offset:3,length:raw.length,sourceOffset:0,chunk:base},{offset:3+raw.length,length:2,sourceOffset:3,chunk:header}]},[`__corrections/${base}`]:{bytes:low.length,sha256:await sha256(low),exactSha256:await sha256(low),pieces:[{offset:0,length:low.length,sourceOffset:0,chunk:correction}]}}};
 const entries=new Map(),disk={open:async()=>({match:async key=>entries.get(key)?.clone(),put:async(key,value)=>entries.set(key,value.clone()),delete:async key=>entries.delete(key)}),delete:async()=>entries.clear()};
 const requests=[];
 const cache=new VerifiedModelCache({cacheStorage:noStorage?null:disk,fetcher:async url=>{const key=url.split('/').at(-1).replace('.pack','');requests.push(key);return new Response(payloads.get(key));}});
 return {manifest,cache,requests,entries,payloads,base,correction,header,old,originalPacked,compact,exact,pack:new ProgressiveModelPackage(manifest,'/test',cache)};
}

test('compact omits correction bundles; exact upgrade reuses base and restores original bytes',async()=>{
 const s=await setup();assert.deepEqual(await s.pack.load('model'),s.compact);
 assert.equal(s.requests.length,2);assert.ok(!s.requests.includes(s.correction));
 const before=s.cache.stats.networkBytes;
 assert.deepEqual(await s.pack.load('model',()=>{},'exact'),s.exact);
 assert.equal(s.requests.length,3);assert.equal(s.cache.stats.networkBytes-before,s.manifest.chunks[s.correction].bytes);
 assert.deepEqual(await s.pack.load('model'),s.compact);
 assert.equal(s.requests.length,3);assert.equal(s.cache.activeTransfers,0);
});

test('previous full-precision cache serves compact and original modes without weight downloads',async()=>{
 const s=await setup();s.entries.set(s.cache.key(s.old.sha256),new Response(s.originalPacked));
 s.entries.set(s.cache.key(s.header),new Response(s.payloads.get(s.header)));
 assert.deepEqual(await s.pack.load('model'),s.compact);
 assert.deepEqual(await s.pack.load('model',()=>{},'exact'),s.exact);
 assert.equal(s.requests.length,0);assert.ok(s.cache.stats.originalWeightCacheHits>=3);
});

test('fresh loader reuses saved correction data with network disabled',async()=>{
 const s=await setup();await s.pack.load('model',()=>{},'exact');s.cache.fetcher=async()=>{throw Error('offline');};
 const fresh=new ProgressiveModelPackage(s.manifest,'/another-revision',s.cache);
 assert.deepEqual(await fresh.load('model',()=>{},'exact'),s.exact);
 assert.equal(s.requests.length,3);
});

test('corrupt correction cannot pass verification or poison the saved base',async()=>{
 const s=await setup();await s.pack.load('model');
 const original=s.payloads.get(s.correction);s.payloads.set(s.correction,new Uint8Array(original.length));
 await assert.rejects(s.pack.load('model',()=>{},'exact'),/checksum/);
 assert.equal(s.cache.activeTransfers,0);assert.ok(s.entries.has(s.cache.key(s.base)));
 s.payloads.set(s.correction,original);assert.deepEqual(await s.pack.load('model',()=>{},'exact'),s.exact);
});

test('no-storage mode supports exact restoration; invalid correction layouts fail early',async()=>{
 const s=await setup({noStorage:true});assert.deepEqual(await s.pack.load('model',()=>{},'exact'),s.exact);
 const bad=structuredClone(s.manifest);bad.chunks[s.base].encodedBytes++;assert.throws(()=>validatePackage(bad),/progressive/);
 const missing=structuredClone(s.manifest);delete missing.files[`__corrections/${s.base}`];assert.throws(()=>validatePackage(missing),/correction/);
 const hash=structuredClone(s.manifest);hash.files.model.exactSha256='bad';assert.throws(()=>validatePackage(hash),/checksum/);
 await assert.rejects(s.pack.load('model',()=>{},'unknown'),/mode/);
});
