import test from 'node:test';
import assert from 'node:assert/strict';
import {SessionCache} from './session-cache.js';
test('reuses in-flight and completed sessions, then releases exactly once',async()=>{
  const cache=new SessionCache();let creates=0,releases=0;
  const create=async()=>{creates++;return {release:async()=>{releases++;}};};
  const [a,b]=await Promise.all([cache.get('dense',create),cache.get('dense',create)]);
  assert.equal(a,b);assert.equal(await cache.get('dense',create),a);assert.equal(creates,1);
  await cache.clear();await cache.clear();assert.equal(releases,1);assert.equal(cache.entries.size,0);
  await cache.get('dense',create);assert.equal(creates,2);await cache.clear();
});
test('a rejected load can be retried and clear waits for a pending load',async()=>{
  const cache=new SessionCache();await assert.rejects(cache.get('x',()=>{throw Error('network');}),/network/);
  let resolve,released=0;
  const loading=cache.get('x',()=>new Promise(r=>{resolve=r;}));await Promise.resolve();
  const clearing=cache.clear();resolve({release:async()=>{released++;}});
  await loading;await clearing;assert.equal(released,1);assert.equal(cache.entries.size,0);
});
test('a failing release does not prevent other sessions from being released',async()=>{
  const cache=new SessionCache();let released=0;
  await cache.get('broken',()=>({release(){throw Error('device lost');}}));
  await cache.get('healthy',()=>({release(){released++;}}));
  await assert.rejects(cache.clear(),/device lost/);
  assert.equal(released,1);assert.equal(cache.snapshot().retained,0);
});
