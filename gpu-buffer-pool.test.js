import test from 'node:test';
import assert from 'node:assert/strict';
import {GpuBufferPool} from './gpu-buffer-pool.js';
function pool(maxBytes=1024){const device={createBuffer:({size,usage})=>({size,usage,mapState:'unmapped',destroyed:false,destroy(){assert.equal(this.destroyed,false);this.destroyed=true;}})};return new GpuBufferPool(device,{maxBytes});}
test('only released compatible leases are reused, with idempotent release',()=>{
 const p=pool(),a=p.acquire(16,1),b=p.acquire(16,1);assert.notEqual(a.buffer,b.buffer);a.release();a.release();
 const c=p.acquire(16,1);assert.equal(c.buffer,a.buffer);const d=p.acquire(16,2);assert.notEqual(c.buffer,d.buffer);
 b.release();c.release();d.release();assert.equal(p.snapshot().hits,1);p.clear();assert.equal(p.bytes,0);
});
test('unload invalidates active leases and destroys free buffers',()=>{
 const p=pool(),a=p.acquire(16,1),b=p.acquire(16,1);a.release();p.clear();assert.ok(a.buffer.destroyed);b.release();assert.ok(b.buffer.destroyed);assert.equal(p.bytes,0);
});
test('retained memory is bounded and mapped or disabled buffers are destroyed',()=>{
 const p=pool(16),a=p.acquire(16,1),b=p.acquire(32,1);a.release();b.release();assert.ok(b.buffer.destroyed);assert.equal(p.bytes,16);
 const c=p.acquire(16,1);c.buffer.mapState='mapped';c.release();assert.ok(c.buffer.destroyed);
 p.enabled=false;const d=p.acquire(16,1);d.release();assert.ok(d.buffer.destroyed);
});
