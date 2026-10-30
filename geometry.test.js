import test from 'node:test';
import assert from 'node:assert/strict';
import {recoverCamera,project} from './geometry.js';

test('recovers a known camera and depth shift from a non-planar scene',()=>{
 const width=256,height=256,focal=1.2,shift=.3,scale=5;
 const points=new Float32Array(width*height*3),mask=new Float32Array(width*height).fill(1);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const i=y*width+x,z=1+2*(x/width)**2+Math.sin(y/height*Math.PI);
  const u=((x+.5)/width*2-1)/Math.SQRT2,v=((y+.5)/height*2-1)/Math.SQRT2;
  points.set([u/focal*(z+shift),v/focal*(z+shift),z],i*3);
 }
 // Excluded points must not influence camera fitting.
 mask[0]=0;points.set([999,999,999],0);
 const camera=recoverCamera(points,mask,width,height);
 assert.ok(Math.abs(camera.focal-focal)<1e-4);
 assert.ok(Math.abs(camera.shift-shift)<1e-4);
 const result=project(points,mask,scale,width,height),i=97*width+73;
 assert.ok(Math.abs(result.depth[i]-(points[i*3+2]+shift)*scale)<1e-4);
 assert.ok(Math.abs(result.points[i*3]-points[i*3]*scale)<1e-4);
 assert.equal(result.mask[0],0);assert.equal(result.depth[0],Infinity);
});

test('empty foreground returns the defined camera fallback and no geometry',()=>{
 const points=new Float32Array(256*256*3),mask=new Float32Array(256*256);
 const result=project(points,mask,1);
 assert.equal(result.focal,1);assert.equal(result.shift,0);
 assert.equal(result.mask.some(Boolean),false);
 assert.ok(result.depth.every(x=>x===Infinity));
});
