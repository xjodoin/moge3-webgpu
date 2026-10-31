// Diagnostic only: never imported by the application build.
import * as ort from 'onnxruntime-web/webgpu';
import {denseSession,releaseSessions} from './runtime.js';
import {refine} from './refiner.js';
import {voxelize} from './sparse-maps.js';
import {compare} from './comparison.js';
const read=async path=>new Float32Array(await(await fetch(path)).arrayBuffer());
export async function traceMultipass(profile='detail',photo='raised-deck',seed='browser'){
  const base=`/models/profiles/${profile}`,folder=`${base}/cases/${photo}`;
  const meta=await(await fetch(`${base}/manifest.json`)).json(),[h,w]=meta.grid;
  const session=await denseSession(profile,meta,()=>{});
  const image=new ort.Tensor('float32',await read(`${folder}/input.f32`),meta.inputShape);
  const out=await session.run({image});const reports=[];
  try{
    let raw=seed==='browser'?out.raw_coord.data:await read(`${folder}/${seed==='native'?'diagnostics/native-raw-0':'reference-raw_coord'}.f32`);
    const encoder=seed==='browser'?out.encoder_feature:await read(`${folder}/${seed==='native'?'diagnostics/native-encoder':'reference-encoder_feature'}.f32`);
    for(let step=0;step<=3;step++){
      const expected=await read(`${folder}/diagnostics/native-raw-${step}.f32`);
      const actualVoxels=voxelize(raw,h,w),expectedVoxels=voxelize(expected,h,w);
      let voxelMismatches=0,maxZ=0,maxIndex=0;
      for(let i=0;i<h*w;i++){
        if(actualVoxels.coords[3*i+2]!==expectedVoxels.coords[3*i+2])voxelMismatches++;
        if(Math.abs(raw[3*i+2]-expected[3*i+2])>maxZ){maxZ=Math.abs(raw[3*i+2]-expected[3*i+2]);maxIndex=i;}
      }
      reports.push({step,seed,voxelMismatches,shape:actualVoxels.shape,expectedShape:expectedVoxels.shape,maxZ,maxIndex,metrics:compare(raw,expected)});
      if(step<3)raw=await refine(raw,encoder,()=>{},{grid:meta.grid,width:meta.inputShape[3],height:meta.inputShape[2],sparseBackend:'wgsl'});
    }
    return reports;
  }finally{image.dispose();for(const t of Object.values(out))t.dispose();await releaseSessions();}
}
