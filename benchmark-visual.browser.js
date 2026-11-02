// Diagnostic images are generated outside timed inference and use CUDA's depth range.
export async function captureVisual(photo,profile='detail',steps=3){
 let images;
 const result=await window.runTest(true,photo,null,profile,'wgsl',steps,true,{render:false,referenceScope:'original-only',onGeometry:async(g,{width,height,input})=>{
 const root=`/original-cuda/${profile}/${photo}/${steps}`;
 const reference=new Float32Array(await(await fetch(`${root}/points.f32`)).arrayBuffer()),mask=new Uint8Array(await(await fetch(`${root}/mask.u8`)).arrayBuffer());
 const n=width*height,depths=[];for(let i=0;i<n;i++)if(mask[i]&&Number.isFinite(reference[3*i+2]))depths.push(reference[3*i+2]);depths.sort((a,b)=>a-b);
 const lo=depths[Math.floor(depths.length*.02)],hi=depths[Math.floor(depths.length*.98)],range=Math.max(hi-lo,1e-8);
 const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');
 const draw=kind=>{const image=ctx.createImageData(width,height);for(let i=0;i<n;i++){
 let rgb;if(kind==='photo')rgb=[input[i],input[n+i],input[2*n+i]].map(v=>255*v);
 else if(kind==='error'){
 if(Boolean(mask[i])!==Boolean(g.mask[i]))rgb=[255,0,255];else if(!mask[i])rgb=[24,24,24];else{const error=Math.min(1,Math.abs(g.points[3*i+2]-reference[3*i+2])/(range*.01));rgb=[255*error,255*(1-error),0];}
 }else{const valid=kind==='cuda'?mask[i]:g.mask[i],z=kind==='cuda'?reference[3*i+2]:g.points[3*i+2];const v=255*(1-Math.max(0,Math.min(1,(z-lo)/range)));rgb=valid&&Number.isFinite(z)?[v,v,v]:[24,24,24];}
 image.data.set([...rgb,255],4*i);
 }ctx.putImageData(image,0,0);return canvas.toDataURL('image/png');};
 images={photo:draw('photo'),cuda:draw('cuda'),webgpu:draw('webgpu'),error:draw('error'),depthRange:[lo,hi],errorSaturation:range*.01};
 }});
 return {photo,profile,steps,passed:result.passed,original:result.original,...images};
}
