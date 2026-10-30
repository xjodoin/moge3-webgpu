// Camera recovery follows MLX-VLM's MoGe-3 damped Gauss-Newton solver.
// These small coordinate operations run in JavaScript, not a learned CPU model.
export function recoverCamera(points, mask, width=256, height=256) {
  const aspect=width/height, sx=aspect/Math.sqrt(1+aspect*aspect), sy=sx/aspect;
  const samples=[];
  for(let y=0;y<64;y++)for(let x=0;x<64;x++){
    const iy=Math.floor(y*height/64),ix=Math.floor(x*width/64),i=iy*width+ix;
    if(mask[i]>.5)samples.push([points[3*i],points[3*i+1],points[3*i+2],((ix+.5)/width*2-1)*sx,((iy+.5)/height*2-1)*sy]);
  }
  function evaluate(shift){
    let a=0,b=0,da=0,db=0;
    for(const [x,y,z,u,v] of samples){
      const inv=1/(z+shift),px=x*inv,py=y*inv,qx=-px*inv,qy=-py*inv;
      a+=px*u+py*v;b+=px*px+py*py;da+=qx*u+qy*v;db+=2*(px*qx+py*qy);
    }
    b=Math.max(b,1e-12);const focal=a/b,df=(da*b-a*db)/(b*b);
    let cost=0,g=0,h=0;
    for(const [x,y,z,u,v] of samples){
      const inv=1/(z+shift),px=x*inv,py=y*inv;
      const rx=focal*px-u,ry=focal*py-v,jx=df*px-focal*px*inv,jy=df*py-focal*py*inv;
      cost+=rx*rx+ry*ry;g+=jx*rx+jy*ry;h+=jx*jx+jy*jy;
    }
    return {cost,g,h,focal};
  }
  let shift=0,lambda=.001,current=evaluate(0);
  if(samples.length<2)current.focal=1;
  else for(let i=0;i<30;i++){
    const step=-current.g/Math.max(current.h*(1+lambda),1e-30),trial=evaluate(shift+step);
    const tiny=Math.abs(step)<1e-8*Math.max(Math.abs(shift),1);
    if(trial.cost<current.cost){
      const converged=current.cost-trial.cost<=.001*trial.cost;
      shift+=step;current=trial;lambda/=3;if(converged||tiny)break;
    }else{lambda*=5;if(tiny)break;}
  }
  const fx=current.focal/(2*sx),fy=current.focal/(2*sy);
  if(!Number.isFinite(fx)||fx<=0||!Number.isFinite(shift))throw Error('Camera recovery failed');
  return {focal:current.focal,shift,intrinsics:[fx,0,.5,0,fy,.5,0,0,1]};
}

export function project(points, mask, metricScale, width=256, height=256){
  const camera=recoverCamera(points,mask,width,height),output=new Float32Array(points.length),depth=new Float32Array(width*height),valid=new Uint8Array(depth.length);
  for(let i=0;i<depth.length;i++){
    const z=(points[3*i+2]+camera.shift)*metricScale;
    valid[i]=mask[i]>.5&&z>0&&Number.isFinite(z);depth[i]=valid[i]?z:Infinity;
    output[i*3]=valid[i]?((i%width+.5)/width-.5)/camera.intrinsics[0]*z:Infinity;
    output[i*3+1]=valid[i]?((Math.floor(i/width)+.5)/height-.5)/camera.intrinsics[4]*z:Infinity;
    output[i*3+2]=depth[i];
  }
  return {points:output,depth,mask:valid,...camera};
}
