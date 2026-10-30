"""Independent MLX stage references AND full native inference on identical input."""
from pathlib import Path
import argparse
import numpy as np,json,time
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.moge3.moge3 import resize
from mlx_vlm.models.moge3.geometry import recover_focal_shift, intrinsics_from_focal_center, depth_map_to_point_map
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--profile',default='fast');parser.add_argument('--mlx-model',type=Path,default=ROOT/'weights/mlx');args=parser.parse_args()
base=ROOT/'public/models' if args.profile=='legacy' else ROOT/'public/models/profiles'/args.profile
meta=json.loads((base/'manifest.json').read_text());_,_,height,width=meta['inputShape']
gh,gw=meta.get('grid',[272,272]);eh,ew=meta.get('encoderGrid',[17,17]);steps=meta.get('refineSteps',1) or 1
mx.set_memory_limit(8_000_000_000);mx.set_cache_limit(256_000_000)
model,_=load(str(args.mlx_model),trust_remote_code=False)
reports=[]
for name in json.loads((base/'cases.json').read_text()):
 p=base/'cases'/name
 raw=mx.array(np.fromfile(p/'reference-raw_coord.f32',np.float32).reshape(1,gh,gw,3))
 enc=mx.array(np.fromfile(p/'reference-encoder_feature.f32',np.float32).reshape(1,eh,ew,1026))
 t=time.perf_counter();refined=raw
 for _ in range(steps):
  logz=model._refine_logz(refined,enc);refined=mx.concatenate([refined[...,:2],logz[...,None]],axis=-1)
  mx.eval(refined)
 points=model._remap_points(resize(refined,(height,width)));mx.eval(points)
 np.array(refined).tofile(p/'reference-refined-raw.f32');np.array(points).tofile(p/'reference-refined-points.f32')
 mask=mx.array(np.fromfile(p/'reference-mask.f32',np.float32).reshape(1,height,width))>.5
 scale=mx.array(np.fromfile(p/'reference-metric_scale.f32',np.float32))
 focal,shift=recover_focal_shift(points,mask);aspect=width/height
 fx=focal/2*(1+aspect**2)**.5/aspect;fy=fx*aspect;half=mx.full(fx.shape,.5)
 intrinsics=intrinsics_from_focal_center(fx,fy,half,half)
 depth=(points[...,2]+shift[:,None,None])*scale[:,None,None]
 projected=depth_map_to_point_map(depth,intrinsics);valid=mask&(depth>0)
 projected=mx.where(valid[...,None],projected,float('inf'));mx.eval(projected)
 np.array(projected).tofile(p/'reference-projected-points.f32')
 (p/'reference-camera.json').write_text(json.dumps({'focal':focal.item(),'shift':shift.item(),'intrinsics':np.array(intrinsics).reshape(-1).tolist()}))
 stage_seconds=time.perf_counter()-t
 del raw,enc,refined,points,projected,depth;mx.clear_cache();mx.reset_peak_memory()
 image=mx.array(np.fromfile(p/'input.f32',np.float32).reshape(1,3,height,width).transpose(0,2,3,1));mx.eval(image)
 t=time.perf_counter()
 native=model.infer(image,num_tokens=meta['numTokens'],refine_steps=steps)
 mx.eval(native)
 seconds=time.perf_counter()-t;peak=mx.get_peak_memory()
 np.array(native['points']).tofile(p/'reference-native-points.f32')
 np.array(native['intrinsics']).tofile(p/'reference-native-intrinsics.f32')
 np.array(native['mask'],dtype=np.uint8).tofile(p/'reference-native-mask.u8')
 report={'case':name,'profile':args.profile,'stageReferenceSeconds':stage_seconds,'nativeInferenceSeconds':seconds,'mlxPeakBytes':peak,'tokens':meta['numTokens'],'refineSteps':steps,'inputShape':meta['inputShape']}
 reports.append(report);print(json.dumps(report),flush=True)
 del native,image;mx.clear_cache()
(base/'native-validation.json').write_text(json.dumps(reports,indent=2))
