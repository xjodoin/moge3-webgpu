"""Independent native MLX references for 2/3 passes; preserves one-pass files."""
from pathlib import Path
import json
import time
import numpy as np
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.moge3.moge3 import resize
from mlx_vlm.models.moge3.geometry import recover_focal_shift, intrinsics_from_focal_center, depth_map_to_point_map

ROOT=Path(__file__).resolve().parent
mx.set_memory_limit(8_000_000_000);mx.set_cache_limit(256_000_000)
model,_=load(str(ROOT/'weights/mlx'),trust_remote_code=False)
reports=[]
for profile in ('fast','detail'):
    base=ROOT/'public/models/profiles'/profile
    meta=json.loads((base/'manifest.json').read_text());_,_,h,w=meta['inputShape'];gh,gw=meta['grid'];eh,ew=meta['encoderGrid']
    for name in json.loads((base/'cases.json').read_text()):
        folder=base/'cases'/name;debug=folder/'diagnostics';debug.mkdir(exist_ok=True)
        image=mx.array(np.fromfile(folder/'input.f32',np.float32).reshape(1,3,h,w).transpose(0,2,3,1));mx.eval(image)
        original=model._refine_logz
        counter=[0]
        def trace(raw,enc):
            step=counter[0];counter[0]+=1
            if step==0:
                np.array(raw).tofile(debug/'native-raw-0.f32');np.array(enc).tofile(debug/'native-encoder.f32')
            z=original(raw,enc)
            coord=mx.concatenate([raw[...,:2],z[...,None]],axis=-1)
            np.array(coord).tofile(debug/f'native-raw-{step+1}.f32')
            return z
        for steps in (2,3):
            dest=folder/'passes'/str(steps);dest.mkdir(parents=True,exist_ok=True)
            if steps==3:model._refine_logz=trace
            t=time.perf_counter();native=model.infer(image,num_tokens=meta['numTokens'],refine_steps=steps);mx.eval(native)
            seconds=time.perf_counter()-t
            model._refine_logz=original
            for key in ('points','intrinsics'):np.array(native[key]).tofile(dest/f'reference-native-{key}.f32')
            np.array(native['mask'],dtype=np.uint8).tofile(dest/'reference-native-mask.u8')
            reports.append(dict(profile=profile,caseName=name,refineSteps=steps,nativeSeconds=seconds,diagnosticTracing=steps==3))
            del native
        raw=mx.array(np.fromfile(folder/'reference-raw_coord.f32',np.float32).reshape(1,gh,gw,3))
        enc=mx.array(np.fromfile(folder/'reference-encoder_feature.f32',np.float32).reshape(1,eh,ew,1026))
        mask=mx.array(np.fromfile(folder/'reference-mask.f32',np.float32).reshape(1,h,w))>.5
        scale=mx.array(np.fromfile(folder/'reference-metric_scale.f32',np.float32))
        for steps in (1,2,3):
            z=model._refine_logz(raw,enc);raw=mx.concatenate([raw[...,:2],z[...,None]],axis=-1);mx.eval(raw)
            if steps==1:continue
            dest=folder/'passes'/str(steps)
            points=model._remap_points(resize(raw,(h,w)));mx.eval(points)
            np.array(raw).tofile(dest/'reference-refined-raw.f32');np.array(points).tofile(dest/'reference-refined-points.f32')
            focal,shift=recover_focal_shift(points,mask);aspect=w/h;fx=focal/2*(1+aspect**2)**.5/aspect;fy=fx*aspect;half=mx.full(fx.shape,.5)
            intrinsics=intrinsics_from_focal_center(fx,fy,half,half)
            depth=(points[...,2]+shift[:,None,None])*scale[:,None,None]
            projected=mx.where((mask&(depth>0))[...,None],depth_map_to_point_map(depth,intrinsics),float('inf'));mx.eval(projected)
            np.array(projected).tofile(dest/'reference-projected-points.f32')
        print(json.dumps(reports[-2:]),flush=True)
        del raw,enc,mask,scale,image,points,projected;mx.clear_cache()
(ROOT/'benchmarks/multipass-native-reference.json').write_text(json.dumps(reports,indent=2)+'\n')
