"""Export the actual MoGe-3 dense pass. Sparse refinement is explicitly excluded."""
import os, sys, types, json, time, argparse, hashlib
from pathlib import Path
os.environ['XFORMERS_DISABLED']='1'
import torch
import numpy as np
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--references-only',action='store_true')
parser.add_argument('--profile',default='fast')
parser.add_argument('--width',type=int,default=512)
parser.add_argument('--height',type=int,default=344)
parser.add_argument('--tokens',type=int,default=1200)
parser.add_argument('--steps',type=int,default=1)
args=parser.parse_args()
DEST=ROOT/'public/models' if args.profile=='legacy' else ROOT/'public/models/profiles'/args.profile
DEST.mkdir(parents=True,exist_ok=True)
WIDTH,HEIGHT=args.width,args.height
ASPECT=WIDTH/HEIGHT
BH,BW=round((args.tokens/ASPECT)**.5),round((args.tokens*ASPECT)**.5)
sys.path.insert(0,str(ROOT/'vendor/MoGe'))
# Upstream imports FlexGEMM even when refiner=None. Avoid importing its CUDA
# extension for this dense-only experiment. Any attempted use fails loudly.
sparse=types.ModuleType('moge.model.modules.sparse_unet')
class UnavailableRefiner:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Sparse refinement is not implemented by this export')
sparse.Sparse3DUNet=UnavailableRefiner
sys.modules[sparse.__name__]=sparse
from moge.model.v3 import MoGeModel
from moge.utils.geometry_torch import normalized_view_plane_uv
ckpt=torch.load(ROOT/'weights/model.pt',map_location='cpu',weights_only=True)
config={**ckpt['model_config'],'refiner':None}
model=MoGeModel(**config).eval()
weights={k:v for k,v in ckpt['model'].items() if not k.startswith('refiner.')}
model.load_state_dict(weights,strict=True)
del ckpt,weights
from native_export import prepare_native_encoder
prepare_native_encoder(model,HEIGHT,WIDTH,BH,BW)
model.onnx_compatible_mode=True
torch.set_num_threads(6)
class Dense(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=model
        model.points_head.register_forward_hook(lambda module,args,out:setattr(self,'raw',out[-1]))
        model.encoder.register_forward_hook(lambda module,args,out:setattr(self,'encoder',out[0]))
    def forward(self,image):
        out=self.model(image,args.tokens,refine_steps=0)
        uv=normalized_view_plane_uv(width=BW,height=BH,aspect_ratio=ASPECT,dtype=image.dtype,device=image.device).permute(2,0,1)[None]
        feature=torch.cat([self.encoder,uv],dim=1).permute(0,2,3,1)
        return out['points'],out['normal'],out['mask'],out['metric_scale'],self.raw.permute(0,2,3,1),feature
wrapper=Dense().eval()

photos=[('sample-backyard','sample-backyard.jpg'),('brick-ranch','test-photos/brick-ranch-yard-v1.jpg'),('raised-deck','test-photos/raised-deck-suburban-v1.jpg')]
keys=['points','normal','mask','metric_scale','raw_coord','encoder_feature']
for name,filename in photos:
    dest=DEST/'cases'/name;dest.mkdir(parents=True,exist_ok=True)
    photo=ROOT/'fixtures'/f'{name}.jpg'
    Image.open(photo).convert('RGB').save(dest/'photo.jpg')
    x=np.asarray(ImageOps.fit(Image.open(photo).convert('RGB'),(WIDTH,HEIGHT),method=Image.Resampling.BICUBIC),dtype=np.float32)/255
    x=np.transpose(x,(2,0,1))[None].copy();x.tofile(dest/'input.f32')
    start=time.perf_counter()
    with torch.inference_mode():
        reference=wrapper(torch.from_numpy(x))
        for key,t in zip(keys,reference):t.numpy().tofile(dest/f'reference-{key}.f32')
    print(name,'reference seconds',time.perf_counter()-start,flush=True)
(DEST/'cases.json').write_text(json.dumps([name for name,_ in photos]))
if not args.references_only:
    with torch.inference_mode():
        torch.onnx.export(wrapper,(torch.from_numpy(x),),str(DEST/'moge3-dense.onnx'),
            input_names=['image'],output_names=keys,opset_version=17,dynamo=False,external_data=True)
metadata={'upstream':json.loads((ROOT/'public/models/upstream.json').read_text()),'sourceCommit':'74fbce054ebed49800de42d0ad0e83495065719a','exportSemantics':'native-resize-and-position-v2','profile':args.profile,'inputShape':[1,3,HEIGHT,WIDTH], 'numTokens':args.tokens,'refineSteps':args.steps,'stage':'dense','preprocessing':'center-crop bicubic', 'grid':[BH*16,BW*16], 'encoderGrid':[BH,BW],'outputs':{k:list(t.shape) for k,t in zip(keys,reference)}}
metadata['modelBytes']=(DEST/'moge3-dense.onnx').stat().st_size if (DEST/'moge3-dense.onnx').exists() else None
(DEST/'manifest.json').write_text(json.dumps(metadata,indent=2))
print('References complete' if args.references_only else 'Export complete',flush=True)
