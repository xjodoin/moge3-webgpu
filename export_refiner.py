"""Export MoGe-3 refiner tensor blocks; coordinate indexing is handled in JS.
Uses the original checkpoint and FlexGEMM (Co,kD,kH,kW,Ci) kernel ordering.
"""
from pathlib import Path
import json,torch
from torch import nn
ROOT=Path(__file__).resolve().parent
out=ROOT/'public/models/refiner';out.mkdir(exist_ok=True)
s=torch.load(ROOT/'weights/model.pt',map_location='cpu',weights_only=True)['model']
s={k.removeprefix('refiner.'):v for k,v in s.items() if k.startswith('refiner.')}
def linear(prefix):
 w=s[prefix+'.weight'];m=nn.Linear(w.shape[1],w.shape[0]);m.load_state_dict({'weight':w,'bias':s[prefix+'.bias']});return m
class Block(nn.Module):
 def __init__(self,p):
  super().__init__();c=s[p+'.norm1.weight'].numel();self.c=c
  self.norm=nn.LayerNorm(c,eps=1e-6);self.norm.load_state_dict({k:s[p+'.norm1.'+k] for k in ['weight','bias']})
  self.w1=nn.Parameter(s[p+'.conv1.weight'].reshape(c,-1).T.contiguous());self.b1=nn.Parameter(s[p+'.conv1.bias'])
  self.w2=nn.Parameter(s[p+'.conv2.weight'].reshape(c,-1).T.contiguous());self.b2=nn.Parameter(s[p+'.conv2.bias'])
 def conv(self,x,neighbors,w,b):
  pad=torch.cat([x,torch.zeros((1,self.c),dtype=x.dtype,device=x.device)],dim=0)
  return pad[neighbors].reshape(-1,27*self.c)@w+b
 def forward(self,x,neighbors):
  h=self.conv(torch.nn.functional.silu(self.norm(x)),neighbors,self.w1,self.b1)
  return x+self.conv(torch.nn.functional.silu(h),neighbors,self.w2,self.b2)
class Fuse(nn.Module):
 def __init__(self):
  super().__init__();self.enc=linear('encoder_fuse');self.a=linear('fuse_proj.0');self.b=linear('fuse_proj.2')
 def forward(self,x,enc):return self.b(torch.nn.functional.silu(self.a(torch.cat([x,self.enc(enc)],dim=1))))
manifest={}
def export(name,module,args,names):
 module.eval();torch.onnx.export(module,args,str(out/(name+'.onnx')),opset_version=17,dynamo=False,input_names=names,output_names=['output'],dynamic_axes={k:{0:'rows'} for k in names+['output']})
 manifest[name]={'inputs':names,'bytes':(out/(name+'.onnx')).stat().st_size}
for name in ['input_proj','out_proj']+[f'downsample_blocks.{i}.linear' for i in range(4)]+[f'upsample_blocks.{i}.linear' for i in range(4)]:
 m=linear(name);export(name,m,(torch.zeros(8,m.in_features),),['features'])
for prefix in [f'down_stages.{i}.0' for i in range(5)]+['bottleneck_stage.0']+[f'up_stages.{i}.0' for i in range(4)]:
 m=Block(prefix);export(prefix,m,(torch.zeros(8,m.c),torch.zeros(8,27,dtype=torch.int32)),['features','neighbors'])
export('fuse',Fuse(),(torch.zeros(8,512),torch.zeros(8,1026)),['features','encoder'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Exported',len(manifest),'refiner blocks')
