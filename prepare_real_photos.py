"""Prepare deterministic real-photo inputs without modifying synthetic fixtures."""
from pathlib import Path
import hashlib,json,tarfile
import numpy as np
from PIL import Image,ImageOps
root=Path(__file__).resolve().parent
photos=json.loads((root/'fixtures/real/manifest.json').read_text())
records=[]
for profile in ('fast','detail'):
 base=root/'public/models/profiles'/profile
 meta=json.loads((base/'manifest.json').read_text());_,_,h,w=meta['inputShape']
 for row in photos:
  source=root/'fixtures/real'/f'{row["id"]}.jpg'
  assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
  dest=base/'cases'/row['id'];dest.mkdir(parents=True,exist_ok=True)
  (dest/'photo.jpg').write_bytes(source.read_bytes())
  image=ImageOps.fit(Image.open(source).convert('RGB'),(w,h),method=Image.Resampling.BICUBIC)
  array=(np.asarray(image,dtype=np.float32)/255).transpose(2,0,1)[None].copy()
  array.astype('<f4').tofile(dest/'input.f32')
  records.append(dict(profile=profile,caseName=row['id'],shape=list(array.shape),sha256=hashlib.sha256((dest/'input.f32').read_bytes()).hexdigest()))
(root/'benchmarks/real-photo-inputs.json').write_text(json.dumps(records,indent=2)+'\n')
with tarfile.open(root/'results/real-photo-inputs.tar.gz','w:gz') as archive:
 for profile in ('fast','detail'):
  base=root/'public/models/profiles'/profile
  archive.add(base/'manifest.json',arcname=f'inputs/{profile}/manifest.json')
  for row in photos:archive.add(base/'cases'/row['id']/'input.f32',arcname=f'inputs/{profile}/cases/{row["id"]}/input.f32')
 archive.add(root/'benchmark_original.py',arcname='benchmark_original.py')
print(json.dumps(records,indent=2))
