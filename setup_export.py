"""Fetch pinned upstream source/checkpoint for the local browser experiment."""
from pathlib import Path
import json,argparse
import shutil
import subprocess
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');args=parser.parse_args()
SOURCE_COMMIT = '74fbce054ebed49800de42d0ad0e83495065719a'
MODEL = 'Ruicheng/moge-3-vitl'
REVISION = '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be'
source = ROOT / 'vendor/MoGe'
if not source.exists():
    source.parent.mkdir(exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/microsoft/MoGe.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', SOURCE_COMMIT], check=True)
actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if actual != SOURCE_COMMIT:
    raise RuntimeError(f'Expected MoGe source {SOURCE_COMMIT}, found {actual}; source was not modified')
weights = ROOT / 'weights'
weights.mkdir(exist_ok=True)
if not (weights / 'model.pt').exists():
    checkpoint = hf_hub_download(MODEL, 'model.pt', revision=REVISION)
    shutil.copyfile(checkpoint, weights / 'model.pt')
public = ROOT / 'public/models'
public.mkdir(parents=True, exist_ok=True)
(public / 'upstream.json').write_text(json.dumps({'repository': MODEL, 'revision': REVISION}))
print('Pinned source and checkpoint ready; run export_dense.py and export_refiner.py next.')

if args.native:
    from huggingface_hub import snapshot_download
    snapshot_download("mlx-community/moge-3-vitl-mlx-fp32",revision="f5ceca7512506731fb1c50307c313da76aac500f",local_dir=weights/"mlx",token=False,allow_patterns=["*.json","*.safetensors","README.md"])
