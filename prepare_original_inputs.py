"""Package exact browser inputs and CUDA benchmark scripts, without credentials."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
dest = root / 'results/original-inputs.tar.gz'
dest.parent.mkdir(exist_ok=True)
with tarfile.open(dest, 'w:gz') as archive:
    for profile in ('fast', 'detail'):
        base = root / 'public/models/profiles' / profile
        archive.add(base / 'manifest.json', arcname=f'inputs/{profile}/manifest.json')
        for path in sorted((base / 'cases').glob('*/input.f32')):
            archive.add(path, arcname=f'inputs/{profile}/cases/{path.parent.name}/input.f32')
    for name in ('benchmark_original.py', 'setup_original_cuda.sh'):
        archive.add(root / name, arcname=name)
print(dest)
