"""Export FP32-only constant-folding controls independently of FP16 casts."""
from pathlib import Path
import hashlib
import json
import onnx
from onnxscript import optimizer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'public/models'


def main():
    records = []
    for profile in ['fast', 'detail']:
        path = MODELS / f'precision/fold-control/{profile}/model.onnx'
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = onnx.load(MODELS / f'profiles/{profile}/moge3-dense.onnx')
        optimizer.fold_constants(model, input_size_limit=8192, output_size_limit=262144)
        onnx.checker.check_model(model)
        onnx.save(model, path)
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        records.append({'profile': profile, 'sha256': sha, 'bytes': path.stat().st_size})
        print(records[-1], flush=True)
        del model
    (ROOT / 'benchmarks/fp16-folding-exports.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
