"""Create experimental half-precision copies; never overwrite FP32 or goldens.

compute: ONNX FP16 operations with FP32 I/O; WGSL FP16 weights/FP32 math.
storage: FP16 ONNX initializers cast to FP32; WGSL FP16 weights/FP32 math.
Usage: .venv/bin/python export_fp16.py --mode compute --profile fast
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'public/models'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def storage_model(model):
    """Store weights as FP16, retain all operations/activations in FP32."""
    casts = []
    for tensor in model.graph.initializer:
        if tensor.data_type != TensorProto.FLOAT:
            continue
        original = tensor.name
        value = numpy_helper.to_array(tensor)
        half = value.astype(np.float16)
        if not np.isfinite(half).all():
            raise ValueError(f'FP16 overflow: {original}')
        tensor.CopyFrom(numpy_helper.from_array(half, original + '__fp16_storage'))
        casts.append(helper.make_node('Cast', [tensor.name], [original],
                                      name=original + '__restore_fp32', to=TensorProto.FLOAT))
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(casts + nodes)
    return model


def convert(source, target, mode):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    if mode == 'compute':
        # Evaluate scalar/shape constants while still FP32. WASM cannot fold
        # FP16 scalar Sqrt/Pow nodes and would otherwise request CPU fallback.
        from onnxscript import optimizer
        optimizer.fold_constants(model, input_size_limit=8192, output_size_limit=262144)
        model = convert_float_to_float16(model, keep_io_types=True,
                                        min_positive_val=5.960464477539063e-8,
                                        max_finite_val=65504.0)
    else:
        model = storage_model(model)
    # ORT's converter appends I/O casts; ONNX requires producer-first order.
    pending = list(model.graph.node)
    known = {t.name for t in model.graph.input} | {t.name for t in model.graph.initializer}
    ordered = []
    while pending:
        remaining = []
        for node in pending:
            if all(not name or name in known for name in node.input):
                ordered.append(node)
                known.update(node.output)
            else:
                remaining.append(node)
        if len(remaining) == len(pending):
            raise ValueError('Unresolved graph dependencies')
        pending = remaining
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    row = {'source': str(source.relative_to(MODELS)),
           'target': str(target.relative_to(MODELS)), 'sourceSha256': sha(source),
           'sha256': sha(target), 'sourceBytes': source.stat().st_size,
           'bytes': target.stat().st_size}
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['compute', 'storage'], required=True)
    p.add_argument('--profile', choices=['fast', 'detail', 'refiner'], required=True)
    args = p.parse_args()
    folder = MODELS / 'precision' / f'fp16-{args.mode}'
    if args.profile == 'refiner':
        packed = MODELS / 'precision/fp16-weights/refiner-wgsl'
        if not packed.exists():
            packed.mkdir(parents=True)
            manifest = json.loads((MODELS / 'refiner-wgsl/manifest.json').read_text())
            manifest.update(format=2, dtype='float16-le', accumulation='float32')
            for name, meta in manifest['blocks'].items():
                source = MODELS / f'refiner-wgsl/{name}.bin'
                if sha(source) != meta['sha256']:
                    raise ValueError(f'Source checksum mismatch: {name}')
                half = np.fromfile(source, dtype='<f4').astype('<f2')
                if not np.isfinite(half).all():
                    raise ValueError(f'FP16 overflow: {name}')
                data = half.tobytes()
                (packed / f'{name}.bin').write_bytes(data)
                meta.update(sourceSha256=meta['sha256'], sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            (packed / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        paths = [path for path in sorted((MODELS / 'refiner').glob('*.onnx'))
                 if path.stem not in json.loads((MODELS / 'refiner-wgsl/manifest.json').read_text())['blocks']]
    else:
        paths = [MODELS / f'profiles/{args.profile}/moge3-dense.onnx']
    rows = [convert(path, folder / path.relative_to(MODELS), args.mode) for path in paths]
    metadata = {'precision': f'fp16-{args.mode}', 'onnxruntimeConverter': __import__('onnxruntime').__version__,
                'keepIoTypes': True, 'files': rows}
    if args.profile != 'refiner':
        meta = json.loads(paths[0].with_name('manifest.json').read_text())
        meta.update(modelBytes=rows[0]['bytes'], sha256=rows[0]['sha256'], precision=metadata['precision'])
        (folder / f'profiles/{args.profile}/manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    (folder / f'conversion-{args.profile}.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
