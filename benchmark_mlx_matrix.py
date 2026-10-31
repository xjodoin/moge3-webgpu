"""Fresh synchronized MLX 1/2/3-pass baseline; no tracing or reference writes."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm import load
import numpy as np

from benchmark_native import agreement

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/subgroup-mlx.json')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output path; existing benchmark evidence is preserved')
    mx.set_memory_limit(8_000_000_000)
    mx.set_cache_limit(256_000_000)
    model, _ = load(str(ROOT / 'weights/mlx'), trust_remote_code=False)
    dtypes = sorted({str(p.dtype) for _, p in tree_flatten(model.parameters())})
    assert dtypes == ['mlx.core.float32'], f'FP32 parameters required: {dtypes}'
    source = Path(inspect.getsourcefile(type(model)))
    revision = (ROOT / 'weights/mlx/.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
    assert revision == 'f5ceca7512506731fb1c50307c313da76aac500f', 'Use the pinned MLX checkpoint'
    report = dict(startedAt=time.time() * 1000, runs=[], warmups=[], diagnosticTracing=False,
                  returnPerStep=False, sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checkpoint='mlx-community/moge-3-vitl-mlx-fp32', revision=revision,
                  versions={k: importlib.metadata.version(k) for k in ['mlx', 'mlx-vlm', 'numpy']},
                  parameterDtypes=dtypes, method='Exact browser FP32 input, synchronized warm inference; no monkeypatches, hooks, per-step outputs or diagnostic dumps. Validation follows timing. Reference files are read-only.')
    for profile in ['fast', 'detail']:
        base = ROOT / 'public/models/profiles' / profile
        meta = json.loads((base / 'manifest.json').read_text())
        _, _, height, width = meta['inputShape']
        names = json.loads((base / 'cases.json').read_text())
        data = {name: (base / 'cases' / name / 'input.f32').read_bytes() for name in names}
        inputs = {name: mx.array(np.frombuffer(data[name], np.float32).reshape(1, 3, height, width).transpose(0, 2, 3, 1)) for name in names}
        mx.eval(list(inputs.values()))
        for steps in [1, 2, 3]:
            for _ in range(3):
                warm = model.infer(inputs[names[0]], num_tokens=meta['numTokens'], refine_steps=steps, return_per_step=False)
                mx.eval(warm)
                del warm
                report['warmups'].append(dict(profile=profile, refineSteps=steps))
            for repeat in range(args.repeats):
                for name in names:
                    mx.reset_peak_memory()
                    started = time.perf_counter()
                    result = model.infer(inputs[name], num_tokens=meta['numTokens'], refine_steps=steps, return_per_step=False)
                    mx.eval(result)
                    elapsed = (time.perf_counter() - started) * 1000
                    peak = mx.get_peak_memory()
                    folder = base / 'cases' / name
                    if steps > 1:
                        folder = folder / 'passes' / str(steps)
                    comparisons = {key: agreement(np.array(result[key]).reshape(-1), np.fromfile(folder / f'reference-native-{key}.f32', np.float32)) for key in ['points', 'intrinsics']}
                    mask_equal = np.array_equal(np.array(result['mask']).reshape(-1), np.fromfile(folder / 'reference-native-mask.u8', np.uint8))
                    row = dict(profile=profile, caseName=name, refineSteps=steps, repeat=repeat, numTokens=meta['numTokens'], inputShape=meta['inputShape'], inputSha256=hashlib.sha256(data[name]).hexdigest(), inferenceMs=elapsed, mlxPeakBytes=peak, outputs=comparisons, maskEqual=mask_equal, passed=bool(mask_equal and all(c['passed'] for c in comparisons.values())))
                    report['runs'].append(row)
                    print(json.dumps(row), flush=True)
                    del result
        del inputs
        mx.clear_cache()
    report['finishedAt'] = time.time() * 1000
    report['passed'] = len(report['runs']) == 18 * args.repeats and all(r['passed'] for r in report['runs'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('Native MLX reference drift; benchmark saved for diagnosis')


if __name__ == '__main__':
    main()
