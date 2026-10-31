"""Warm, repeated native MLX timings on the browser's exact input fixtures.

Run separately from browser benchmarks to avoid GPU contention. Does not rewrite
reference files; verifies each fresh native result against the saved reference.
"""
import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_vlm import load
import numpy as np

ROOT = Path(__file__).resolve().parent


def agreement(actual, expected):
    finite = np.isfinite(actual) & np.isfinite(expected)
    mask_equal = np.array_equal(np.isposinf(actual), np.isposinf(expected))
    valid = not (np.isnan(actual).any() or np.isneginf(actual).any())
    a, b = actual[finite].astype(np.float64), expected[finite].astype(np.float64)
    error = float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-12))
    return {"nrmse": error, "maskEqual": mask_equal, "passed": bool(valid and mask_equal and error < .001)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/performance-native.json')
    args = parser.parse_args()
    mx.set_memory_limit(8_000_000_000)
    mx.set_cache_limit(256_000_000)
    model, _ = load(str(ROOT / 'weights/mlx'), trust_remote_code=False)
    rows = []
    for profile in ('fast', 'detail'):
        base = ROOT / 'public/models/profiles' / profile
        meta = json.loads((base / 'manifest.json').read_text())
        _, _, height, width = meta['inputShape']
        names = json.loads((base / 'cases.json').read_text())
        inputs = {name: mx.array(np.fromfile(base / 'cases' / name / 'input.f32', np.float32)
                                .reshape(1, 3, height, width).transpose(0, 2, 3, 1)) for name in names}
        mx.eval(list(inputs.values()))
        warm = model.infer(inputs[names[0]], num_tokens=meta['numTokens'], refine_steps=1)
        mx.eval(warm)
        del warm
        for repeat in range(args.repeats):
            for name in names:
                mx.reset_peak_memory()
                started = time.perf_counter()
                result = model.infer(inputs[name], num_tokens=meta['numTokens'], refine_steps=1)
                mx.eval(result)
                elapsed = (time.perf_counter() - started) * 1000
                folder = base / 'cases' / name
                comparisons = {key: agreement(np.array(result[key]).reshape(-1),
                               np.fromfile(folder / f'reference-native-{key}.f32', np.float32))
                               for key in ('points', 'intrinsics')}
                mask_equal = np.array_equal(np.array(result['mask']).reshape(-1),
                                           np.fromfile(folder / 'reference-native-mask.u8', np.uint8))
                row = dict(profile=profile, caseName=name, repeat=repeat, inferenceMs=elapsed,
                           mlxPeakBytes=mx.get_peak_memory(), outputs=comparisons,
                           maskEqual=mask_equal, passed=bool(mask_equal and all(v['passed'] for v in comparisons.values())))
                rows.append(row)
                print(json.dumps(row), flush=True)
                del result
        del inputs
        mx.clear_cache()
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    if not all(row['passed'] for row in rows):
        raise SystemExit('Native reference drift: accuracy comparison failed')


if __name__ == '__main__':
    main()
