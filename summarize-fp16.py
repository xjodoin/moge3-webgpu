"""Validate and summarize the AC-power FP16 experiment without changing gates."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'benchmarks'
MODELS = ROOT / 'public/models'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    target = BENCH / 'fp16-summary.json'
    if target.exists():
        raise FileExistsError(target)
    report = json.loads((BENCH / 'fp16-browser.json').read_text())
    power = [json.loads(row) for row in (BENCH / 'fp16-power.jsonl').read_text().splitlines()]
    assert len(report['runs']) == 162 and len(report['warmups']) == 54
    assert power and all(row['ac'] for row in power)
    assert power[0]['time'] * 1000 <= report['startedAt'] + 20000
    assert power[-1]['time'] * 1000 >= report['finishedAt'] - 20000
    assert max(b['time'] - a['time'] for a, b in zip(power, power[1:])) <= 30
    assert len(report['unloaded']) == 6
    assert all(row['sessions']['retained'] == 0 and row['buffers'] == [] for row in report['unloaded'])
    provenance, sizes, quality, profiles = {}, {}, {}, []
    packed = json.loads((MODELS / 'refiner-wgsl/manifest.json').read_text())
    half_packed = json.loads((MODELS / 'precision/fp16-weights/refiner-wgsl/manifest.json').read_text())
    assert half_packed['dtype'] == 'float16-le' and half_packed['accumulation'] == 'float32'
    for mode in ['storage', 'compute']:
        provenance[mode] = {p: json.loads((MODELS / f'precision/fp16-{mode}/conversion-{p}.json').read_text())
                            for p in ['fast', 'detail', 'refiner']}
        for part in provenance[mode].values():
            for file in part['files']:
                assert digest(MODELS / file['target']) == file['sha256']
    for name, meta in half_packed['blocks'].items():
        assert digest(MODELS / f'precision/fp16-weights/refiner-wgsl/{name}.bin') == meta['sha256']
        assert meta['bytes'] * 2 == packed['blocks'][name]['bytes']
    for precision in ['fp32', 'fp16-storage', 'fp16-compute']:
        rows = [r for r in report['runs'] if r['precision'] == precision]
        signatures = {(r['profile'], r['refineSteps'], r['caseName'], r['repeat']) for r in rows}
        assert len(rows) == len(signatures) == 54
        assert signatures == {(p, s, c, r) for p in ['fast', 'detail'] for s in [1, 2, 3]
                              for c in ['sample-backyard', 'brick-ranch', 'raised-deck'] for r in range(3)}
        assert all(r['sessions']['precision'] == precision and r['sessions']['indexingBackend'] == 'gpu'
                   and not r['sessions']['denseGraphCapture'] and not r['cpuLearnedFallback']
                   and r['validationReference'] == 'original-cuda-strict-fp32' for r in rows)
        quality[precision] = {
            'comparisons': len(rows), 'passedUnchangedGate': sum(r['passed'] for r in rows),
            'worstPointsNrmse': max(r['original']['points']['nrmse'] for r in rows),
            'medianPointsNrmse': statistics.median(r['original']['points']['nrmse'] for r in rows),
            'worstIntrinsicsNrmse': max(r['original']['intrinsics']['nrmse'] for r in rows),
            'worstMaskChangedPixels': max(r['original']['maskMismatches'] for r in rows),
            'worstMaskChangedFraction': max(r['original']['maskMismatches'] / math.prod(r['inputShape'][2:]) for r in rows),
            'allOutputsFiniteOnValidPixels': all(r['original']['points']['finite'] and all(o['finite'] for o in r['outputs'].values()) for r in rows),
            'worstDenseNrmse': {name: max(r['outputs'][name]['nrmse'] for r in rows) for name in ['points', 'normal', 'mask', 'metric_scale', 'raw_coord', 'encoder_feature']},
        }
    assert quality['fp32']['passedUnchangedGate'] == 54
    control = json.loads((BENCH / 'fp16-folding-control.json').read_text())
    assert control['passed'] and len(control['runs']) == 6 and not control['cpuLearnedFallback']
    for row in control['runs']:
        baseline = next(r for r in report['runs'] if r['precision'] == 'fp32' and r['profile'] == row['profile'] and r['caseName'] == row['photo'])
        assert all(result == baseline['outputs'][name] for name, result in row['comparisons'].items())
    for profile in ['fast', 'detail']:
        source = provenance['storage'][profile]['files'][0]
        linears = provenance['storage']['refiner']['files']
        sizes[profile] = {'fp32': source['sourceBytes'] + sum(f['sourceBytes'] for f in linears) + sum(b['bytes'] for b in packed['blocks'].values())}
        for mode in ['storage', 'compute']:
            sizes[profile][f'fp16-{mode}'] = provenance[mode][profile]['files'][0]['bytes'] + sum(f['bytes'] for f in provenance[mode]['refiner']['files']) + sum(b['bytes'] for b in half_packed['blocks'].values())
        for steps in [1, 2, 3]:
            row = {'profile': profile, 'refineSteps': steps}
            for precision in quality:
                runs = [r for r in report['runs'] if r['profile'] == profile and r['refineSteps'] == steps and r['precision'] == precision]
                assert len(runs) == 9
                row[precision] = {'computeMs': statistics.median(r['inferenceMs'] + r['refinementMs'] + r['cameraMs'] for r in runs),
                                  'denseMs': statistics.median(r['inferenceMs'] for r in runs),
                                  'refinementMs': statistics.median(r['refinementMs'] for r in runs),
                                  'peakExplicitGpuBufferBytes': max(r['gpuMemory']['peakExplicitBufferBytes'] for r in runs)}
            profiles.append(row)
    summary = {'device': report['userAgent'], 'startedAt': report['startedAt'], 'finishedAt': report['finishedAt'],
               'power': {'source': 'AC verified throughout', 'samples': len(power)},
               'method': 'Three synthetic photos, two resolutions, 1/2/3 passes, three repeats per photo and three warmups per configuration. Grouped mode order reversed by resolution. Default subgroup FP32 sparse accumulation and GPU indexing, reuse enabled, graph capture disabled, no CPU learned fallback. Compute includes upload, required readbacks and CPU camera fitting; loading, validation and rendering excluded. Explicit WebGPU buffers exclude WebGL, driver/browser overhead and process RSS. Same immutable original NVIDIA CUDA FP32 goldens and unchanged gates. No retraining, pruning, architecture or parameter-count change.',
               'gate': {'nrmseBelow': 0.001, 'maskConfidenceBoundary': 0.00001, 'maskMismatchFractionAtMost': 0.00001},
               'modes': {'fp32': 'Original FP32 storage and computation; remains default',
                         'fp16-storage': 'FP16 ONNX initializers restored to FP32 by Cast; FP32 operations and activations. Packed WGSL weights stay FP16 on GPU and widen before FP32 arithmetic.',
                         'fp16-compute': 'Convertible dense/linear ONNX operations and weights use FP16 with FP32 I/O. Packed WGSL weights use FP16 storage and FP32 arithmetic; geometry, camera and indexing remain FP32. Scalar/shape constants folded in FP32 before conversion.'},
               'quality': quality, 'fp32FoldingControl': {'comparisons': 6, 'passed': True, 'denseMetricsIdenticalToUnfoldedFp32': True, 'reportSha256': digest(BENCH / 'fp16-folding-control.json')},
               'activeWgslInferenceModelBytes': sizes, 'profiles': profiles,
               'medianComputeReductionGeometricMean': {p: 1 - statistics.geometric_mean(r[p]['computeMs'] / r['fp32']['computeMs'] for r in profiles) for p in ['fp16-storage', 'fp16-compute']},
               'provenance': {'reportSha256': digest(BENCH / 'fp16-browser.json'), 'powerSha256': digest(BENCH / 'fp16-power.jsonl'),
                              'sourceFiles': {p: digest(ROOT / p) for p in ['main.js', 'runtime.js', 'sparse-block.js', 'sparse-layout.js', 'fp16-weights.js', 'export_fp16.py', 'benchmark-fp16.browser.js']},
                              'exports': provenance, 'packedHalfWeights': half_packed},
               'decision': 'Keep both FP16 modes experimental and FP32 as the published default. Neither FP16 mode passes the existing release gate. Storage-only minimizes drift and halves inference downloads; FP16 compute loses too much geometry precision in this sample set.'}
    target.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['quality', 'activeWgslInferenceModelBytes', 'medianComputeReductionGeometricMean', 'profiles']}, indent=2))


if __name__ == '__main__':
    main()
