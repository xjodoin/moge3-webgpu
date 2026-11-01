"""Gate the lossless download package on bytes, cache behavior and CUDA parity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'benchmarks'


def load(name):
    return json.loads((BENCH / name).read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    out = BENCH / 'download-summary.json'
    if out.exists():
        raise FileExistsError(out)
    package_path = ROOT / 'public/models/packed/manifest.json'
    package = json.loads(package_path.read_text())
    report = load('download-browser.json')
    warm = load('download-cache-reload.json')
    detail = load('download-detail-cold.json')
    detail_warm = load('download-detail-reload.json')
    verification = load('packed-file-verification.json')
    cleared = load('download-cache-clear.json')
    package_hash = digest(package_path)
    assert report['packageSha256'] == verification['packageSha256'] == package_hash
    assert verification['passed'] and len(verification['files']) == len(package['files']) == 33
    assert all(r['byteExact'] and r['sha256'] == package['files'][r['path']]['sha256'] for r in verification['files'])
    expected = {(p, c, s, r) for p in ['fast', 'detail'] for c in ['sample-backyard', 'brick-ranch', 'raised-deck'] for s in [1, 2, 3] for r in range(3)}
    runs = report['runs']
    assert len(runs) == 54 and {(r['profile'], r['caseName'], r['refineSteps'], r['repeat']) for r in runs} == expected
    assert len(report['warmups']) == 6
    assert all(r['passed'] and r['original']['passed'] and r['original']['maskMismatches'] == 0 for r in runs + report['warmups'])
    assert all(r['downloads']['after']['packaged'] and r['sessions']['precision'] == 'fp32' and not r['cpuLearnedFallback'] and r['sessions']['indexingBackend'] == 'gpu' for r in runs)
    baseline = [r for r in load('fp16-browser.json')['runs'] if r['precision'] == 'fp32']
    for r in runs:
        b = next(b for b in baseline if all(b[k] == r[k] for k in ['profile', 'caseName', 'refineSteps', 'repeat']))
        assert r['original'] == b['original'], 'Lossless transport must preserve all original CUDA error metrics'
    assert not report['blockedRawRequests']
    for row in report['unloaded']:
        assert row['sessions']['retained'] == 0 and row['buffers'] == []
    for row in [warm, detail_warm]:
        assert row['modelNetworkBlocked'] and row['result']['passed'] and row['downloads']['networkBytes'] == row['downloads']['networkRequests'] == 0
        assert not row['resources'] and row['downloads']['cacheHits'] > 0 and row['downloads']['verifiedFiles'] == 22
        assert row['downloads']['packageSha256'] == package_hash and row['unloaded']['retained'] == 0 and row['buffers'] == []
    assert detail['result']['passed'] and detail['downloads']['cacheHits'] == 0
    assert detail['unloaded']['retained'] == 0 and detail['buffers'] == []
    assert cleared['passed'] and not cleared['cachePresentAfterClear']
    for d in [report['downloads'], warm['downloads'], detail['downloads'], detail_warm['downloads']]:
        assert d['cacheReadFailures'] == d['cacheWriteFailures'] == d['cacheUnavailable'] == d['corruptCacheEntries'] == 0
    blocks = set(json.loads((ROOT / 'public/models/refiner-wgsl/manifest.json').read_text())['blocks'])
    common_files = [name for name in package['files'] if name.startswith('refiner-wgsl/') or (name.startswith('refiner/') and Path(name).stem not in blocks)]
    assert len(common_files) == 21
    files = {p: [f'profiles/{p}/moge3-dense.onnx'] + common_files for p in ['fast', 'detail']}
    chunks = {p: {piece['chunk'] for name in names for piece in package['files'][name]['pieces']} for p, names in files.items()}
    packed_bytes = lambda keys: sum(package['chunks'][k]['bytes'] for k in keys)
    expected_cold = {p: packed_bytes(keys) + package_path.stat().st_size for p, keys in chunks.items()}
    cold_fast = report['warmups'][0]['downloads']['after']['networkBytes'] - report['warmups'][0]['downloads']['before']['networkBytes']
    assert cold_fast == expected_cold['fast']
    assert detail['downloads']['networkBytes'] == expected_cold['detail']
    switch = report['warmups'][3]['downloads']['after']['networkBytes'] - report['warmups'][3]['downloads']['before']['networkBytes']
    assert switch == packed_bytes(chunks['detail'] - chunks['fast'])
    assert report['downloads']['networkBytes'] == packed_bytes(chunks['fast'] | chunks['detail']) + package_path.stat().st_size
    assert report['downloads']['networkRequests'] == len(report['resources']) == 1 + len(chunks['fast'] | chunks['detail'])
    resource_keys = {r['name'] for r in report['resources']}
    assert len(resource_keys) == len(report['resources'])
    power = [json.loads(r) for r in (BENCH / 'download-power.jsonl').read_text().splitlines()]
    assert power and all(r['ac'] for r in power)
    assert any(abs(r['time'] * 1000 - report['startedAt']) <= 20000 for r in power)
    assert any(abs(r['time'] * 1000 - report['finishedAt']) <= 20000 for r in power)
    assert max(b['time'] - a['time'] for a, b in zip(power, power[1:])) <= 30
    sizes = {}
    for profile in ['fast', 'detail']:
        raw_bytes = sum(package['files'][name]['bytes'] for name in files[profile])
        resources = detail['resources'] if profile == 'detail' else [r for r in report['resources'] if r['name'].endswith('manifest.json') or Path(r['name']).stem in chunks['fast']]
        sizes[profile] = {'originalModelBytes': raw_bytes, 'compressedPackageBytes': expected_cold[profile],
                          'reductionFraction': 1 - expected_cold[profile] / raw_bytes,
                          'browserEncodedBodyBytes': sum(r['encodedBodySize'] for r in resources),
                          'modelRequestsIncludingManifest': len(chunks[profile]) + 1}
    raw_both = sum(package['files'][name]['bytes'] for name in set(files['fast'] + files['detail']))
    encoded_both = sum(r['encodedBodySize'] for r in report['resources'])
    entry = next((ROOT / 'dist/assets').glob('index-*.js'))
    summary = {'passed': True, 'packageSha256': package_hash, 'device': report['userAgent'],
               'power': {'source': 'AC verified throughout', 'samples': len(power)},
               'accuracy': {'comparisons': 54, 'allPassed': True, 'allMasksMatch': True,
                            'worstPointsNrmse': max(r['original']['points']['nrmse'] for r in runs), 'cudaMetricsIdenticalToFp32Baseline': True},
               'byteExactOriginalFiles': 33, 'profiles': sizes,
               'detailAfterFast': {'additionalModelBytes': switch, 'newChunks': len(chunks['detail'] - chunks['fast'])},
               'bothProfiles': {'originalUniqueModelBytes': raw_both, 'compressedPackageBytes': report['downloads']['networkBytes'],
                                'browserEncodedBodyBytes': encoded_both, 'reductionFraction': 1 - report['downloads']['networkBytes'] / raw_both},
               'persistentReload': {'passed': True, 'profiles': ['fast', 'detail'], 'networkBytes': 0, 'networkRequests': 0, 'modelNetworkBlocked': True, 'verifiedModelFilesPerProfile': 22},
               'clearCacheUiPassed': True,
               'method': 'Production build on localhost. SHA-256 verification of compressed chunks, decoded chunks and assembled byte-identical original files. Dedicated Cache Storage persists compressed content-addressed chunks across page reloads. HTTP cache bypassed for missing chunks. Cold Fast, incremental Detail, independent cold Detail and fresh-page cached inference at both resolutions with all model URLs blocked. 3 photos x 2 profiles x 1/2/3 passes x 3 repetitions = 54 original CUDA comparisons, plus warmups/cache checks. Default FP32 subgroup sparse kernels, GPU indexing, reuse enabled, graph capture disabled. No precision, learned weights, model graph or accuracy gate change. Byte totals cover active model payloads and package manifest; exclude app JS/WASM, photos, diagnostic goldens and HTTP headers. Browser encoded-body totals additionally reflect local server HTTP compression. Local timings do not predict Internet download speed. Browser cache remains subject to quota and eviction.',
               'provenance': {'bundleSha256': digest(entry), 'bundleFile': entry.name,
                              'sourceFiles': {name: digest(ROOT / name) for name in ['pack_models.py', 'model-cache.js', 'model-package.js', 'runtime.js', 'sparse-block.js', 'main.js', 'vite.config.js', 'benchmark-downloads.browser.js']},
                              'reports': {name: digest(BENCH / name) for name in ['download-browser.json', 'download-detail-cold.json', 'download-cache-reload.json', 'download-detail-reload.json', 'download-power.jsonl', 'packed-file-verification.json']}}}
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
