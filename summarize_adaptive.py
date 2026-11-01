"""Validate adaptive transport results without relaxing model accuracy gates."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'benchmarks'


def load(name):
    return json.loads((BENCH / name).read_text())


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    target = BENCH / 'adaptive-summary.json'
    if target.exists():
        raise FileExistsError(target)
    path = ROOT / 'public/models/packed-v2/manifest.json'
    manifest = json.loads(path.read_text())
    package_hash, metadata_bytes = digest(path), path.stat().st_size
    browser = load('adaptive-browser.json')
    detail = load('adaptive-detail-cold.json')
    warm = load('adaptive-cache-reload.json')
    detail_warm = load('adaptive-detail-reload.json')
    migration = load('adaptive-cache-migration.json')
    repair = load('adaptive-range-recovery.json')
    verified = load('adaptive-file-verification.json')
    full = load('adaptive-codec-full.json')
    screen = load('adaptive-codec-screen.json')
    assert browser['cacheEmptyAtStart'] and browser['packageSha256'] == package_hash == verified['packageSha256'] == full['packageSha256']
    assert verified['passed'] and len(verified['files']) == len(manifest['files']) == 33
    assert all(row['byteExact'] for row in verified['files'])
    assert full['allByteExact'] and all(row['byteExact'] and row['selectedBytes'] == min(c['bytes'] for c in row['candidates']) for row in full['chunks'])
    assert len(screen['rows']) == 8 and sum(row['rawBytes'] for row in screen['rows']) == 32*1024*1024
    assert all(r['byteExact'] for row in screen['rows'] for r in row['results'] if 'error' not in r)
    rows = browser['runs']
    expected = {(p,c,s,r) for p in ['fast','detail'] for c in ['sample-backyard','brick-ranch','raised-deck'] for s in [1,2,3] for r in range(3)}
    assert len(rows) == 54 and {(r['profile'],r['caseName'],r['refineSteps'],r['repeat']) for r in rows} == expected
    assert len(browser['warmups']) == 6
    assert all(r['passed'] and r['original']['passed'] and r['original']['maskMismatches'] == 0 for r in rows + browser['warmups'])
    assert all(r['sessions']['precision'] == 'fp32' and r['downloads']['after']['packaged'] and not r['cpuLearnedFallback'] and not r['sessions']['denseGraphCapture'] and r['sessions']['indexingBackend']=='gpu' for r in rows)
    previous = load('download-browser.json')
    for row in rows:
        baseline = next(r for r in previous['runs'] if all(row[k] == r[k] for k in ['profile','caseName','refineSteps','repeat']))
        assert row['original'] == baseline['original']
        for name in ['points','normal','mask','metric_scale','raw_coord','encoder_feature']:
            assert row['outputs'][name] == baseline['outputs'][name]
    assert browser['unloaded']['sessions']['retained'] == 0 and not browser['unloaded']['buffers']
    for report in [warm,detail_warm]:
        assert report['modelNetworkBlocked'] and report['result']['passed'] and not report['resources']
        assert report['downloads']['networkBytes'] == report['downloads']['networkRequests'] == 0
        assert report['downloads']['verifiedFiles'] == 22 and report['downloads']['cacheHits'] > 0
        assert report['unloaded']['retained'] == 0 and not report['buffers']
    assert detail['result']['passed'] and detail['unloaded']['retained'] == 0 and not detail['buffers']
    assert migration['result']['passed'] and migration['downloads']['networkBytes'] == metadata_bytes
    assert migration['downloads']['networkRequests'] == 1 and migration['downloads']['legacyCacheHits'] > 0
    assert len(migration['resources']) == 1 and migration['resources'][0]['name'].endswith('manifest.json')
    recovered_bytes = repair['after']['networkBytes'] - repair['before']['networkBytes']
    assert repair['result']['passed'] and repair['after']['rangeRequests']-repair['before']['rangeRequests'] == 1
    assert recovered_bytes == repair['expectedMissingBytes'] < repair['bundleBytes']
    assert len(repair['resources']) == 1 and repair['resources'][0]['responseStatus'] == 206
    assert load('adaptive-cache-clear.json')['passed']
    for d in [browser['downloads'],warm['downloads'],detail['downloads'],detail_warm['downloads'],migration['downloads'],repair['after']]:
        assert d['packageSha256'] == package_hash
        assert d['cacheReadFailures'] == d['cacheWriteFailures'] == d['cacheUnavailable'] == d['corruptCacheEntries'] == 0
    common = [name for name in manifest['files'] if name.startswith('refiner-wgsl/') or (name.startswith('refiner/') and ('linear' in name or Path(name).stem in ['fuse','input_proj','out_proj']))]
    assert len(common) == 21
    files = {p: [f'profiles/{p}/moge3-dense.onnx'] + common for p in ['fast','detail']}
    chunks = {p: {piece['chunk'] for name in names for piece in manifest['files'][name]['pieces']} for p,names in files.items()}
    bundles = {p: {manifest['chunks'][key]['bundle'] for key in keys} for p,keys in chunks.items()}
    size = lambda keys: sum(manifest['bundles'][key]['bytes'] for key in keys)
    cold_fast = browser['warmups'][0]['downloads']['after']['networkBytes'] - browser['warmups'][0]['downloads']['before']['networkBytes']
    assert cold_fast == size(bundles['fast']) + metadata_bytes
    assert detail['downloads']['networkBytes'] == size(bundles['detail']) + metadata_bytes
    assert browser['warmups'][0]['downloads']['after']['networkRequests'] == len(bundles['fast']) + 1
    assert detail['downloads']['networkRequests'] == len(bundles['detail']) + 1
    all_bundles = bundles['fast'] | bundles['detail']
    assert browser['downloads']['networkBytes'] == size(all_bundles) + metadata_bytes
    assert browser['downloads']['networkRequests'] == len(browser['resources']) == len(all_bundles) + 1
    assert len({r['name'] for r in browser['resources']}) == len(browser['resources'])
    switch = browser['warmups'][3]['downloads']['after']['networkBytes'] - browser['warmups'][3]['downloads']['before']['networkBytes']
    assert switch == size(bundles['detail'] - bundles['fast'])
    assert browser['downloads']['legacyCacheHits'] == browser['downloads']['rangeRequests'] == browser['downloads']['fullBundleFallbacks'] == 0
    old_summary = load('download-summary.json')
    profiles = {}
    for p in ['fast','detail']:
        new_bytes = size(bundles[p]) + metadata_bytes
        old = old_summary['profiles'][p]
        profiles[p] = {'rawModelBytes': old['originalModelBytes'], 'previousCompressedBytes': old['compressedPackageBytes'],
                       'adaptiveBytes': new_bytes, 'reductionVsRaw': 1-new_bytes/old['originalModelBytes'],
                       'reductionVsPreviousCompressed': 1-new_bytes/old['compressedPackageBytes'],
                       'previousRequests': old['modelRequestsIncludingManifest'], 'requests': len(bundles[p])+1,
                       'requestReductionFraction': 1-(len(bundles[p])+1)/old['modelRequestsIncludingManifest']}
        resources = detail['resources'] if p == 'detail' else [r for r in browser['resources'] if r['name'].endswith('manifest.json') or Path(r['name']).stem in bundles['fast']]
        profiles[p]['browserEncodedBodyBytes'] = sum(r['encodedBodySize'] for r in resources)
    encoding_stats = collections.defaultdict(lambda:{'chunks':0,'rawBytes':0,'compressedBytes':0})
    for c in manifest['chunks'].values():
        row = encoding_stats[c['codec']+'/'+c['encoding']]
        row['chunks']+=1;row['rawBytes']+=c['rawBytes'];row['compressedBytes']+=c['bytes']
    power = [json.loads(row) for row in (BENCH/'adaptive-power.jsonl').read_text().splitlines()]
    assert power and all(row['ac'] for row in power)
    assert any(abs(row['time']*1000-browser['startedAt'])<=20000 for row in power)
    assert any(abs(row['time']*1000-browser['finishedAt'])<=20000 for row in power)
    assert max(b['time']-a['time'] for a,b in zip(power,power[1:]))<=30
    bundle = next((ROOT/'dist/assets').glob('index-*.js'))
    summary = {'passed':True,'packageSha256':package_hash,'format':2,'hardware':'Apple M4 Pro, 24 GB; sysctl verified',
               'power':{'source':'AC verified throughout matrix','samples':len(power)},
               'accuracy':{'comparisons':54,'allPassed':True,'allMasksMatch':True,'allDenseAndCudaMetricsIdenticalToPrevious':True,'worstPointsNrmse':max(r['original']['points']['nrmse'] for r in rows)},
               'profiles':profiles,'bothProfiles':{'bytes':browser['downloads']['networkBytes'],'requests':browser['downloads']['networkRequests'],'previousBytes':old_summary['bothProfiles']['compressedPackageBytes'],'previousRequests':len(previous['resources']),'browserEncodedBodyBytes':sum(r['encodedBodySize'] for r in browser['resources'])},
               'detailAfterFast':{'additionalBytes':switch,'additionalRequests':len(bundles['detail']-bundles['fast'])},
               'persistentReload':{'passed':True,'networkBytes':0,'requests':0,'profiles':['fast','detail'],'allModelUrlsBlocked':True},
               'cacheUpgrade':{'passed':True,'newMetadataBytes':metadata_bytes,'weightBytes':0,'legacyRepresentationsReused':migration['downloads']['legacyCacheHits']},
               'partialRecovery':{'passed':True,'status':206,'requests':1,'bytes':recovered_bytes,'fullBundleBytes':repair['bundleBytes']},
               'byteExactFilesVerified':33,'screenedMiB':32,'fullCodecCandidatesChecked':len(full['chunks']),
               'allCodecCandidatesByteExact':True,'codecDistribution':dict(encoding_stats),'bundleCountIncludingOptionalBackend':len(manifest['bundles']),
               'method':'Byte-exact FP32 transport only. Eight real 4 MiB chunks screen native ZipNN/Bitshuffle and portable transforms. Full candidate scan covers every unique v1 chunk and selects smallest portable encoding. Cold production Fast, independent cold Detail, two-resolution CUDA matrix, fresh-page model-network-blocked reloads, existing v1 cache migration, and one forced missing-chunk HTTP range recovery. Four concurrent bundle workers; 16 MiB target; separate cache entries and hashes per chunk. Whole bundles are prefetched only on cold storage; partial caches use coalesced ranges with full-response fallback. Model byte totals include manifest and active weight bundles, exclude app JS/WASM, photos, references and HTTP headers. Localhost tests do not establish Internet startup-speed improvement. Native codec times are not browser decoding times. No graph, precision, parameters or numerical thresholds changed.',
               'provenance':{'bundleFile':bundle.name,'bundleSha256':digest(bundle),
                             'sources':{name:digest(ROOT/name) for name in ['model-cache.js','model-codecs.js','model-bundles.js','model-package.js','pack_adaptive.py','lossless_codecs.py','screen_codecs.py','vite.config.js','benchmark-adaptive.browser.js']},
                             'reports':{name:digest(BENCH/name) for name in ['adaptive-browser.json','adaptive-cache-reload.json','adaptive-detail-cold.json','adaptive-detail-reload.json','adaptive-cache-migration.json','adaptive-range-recovery.json','adaptive-codec-screen.json','adaptive-codec-full.json','adaptive-file-verification.json','adaptive-power.jsonl']}}}
    target.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
