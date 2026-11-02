"""Validate precision, transport, cache and matrix coverage before release."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(name):return json.loads((ROOT/'benchmarks'/name).read_text())


def main():
    compact=read('progressive-compact-browser.json');exact=read('progressive-exact-browser.json');firefox=read('progressive-firefox-browser.json')
    previous=read('shader-compat-chrome.json');verification=read('progressive-file-verification.json')
    cache=read('progressive-cache-reload.json');detail=read('progressive-detail-cold.json');migration=read('progressive-cache-migration.json');recovery=read('progressive-correction-recovery.json');ui=read('progressive-ui.json')
    manifest_bytes=(ROOT/'public/models/packed-v3/manifest.json').read_bytes();manifest=json.loads(manifest_bytes);package_hash=hashlib.sha256(manifest_bytes).hexdigest()
    expected={(p,c,s,r) for p in ['fast','detail'] for c in ['sample-backyard','brick-ranch','raised-deck'] for s in [1,2,3] for r in range(3)}
    for report,mode in [(compact,'compact'),(exact,'exact')]:
        assert len(report['runs'])==54 and {(r['profile'],r['caseName'],r['refineSteps'],r['repeat']) for r in report['runs']}==expected
        assert all(r['passed'] and r['original']['passed'] and r['sessions']['precision']=='fp32' and r['sessions']['weightMode']==mode and r['downloads']['after']['packageSha256']==package_hash for r in report['runs'])
        assert report['unloaded']['retained']==0 and not report['pool']
    assert len(firefox['runs'])==18 and {(r['profile'],r['caseName'],r['refineSteps']) for r in firefox['runs']}=={x[:3] for x in expected}
    assert all(r['passed'] and r['sessions']['weightMode']=='compact' for r in firefox['runs'])
    assert firefox['unloaded']['retained']==0 and not firefox['pool']
    assert ui['error'] is None and ui['validated'] is False and ui['passed'] is None and ui['mode']=='compact' and ui['locked'] and ui['layout']['scrollWidth']<=ui['layout']['viewport']
    lookup={(r['profile'],r['caseName'],r['refineSteps'],r['repeat']):r for r in previous['runs']}
    for row in exact['runs']:
        old=lookup[row['profile'],row['caseName'],row['refineSteps'],row['repeat']]
        assert row['original']==old['original']
        assert all(row['outputs'][n]==old['outputs'][n] for n in ['points','normal','mask','metric_scale','raw_coord','encoder_feature'])
    assert verification['passed'] and verification['packageSha256']==package_hash and len(verification['files'])==33 and all(r['exactUpgradeByteExact'] for r in verification['files'])
    assert cache['modelNetworkBlocked'] and cache['downloads']['networkBytes']==0 and cache['downloads']['networkRequests']==0 and len(cache['runs'])==2 and all(r['passed'] for r in cache['runs'])
    assert detail['before']['networkBytes']==0 and detail['result']['passed']
    assert migration['after']['networkBytes']==len(manifest_bytes) and migration['after']['networkRequests']==1 and len(migration['runs'])==4 and all(r['passed'] for r in migration['runs'])
    assert recovery['result']['passed'] and recovery['rangeRequests']==1 and recovery['networkBytes']==recovery['expectedBytes'] and recovery['fullBundleFallbacks']==0
    fast=compact['warmups'][0]['downloads']['after'];both=compact['downloads'];delta=exact['after']['networkBytes']-exact['before']['networkBytes']
    # Audit bundle sets independently of browser counters: no corrections may be
    # present in a compact cold load, and upgrade may fetch only corrections.
    default=[n for n in manifest['files'] if n.startswith('refiner-wgsl/') or (n.startswith('refiner/') and ('linear' in n or Path(n).stem in ['fuse','input_proj','out_proj']))]
    def bundles(profile):return {manifest['chunks'][p['chunk']]['bundle'] for n in [f'profiles/{profile}/moge3-dense.onnx']+default for p in manifest['files'][n]['pieces']}
    fb,db=bundles('fast'),bundles('detail');cb={c['bundle'] for c in manifest['chunks'].values() if c['encoding']=='identity' and c['bundle'] in {k for k,b in manifest['bundles'].items() if b['usage']=='correction'}}
    size=lambda keys:sum(manifest['bundles'][k]['bytes'] for k in keys)
    assert size(fb)+len(manifest_bytes)==fast['networkBytes'] and size(db)+len(manifest_bytes)==detail['after']['networkBytes']
    assert size(fb|db)+len(manifest_bytes)==both['networkBytes'] and size(cb)==delta and not (fb|db)&cb
    sources=['runtime.js','main.js','index.html','model-package.js','progressive-package.js','model-bundles.js','model-cache.js','model-codecs.js','weight-precision.js','gpu-shader-compat.js','fp24.py','pack_progressive.py','screen_fp24.py','vite.config.js']
    summary={'passed':True,'packageSha256':package_hash,'format':3,'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},'compact':{'comparisons':54,'allPassed':True,'allMasksMatch':all(r['original']['maskMismatches']==0 for r in compact['runs']),'worstPointsNrmse':max(r['original']['points']['nrmse'] for r in compact['runs'])},'exact':{'comparisons':54,'allPassed':True,'metricsIdenticalToOriginal':True,'worstPointsNrmse':max(r['original']['points']['nrmse'] for r in exact['runs'])},'firefox':{'comparisons':18,'allPassed':True,'userAgent':firefox['browser']['ua']},'downloads':{'coldFastBytes':fast['networkBytes'],'coldFastRequests':fast['networkRequests'],'coldDetailBytes':detail['after']['networkBytes'],'coldDetailRequests':detail['after']['networkRequests'],'bothProfilesBytes':both['networkBytes'],'detailAfterFastBytes':both['networkBytes']-fast['networkBytes'],'exactUpgradeBytes':delta,'exactUpgradeRequests':exact['after']['networkRequests']-exact['before']['networkRequests'],'coldFastReductionVsV2':1-fast['networkBytes']/read('adaptive-summary.json')['profiles']['fast']['adaptiveBytes']},'persistentReload':{'passed':True,'networkBytes':0},'cacheMigration':{'passed':True,'weightBytes':0,'metadataBytes':len(manifest_bytes)},'partialCorrectionRecovery':{'passed':True,'bytes':recovery['networkBytes']},'files':{'count':33,'allExactUpgradesByteExact':True},'method':'Selective custom 24-bit encoder MatMul storage, expanded before FP32 computation. Compact is approximate; exact adds low-byte corrections and verifies original file hashes. No retraining, FP16 math, threshold changes or regenerated goldens. Accuracy set: three synthetic photos, two resolutions, 1/2/3 passes, three Chrome repeats and one Firefox pass per combination. Download counts cover manifest/model payload only; exclude app, photos, reference data and HTTP headers. Not a performance or physical-measurement benchmark.'}
    (ROOT/'benchmarks/progressive-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['downloads']))


if __name__=='__main__':main()
