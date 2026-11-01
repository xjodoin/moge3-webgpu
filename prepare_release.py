"""Prepare a reviewable static Hugging Face Space; does not publish anything."""
from pathlib import Path
import json,os,shutil
ROOT=Path(__file__).resolve().parent
if (ROOT/'gpu-shader-compat.js').exists():
    import hashlib
    compatibility=json.loads((ROOT/'benchmarks/shader-compat-summary.json').read_text())
    assert compatibility['passed'], 'Require Firefox/Chrome shader compatibility validation'
    assert compatibility['chrome']['comparisons']==54 and compatibility['firefox']['comparisons']==18, 'Require complete browser accuracy matrices'
    for name,digest in compatibility['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Validate the current shader compatibility sources before release'
out=ROOT/'release/space'
if out.exists():
    raise SystemExit('release/space already exists; move it aside before preparing a new snapshot')
report=json.loads((ROOT/'benchmarks/release-validation.json').read_text())
assert len(report['results'])==6 and all(r['passed'] for r in report['results']), 'All six release comparisons must pass'
sparse_report=json.loads((ROOT/'benchmarks/sparse-validation.json').read_text())
assert len(sparse_report['runs'])==18 and all(r['passed'] and r['sparseBackend']=='wgsl' for r in sparse_report['runs']), 'All repeated direct sparse kernel comparisons must pass'
subgroup_report=json.loads((ROOT/'benchmarks/subgroup-browser.json').read_text())
subgroup_summary=json.loads((ROOT/'benchmarks/subgroup-summary.json').read_text())
subgroup_mlx=json.loads((ROOT/'benchmarks/subgroup-mlx.json').read_text())
subgroup_screen=json.loads((ROOT/'benchmarks/subgroup-broadcast-screen.json').read_text())
subgroup_blocks=json.loads((ROOT/'benchmarks/subgroup-blocks.json').read_text())
subgroup_ui=json.loads((ROOT/'benchmarks/subgroup-ui.json').read_text())
assert subgroup_ui['done'] and subgroup_ui['result']['passed'] and subgroup_ui['result']['sessions']['sparseVariant']=='subgroup' and subgroup_ui['unloaded']['retained']==0, 'Default UI inference and unload must pass'
assert subgroup_summary['passed'] and subgroup_summary['powerSource']=='AC verified throughout both runs', 'Verify AC coverage for browser and MLX'
assert len(subgroup_report['runs'])==108 and all(r['passed'] and r['original']['passed'] for r in subgroup_report['runs']), 'All subgroup A/B CUDA comparisons must pass'
subgroup_rows=[r for r in subgroup_report['runs'] if r['variant']=='subgroup']
assert len(subgroup_rows)==54 and all(r['sessions']['subgroupSupport']['supported'] for r in subgroup_rows), 'All 54 comparisons must execute supported subgroup kernels'
assert len(subgroup_mlx['runs'])==54 and subgroup_mlx['passed'] and not subgroup_mlx['diagnosticTracing'] and not subgroup_mlx['returnPerStep'], 'Require a fresh MLX baseline without tracing'
assert subgroup_screen['done'] and len(subgroup_screen['rows'])==12 and all(len(r['candidates'])==6 and all(c['parity']['maxAbs']==0 for c in r['candidates']) for r in subgroup_screen['rows']), 'All 72 isolated subgroup checks must have zero FP32 numerical error'
assert subgroup_blocks['done'] and len(subgroup_blocks['rows'])==20 and all(r['passed'] for r in subgroup_blocks['rows']), 'All 20 partial/empty ONNX checks must pass'
graph_report=json.loads((ROOT/'benchmarks/graph-browser.json').read_text())
graph_summary=json.loads((ROOT/'benchmarks/graph-summary.json').read_text())
graph_proof=json.loads((ROOT/'benchmarks/graph-capture-checks.json').read_text())
assert graph_summary['passed'] and graph_summary['powerSource']=='AC verified throughout run', 'Verify AC power for graph capture'
assert len(graph_report['runs'])==108 and all(r['passed'] and r['original']['passed'] for r in graph_report['runs']), 'All graph capture A/B CUDA comparisons must pass'
assert len([r for r in graph_report['runs'] if r['capture']])==54, 'All 54 captured comparisons are required'
assert graph_proof['done'] and len(graph_proof['rows'])==16 and all(r['passed'] for r in graph_proof['rows']), 'Dense capture checks must pass at both resolutions'
assert all(r['bindGroups']==0 for r in graph_proof['rows'] if r['capture'] and r['iteration']>=2), 'Warm dense runs must use native graph replay'
index_report=json.loads((ROOT/'benchmarks/index-browser.json').read_text())
index_summary=json.loads((ROOT/'benchmarks/index-summary.json').read_text())
index_parity=json.loads((ROOT/'benchmarks/gpu-indexing-parity.json').read_text())
assert index_summary['passed'] and index_summary['powerSource']=='AC verified throughout run', 'Verify AC power for GPU indexing'
assert len(index_report['runs'])==108 and all(r['passed'] and r['original']['passed'] for r in index_report['runs']), 'All CPU/GPU indexing comparisons must pass'
index_gpu=[r for r in index_report['runs'] if r['indexing']=='gpu']
assert len(index_gpu)==54 and all(r['transfers']['indexUploadBytes']==0 and r['transfers']['indexShapeReadbackBytes']==16*r['refineSteps'] for r in index_gpu), 'All 54 GPU checks must keep index arrays resident'
assert index_parity['done'] and len(index_parity['rows'])==23 and all(r['exact'] and r['levels']==5 for r in index_parity['rows']), 'Exact map and ordered FP32 pooling parity is required'
wide_report=json.loads((ROOT/'benchmarks/wide-browser.json').read_text())
wide_summary=json.loads((ROOT/'benchmarks/wide-summary.json').read_text())
wide_blocks=json.loads((ROOT/'benchmarks/wide-blocks.json').read_text())
assert wide_summary['passed'] and wide_summary['powerSource']=='AC verified throughout run', 'Verify AC power for the wide-kernel benchmark'
assert len(wide_report['runs'])==108 and all(r['passed'] and r['original']['passed'] for r in wide_report['runs']), 'All wide-kernel CUDA comparisons must pass'
assert len([r for r in wide_report['runs'] if r['variant']=='optimized'])==54, 'All 54 optimized comparisons are required'
assert wide_blocks['done'] and len(wide_blocks['rows'])==20 and all(r['passed'] for r in wide_blocks['rows']), 'Partial tiles and empty neighborhoods must pass'
original_report=json.loads((ROOT/'benchmarks/original-browser.json').read_text())
coordinate_report=json.loads((ROOT/'benchmarks/coordinate-browser.json').read_text())
buffer_report=json.loads((ROOT/'benchmarks/buffer-browser-ac.json').read_text())
buffer_summary=json.loads((ROOT/'benchmarks/buffer-summary-ac.json').read_text())
assert buffer_summary['passed'] and buffer_summary['powerSource']=='AC verified throughout run', 'Verify AC power for the buffer benchmark'
assert len(buffer_report['runs'])==108 and all(r['passed'] and r['original']['passed'] for r in buffer_report['runs']), 'Both buffer modes must pass all CUDA comparisons'
assert len([r for r in buffer_report['runs'] if r['reuse']])==54, 'All 54 reuse comparisons are required'
assert len(coordinate_report['runs'])==54 and all(r['passed'] and r['original']['passed'] for r in coordinate_report['runs']), 'All coordinate-optimized CUDA comparisons must pass'
assert len(original_report['runs'])==54 and all(r['original']['passed'] for r in original_report['runs']), 'All repeated original CUDA comparisons must pass before enabling multipass'
assert {(r['profile'],r['caseName'],r['refineSteps'],r['repeat']) for r in original_report['runs']}=={(p,c,s,r) for p in ['fast','detail'] for c in ['sample-backyard','brick-ranch','raised-deck'] for s in [1,2,3] for r in range(3)}, 'Incomplete CUDA validation matrix'
for row in original_report['runs']:
    for field in ['points.f32','intrinsics.f32','mask.u8']:
        assert (ROOT/f"dist/original-cuda/{row['profile']}/{row['caseName']}/{row['refineSteps']}/{field}").is_file(), 'Build original CUDA goldens first'
source=json.loads((ROOT/'dist/model-source.json').read_text())
if source.get('modelPackageBaseUrl'):
    import hashlib
    package_bytes=(ROOT/'dist/models/packed/manifest.json').read_bytes()
    package=json.loads(package_bytes)
    delivery=json.loads((ROOT/('benchmarks/adaptive-summary.json' if package['format']==2 else 'benchmarks/download-summary.json')).read_text())
    assert delivery['passed'] and delivery['accuracy']['comparisons']==54 and delivery['accuracy']['allPassed'], 'Require all 54 compressed-package CUDA comparisons'
    assert delivery['packageSha256']==hashlib.sha256(package_bytes).hexdigest(), 'Validate this exact model package before release'
    assert delivery['persistentReload']['passed'] and delivery['persistentReload']['networkBytes']==0, 'Require persistent model-cache validation after reload'
    if source['modelPackageBaseUrl'] in ['/models/packed','/models/packed-v2']:
        folder=source['modelPackageBaseUrl'].lstrip('/')
        for key,meta in (package['bundles'] if package['format']==2 else package['chunks']).items():
            path=f"{folder}/bundles/{key}.pack" if package['format']==2 else f"{folder}/chunks/{key}.zlib"
            assert (ROOT/f'dist/{path}').stat().st_size==meta['bytes'], 'Missing compressed model payload'
if not source.get('modelPackageBaseUrl') and source.get('sparseModelBaseUrl','/models')=='/models':
    sparse_folder=ROOT/'dist/models/refiner-wgsl'
    sparse_manifest=json.loads((sparse_folder/'manifest.json').read_text())
    for name,meta in sparse_manifest['blocks'].items():
        assert (sparse_folder/f'{name}.bin').stat().st_size==meta['bytes'], 'Build the direct sparse weights first'
for name in ['fast','detail']:
    assert (ROOT/f'dist/models/profiles/{name}/manifest.json').is_file(), 'Build exported model manifests first'
    if not source.get('modelPackageBaseUrl') and source['modelBaseUrl']=='/models':
        assert (ROOT/f'dist/models/profiles/{name}/moge3-dense.onnx').is_file(), 'Build exported models first'
shutil.copytree(ROOT/'dist',out,copy_function=os.link)
for folder in ['licenses','benchmarks']:
    shutil.copytree(ROOT/folder,out/folder)
shutil.copy(ROOT/'LICENSE',out/'LICENSE')
for name in ['DOWNLOADS.md','ADAPTIVE.md','FP16.md']:
    if (ROOT/name).exists():shutil.copy(ROOT/name,out/name)
readme='''---
title: MoGe-3 WebGPU
emoji: 🏡
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: mit
models:
  - Ruicheng/moge-3-vitl
  - xjodoin/moge-3-vitl-webgpu
tags:
  - webgpu
  - onnx
  - depth-estimation
  - 3d-reconstruction
short_description: MoGe-3 with sparse refinement on WebGPU
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

'''+(ROOT/'README.md').read_text()
(out/'README.md').write_text(readme)
size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
if size>=1_000_000_000:raise SystemExit('Static Space exceeds 1 GB. Build with VITE_MODEL_BASE_URL pointing to a model repository.')
print(out, size, 'bytes')
