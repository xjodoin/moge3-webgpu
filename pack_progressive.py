"""Selective FP24 encoder transport with byte-exact optional FP32 correction.

Read the validated v2 package without modifying it. Reuse its untouched chunks;
store correction bytes in separate bundles so compact loads never fetch them.
"""
import argparse,collections,concurrent.futures,hashlib,json,shutil,time
from pathlib import Path
from fp24 import split24,expand24,restore32
from lossless_codecs import compress,decompress,inverse


def sha(raw):return hashlib.sha256(raw).hexdigest()


def build(source,plan_path,out,workers=4):
    source,out=Path(source),Path(out)
    old_raw=(source/'manifest.json').read_bytes();old=json.loads(old_raw)
    plan=json.loads(Path(plan_path).read_text());assert plan['policy']=='encoder-matmul-fp24-half-up-v1'
    selected=set()
    for name,policy in plan['files'].items():
        file=old['files'][name];assert file['sha256']==policy['sourceSha256'] and file['bytes']==policy['bytes']
        for piece in file['pieces']:
            a=piece['offset'];b=a+piece['length']
            if any(a>=s['offset'] and b<=s['offset']+s['bytes'] for s in policy['spans']):
                assert piece['sourceOffset']==0 and piece['length']==old['chunks'][piece['chunk']]['rawBytes']
                selected.add(piece['chunk'])
    out.mkdir(parents=True,exist_ok=False);temporary=out/'temporary-chunks';temporary.mkdir();(out/'bundles').mkdir()
    def read(key,c):
        with (source/f"bundles/{c['bundle']}.pack").open('rb') as f:
            f.seek(c['offset']);raw=f.read(c['bytes'])
        assert sha(raw)==key
        return raw
    def encode(item):
        key,c=item;packed=read(key,c)
        if key not in selected:
            meta={k:v for k,v in c.items() if k not in ['bundle','offset']}
            (temporary/key).write_bytes(packed)
            return key,key,[(key,meta)],None
        raw=inverse(decompress(packed,c['codec'],c['rawBytes']),c['encoding']);assert sha(raw)==c['rawSha256']
        planes,low,rounded=split24(raw)
        assert expand24(planes)==rounded and restore32(rounded,low)==raw
        base=compress(planes,'zstd',19);correction=compress(low,'zstd',6)
        base_key=sha(base);correction_key=sha(correction)
        original={k:v for k,v in c.items() if k not in ['bundle','offset']};original['sha256']=key
        base_meta={'bytes':len(base),'rawBytes':len(raw),'encodedBytes':len(planes),'rawSha256':sha(rounded),'codec':'zstd','encoding':'fp24','legacy':[],'correction':correction_key,'original':original}
        correction_meta={'bytes':len(correction),'rawBytes':len(low),'rawSha256':sha(low),'codec':'zstd','encoding':'identity','legacy':[]}
        (temporary/base_key).write_bytes(base);(temporary/correction_key).write_bytes(correction)
        return key,base_key,[(base_key,base_meta),(correction_key,correction_meta)],{'source':key,'base':base_key,'correction':correction_key,'originalBytes':c['bytes'],'compactBytes':len(base),'correctionBytes':len(correction),'byteExactUpgrade':True}
    chunks,remap,rows={},{},[]
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(old_key,new_key,entries,row) in enumerate(pool.map(encode,old['chunks'].items())):
            remap[old_key]=new_key
            for key,meta in entries:
                if key in chunks:assert chunks[key]==meta
                chunks[key]=meta
            if row:rows.append(row)
            if i%100==0:print('Packed',i+1,'chunks',flush=True)
    files=json.loads(json.dumps(old['files']))
    for name,file in files.items():
        file['exactSha256']=file['sha256']
        if name in plan['files']:file['sha256']=plan['files'][name]['compactSha256']
        for piece in file['pieces']:piece['chunk']=remap[piece['chunk']]
    for key,c in list(chunks.items()):
        if c['encoding']!='fp24':continue
        correction=chunks[c['correction']]
        files[f'__corrections/{key}']={'bytes':correction['rawBytes'],'sha256':correction['rawSha256'],'exactSha256':correction['rawSha256'],'pieces':[{'offset':0,'length':correction['rawBytes'],'sourceOffset':0,'chunk':c['correction']}]}
    default=[name for name in old['files'] if name.startswith('refiner-wgsl/') or (name.startswith('refiner/') and ('linear' in name or Path(name).stem in ['fuse','input_proj','out_proj']))]
    active={p:{q['chunk'] for name in [f'profiles/{p}/moge3-dense.onnx']+default for q in files[name]['pieces']} for p in ['fast','detail']}
    corrections={c['correction'] for c in chunks.values() if c['encoding']=='fp24'}
    groups=collections.defaultdict(list)
    for key in chunks:
        group='correction' if key in corrections else {(True,True):'shared',(True,False):'fast',(False,True):'detail',(False,False):'comparison'}[(key in active['fast'],key in active['detail'])]
        groups[group].append(key)
    bundles={}
    def bundle(keys,group):
        raw=b''.join((temporary/key).read_bytes() for key in keys);hash=sha(raw);(out/f'bundles/{hash}.pack').write_bytes(raw)
        bundles[hash]={'bytes':len(raw),'members':keys,'usage':group};offset=0
        for key in keys:chunks[key].update(bundle=hash,offset=offset);offset+=chunks[key]['bytes']
    for group,keys in groups.items():
        batch,size=[],0
        for key in keys:
            if batch and size+chunks[key]['bytes']>16*1024*1024:bundle(batch,group);batch,size=[],0
            batch.append(key);size+=chunks[key]['bytes']
        if batch:bundle(batch,group)
    manifest={'format':3,'codec':'progressive-fp24-v1','policy':plan['policy'],'checkpointRevision':old['checkpointRevision'],'previousPackageSha256':sha(old_raw),'files':files,'chunks':chunks,'bundles':bundles}
    (out/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n');shutil.rmtree(temporary)
    report={'packageSha256':sha((out/'manifest.json').read_bytes()),'sourcePackageSha256':sha(old_raw),'policy':plan['policy'],'selectedUniqueChunks':len(rows),'selectedUniqueRawBytes':sum(chunks[r['base']]['rawBytes'] for r in rows),'allCorrectionsByteExact':True,'rows':rows,'seconds':time.time()-started}
    return manifest,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('public/models/packed-v2'));p.add_argument('--plan',type=Path,default=Path('public/models/precision/fp24-screen/plan.json'));p.add_argument('--out',type=Path,default=Path('public/models/packed-v3'));p.add_argument('--report',type=Path,default=Path('benchmarks/progressive-packing.json'));args=p.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    manifest,report=build(args.source,args.plan,args.out);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
