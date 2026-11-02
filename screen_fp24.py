"""Find encoder MatMul initializers and hash a selectively rounded candidate.

Only a small plan is written. Browser screening applies it to a verified copy
of the original model; no original ONNX or reference file is modified.
"""
import argparse,hashlib,json,mmap
from pathlib import Path
from pack_models import fields
from fp24 import split24


def selected_spans(data):
    tensors,consumers={},{}
    for n,w,a,b,_ in fields(data,0,len(data)):
        if n!=7 or w!=2:continue
        for n,w,a,b,_ in fields(data,a,b):
            if n==1 and w==2:
                fs=list(fields(data,a,b))
                name=next((data[x:y].decode() for n,w,x,y,v in fs if n==3),'')
                op=next((data[x:y].decode() for n,w,x,y,v in fs if n==4),'')
                for n,w,x,y,v in fs:
                    if n==1:consumers.setdefault(data[x:y].decode(),[]).append((name,op))
            elif n==5 and w==2:
                fs=list(fields(data,a,b))
                name=next((data[x:y].decode() for n,w,x,y,v in fs if n==8),'')
                dtype=next((v for n,w,x,y,v in fs if n==2),None)
                raw=next(((x,y) for n,w,x,y,v in fs if n==9 and w==2),None)
                if dtype==1 and raw:tensors[name]=raw
    result=[]
    for name,(a,b) in tensors.items():
        uses=consumers.get(name,[])
        if b-a>=262144 and uses and all(op=='MatMul' and node.startswith('/model/encoder/blocks.') for node,op in uses):
            result.append({'name':name,'offset':a,'bytes':b-a})
    return sorted(result,key=lambda r:r['offset'])


def plan(source):
    with source.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        spans=selected_spans(data);digest=hashlib.sha256();cursor=0
        for row in spans:
            a=row['offset'];b=a+row['bytes'];digest.update(data[cursor:a])
            for start in range(a,b,4*1024*1024):
                _,_,rounded=split24(data[start:min(b,start+4*1024*1024)])
                digest.update(rounded)
            cursor=b
        digest.update(data[cursor:])
        return {'bytes':len(data),'sourceSha256':hashlib.sha256(data).hexdigest(),'compactSha256':digest.hexdigest(),'selectedBytes':sum(r['bytes'] for r in spans),'spans':spans}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('public/models/precision/fp24-screen/plan.json'));args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result={'policy':'encoder-matmul-fp24-half-up-v1','files':{f'profiles/{profile}/moge3-dense.onnx':plan(Path(f'public/models/profiles/{profile}/moge3-dense.onnx')) for profile in ['fast','detail']}}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='spans'} for name,row in result['files'].items()}))
