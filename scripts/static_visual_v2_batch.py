"""Bounded worker over predeclared Source/Dev cases; no method training or Blind access."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.static_visual_v2 import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--libero-src',type=Path,required=True);p.add_argument('--corp-src',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    m=json.loads(a.manifest.read_text())
    if m.get('kind')=='blind':raise PermissionError('Blind denied before model imports')
    if not 0<=a.worker<a.workers:raise ValueError('worker bounds')
    selected=sorted(m['cases'],key=lambda x:(x['base_task'],x['role'],x['case_id']))[a.worker::a.workers]
    for case in selected:
        path=a.out/(case['case_id']+'.json')
        if path.exists():raise FileExistsError('no silent skipping or retries: '+str(path))
        a.case_id=case['case_id']
        try:run(a)
        except SystemExit as e:
            if e.code!=2:raise
            print(json.dumps({'case':a.case_id,'recorded_exception':True}),flush=True)
    print(json.dumps({'worker_complete':a.worker,'planned_cases':len(selected)}),flush=True)
