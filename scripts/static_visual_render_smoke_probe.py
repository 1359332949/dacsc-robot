"""One predeclared failed Clean case, collision-display override only, labeled forensic."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.static_visual_v2 import run

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.manifest=args.audit/'dev_manifest.json';m=json.loads(args.manifest.read_text());args.case_id=sorted(m['cases'],key=lambda c:(c['base_task'],c['role'],c['case_id']))[0]['case_id']
    args.libero_src=Path('/hy-tmp/LIBERO-plus');args.corp_src=ROOT/'repos/CoRP';args.checkpoint=args.corp_src/'checkpoints/corp_libero.pt'
    sys.path[:0]=[str(args.libero_src),str(args.corp_src),str(args.corp_src/'scripts'),str(ROOT/'scripts')]
    import libero.libero.envs as envs
    original=envs.OffScreenRenderEnv
    def override(**kwargs):return original(**kwargs,render_collision_mesh=True,render_visual_mesh=True)
    envs.OffScreenRenderEnv=override
    try:run(args)
    finally:envs.OffScreenRenderEnv=original
    p=args.out/(args.case_id+'.json')
    receipt={'case_id':args.case_id,'result_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest(),
             'forensic_render_override':{'collision':True,'visual':True},'formal_clean_compatibility':False,
             'reason':'one fixed first case, isolate display difference; never add to main50 denominator',
             'forensic_script_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'forensic_annotation.json').write_text(json.dumps(receipt,indent=2)+'\n')
