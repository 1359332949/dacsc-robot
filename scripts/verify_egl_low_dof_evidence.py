"""Verify case identities, trace lengths and random-noise provenance from archived receipts."""
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash

def verify(root):
    hashes={};seeds={};manifest=ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json';cases={r['case_id']:r for r in json.loads(manifest.read_text())['cases']}
    for archive in ['seed0_evidence.tar.gz','repeat_evidence.tar.gz']:
        seen=set();fit_by_case={};ranges={}
        with tarfile.open(root/archive) as f:
            for m in f.getmembers():
                if not m.isfile() or not Path(m.name).name.startswith('run_s') or not m.name.endswith('.json'):continue
                r=json.load(f.extractfile(m));cid=r['case_id'];c=cases[cid];assert c['role'] in ('target_dev','dev_clean') and r['factor']==c['factor'] and r['base_task']==c['base_task']
                k=(r['seed'],cid,r['method']);assert k not in seen and len(r['trace'])==(r['steps']+11)//12;seen.add(k)
                assert all(len(x)==64 for t in r['trace'] for x in [*t['images'],t['state'],t['plan']]);assert type(r['success']) is bool
                if r['gate']:
                    g=r['gate'];z=g['training_noise_hashes']+g['validation_noise_hashes'];assert len(set(z))==8
                    if cid in fit_by_case:assert fit_by_case[cid]==z
                    fit_by_case[cid]=z
                t=ranges.setdefault(r['seed'],[float('inf'),0]);t[0]=min(t[0],m.mtime-r['seconds']);t[1]=max(t[1],m.mtime)
        for seed,times in ranges.items():
            methods=('frozen','trust','joint','rgb') if seed==0 else ('frozen','trust');assert {(s,c,m) for s,c,m in seen if s==seed}=={(seed,c,m) for c in cases for m in methods}
            seeds[seed]={'methods':methods,'rollouts':len(cases)*len(methods),'observed_wall_span_minutes':round((times[1]-times[0])/60,2)}
        hashes[archive]=file_hash(root/archive)
    assert set(seeds)=={0,1,2}
    return {'manifest_sha256':file_hash(manifest),'archives':hashes,'seeds':seeds,'scope':'integrity and completeness; not an independent confirmation dataset'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();(a.root/'verification.json').write_text(json.dumps(verify(a.root),indent=2)+'\n')
