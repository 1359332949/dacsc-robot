"""Independent output/receipt and original Euler reconstruction verifier."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash
from dacsc.flow_geometry import geometry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());cases={c['case_id']:c for c in m['cases']}
    records=[json.loads(p.read_text()) for p in a.results.glob('*.json')];ledger=[json.loads(x) for x in (a.results/'access_ledger.jsonl').read_text().splitlines()]
    if len(records)!=len(cases) or {r['case_id'] for r in records}!=set(cases):raise ValueError('coverage mismatch')
    maxerr=0
    for r in records:
        cid=r['case_id'];p=a.results/(cid+'.json');bpath=a.baseline/(cid+'.json');b=json.loads(bpath.read_text())
        if r['status']!='completed' or r['manifest_sha256']!=file_hash(a.manifest):raise ValueError('invalid completion')
        if r['baseline_result_sha256']!=file_hash(bpath):raise ValueError('baseline changed')
        if not any(e.get('action')=='flow_diagnostic_end' and e.get('case_id')==cid and e.get('result_sha256')==file_hash(p) for e in ledger):raise ValueError('missing receipt')
        if r['z0_hashes'][0]!=b['chunks'][0]['z0_hash']:raise ValueError('policy RNG drift')
        if any(r['initial_rgb_hashes'][v]!=b['initial_observation'][v]['sha256'] for v in ('agentview','eye_in_hand')):raise ValueError('pixels drift')
        if r['baseline_success']!=b['success'] or r['policy_weights_updated'] or r['blind_accessed']:raise ValueError('role/label/update drift')
        v=np.asarray(r['velocity_trace'])
        if v.shape!=(4,10,1,12,7):raise ValueError('trace shape')
        for j in range(4):
            for name,sl in [('all',slice(None)),('xyz',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
                d=geometry(v[j,:,0][:,:,sl])
                for k,value in d.items():
                    stored=r['metrics'][j][name][k]
                    if value is None:
                        if stored is not None:raise ValueError('undefined metric changed')
                    elif not np.isclose(value,stored,atol=1e-10,rtol=1e-8):raise ValueError('metric recompute mismatch '+k)
        maxerr=max(maxerr,r['official_controller_max_abs'])
        if r['official_controller_max_abs']>1e-5:raise ValueError('native sampler mismatch')
    a.out.write_text(json.dumps({'pass':True,'cases':len(records),'native_predict_max_abs':maxerr,'both_rgb_hashes_verified':True,
        'all_trace_metrics_recomputed':True,'result_receipts_verified':True,'manifest_sha256':file_hash(a.manifest),
        'scope':'execution/data consistency, not scientific validation'},indent=2)+'\n')
    print(a.out.read_text())
if __name__=='__main__':main()
