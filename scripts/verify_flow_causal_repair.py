"""Independent fixed-grid/constraints/null norm and result receipt checks."""
import argparse,json,sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    sel=json.loads(a.selection.read_text());files=sorted(a.results.glob('*.json'));rr=[json.loads(p.read_text()) for p in files]
    ledger=[json.loads(x) for x in (a.results/'access_ledger.jsonl').read_text().splitlines()]
    if len(rr)!=10 or {r['case_id'] for r in rr}!=set(sel['case_ids']):raise ValueError('coverage mismatch')
    matrix=set(itertools.product(sel['grid']['knobs'],sel['grid']['objectives'],sel['grid']['learning_rates']))
    norm_error=0.;maxparity=0.;maxnative=0.;maxv1=0.;count=0
    def check_eval(x):
        d=np.asarray(x['spatial_bias']);s=float(x['view_log_trust'])
        if d.shape!=(1,256) or not np.isfinite(d).all() or d.max()>1e-6 or d.min()<-8.00001 or abs(d.max())>1e-5:raise ValueError('spatial constraint/gauge')
        if not -4.60518<=s<=1e-6 or not np.isclose(np.exp(s),x['view_trust'],rtol=1e-6):raise ValueError('view constraint')
        for k in ['controller_actions','clipped_controller_actions']:
            arr=np.asarray(x[k])
            if arr.shape!=(4,12,7) or not np.isfinite(arr).all():raise ValueError('action shape/finite')
        if not np.allclose(np.clip(x['controller_actions'],-1,1),x['clipped_controller_actions'],atol=1e-7):raise ValueError('clipping parity')
        return d,s
    for p,r in zip(files,rr):
        if r['status']!='completed' or r['selection_sha256']!=file_hash(a.selection):raise ValueError('status/selection')
        if r['policy_weights_updated'] or r['adapted_actions_executed'] or r['blind_accessed']:raise ValueError('forbidden action')
        if not any(e.get('action')=='repair_end' and e.get('case_id')==r['case_id'] and e.get('result_sha256')==file_hash(p) for e in ledger):raise ValueError('missing receipt')
        if set(r['training_z0_hashes'])&set(r['heldout_z0_hashes']) or len(set(r['heldout_z0_hashes']))!=4:raise ValueError('noise separation')
        maxparity=max(maxparity,r['zero_gate_max_abs']);maxnative=max(maxnative,r['native_sampler_max_abs']);maxv1=max(maxv1,r['batched_v1_controller_max_abs'])
        if r['zero_gate_max_abs']>1e-5 or r['native_sampler_max_abs']>1e-5 or r['batched_v1_controller_max_abs']>1e-4:raise ValueError('forward parity')
        actual={(c['knob'],c['objective'],c['learning_rate']) for c in r['configurations']}
        if len(r['configurations'])!=18 or actual!=matrix:raise ValueError('grid changed')
        d,s=check_eval(r['step0'])
        if np.any(d) or s!=0:raise ValueError('step0 not identity')
        for c in r['configurations']:
            if len(c['history'])!=10 or set(c['checkpoints'])!={'5','10'}:raise ValueError('update count')
            for step in ('5','10'):
                ck=c['checkpoints'][step];d,s=check_eval(ck['adapted'])
                if len(ck['random_nulls'])!=5:raise ValueError('null count')
                if c['knob']=='view' and np.any(d):raise ValueError('view changed spatial')
                if c['knob']=='spatial' and s!=0:raise ValueError('spatial changed trust')
                for n in ck['random_nulls']:
                    nd,ns=check_eval(n);err=abs(np.linalg.norm(d)-np.linalg.norm(nd));norm_error=max(norm_error,err)
                    if err>max(1e-4,np.linalg.norm(d)*1e-5) or abs(ns-s)>1e-7:raise ValueError('unmatched null magnitude')
                    if c['knob']=='view' and not np.array_equal(n['controller_actions'],ck['adapted']['controller_actions']):raise ValueError('scalar null should be identical')
                count+=1
    a.out.write_text(json.dumps({'pass':True,'cases':10,'configurations':180,'checkpoint_evaluations':count,'matched_null_evaluations':count*5,
        'max_null_spatial_norm_error':float(norm_error),'max_zero_gate_controller_error':maxparity,'max_native_sampler_error':maxnative,'max_batched_v1_error':maxv1,
        'receipt_and_constraints_verified':True,'heldout_noise_separation_verified':True,'scope':'consistency only; no task-repair inference'},indent=2)+'\n')
    print(a.out.read_text())
