"""Reopen completed V2 source/Dev results; verify receipts and lock status."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import file_hash,PROTECTED

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());mh=file_hash(a.manifest)
    if m['kind']=='blind':raise PermissionError('validator must not read Blind results')
    cases={r['case_id']:r for r in m['cases']};receipts=[json.loads(l) for l in (a.manifest.parent/'access_ledger.jsonl').read_text().splitlines()]
    rows=[]
    for cid,c in cases.items():
        p=a.results/(cid+'.json')
        if not p.exists():raise ValueError('missing planned case: '+cid)
        r=json.loads(p.read_text())
        if r['role'] in PROTECTED or r['case_id']!=cid or r['role']!=c['role']:raise ValueError('case/role mismatch')
        if r['manifest_sha256']!=mh:raise ValueError('manifest receipt mismatch')
        if not any(e.get('case_id')==cid and e.get('action')=='rollout_end' and e.get('result_sha256')==file_hash(p) for e in receipts):raise ValueError('result hash absent from append-only receipt')
        if r['status']!='completed':raise ValueError('case exception retained: '+r.get('error',''))
        if not r['physical_pair']['raw_state_equal'] or not r['physical_pair']['model_physics_compatible']:raise ValueError('physical pairing failed')
        end=0
        for chunk in r['chunks']:
            if chunk['start']!=end or not 0<chunk['end']-chunk['start']<=12:raise ValueError('bad executed time span')
            end=chunk['end']
            for k in ('x_max_abs','velocity_max_abs','controller_max_abs','clipped_max_abs'):
                if chunk['official_parity'] and chunk['official_parity'][k]>1e-5:raise ValueError('sampler parity')
        if end!=r['steps'] or end>220:raise ValueError('step accounting')
        if set(r['seeds'])!={'environment_seed','sensor_noise_seed','policy_flow_seed','method_augmentation_seed'}:raise ValueError('RNG stream fields')
        if not all(k in r['initial_observation'] for k in ('agentview','eye_in_hand')):raise ValueError('missing returned RGB receipt')
        rows.append({'case_id':cid,'result_sha256':file_hash(p),'role':r['role'],'success':r['success'],'steps':r['steps']})
    extras={p.stem for p in a.results.glob('*.json')}-set(cases)
    if extras:raise ValueError('undeclared result files: '+str(extras))
    blind=a.manifest.parent/'blind_manifest.json'
    bm=json.loads(blind.read_text()) if blind.exists() else None
    report={'pass':True,'manifest_sha256':mh,'cases':len(rows),'receipt_checked':True,'results':rows,
        'blind_cases':len(bm['cases']) if bm is not None else None,'blind_status':bm.get('status') if bm else None,
        'claim_scope':'data and execution integrity only; no statistical independence or performance guarantee'}
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'}))
