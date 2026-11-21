"""Audit all real source shards against saved V1 inputs and collector source."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.source_provenance import FIELDS,require_same_collection
from dacsc.source_observer import load_role

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    model=json.loads((ROOT/'docs/data/source_observer_v1/model.json').read_text());val=json.loads((ROOT/'docs/data/source_observer_v1/validation_report.json').read_text())
    saved={r['state_id']:r for r in model['inputs']+val['validation_inputs']}
    rows=[];signature=None
    for path in sorted(a.root.glob('*/*/metadata.json')):
        m=json.loads(path.read_text());signature=require_same_collection(m,signature)
        old=saved[m['state_id']]
        if old['metadata_sha256']!=sha(path) or old['npz_sha256']!=sha(path.parent/'episode.npz'):raise ValueError('V1 data changed')
        for field,source in [('script_sha256','scripts/collect_source_interactions.py'),('source_contract_sha256','dacsc/source_interactions.py'),('transition_contract_sha256','dacsc/interaction_records.py'),('shared_inference_sha256','scripts/collect_interaction_replay.py')]:
            if m[field]!=sha(ROOT/source):raise ValueError('collector fingerprint mismatch '+source)
        rows.append({'state_id':m['state_id'],'role':m['role'],'path':str(path),'metadata_sha256':sha(path),
                     'npz_sha256':old['npz_sha256'],'signature':signature,'steps':m['steps'],'success':m['success'],
                     'validation':m['validation'],'official_sampler_parity':m['official_sampler_parity']})
    actual={}
    for role in ('source_fit','source_validation'):
        d=load_role(a.root,a.split,role,('agentview','eye_in_hand'))
        actual[role]={'states':sorted(set(d.states.tolist())),'full_rows':len(d.states),'partial_excluded':d.excluded_partial}
    out={'pass':True,'common_signature':signature,'fields':FIELDS,'roles':actual,'records':rows,
         'smoke_conclusion':'fit_smoke and val_smoke are formal full-horizon acquisition shards; name denotes staged instrumentation check only',
         'V1_model_unchanged':sha(ROOT/'docs/data/source_observer_v1/model.npz')==model['model_sha256'],
         'retrain_required':False,'V1_decision':'STOP_V1',
         'claim_scope':'persisted metadata/data/source fingerprint verification; not proof of undocumented actions outside recorded provenance'}
    if not out['V1_model_unchanged']:raise ValueError('V1 model changed')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('records','fields')},indent=2))
