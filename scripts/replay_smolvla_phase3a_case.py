"""One diagnostic replay of the first harmed case; never replaces main outcomes."""
import argparse,json
from phase3_smolvla_common import load_actor,rollout,reference_for,save_json


def main(a):
    original=json.loads((a.out/'guard/task0.json').read_text())
    actor=load_actor();refs=reference_for(a.out/'spatial_init0_calib.npz',0)
    rows=[];checks=[]
    for old,mode in zip(original['rows'],('frozen','v1')):
        row,_,_=rollout(actor,0,1,0,mode,refs)
        rows.append(row)
        checks.append({k:row[k]==old[k] for k in ('success','steps','start','actions_sha256','rng_before_replans')})
    save_json(a.out/'replay_task0.json',{'scope':'single post-pilot diagnostic replay, not additional independent scoring',
              'repeat_count':1,'rows':rows,'matches_original':checks,
              'all_fields_exact':all(all(c.values()) for c in checks)})
    print('REPLAY_DONE',checks,flush=True)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
