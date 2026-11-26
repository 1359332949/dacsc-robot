"""Native Frozen init0 trajectories and nested task-LOTO SigLIP reference."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from phase3_smolvla_common import load_actor,rollout,save_json,sha,LAYERS
from dacsc.smolvla_v1_hook import fit_loto


def main(a):
    if a.mode=='source':
        actor=load_actor();before=[p._version for p in actor._model.parameters()]
        dest=a.out/'source';dest.mkdir(parents=True,exist_ok=True)
        for task in range(a.shard,10,a.workers):
            f=dest/f'task{task}.npz';assert not f.exists()
            row,raw,_=rollout(actor,task,0,0,'source',out=dest/f'task{task}_initial.npz')
            np.savez_compressed(f,**raw)
            row['moments_sha256']=sha(f);row['frames']=raw['mean'].shape[0]
            save_json(dest/f'task{task}.json',row)
            print('SOURCE_DONE',task,row['steps'],row['seconds'],flush=True)
        assert before==[p._version for p in actor._model.parameters()]
        assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
    else:
        raw={};provenance={}
        for task in range(10):
            f=a.out/'source'/f'task{task}.npz';r=json.loads(f.with_suffix('.json').read_text())
            assert sha(f)==r['moments_sha256'] and r['init_index']==0
            with np.load(f,allow_pickle=False) as z:raw[task]={k:torch.from_numpy(z[k].copy()) for k in ('mean','second')}
            assert raw[task]['mean'].shape[1:]==(4,2,768)
            provenance[task]={'task':r['task'],'frames':r['steps'],'source_sha256':sha(f)}
        refs,audit=fit_loto(raw)
        payload={f'task{t}_layer{l}_{k}':x.numpy() for (t,l),v in refs.items() for k,x in zip(('mu','ss','tau'),v)}
        payload['schema']=np.array('smolvla_v1_init0_loto_v1')
        payload['task_names']=np.array([provenance[t]['task'] for t in range(10)])
        f=a.out/'spatial_init0_calib.npz';assert not f.exists();np.savez_compressed(f,**payload)
        save_json(a.out/'folds.json',{'folds':audit,'source':provenance,'calib_sha256':sha(f),
                                    'weighting':'all actual source frames equally; no success filtering'})
        print('FIT_DONE',sha(f),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('source','fit'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--shard',type=int,default=0);main(p.parse_args())
