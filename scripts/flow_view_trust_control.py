"""Post-grid fixed attenuation control: test whether the view knob actually moves outputs."""
import argparse,json,sys,os,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from dacsc.static_visual_v2 import file_hash,authorize,seed_for,append_event

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['manifest','results','selection','repair-results','checkpoint','libero-src','corp-src','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--num-shards',type=int,default=1);ap.add_argument('--shard-id',type=int,default=0);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());sel=json.loads(a.selection.read_text());cases={c['case_id']:c for c in m['cases']}
    mh=file_hash(a.manifest);expected=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())[a.manifest.name]
    if mh!=sel['manifest_sha256']:raise ValueError('selection mismatch')
    for c in m['cases']:authorize(m,c,'view attenuation mechanism control',expected,a.manifest)
    sys.path[:0]=[str(a.libero_src),str(a.corp_src),str(a.corp_src/'scripts')]
    os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from evaluate_libero import policy_observation
    from libero.libero.benchmark import grab_language_from_filename
    from flow_causal_repair_probe import observe,encode_pieces,evaluate
    from dacsc.flow_repair import integrate,energy,build_representation
    if file_hash(a.checkpoint)!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint drift')
    policy=load_policy(a.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False);a.out.mkdir(parents=True,exist_ok=True)
    for cid in sel['case_ids'][a.shard_id::a.num_shards]:
        began=time.monotonic();c=cases[cid];ref=json.loads((a.results/(cid+'.json')).read_text());rp=json.loads((a.repair_results/(cid+'.json')).read_text());p=a.out/(cid+'.json')
        if p.exists():raise FileExistsError(p)
        obs=observe(c,ref,a);lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl');pieces=encode_pieces(policy,policy_observation(obs,lang,policy))
        held=torch.cat([torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed_for(ref['seeds']['method_augmentation_seed'],'flow_causal_repair_v1','heldout',j))) for j in range(4)])
        delta=torch.zeros((1,256),device='cuda:0');s=torch.zeros((),device='cuda:0',requires_grad=True)
        _,v=integrate(policy,build_representation(policy,pieces,delta,s,4),held);e,_=energy(v);g=torch.autograd.grad(e,s)[0]
        trials=[]
        for trust in [1.,.75,.5,.25,.1]:
            ev=evaluate(policy,pieces,held,delta,torch.tensor(np.log(trust),device='cuda:0',dtype=torch.float32));trials.append({'trust':trust,'evaluation':ev})
        if np.max(np.abs(np.asarray(trials[0]['evaluation']['controller_actions'])-np.asarray(rp['step0']['controller_actions'])))>1e-5:raise ValueError('step0 drift')
        record={'case_id':cid,'factor':c['factor'],'heldout_relative_energy_derivative_at_identity':float(g),'n_dit_blocks':len(policy.action_generator.blocks),
                'trials':trials,'source_repair_result_sha256':file_hash(a.repair_results/(cid+'.json')),'script_sha256':file_hash(Path(__file__)),
                'scope':'post-grid fixed mechanism control; not hyperparameter selection or policy rollout','wall_seconds':time.monotonic()-began}
        p.write_text(json.dumps(record,indent=2)+'\n');append_event(a.out/'access_ledger.jsonl',{'case_id':cid,'result_sha256':file_hash(p),'action':'view_control_end'})
        print(json.dumps({'case_id':cid,'done':True,'seconds':record['wall_seconds']}),flush=True)
