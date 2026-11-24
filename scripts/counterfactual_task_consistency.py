"""Frozen CoRP, existing paired RGB only. No simulator creation or policy training."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts'),str(ROOT/'scripts')]
from corp import load_policy
from collect_interaction_replay import encode,official_plan,probe,cpu,file_hash
from validate_interaction_replay import validate as validate_replay
from dacsc.interaction_records import array_hash
from dacsc.counterfactual_diagnostic import rms,require_current_pair,command_difference,separation,interaction


def read_pair(root,relative):
    folder=root/relative;data={};meta={}
    for dom in ('clean','shift'):
        meta[dom]=json.loads((folder/(dom+'.json')).read_text());p=folder/(dom+'.npz')
        if file_hash(p)!=meta[dom]['npz_sha256']:raise ValueError('input hash mismatch')
        with np.load(p,allow_pickle=False) as z:data[dom]={k:z[k] for k in z.files}
    return data,meta


def parameter_hash(policy):
    h=hashlib.sha256()
    for name,p in policy.state_dict().items():
        h.update(name.encode());h.update(np.ascontiguousarray(cpu(p)).tobytes())
    return h.hexdigest()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired-root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();config=json.loads(args.config.read_text())
    if args.out.exists():raise FileExistsError(args.out)
    input_checks={name:validate_replay(args.paired_root/name) for name in ('20261121_pilot','20261121_terminal')}
    pairs={};metas={}
    for sid,rel in config['canonical_sources'].items():pairs[int(sid)],metas[int(sid)]=read_pair(args.paired_root,rel)
    duplicate,_=read_pair(args.paired_root,config['duplicate_source']);repeat={}
    for dom in ('clean','shift'):
        repeat[dom]={}
        for key in ('state8','rgb_agentview','rgb_eye_in_hand','tokens_agentview','tokens_eye_in_hand','actions_executed','z0'):
            x=duplicate[dom][key];y=pairs[36][dom][key][:len(x)]
            if not np.array_equal(x,y):raise ValueError('duplicate prefix not identical '+key)
            repeat[dom][key]=0.
    for sid in pairs:
        for dom in ('clean','shift'):
            if metas[sid][dom]['checkpoint_sha256']!=file_hash(args.checkpoint):raise ValueError('checkpoint mismatch')
    policy=load_policy(args.checkpoint,device=torch.device('cuda:0'));policy.eval();policy.requires_grad_(False)
    if (policy.action_generator.action_horizon,policy.action_generator.action_dim,policy.action_generator.num_sample_steps)!=(12,7,10):raise ValueError('policy protocol drift')
    first=next(iter(metas.values()))['clean']
    for relative,digest in first['upstream_sha256'].items():
        if file_hash(ROOT/'repos/CoRP'/relative)!=digest:raise ValueError('upstream source drift '+relative)
    before=parameter_hash(policy);args.out.mkdir(parents=True)
    gen=torch.Generator(device='cuda:0');gen.manual_seed(config['z0_seed'])
    z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen)
    xp=pairs[34]['clean']['x_probe']
    for pair in pairs.values():
        for data in pair.values():
            if not np.array_equal(xp,data['x_probe']):raise ValueError('common probe differs')
    x=torch.tensor(xp,device='cuda:0',dtype=torch.float32)
    outputs={};records=[];all_checks=[]
    for sid,pair in pairs.items():
        c,s=pair['clean'],pair['shift'];language=metas[sid]['clean']['language']
        for i,step in enumerate(c['boundary_step']):
            q=require_current_pair(c,s,i);row={'state_id':sid,'boundary_index':i,'step':int(step),
                'current_state8':q.tolist(),'current_minus_initial_anchor_l2':float(np.linalg.norm(q-c['probe_anchor_state8'][i])),
                'subsequent_recorded_executed_steps':int(c['executed_steps'][i]) if i<len(c['executed_steps']) else 0,
                'current_is_recording_end':i==len(c['boundary_step'])-1,
                'preceding_step_flags':({key:bool(c[key][i-1,int(c['executed_steps'][i-1])-1]) for key in ('success','terminated','truncated','done')} if i else None),
                'variants':{},'checks':{}}
            per={}
            for variant in config['variant_order']:
                domains=config['variants'][variant]
                po={'images':{v:torch.from_numpy(pair[dom]['rgb_'+v][i].copy()).unsqueeze(0) for v,dom in zip(config['views'],domains)},
                    'state':torch.from_numpy(q.copy()).unsqueeze(0),'instruction':[language]}
                if not np.array_equal(cpu(po['state'][0]),q):raise ValueError('wrong proprio supplied')
                tok,rep=encode(policy,po)
                h,vel=probe(policy.action_generator,rep,x,config['probe_tau'],config['probe_action_index'])
                actions,parity=official_plan(policy,po,z0,audit=True)
                per[variant]={'tokens':cpu(tok[0]),'hidden':h,'velocity':vel,'actions':actions}
                row['checks'][variant+'_official_parity']=parity
                row['checks'][variant+'_state_input_hash']=array_hash(cpu(po['state'][0]))
                if variant=='CC':
                    h2,v2=probe(policy.action_generator,rep,x,config['probe_tau'],config['probe_action_index'])
                    repeat_error=max(float(np.max(np.abs(h2-h))),float(np.max(np.abs(v2-vel))))
                    if repeat_error>1e-6:raise ValueError('probe repeat failed')
                    row['checks']['same_x_repeat_max_abs']=repeat_error
                    if i<len(c['z0']):
                        original,_=official_plan(policy,po,torch.tensor(c['z0'][i:i+1],device='cuda:0'),audit=False)
                        e=float(np.max(np.abs(original-c['actions_planned'][i])))
                        if e>1e-5:raise ValueError('saved clean plan not reproduced '+str(e))
                        row['checks']['saved_clean_plan_max_abs']=e
                if variant in ('CC','SS'):
                    ref=c if variant=='CC' else s
                    saved=np.concatenate([ref['tokens_'+v][i] for v in config['views']])
                    error=float(np.max(np.abs(saved-per[variant]['tokens'])))
                    if error>1e-5:raise ValueError('Resampler stored parity failed '+str(error))
                    row['checks'][variant+'_stored_tokens_max_abs']=error
            for variant,sl in [('SC',slice(48,96)),('CS',slice(0,48))]:
                e=float(np.max(np.abs(per[variant]['tokens'][sl]-per['CC']['tokens'][sl])))
                if e>1e-5:raise ValueError('cross-view contamination')
                row['checks'][variant+'_other_view_max_abs']=e
            for variant in ('SC','CS','SS'):
                a=per[variant];b=per['CC'];metric={'same_x_velocity_rms':rms(a['velocity']-b['velocity']),
                    'same_x_hidden_k0_l2':float(np.linalg.norm(a['hidden']-b['hidden'])),
                    'same_x_velocity_k0_l2':float(np.linalg.norm(a['velocity'][0]-b['velocity'][0])),
                    'command':command_difference(a['actions'],b['actions'])}
                for vi,v in enumerate(config['views']):
                    sl=slice(vi*48,(vi+1)*48);delta=a['tokens'][sl]-b['tokens'][sl]
                    metric[v]={'token_rms':rms(delta),'pooled_l2':float(np.linalg.norm(delta.mean(0)))}
                row['variants'][variant]=metric
            row['nonadditive_interaction']={key:interaction(*[per[v][key] for v in config['variant_order']]) for key in ('velocity','actions','hidden')}
            row['nonadditive_interaction']['actions_postclip']=interaction(*[np.clip(per[v]['actions'],-1,1) for v in config['variant_order']])
            for variant,a in per.items():
                key=f's{sid}_t{int(step)}_{variant}'
                for name,value in a.items():
                    if not np.isfinite(value).all():raise ValueError('nonfinite output')
                    outputs[key+'_'+name]=value
            records.append(row)
            print(json.dumps({'state':sid,'step':int(step),'SS_velocity_rms':row['variants']['SS']['same_x_velocity_rms'],
                'SS_postclip_command_rms':row['variants']['SS']['command']['postclip']['all_rms']}),flush=True)
    # Descriptive between-state geometry at matched environment steps; do not fabricate shared actions.
    shared=sorted(set(pairs[34]['clean']['boundary_step'])&set(pairs[36]['clean']['boundary_step']))
    between=[]
    for step in shared:
        indices={sid:int(np.flatnonzero(pairs[sid]['clean']['boundary_step']==step)[0]) for sid in pairs}
        prior={sid:pairs[sid]['clean']['actions_executed'][:indices[sid]][pairs[sid]['clean']['executed_mask'][:indices[sid]]] for sid in pairs}
        same=prior[34].shape==prior[36].shape and np.array_equal(prior[34],prior[36])
        d={'step':int(step),'shared_past_action_sequence':same,'executed_past_steps':len(prior[34]),
           'note':'zero-step history is vacuously shared; future action consequence separation unmeasured',
           'current_state8_l2':float(np.linalg.norm(pairs[34]['clean']['state8'][indices[34]]-pairs[36]['clean']['state8'][indices[36]]))}
        for vi,v in enumerate(config['views']):
            vectors=[outputs[f's{sid}_t{int(step)}_{dom}_tokens'][vi*48:(vi+1)*48].mean(0) for dom in ('CC','SS') for sid in (34,36)]
            d[v]=separation(*vectors)
        for name in ('velocity','hidden','actions'):
            d[name]=separation(*[outputs[f's{sid}_t{int(step)}_{dom}_{name}'] for dom in ('CC','SS') for sid in (34,36)])
        between.append(d)
    temporal=[]
    for sid in pairs:
        steps=pairs[sid]['clean']['boundary_step'].tolist()
        for j,st in enumerate(steps):
            for en in steps[j+1:]:
                d={'state_id':sid,'start_step':st,'end_step':en}
                for vi,v in enumerate(config['views']):
                    d[v]=separation(*[outputs[f's{sid}_t{time}_{dom}_tokens'][vi*48:(vi+1)*48].mean(0) for dom in ('CC','SS') for time in (st,en)])
                temporal.append(d)
    aggregated={}
    for variant in ('SC','CS','SS'):
        by={}
        for sid in pairs:
            rr=[r for r in records if r['state_id']==sid]
            by[str(sid)]={'boundaries':len(rr),'mean_velocity_rms':float(np.mean([r['variants'][variant]['same_x_velocity_rms'] for r in rr])),
                **{key:float(np.mean([r['variants'][variant]['command']['postclip'][key] for r in rr])) for key in ('all_rms','xyz_l2_mean','rotation_channels_l2_mean','gripper_abs_mean')}}
        aggregated[variant]={'states':by,'equal_initial_state_mean':{k:float(np.mean([v[k] for v in by.values()])) for k in next(iter(by.values())) if k!='boundaries'}}
    after=parameter_hash(policy)
    if before!=after:raise ValueError('policy parameters mutated')
    np.savez_compressed(args.out/'outputs.npz',**outputs,x_probe=xp,z0=cpu(z0))
    out={'schema':'counterfactual_task_consistency_v1','pass':True,'config':config,'config_sha256':file_hash(args.config),
        'checkpoint_sha256':file_hash(args.checkpoint),'parameters_before':before,'parameters_after':after,
        'x_probe_sha256':array_hash(xp),'z0_sha256':array_hash(cpu(z0)),'source_deduplication':repeat,
        'input_validation':input_checks,'input_metadata':metas,
        'records':records,'aggregate':aggregated,'between_state_separation':between,'within_state_temporal_separation':temporal,
        'outputs_sha256':file_hash(args.out/'outputs.npz'),'independent_initial_states':2,'unique_boundaries':len(records),
        'unmeasured':['counterfactual physical futures under common cross-state actions','future task-risk labels','causal failure impact','effective intervention window'],
        'STOP_V1_preserved':True,'new_rollouts':0,'policy_updates':0,'observer_used':False,'target_test_accessed':False,
        'code_sha256':{p:file_hash(ROOT/p) for p in ['scripts/counterfactual_task_consistency.py','dacsc/counterfactual_diagnostic.py','scripts/collect_interaction_replay.py','scripts/validate_interaction_replay.py','dacsc/interaction_records.py']}}
    (args.out/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pass':True,'unique_boundaries':len(records),'aggregate':aggregated}),flush=True)

if __name__=='__main__':main()
