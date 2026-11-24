"""Instrumentation-only Clean/Lighting replay; no adaptation or risk classifier.

Actual frozen Clean commands drive both simulators. Native returned observations
are retained without sim.forward()/state restore at chunk boundaries. A common
x_probe, tau, action index, initial proprioceptive anchor and language isolate
visual conditioning. Features/hidden come from hooks on the official forward.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'repos/CoRP'), str(ROOT/'repos/CoRP/scripts'), str(ROOT/'scripts')]
from corp import load_policy
from corp.preprocessing import prepare_observation
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation
from target_only_action_validation import configure
from dacsc.interaction_records import VIEWS, array_hash, step_flags, validate_pair
from dacsc.state_pairing import snapshot, model_compatible


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def cpu(t): return t.detach().cpu().numpy().copy()


@torch.inference_mode()
def encode(policy, po):
    batch=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
    captured=[]
    handle=policy.representation_extractor.cross_attention_resampler.register_forward_hook(
        lambda mod,inp,out:captured.append(out.detach()))
    try: rep=policy.representation_extractor(**batch)
    finally: handle.remove()
    if len(captured)!=1 or captured[0].shape != (1,96,768): raise ValueError('unexpected visual tokens')
    return captured[0],rep


@torch.inference_mode()
def probe(gen, rep, x_probe, tau, index):
    captured=[]
    handle=gen.output_norm.register_forward_hook(lambda mod,inp,out:captured.append(out.detach()))
    x=x_probe.clone(); before=cpu(x)
    try:
        velocity=gen(x,torch.full((1,),tau,device=x.device,dtype=x.dtype),gen.prepare_prefix(rep))
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('unexpected hidden hook calls')
    if not np.array_equal(cpu(x),before):raise ValueError('probe action state mutated')
    return cpu(captured[0][0,index]),cpu(velocity[0])


@torch.inference_mode()
def official_plan(policy, po, z0, audit=False):
    """Inject ONLY the initial random draw, leaving policy.predict intact."""
    draws=[]; velocities=[]; inputs=[]
    def fixed_randn(*args,**kwargs):
        shape=args[0] if len(args)==1 else args
        if tuple(shape)!=tuple(z0.shape):raise ValueError('unexpected random draw in predict')
        out=z0.to(device=kwargs.get('device',z0.device),dtype=kwargs.get('dtype',z0.dtype)).clone()
        draws.append(array_hash(cpu(out)));return out
    handles=[]
    if audit:
        handles=[policy.action_generator.register_forward_hook(
            lambda mod,args,out:(inputs.append(cpu(args[0])),velocities.append(cpu(out))) and None)]
    try:
        with patch('torch.randn',side_effect=fixed_randn):controller=policy.predict(po)
    finally:
        for handle in handles:handle.remove()
    if len(draws)!=1:raise ValueError('predict must use exactly one initial noise draw')
    report={}
    if audit:
        _,rep=encode(policy,po);gen=policy.action_generator;prefix=gen.prepare_prefix(rep)
        x=z0.to(rep.action_dtype).clone(); dx=[];dv=[]
        if len(inputs)!=gen.num_sample_steps:raise ValueError('official sampler step count')
        for i in range(gen.num_sample_steps):
            t=torch.full((1,),i/gen.num_sample_steps,device=x.device,dtype=x.dtype)
            v=gen(x,t,prefix)
            dx.append(float(np.max(np.abs(cpu(x)-inputs[i]))))
            dv.append(float(np.max(np.abs(cpu(v)-velocities[i]))))
            x=x+v*(1./gen.num_sample_steps)
        manual=policy.normalizer.unnormalize_action(x)
        report={'x_max_abs':max(dx),'velocity_max_abs':max(dv),
                'controller_max_abs':float((manual-controller).abs().max()),
                'clipped_max_abs':float((manual.clamp(-1,1)-controller.clamp(-1,1)).abs().max()),
                'official_predict_calls':1,'steps':len(inputs)}
        if max(report[k] for k in ('x_max_abs','velocity_max_abs','controller_max_abs','clipped_max_abs'))>1e-5:
            raise ValueError('official sampler parity failed '+str(report))
    return cpu(controller[0]),report


def physics_pair(clean,shift):
    """Read-only measured state check; never refresh or restore either env."""
    a,b=snapshot(clean),snapshot(shift);diff={}
    for key in ('qpos','qvel','time','body_xpos','body_xquat','eef_pos','site_xpos','site_xmat','ctrl'):
        if a.get(key) is None or b.get(key) is None:raise ValueError('missing physics '+key)
        av,bv=np.asarray(a[key]),np.asarray(b[key])
        if av.shape!=bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():raise ValueError('invalid physics '+key)
        diff[key]=float(np.max(np.abs(av-bv))) if av.size else 0.
    if max(diff.values())>1e-6:raise ValueError('measured pair divergence '+str(diff))
    return diff


@torch.inference_mode()
def boundary(policy, obs, language, anchor, x_probe, args):
    po=policy_observation(obs,language,policy)
    tokens,_=encode(policy,po)
    anchored=dict(po,state=anchor.clone())
    _,rep=encode(policy,anchored)
    h,v=probe(policy.action_generator,rep,x_probe,args.probe_tau,args.probe_index)
    return po,{'state8':cpu(po['state'][0]),'probe_anchor_state8':cpu(anchor[0]),
               'tokens_agentview':cpu(tokens[0,:48]),'tokens_eye_in_hand':cpu(tokens[0,48:]),
               'rgb_agentview':cpu(po['images']['agentview'][0]),
               'rgb_eye_in_hand':cpu(po['images']['eye_in_hand'][0]),
               'probe_hidden':h,'probe_velocity':v}


@torch.inference_mode()
def probe_controls(policy, clean_po, shift_po, anchor, x_probe,args):
    c=dict(clean_po,state=anchor.clone());s=dict(shift_po,state=anchor.clone())
    tc,rc=encode(policy,c);ts,rs=encode(policy,s)
    hc,vc=probe(policy.action_generator,rc,x_probe,args.probe_tau,args.probe_index)
    hc2,vc2=probe(policy.action_generator,rc,x_probe,args.probe_tau,args.probe_index)
    hs,vs=probe(policy.action_generator,rs,x_probe,args.probe_tau,args.probe_index)
    repeat=max(float(np.max(np.abs(hc-hc2))),float(np.max(np.abs(vc-vc2))))
    if repeat>1e-6:raise ValueError('fixed probe not repeatable')
    ownership={}
    for view,sl,other in [('agentview',slice(0,48),slice(48,96)),('eye_in_hand',slice(48,96),slice(0,48))]:
        images=dict(c['images']);images[view]=s['images'][view]
        tv,_=encode(policy,dict(c,images=images))
        changed=float((tv[0,sl]-tc[0,sl]).abs().max())
        unchanged=float((tv[0,other]-tc[0,other]).abs().max())
        if unchanged>1e-5 or changed<=0:raise ValueError('view ownership failed '+view)
        ownership[view]={'changed_view_max_abs':changed,'other_view_max_abs':unchanged}
    return {'repeat_max_abs':repeat,'clean_shift_hidden_l2':float(np.linalg.norm(hc-hs)),
            'clean_shift_velocity_l2':float(np.linalg.norm(vc-vs)),'view_ownership':ownership,
            'only_images_vary':True}


def collect_pair(policy,variant,states,sid,args,language,selected_from,base_meta):
    from libero.libero.envs import OffScreenRenderEnv
    envs={}; observations={}; data={d:[] for d in ('clean','shift')}; transitions={d:[] for d in data}
    max_physics={};pair_id=f'state{sid}_seed{args.seed}'; out=args.out/pair_id;out.mkdir()
    try:
        for domain,bddl in [('clean',variant['base_bddl']),('shift',variant['env_bddl_name'])]:
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            envs[domain]=env
            np.random.seed(args.seed);torch.manual_seed(args.seed);env.seed(args.seed);env.reset()
            obs=env.set_init_state(states[sid])
            for _ in range(SETTLING_STEPS):
                obs,flags=step_flags(env.step(DUMMY_ACTION),env)
                if any(flags[k] for k in ('success','done','terminated','truncated')):raise ValueError('episode ended during settling')
            observations[domain]=obs
        compat=model_compatible(snapshot(envs['clean']),snapshot(envs['shift']),strict_bodies=True,strict_geometry=True,strict_physics=True)
        if not compat['pass']:raise ValueError('static model incompatibility '+str([k for k,v in compat['checks'].items() if not v]))
        max_physics=physics_pair(envs['clean'],envs['shift'])
        anchor=policy_observation(observations['clean'],language,policy)['state'].clone()
        generator=torch.Generator(device='cuda:0');generator.manual_seed(args.probe_seed)
        x_probe=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=generator)
        # Probe tensor is independent of all observations and execution seeds.
        zg=torch.Generator(device='cuda:0');zg.manual_seed(args.seed)
        plans=[];z0s=[];timesteps=[0];simtimes={d:[float(envs[d].sim.data.time)] for d in data}
        pos={}
        for d in data:
            pos[d],b=boundary(policy,observations[d],language,anchor,x_probe,args);data[d].append(b)
        controls=probe_controls(policy,pos['clean'],pos['shift'],anchor,x_probe,args)
        _,shift_parity=official_plan(policy,pos['shift'],x_probe,audit=True)
        parity=[];total=0;stopped=False
        for chunk in range(args.chunks):
            z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=zg);z0s.append(cpu(z0[0]))
            acts,par=official_plan(policy,pos['clean'],z0,audit=(chunk==0))
            if par:parity.append(par)
            plans.append(acts); clipped=np.clip(acts,-1,1)
            records={d:{k:np.zeros(12,bool) for k in ('success','terminated','truncated','done')} for d in data}
            for d in data:records[d]['reward']=np.zeros(12,np.float32)
            n=0
            for k in range(min(12,args.max_steps-total)):
                for d in data:
                    observations[d],flags=step_flags(envs[d].step(clipped[k].copy()),envs[d])
                    for key in records[d]:records[d][key][k]=flags[key]
                n+=1;total+=1
                diff=physics_pair(envs['clean'],envs['shift'])
                for key,value in diff.items():max_physics[key]=max(max_physics[key],value)
                for key in ('success','terminated','truncated','done'):
                    if records['clean'][key][k]!=records['shift'][key][k]:raise ValueError('paired termination mismatch '+key)
                if any(records['clean'][key][k] for key in ('success','terminated','truncated','done')):
                    stopped=True;break
                if total>=args.max_steps:
                    # Collector cutoff is truncation, not simulator done or failure.
                    for d in data:records[d]['truncated'][k]=True
                    stopped=True;break
            if chunk==args.chunks-1 and not stopped:
                for d in data:records[d]['truncated'][n-1]=True
                stopped=True
            mask=np.arange(12)<n;executed=np.zeros_like(clipped);executed[mask]=clipped[mask]
            timesteps.append(total)
            for d in data:
                records[d].update(actions_planned=acts.copy(),actions_executed=executed.copy(),executed_mask=mask.copy(),
                                  executed_steps=n,full_chunk=(n==12),fixed_horizon_eligible=(n==12))
                transitions[d].append(records[d]);simtimes[d].append(float(envs[d].sim.data.time))
                pos[d],b=boundary(policy,observations[d],language,anchor,x_probe,args);data[d].append(b)
            print(json.dumps({'pair':pair_id,'chunk':chunk,'steps':n,'total':total,'physics_max':max(max_physics.values()),
                              'success':bool(records['clean']['success'][n-1])}),flush=True)
            if stopped:break
        arrays={};metas={}
        for d in data:
            a={key:np.stack([b[key] for b in data[d]]) for key in data[d][0]}
            a.update({key:np.asarray([r[key] for r in transitions[d]]) for key in transitions[d][0]})
            a.update(boundary_step=np.asarray(timesteps),boundary_sim_time=np.asarray(simtimes[d]),x_probe=cpu(x_probe),z0=np.stack(z0s))
            m=dict(base_meta,domain=d,pair_id=pair_id,state_id=sid,seed=args.seed,language=language,
                   control_dt=1/float(envs[d].env.control_freq),probe_tau=args.probe_tau,probe_action_index=args.probe_index,
                   probe_seed=args.probe_seed,x_probe_sha256=array_hash(a['x_probe']),
                   initial_state_sha256=array_hash(np.asarray(states[sid])),
                   selected_from=selected_from,driver_domain='clean',
                   flags_semantics='success from checker; legacy done preserved; collector cap sets truncated only',
                   observation_semantics='native returned observations; no boundary refresh or state restore',
                   probe_semantics='common x,tau,k plus fixed initial state8 and language; only images vary',
                   risk_labels='not assigned; episode outcome is not copied to chunks',
                   rollout_outcome='controlled Clean-command replay, not independent Lighting policy success')
            arrays[d]=a;metas[d]=m
        check=validate_pair(arrays['clean'],arrays['shift'],metas['clean'],metas['shift'])
        for d in data:
            path=out/(d+'.npz');np.savez_compressed(path,**arrays[d]);metas[d]['npz_sha256']=file_hash(path)
            (out/(d+'.json')).write_text(json.dumps(metas[d],indent=2,allow_nan=False)+'\n')
        report={'pair_id':pair_id,'state_id':sid,'seed':args.seed,'contract':check,'physics_max_abs_by_field':max_physics,
                'static_model_checks':compat,'probe_controls':controls,'official_sampler_parity_clean':parity,
                'official_sampler_parity_shift':shift_parity,
                'final_success':bool(transitions['clean'][-1]['success'][transitions['clean'][-1]['executed_steps']-1])}
        for view in VIEWS:
            c=arrays['clean']['tokens_'+view];s=arrays['shift']['tokens_'+view]
            report[view]={'paired_token_rms':float(np.sqrt(np.mean((c-s)**2))),
                          'paired_transition_delta_rms':float(np.sqrt(np.mean((np.diff(c,axis=0)-np.diff(s,axis=0))**2)))}
        (out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    finally:
        for env in envs.values():env.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True);ap.add_argument('--episode-log',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--states',default='')
    ap.add_argument('--seed',type=int,default=7777);ap.add_argument('--chunks',type=int,default=3)
    ap.add_argument('--max-steps',type=int,default=36);ap.add_argument('--probe-seed',type=int,default=20261121)
    ap.add_argument('--probe-tau',type=float,default=.5);ap.add_argument('--probe-index',type=int,default=0)
    ap.add_argument('--suite',default='libero_spatial');ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus'))
    ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    ap.add_argument('--render-gpu',type=int,default=0)
    args=ap.parse_args()
    if os.environ.get('MUJOCO_GL')!='osmesa' or os.environ.get('PYOPENGL_PLATFORM')!='osmesa':raise ValueError('OSMesa required')
    if args.chunks<=0 or args.max_steps<=0 or not 0<=args.probe_tau<1 or not 0<=args.probe_index<12:raise ValueError('invalid protocol')
    if args.out.exists():raise FileExistsError(args.out)
    configure(args)
    import libero.libero.envs.bddl_utils as BDDLUtils
    split=json.loads(args.split.read_text());matrix=json.loads(args.matrix.read_text());log=json.loads(args.episode_log.read_text())
    if args.lighting_id!=split['target_dev_lighting_id']:raise ValueError('only target-dev perturbation allowed')
    if args.seed not in split['screening_seeds']:raise ValueError('only screening seeds allowed')
    candidates=[r for r in log['records'] if r.get('method')=='frozen' and r.get('seed')==args.seed]
    ids=[r['init_index'] for r in candidates]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate episode keys')
    by_id={r['init_index']:r for r in candidates}
    if args.states:
        selected=[int(s) for s in args.states.split(',')]
    else:
        # Deterministic minimal instrumentation sample from actual logged outcomes.
        selected=[min(r['init_index'] for r in candidates if bool(r['success'])==success) for success in (False,True)]
    if len(selected)!=len(set(selected)) or not set(selected)<=set(split['target_dev_state_ids']):raise ValueError('invalid dev state selection')
    if not set(selected)<=set(by_id):raise ValueError('state/seed absent from actual episode log')
    if log['provenance']['lighting_id']!=args.lighting_id:raise ValueError('wrong episode-log perturbation')
    task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False)
    language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    checkpoint_hash=file_hash(args.checkpoint)
    if checkpoint_hash!=split['policy']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    policy=load_policy(args.checkpoint,device=torch.device('cuda:0'));policy.eval();policy.requires_grad_(False)
    gen=policy.action_generator
    if (gen.action_horizon,gen.action_dim,gen.num_sample_steps)!=(12,7,10):raise ValueError('CoRP protocol mismatch')
    if tuple(policy.representation_extractor.view_names)!=VIEWS:raise ValueError('view order mismatch')
    args.out.mkdir(parents=True)
    metadata={'schema':'controlled_interaction_replay_v1','role':'target_dev_instrumentation_only','H':12,'E':12,
              'policy_weights_updated':False,'target_test_accessed_this_run':False,'checkpoint_sha256':checkpoint_hash,
              'split_sha256':file_hash(args.split),'episode_log_sha256':file_hash(args.episode_log),
              'matrix_sha256':file_hash(args.matrix),'initial_state_file_sha256':file_hash(task['state_split']['source_file']),
              'base_bddl_sha256':file_hash(variant['base_bddl']),'shift_bddl_sha256':file_hash(variant['env_bddl_name']),
              'script_sha256':file_hash(__file__),'contracts_sha256':file_hash(ROOT/'dacsc/interaction_records.py'),
              'torch_version':torch.__version__,'numpy_version':np.__version__,'precision':'fp32','renderer':'osmesa',
              'upstream_sha256':{p:file_hash(ROOT/'repos/CoRP'/p) for p in ['corp/policy.py','corp/action_generator.py','corp/representation/extractor.py','scripts/evaluate_libero.py']},
              'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    reports=[]
    for sid in selected:
        record=by_id[sid]
        reports.append(collect_pair(policy,variant,states,sid,args,language,
                       {'logged_lighting_success':record['success'],'logged_steps':record['steps'],'action_sha256':record.get('action_sha256')},metadata))
    result={'metadata':metadata,'selected_states':selected,'reports':reports,'pass':True,
            'claim_scope':'data/probe correctness only; no learned observer, prediction-risk AUROC, or TTA effectiveness claim'}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pass':True,'out':str(args.out),'states':selected}),flush=True)

if __name__=='__main__':main()
