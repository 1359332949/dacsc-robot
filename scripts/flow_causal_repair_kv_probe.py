"""Selected Dev t=0 flow-repair mechanism probe; reference actions only after updates.

No candidate actions executed. Full fixed 3-knob x 3-objective x 2-LR grid,
steps 0/5/10 and five matched-norm null controls. No best-Clean-distance selection.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import (file_hash,array_hash,authorize,EnvironmentRNG,sensor_transform,seed_for,append_event)


def freeze_selection(args):
    m=json.loads(args.manifest.read_text());cases={c['case_id']:c for c in m['cases']}
    rr=[json.loads(p.read_text()) for p in args.flow_results.glob('*.json')]
    clean={cases[r['case_id']]['pair_group_id']:r for r in rr if r['factor']=='clean'}
    chosen=[];ratios={}
    for factor,n in [('sensor_noise',5),('background',3)]:
        candidates=[]
        for r in rr:
            if r['factor']!=factor or r['baseline_success']:continue
            cr=clean[cases[r['case_id']]['pair_group_id']]
            ratio=np.mean([v['all']['velocity_variance'] for v in r['metrics']])/np.mean([v['all']['velocity_variance'] for v in cr['metrics']])
            candidates.append((float(ratio),r['case_id']))
        for ratio,cid in sorted(candidates,key=lambda x:(-x[0],x[1]))[:n]:chosen.append(cid);ratios[cid]=ratio
    chosen += sorted(r['case_id'] for r in rr if r['factor']=='clean')[:2]
    if len({cases[c]['base_task'] for c in chosen[-2:]})!=2:raise ValueError('Clean controls must have different tasks')
    result={'schema':'flow_causal_repair_selection_v1','manifest_sha256':file_hash(args.manifest),'case_ids':chosen,'historical_clean_ratios_for_selection_only':ratios,
            'selection_scope':'outcome and paired-Clean selected exploratory extreme cases; not a risk evaluation cohort',
            'grid':{'knobs':['spatial','view','joint'],'objectives':['relative','late','anchored'],'learning_rates':[.3,.8],'checkpoints':[0,5,10],'train_noises':4,'heldout_noises':4,'null_draws':5},
            'flow_input_sha256':{r['case_id']:file_hash(args.flow_results/(r['case_id']+'.json')) for r in rr}}
    args.selection.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'selected':chosen,'ratios':ratios}))


def encode_pieces(policy,po):
    import torch
    from corp.preprocessing import prepare_observation
    with torch.no_grad():
        b=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0');ex=policy.representation_extractor
        p=ex.visual_encoder(torch.cat([b['images'][n] for n in ex.view_names],dim=0))
        patches={n:p[i:i+1].detach() for i,n in enumerate(ex.view_names)}
        lang,pool=ex.language_encoder(b['instruction']);state=ex.state_encoder(b['state'])
    return {'patches':patches,'language':lang.detach(),'pool':pool.detach(),'state':state.detach()}


def observe(c,ref,args):
    import torch
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.env_wrapper as wrapper
    rng=EnvironmentRNG(ref['seeds']['environment_seed']);env=None
    try:
        with rng.use():
            env=OffScreenRenderEnv(bddl_file_name=str(args.libero_src/'libero/libero/bddl_files'/c['suite']/(c['logical_task_name']+'.bddl')),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            if int(env.noise)!=int(c['runtime_params']['noise']):raise ValueError('noise parse mismatch')
            env.noise=0;env.seed(ref['seeds']['environment_seed']);env.reset()
            initial=torch.load(args.libero_src/c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
            if array_hash(np.asarray(initial))!=c['raw_init_hash']:raise ValueError('initial asset drift')
            obs=env.set_init_state(initial)
            for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
        obs=sensor_transform(obs,int(c['runtime_params']['noise']),ref['seeds']['sensor_noise_seed'],0,wrapper)
        for view,key in [('agentview','agentview_image'),('eye_in_hand','robot0_eye_in_hand_image')]:
            if array_hash(np.asarray(obs[key]))!=ref['initial_observation'][view]['sha256']:raise ValueError('original RGB mismatch '+view)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()}
    finally:
        if env is not None:env.close()


def evaluate(policy,pieces,z,delta,s):
    import torch
    from dacsc.flow_repair_kv import build_representation,integrate,energy
    with torch.no_grad():
        rep=build_representation(policy,pieces,delta,s,len(z));x,v=integrate(policy,rep,z);_,e=energy(v)
        actions=policy.normalizer.unnormalize_action(x)
    return {'controller_actions':actions.cpu().tolist(),'clipped_controller_actions':actions.clamp(-1,1).cpu().tolist(),
            'raw_curvature':e['raw'].cpu().tolist(),'relative_curvature':e['relative'].cpu().tolist(),
            'mean_speed_norm':e['mean_speed_norm'].cpu().tolist(),'spatial_bias':delta.cpu().tolist(),'view_log_trust':float(s),'view_trust':float(s.exp())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--flow-results',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--freeze-selection',action='store_true');ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--libero-src',type=Path);ap.add_argument('--corp-src',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--num-shards',type=int,default=1);ap.add_argument('--shard-id',type=int,default=0);ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args()
    if a.freeze_selection:return freeze_selection(a)
    m=json.loads(a.manifest.read_text());sel=json.loads(a.selection.read_text());mh=file_hash(a.manifest)
    if sel['manifest_sha256']!=mh or m.get('kind')!='dev' or m.get('status')!='FROZEN':raise PermissionError('frozen Dev selection only')
    cases={c['case_id']:c for c in m['cases']};hashes=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())
    for c in m['cases']:authorize(m,c,'offline flow intervention probe',hashes[a.manifest.name],a.manifest)
    sys.path[:0]=[str(a.libero_src),str(a.corp_src),str(a.corp_src/'scripts'),str(ROOT/'scripts')]
    os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from evaluate_libero import policy_observation
    from libero.libero.benchmark import grab_language_from_filename
    from dacsc.flow_repair_kv import build_representation,integrate,optimize,random_null
    from unittest.mock import patch
    ch=file_hash(a.checkpoint)
    if ch!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint drift')
    policy=load_policy(a.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False)
    refs={cid:json.loads((a.results/(cid+'.json')).read_text()) for cid in cases}
    clean_cases={c['pair_group_id']:c for c in m['cases'] if c['factor']=='clean'}
    chosen=sel['case_ids'][a.shard_id::a.num_shards]
    if a.pilot:chosen=chosen[:1]
    a.out.mkdir(parents=True,exist_ok=True)
    for cid in chosen:
        c=cases[cid];ref=refs[cid];flow=json.loads((a.flow_results/(cid+'.json')).read_text())
        if file_hash(a.flow_results/(cid+'.json'))!=sel['flow_input_sha256'][cid]:raise ValueError('flow evidence changed')
        if ref['manifest_sha256']!=mh or ref['checkpoint_sha256']!=ch or ref['status']!='completed':raise ValueError('baseline drift')
        for field,key in [('resolved_bddl_hash','bddl'),('init_state_file_sha256','init_state_file')]:
            if file_hash(a.libero_src/c[key])!=c[field]:raise ValueError('asset drift')
        out=a.out/(cid+'.json')
        if out.exists():raise FileExistsError(out)
        began=time.monotonic();append_event(a.out/'access_ledger.jsonl',{'action':'repair_start','case_id':cid,'selection_sha256':file_hash(a.selection)})
        try:
            obs=observe(c,ref,a);language=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
            if language!=ref['language_instruction']:raise ValueError('language mismatch')
            po=policy_observation(obs,language,policy);pieces=encode_pieces(policy,po)
            zlist=[]
            for j in range(4):
                seed=ref['seeds']['policy_flow_seed'] if j==0 else seed_for(ref['seeds']['method_augmentation_seed'],'flow_geometry_v1',j)
                g=torch.Generator(device='cuda:0').manual_seed(seed);z=torch.randn((1,12,7),device='cuda:0',generator=g)
                if array_hash(z.cpu().numpy())!=flow['z0_hashes'][j]:raise ValueError('training z0 differs from V1')
                zlist.append(z)
            train=torch.cat(zlist)
            held=torch.cat([torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed_for(ref['seeds']['method_augmentation_seed'],'flow_causal_repair_v1','heldout',j))) for j in range(4)])
            zero=torch.zeros((1,256),device='cuda:0');scalar=torch.zeros((),device='cuda:0')
            with torch.no_grad():
                rep=build_representation(policy,pieces,zero,scalar,4,original=True);orig,_=integrate(policy,rep,train)
                manual,_=integrate(policy,build_representation(policy,pieces,zero,scalar,4),train)
                parity=float((policy.normalizer.unnormalize_action(manual)-policy.normalizer.unnormalize_action(orig)).abs().max())
                if parity>1e-5:raise ValueError('zero gate not identity')
                v1=torch.tensor(flow['controller_actions'],device='cuda:0')
                v1_error=float((policy.normalizer.unnormalize_action(orig)-v1).abs().max())
                if v1_error>1e-4:raise ValueError('batched action differs from original V1')
                single,_=integrate(policy,build_representation(policy,pieces,zero,scalar,1,original=True),train[:1])
                with patch('torch.randn',side_effect=lambda *args,**kwargs:train[:1].clone()):official=policy.predict(po)
                native=float((policy.normalizer.unnormalize_action(single)-official).abs().max())
                if native>1e-5:raise ValueError('native sampler mismatch')
            base=evaluate(policy,pieces,held,zero,scalar);outputs=[]
            grid=[(k,o,lr) for k in sel['grid']['knobs'] for o in sel['grid']['objectives'] for lr in sel['grid']['learning_rates']]
            if a.pilot:grid=[('joint','relative',.3)]
            for knob,obj,lr in grid:
                snaps,history=optimize(policy,pieces,train,knob,obj,lr,steps=10)
                ck={}
                for step in (5,10):
                    delta,s=snaps[step];primary=evaluate(policy,pieces,held,delta,s);null=[]
                    for j in range(5):
                        g=torch.Generator(device='cuda:0').manual_seed(seed_for(ref['seeds']['method_augmentation_seed'],'flow_null',knob,obj,lr,step,j))
                        nd,ns=random_null(delta,s,g);null.append(evaluate(policy,pieces,held,nd,ns))
                    ck[str(step)]={'adapted':primary,'random_nulls':null}
                outputs.append({'knob':knob,'objective':obj,'learning_rate':lr,'history':history,'checkpoints':ck})
                print(json.dumps({'case_id':cid,'knob':knob,'objective':obj,'lr':lr,'elapsed':time.monotonic()-began}),flush=True)
            # Only now read and evaluate paired Clean. It never enters optimize().
            cc=clean_cases[c['pair_group_id']];cref=refs[cc['case_id']]
            cobs=observe(cc,cref,a);cpo=policy_observation(cobs,language,policy);cpieces=encode_pieces(policy,cpo)
            clean_eval=evaluate(policy,cpieces,held,zero,scalar)
            # Reference uses exact native resampler; record the tiny manual-path parity.
            with torch.no_grad():
                cx,_=integrate(policy,build_representation(policy,cpieces,zero,scalar,4,original=True),held)
                native_clean=policy.normalizer.unnormalize_action(cx).cpu().tolist()
            row={'case_id':cid,'status':'completed','factor':c['factor'],'base_task':c['base_task'],'baseline_success':ref['success'],
                 'step0':base,'configurations':outputs,'offline_clean_reference':{'case_id':cc['case_id'],'actions':native_clean,'manual_zero_gate_actions':clean_eval['controller_actions']},
                 'initial_rgb_receipts':ref['initial_observation'],'zero_gate_max_abs':parity,'native_sampler_max_abs':native,'batched_v1_controller_max_abs':v1_error,
                 'training_z0_hashes':[array_hash(z.cpu().numpy()) for z in zlist],
                 'heldout_z0_hashes':[array_hash(z.cpu().numpy()) for z in held.split(1)],'seeds':ref['seeds']}
        except Exception as exc:
            import traceback
            row={'case_id':cid,'status':'exception','error':repr(exc),'traceback':traceback.format_exc()}
        row.update(wall_seconds=time.monotonic()-began,manifest_sha256=mh,selection_sha256=file_hash(a.selection),checkpoint_sha256=ch,
                   script_sha256=file_hash(Path(__file__)),optimization_module_sha256=file_hash(ROOT/'dacsc/flow_repair_kv.py'),
                   intervention_location='post_layernorm_agentview_prefix_values',policy_weights_updated=False,adapted_actions_executed=False,blind_accessed=False)
        out.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');append_event(a.out/'access_ledger.jsonl',{'action':'repair_end','case_id':cid,'status':row['status'],'result_sha256':file_hash(out)})
        print(json.dumps({'case_id':cid,'done':row['status'],'seconds':row['wall_seconds'],'error':row.get('error')}),flush=True)
        if row['status']!='completed':raise RuntimeError(row['error'])

if __name__=='__main__':main()
