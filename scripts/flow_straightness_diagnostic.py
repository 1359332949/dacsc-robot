"""V2 Dev t=0 native-flow geometry and optional frozen-weight gradient diagnostic.

Never executes candidate actions, optimizes policy weights, or reads protected roles.
Uses exact official RGB, canonical language, original first policy z0 and 3 isolated
additional diagnostic z0. Labels are attached AFTER inference, never to select scores.
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dacsc.static_visual_v2 import (authorize,file_hash,array_hash,seed_for,EnvironmentRNG,
                                    sensor_transform,append_event)
from dacsc.flow_geometry import geometry


def trace(policy,rep,z):
    import torch
    gen=policy.action_generator;prefix=gen.prepare_prefix(rep);x=z.to(rep.action_dtype).clone();vv=[]
    for k in range(gen.num_sample_steps):
        t=torch.full((x.shape[0],),k/gen.num_sample_steps,device=x.device,dtype=x.dtype)
        v=gen(x,t,prefix);vv.append(v);x=x+v/gen.num_sample_steps
    return x,torch.stack(vv)


def metrics(v):
    a=v.detach().float().cpu().numpy()[:,0]
    return {name:geometry(a[:,:,sl]) for name,sl in [('all',slice(None)),('xyz',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]}


def encode(policy,po):
    import torch
    from corp.preprocessing import prepare_observation
    from corp.representation.extractor import Representation
    with torch.no_grad():
        b=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
        ex=policy.representation_extractor;pp=ex.visual_encoder(torch.cat([b['images'][n] for n in ex.view_names],dim=0))
        patches={n:pp[i:i+1].detach() for i,n in enumerate(ex.view_names)}
        lang,pool=ex.language_encoder(b['instruction']);state=ex.state_encoder(b['state'])
    def rep(p):
        vis=ex.cross_attention_resampler(p,pool)
        return Representation(tokens=torch.cat([vis,lang.to(vis.dtype),state],dim=1),pooled_language=pool,action_dtype=vis.dtype)
    return patches,rep


def gradient_check(policy,po,z,held_z,base_action):
    import torch
    patches,build=encode(policy,po)
    # Exact original resampler: gate=0 is identity, no manual attention substitution.
    p=patches['agentview'].detach().clone().requires_grad_(True)
    delta=torch.zeros((*p.shape[:2],1),device=p.device,requires_grad=True)
    pp=dict(patches,agentview=p*torch.exp(delta));rep=build(pp)
    x,v=trace(policy,rep,z)
    loss=((v-v.mean(0,keepdim=True))**2).mean()
    gp,gd=torch.autograd.grad(loss,(p,delta))
    if not torch.isfinite(gp).all() or not torch.isfinite(gd).all():raise ValueError('nonfinite gradient')
    identity=float((policy.normalizer.unnormalize_action(x).detach()-base_action).abs().max())
    if identity>1e-5:raise ValueError('identity gate mismatch')
    # One preregistered RMS-normalized descent; not a learned deployment gate.
    step=(-.01*gd/(gd.square().mean().sqrt()+1e-12)).clamp(-.05,.05).detach()
    with torch.no_grad():
        updated=build(dict(patches,agentview=patches['agentview']*step.exp()))
        ax,av=trace(policy,updated,z);before_h,bhv=trace(policy,build(patches),held_z);after_h,ahv=trace(policy,updated,held_z)
        ctrl=policy.normalizer.unnormalize_action(ax)
    # Independent loss recomputation at symmetric perturbations tests gradient correctness.
    direction=gd.detach()/(gd.detach().square().mean().sqrt()+1e-12);h=1e-3
    vals=[]
    with torch.no_grad():
        for sign in (-1,1):
            r=build(dict(patches,agentview=patches['agentview']*torch.exp(sign*h*direction)))
            _,sv=trace(policy,r,z);vals.append(float(((sv-sv.mean(0,keepdim=True))**2).mean()))
    finite_diff=(vals[1]-vals[0])/(2*h);analytic=float((gd*direction).sum())
    return {'identity_controller_max_abs':identity,'before':metrics(v),'after':metrics(av),
            'heldout_before':metrics(bhv),'heldout_after':metrics(ahv),
            'xyz_command_rms_before':float(base_action[...,:3].square().mean().sqrt()),
            'xyz_command_rms_after':float(ctrl[...,:3].square().mean().sqrt()),
            'controller_max_change':float((ctrl-base_action).abs().max()),
            'patch_gradient_norm':gp.norm(dim=-1).detach().cpu().tolist(),
            'gate_gradient':gd.detach().cpu().reshape(-1).tolist(),'gate_step':step.cpu().reshape(-1).tolist(),
            'directional_derivative':analytic,'finite_difference':finite_diff,
            'derivative_relative_error':abs(finite_diff-analytic)/max(abs(analytic),1e-10),
            'warning':'no spatial ground truth; saliency is sensitivity, not causal localization; no rollout performed'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--libero-src',type=Path,required=True);ap.add_argument('--corp-src',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--num-shards',type=int,default=1);ap.add_argument('--shard-id',type=int,default=0)
    ap.add_argument('--gradient',action='store_true');ap.add_argument('--limit',type=int);args=ap.parse_args()
    man=json.loads(args.manifest.read_text());mh=file_hash(args.manifest)
    if man.get('kind')!='dev' or man.get('status')!='FROZEN':raise PermissionError('frozen Dev only')
    expected=json.loads((args.manifest.parent/'manifest_hashes.json').read_text())[args.manifest.name]
    for c in man['cases']:
        authorize(man,c,'flow geometry diagnostic',expected,args.manifest)
        if c['role'] not in ('dev_clean','target_dev') or c['factor'] not in ('clean','lighting','background','sensor_noise'):raise PermissionError('unsupported role/factor')
    sys.path[:0]=[str(args.libero_src),str(args.corp_src),str(args.corp_src/'scripts'),str(ROOT/'scripts')]
    os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    from unittest.mock import patch
    checkpoint_hash=file_hash(args.checkpoint)
    if checkpoint_hash!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint changed')
    policy=load_policy(args.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False)
    if policy.action_generator.num_sample_steps!=10:raise ValueError('native sampler must use 10 steps')
    # A fixed illustrative selection by factor/outcome, not a predictive evaluation subset.
    refs={c['case_id']:json.loads((args.results/(c['case_id']+'.json')).read_text()) for c in man['cases']}
    selected=[]
    if args.gradient:
        for factor in ('clean','lighting','background','sensor_noise'):
            for success in ((True,) if factor=='clean' else (True,False)):
                cc=sorted([c for c in man['cases'] if c['factor']==factor and refs[c['case_id']]['success']==success],key=lambda c:c['case_id'])
                selected.extend(cc[:1])
    else:selected=man['cases']
    selected=selected[args.shard_id::args.num_shards]
    if args.limit is not None:selected=selected[:args.limit]
    args.out.mkdir(parents=True,exist_ok=True);ledger=args.out/'access_ledger.jsonl'
    for c in selected:
        out=args.out/(c['case_id']+'.json')
        if out.exists():raise FileExistsError(out)
        ref=refs[c['case_id']]
        if ref['manifest_sha256']!=mh or ref['checkpoint_sha256']!=checkpoint_hash or ref['status']!='completed':raise ValueError('baseline receipt drift')
        for field,key in [('resolved_bddl_hash','bddl'),('init_state_file_sha256','init_state_file')]:
            if file_hash(args.libero_src/c[key])!=c[field]:raise ValueError('asset drift')
        append_event(ledger,{'action':'flow_diagnostic_start','case_id':c['case_id'],'manifest_sha256':mh,'gradient':args.gradient})
        began=time.monotonic();env=None
        try:
            initial=torch.load(args.libero_src/c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
            if array_hash(np.asarray(initial))!=c['raw_init_hash']:raise ValueError('init hash changed')
            seeds=ref['seeds'];rng=EnvironmentRNG(seeds['environment_seed'])
            # Exactly one live environment, wrapper noise disabled before any reset.
            with rng.use():
                env=OffScreenRenderEnv(bddl_file_name=str(args.libero_src/'libero/libero/bddl_files'/c['suite']/(c['logical_task_name']+'.bddl')),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
                if int(env.noise)!=int(c['runtime_params']['noise']):raise ValueError('wrapper noise code mismatch')
                env.noise=0;env.seed(seeds['environment_seed']);env.reset();obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
            obs=sensor_transform(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],0,wrapper)
            hashes={}
            for view,key in [('agentview','agentview_image'),('eye_in_hand','robot0_eye_in_hand_image')]:
                hashes[view]=array_hash(np.asarray(obs[key]))
                if hashes[view]!=ref['initial_observation'][view]['sha256']:raise ValueError('baseline RGB mismatch '+view)
            language=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
            if language!=ref['language_instruction']:raise ValueError('language mismatch')
            po=policy_observation(obs,language,policy)
            gen=torch.Generator(device='cuda:0').manual_seed(seeds['policy_flow_seed']);z=torch.randn((1,12,7),device='cuda:0',generator=gen)
            if array_hash(z.cpu().numpy())!=ref['chunks'][0]['z0_hash']:raise ValueError('original z0 mismatch')
            with torch.no_grad():
                patches,build=encode(policy,po);rep=build(patches);x,v=trace(policy,rep,z);ctrl=policy.normalizer.unnormalize_action(x)
                with patch('torch.randn',side_effect=lambda *a,**kw:z.clone()):official=policy.predict(po)
                parity=float((ctrl-official).abs().max())
                if parity>1e-5:raise ValueError('native predict parity failed')
                all_metrics=[metrics(v)];vels=[v.cpu().numpy()];actions=[ctrl.cpu().numpy()];z_hash=[array_hash(z.cpu().numpy())]
                for j in range(1,4):
                    g=torch.Generator(device='cuda:0').manual_seed(seed_for(seeds['method_augmentation_seed'],'flow_geometry_v1',j))
                    zj=torch.randn((1,12,7),device='cuda:0',generator=g);ax,av=trace(policy,rep,zj)
                    actions.append(policy.normalizer.unnormalize_action(ax).cpu().numpy());vels.append(av.cpu().numpy());all_metrics.append(metrics(av));z_hash.append(array_hash(zj.cpu().numpy()))
                heldg=torch.Generator(device='cuda:0').manual_seed(seed_for(seeds['method_augmentation_seed'],'flow_geometry_heldout'))
                heldz=torch.randn((1,12,7),device='cuda:0',generator=heldg)
            gc=gradient_check(policy,po,z,heldz,ctrl) if args.gradient else None
            aa=np.stack(actions)[:,0];clip=np.clip(aa,-1,1)
            row={'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'sub_dimension':c.get('sub_dimension'),'difficulty':c.get('difficulty'),
                 'noise_code':int(c['runtime_params']['noise']),'status':'completed','metrics':all_metrics,
                 'endpoint_variance_xyz':float(clip[...,:3].var(0).mean()),'endpoint_variance_rotation':float(clip[...,3:6].var(0).mean()),
                 'velocity_trace':np.stack(vels).tolist(),'controller_actions':aa.tolist(),'z0_hashes':z_hash,
                 'initial_rgb_hashes':hashes,'official_controller_max_abs':parity,'gradient':gc,
                 'baseline_success':ref['success'],'baseline_result_sha256':file_hash(args.results/(c['case_id']+'.json'))}
        except Exception as exc:
            import traceback
            row={'case_id':c['case_id'],'status':'exception','error':repr(exc),'traceback':traceback.format_exc()}
        finally:
            if env is not None:env.close()
        row.update(manifest_sha256=mh,checkpoint_sha256=checkpoint_hash,script_sha256=file_hash(Path(__file__)),
                   geometry_sha256=file_hash(ROOT/'dacsc/flow_geometry.py'),wall_seconds=time.monotonic()-began,policy_weights_updated=False,blind_accessed=False)
        out.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');append_event(ledger,{'action':'flow_diagnostic_end','case_id':c['case_id'],'result_sha256':file_hash(out),'status':row['status']})
        print(json.dumps({'case_id':c['case_id'],'status':row['status'],'seconds':row['wall_seconds'],'error':row.get('error')}),flush=True)
        if row['status']!='completed':raise RuntimeError(row['error'])

if __name__=='__main__':main()
