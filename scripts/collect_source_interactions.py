"""Collect independent frozen-policy Source Clean episodes with disjoint roles.

No target environment, paired replay, probe, observer fitting or risk labels.
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts'),str(ROOT/'scripts')]
from corp import load_policy
from evaluate_libero import policy_observation,state_from_observation,DUMMY_ACTION,SETTLING_STEPS
from target_only_action_validation import configure
from collect_interaction_replay import encode,official_plan,cpu,file_hash
from dacsc.interaction_records import step_flags,array_hash,VIEWS
from dacsc.source_interactions import role_ids,validate_source_episode


@torch.inference_mode()
def boundary(policy,obs,language):
    po=policy_observation(obs,language,policy);tokens,_=encode(policy,po)
    return po,{'state8':cpu(po['state'][0]),'tokens_agentview':cpu(tokens[0,:48]),'tokens_eye_in_hand':cpu(tokens[0,48:]),
               'rgb_agentview':cpu(po['images']['agentview'][0]),'rgb_eye_in_hand':cpu(po['images']['eye_in_hand'][0])}


def collect(policy,args,state,sid,bddl,language,meta,split):
    from libero.libero.envs import OffScreenRenderEnv
    folder=args.out/f'{args.role}_state{sid}_seed{args.seed}';folder.mkdir()
    env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    began=time.monotonic()
    try:
        np.random.seed(args.seed);torch.manual_seed(args.seed);env.seed(args.seed);env.reset();obs=env.set_init_state(state)
        for _ in range(SETTLING_STEPS):
            obs,flags=step_flags(env.step(DUMMY_ACTION),env)
            if any(flags[k] for k in ('success','done','terminated','truncated')):raise ValueError('terminal settling episode; retain failure, do not silently drop')
        po,b=boundary(policy,obs,language);boundaries=[b];times=[float(env.sim.data.time)];steps=[0];records=[];zs=[]
        generator=torch.Generator(device='cuda:0');generator.manual_seed(args.seed)
        total=0;parity=[]
        while total<args.max_steps:
            z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=generator);zs.append(cpu(z0[0]))
            actions,check=official_plan(policy,po,z0,audit=(total==0))
            if check:parity.append(check)
            clipped=np.clip(actions,-1,1);r={k:np.zeros(12,bool) for k in ('success','terminated','truncated','done','collector_truncated','env_truncated')}
            r['reward']=np.zeros(12,np.float32);r['state8_executed']=np.zeros((12,8),np.float32);r['sim_time_executed']=np.zeros(12,np.float64)
            stopped=False;n=0
            for k in range(min(12,args.max_steps-total)):
                obs,flags=step_flags(env.step(clipped[k].copy()),env);n+=1;total+=1
                for key in ('success','terminated','truncated','done','reward'):r[key][k]=flags[key]
                r['env_truncated'][k]=flags['truncated']
                r['state8_executed'][k]=state_from_observation(obs,policy.normalizer)
                r['sim_time_executed'][k]=float(env.sim.data.time)
                if any(flags[key] for key in ('success','terminated','truncated','done')):stopped=True;break
                if total>=args.max_steps:r['truncated'][k]=True;r['collector_truncated'][k]=True;stopped=True;break
            mask=np.arange(12)<n;executed=np.zeros_like(clipped);executed[mask]=clipped[mask]
            r.update(actions_planned=actions,actions_executed=executed,executed_mask=mask,executed_steps=n,full_chunk=n==12,fixed_horizon_eligible=n==12)
            records.append(r);po,b=boundary(policy,obs,language);boundaries.append(b);times.append(float(env.sim.data.time));steps.append(total)
            print(json.dumps({'role':args.role,'state':sid,'chunk':len(records)-1,'step':total,'success':bool(r['success'][n-1])}),flush=True)
            if stopped:break
        a={key:np.stack([b[key] for b in boundaries]) for key in boundaries[0]}
        a.update({key:np.asarray([r[key] for r in records]) for key in records[0]})
        a.update(boundary_step=np.asarray(steps),boundary_sim_time=np.asarray(times),z0=np.stack(zs))
        m=dict(meta,state_id=sid,initial_state_sha256=array_hash(np.asarray(state)),control_dt=1/float(env.env.control_freq),
               language=language,success=bool(records[-1]['success'][records[-1]['executed_steps']-1]),steps=total,
               official_sampler_parity=parity,wall_seconds=time.monotonic()-began)
        check=validate_source_episode(a,m,split)
        np.savez_compressed(folder/'episode.npz',**a);m['npz_sha256']=file_hash(folder/'episode.npz');m['validation']=check
        (folder/'metadata.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
        return {'directory':folder.name,'state_id':sid,'seed':args.seed,'success':m['success'],'steps':total,
                'npz_sha256':m['npz_sha256'],'validation':check,'wall_seconds':m['wall_seconds']}
    finally:env.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--matrix',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--role',choices=['source_fit','source_validation'],required=True);ap.add_argument('--states',default='')
    ap.add_argument('--seed',type=int,default=2014);ap.add_argument('--max-steps',type=int,default=220);ap.add_argument('--suite',default='libero_spatial')
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus'));ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'));ap.add_argument('--render-gpu',type=int,default=0)
    args=ap.parse_args()
    if os.environ.get('MUJOCO_GL')!='osmesa' or os.environ.get('PYOPENGL_PLATFORM')!='osmesa':raise ValueError('OSMesa required')
    if args.out.exists():raise FileExistsError(args.out)
    if args.max_steps!=220:raise ValueError('frozen source horizon is 220; no short-trajectory selection')
    split=json.loads(args.split.read_text());ids=role_ids(split,args.role,[int(i) for i in args.states.split(',')] if args.states else None)
    if args.seed not in split['evaluation_seeds']:raise ValueError('seed outside frozen source evaluation pool')
    configure(args)
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text());task=next(t for t in matrix['base_tasks'] if t['suite']==args.suite)
    # Read the base BDDL reference only. Never construct a perturbation environment.
    bddls={v['base_bddl'] for v in task['variants']['lighting']}
    if len(bddls)!=1:raise ValueError('ambiguous source task')
    bddl=next(iter(bddls));state_path=task['state_split']['source_file']
    if file_hash(bddl)!=split['benchmark']['bddl_sha256']:raise ValueError('source BDDL mismatch')
    if file_hash(state_path)!=split['benchmark']['initial_state_file_sha256']:raise ValueError('initial state source mismatch')
    checkpoint_hash=file_hash(args.checkpoint)
    if checkpoint_hash!=split['policy']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    states=torch.load(state_path,map_location='cpu',weights_only=False);language=BDDLUtils.get_problem_info(bddl)['language_instruction']
    policy=load_policy(args.checkpoint,device=torch.device('cuda:0'));policy.eval();policy.requires_grad_(False)
    if (policy.action_generator.action_horizon,policy.action_generator.action_dim,policy.action_generator.num_sample_steps)!=(12,7,10):raise ValueError('policy dimensions changed')
    if tuple(policy.representation_extractor.view_names)!=VIEWS:raise ValueError('view order changed')
    args.out.mkdir(parents=True)
    meta={'schema':'source_clean_interactions_v1','role':args.role,'domain':'clean','data_origin':'independent_source_policy_rollout',
          'seed':args.seed,'H':12,'E':12,'max_steps':args.max_steps,'split_sha256':file_hash(args.split),'checkpoint_sha256':checkpoint_hash,
          'base_bddl_sha256':file_hash(bddl),'initial_state_file_sha256':file_hash(state_path),'matrix_sha256':file_hash(args.matrix),
          'script_sha256':file_hash(__file__),'source_contract_sha256':file_hash(ROOT/'dacsc/source_interactions.py'),
          'transition_contract_sha256':file_hash(ROOT/'dacsc/interaction_records.py'),'shared_inference_sha256':file_hash(ROOT/'scripts/collect_interaction_replay.py'),
          'renderer':'osmesa','precision':'fp32','torch_version':torch.__version__,'numpy_version':np.__version__,
          'target_environment_accessed':False,'weights_updated':False,'chunk_risk_labels_assigned':False,
          'selection':'all predeclared source states; keep successes and non-successes without post-hoc filtering',
          'training_data_provenance':split['policy']['training_data_provenance'],
          'state_ids':ids,'observation_semantics':'native step-returned observations; no refresh/restore',
          'upstream_sha256':{p:file_hash(ROOT/'repos/CoRP'/p) for p in ('corp/policy.py','corp/action_generator.py','scripts/evaluate_libero.py')}}
    reports=[]
    for sid in ids:reports.append(collect(policy,args,states[sid],sid,bddl,language,meta,split))
    (args.out/'summary.json').write_text(json.dumps({'metadata':meta,'reports':reports,'pass':True},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pass':True,'role':args.role,'states':ids,'out':str(args.out)}),flush=True)
if __name__=='__main__':main()
