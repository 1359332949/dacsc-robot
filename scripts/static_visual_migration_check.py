"""Migration check on two recorded SOURCE episodes only, never a fresh test."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from collect_interaction_replay import official_plan,file_hash
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,state_from_observation,policy_observation
from dacsc.static_visual_v2 import EnvironmentRNG,digest

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--libero-src',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.libero_src))
    from libero.libero.envs import OffScreenRenderEnv
    cases=json.loads(a.cases.read_text());checkpoint=ROOT/'repos/CoRP/checkpoints/corp_libero.pt'
    policy=load_policy(checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False)
    report=[]
    for case in cases:
        folder=a.cases.parent/case['folder'];meta=json.loads((folder/'metadata.json').read_text())
        if file_hash(folder/'episode.npz')!=meta['npz_sha256']:raise ValueError('old input altered')
        with np.load(folder/'episode.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        if file_hash(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        # Same saved RGB/state8/z0 is an input-level cross-runtime model audit.
        errors=[];parities=[]
        for i in range(len(d['z0'])):
            po={'images':{v:torch.tensor(d['rgb_'+v][i]).unsqueeze(0) for v in ('agentview','eye_in_hand')},
                'state':torch.tensor(d['state8'][i]).unsqueeze(0),'instruction':[meta['language']]}
            predicted,parity=official_plan(policy,po,torch.tensor(d['z0'][i:i+1],device='cuda:0'),audit=(i==0))
            errors.append(float(np.max(np.abs(predicted-d['actions_planned'][i]))))
            if parity:parities.append(parity)
        base=Path(case['base_task']);init=a.libero_src/'libero/libero/init_files/libero_spatial'/(base.name+'.pruned_init')
        initial=torch.load(init,map_location='cpu',weights_only=False)[meta['state_id']]
        env=OffScreenRenderEnv(bddl_file_name=str(a.libero_src/'libero/libero/bddl_files/libero_spatial'/(base.name+'.bddl')),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
        state_errors=[];state8_end=[];render_metrics={}
        try:
            np.random.seed(meta['seed']);torch.manual_seed(meta['seed']);env.seed(meta['seed']);env.reset();obs=env.set_init_state(initial)
            for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
            state_errors.append(float(np.max(np.abs(state_from_observation(obs,policy.normalizer)-d['state8'][0]))))
            po=policy_observation(obs,meta['language'],policy)
            from PIL import Image
            a.out.parent.mkdir(parents=True,exist_ok=True)
            for view in ('agentview','eye_in_hand'):
                current=po['images'][view][0].numpy();old=d['rgb_'+view][0]
                render_metrics[view]={'mae':float(np.abs(current.astype(float)-old).mean()),'max_abs':float(np.abs(current.astype(float)-old).max())}
                Image.fromarray(np.concatenate([old,current],axis=1)).save(a.out.parent/('state'+str(meta['state_id'])+'_'+view+'.png'))
            actual,_=official_plan(policy,po,torch.tensor(d['z0'][0:1],device='cuda:0'))
            render_metrics['first_actual_observation_action_max_abs']=float(np.abs(actual-d['actions_planned'][0]).max())
            # Replay EXACT historical actions; not a new policy outcome comparison.
            n=0
            for ci,length in enumerate(d['executed_steps']):
                for k in range(int(length)):
                    obs,reward,done,info=env.step(d['actions_executed'][ci,k]);n+=1
                    state_errors.append(float(np.max(np.abs(state_from_observation(obs,policy.normalizer)-d['state8_executed'][ci,k]))))
                state8_end.append(state_from_observation(obs,policy.normalizer).tolist())
            success=bool(env.check_success())
        finally:env.close()
        report.append({'source_role':meta['role'],'state_id':meta['state_id'],'old_steps':meta['steps'],'old_success':meta['success'],
            'saved_input_controller_max_abs':max(errors),'same_runtime_official_parity':parities,
            'initial_render_metrics':render_metrics,'exact_historical_action_replay_state8_max_abs':max(state_errors),'replay_steps':n,'replay_success':success,
            'interpretation':'replay success is not new policy success; numerical drift must not be hidden',
            'old_torch':meta['torch_version'],'new_torch':torch.__version__,'new_robosuite':__import__('robosuite').__version__,'new_mujoco':__import__('mujoco').__version__,'old_input_sha256':meta['npz_sha256']})
        print(json.dumps(report[-1]),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'schema':'migration_input_and_action_replay_v1','cases':report,'new_rollout_seed':False,'scope':'old Source episodes, saved-input policy inference and controlled old-action replay only'},indent=2)+'\n')
if __name__=='__main__':main()
