"""Two legacy Source states x two renderer flags; forensic diagnostic, not method selection."""
import sys,json,argparse
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from collect_interaction_replay import official_plan,file_hash
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation,state_from_observation
from dacsc.interaction_records import step_flags

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--libero-src',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(a.libero_src))
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    policy=load_policy(ROOT/'repos/CoRP/checkpoints/corp_libero.pt',device='cuda:0');policy.eval();policy.requires_grad_(False)
    rows=[];a.out.mkdir(parents=True,exist_ok=False)
    for case in json.loads(a.cases.read_text()):
        folder=a.cases.parent/case['folder'];m=json.load(open(folder/'metadata.json'));z=np.load(folder/'episode.npz');base=case['base_task'];src=a.libero_src/'libero/libero'
        for collision in (False,True):
            env=OffScreenRenderEnv(bddl_file_name=str(src/'bddl_files/libero_spatial'/(base+'.bddl')),camera_heights=256,camera_widths=256,render_gpu_device_id=0,render_collision_mesh=collision,render_visual_mesh=True)
            try:
                np.random.seed(m['seed']);torch.manual_seed(m['seed']);env.seed(m['seed']);env.reset();obs=env.set_init_state(torch.load(src/'init_files/libero_spatial'/(base+'.pruned_init'),weights_only=False)[m['state_id']])
                for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
                seed=torch.Generator(device='cuda:0');seed.manual_seed(m['seed']);n=0;ci=0;success=False;first={};executed=[]
                while n<220:
                    po=policy_observation(obs,m['language'],policy);z0=torch.randn((1,12,7),device='cuda:0',generator=seed)
                    plan,parity=official_plan(policy,po,z0,audit=(ci==0))
                    if ci==0:
                        first['state8_max_abs']=float(np.abs(po['state'].numpy()[0]-z['state8'][0]).max())
                        first['action_max_abs']=float(np.abs(plan-z['actions_planned'][0]).max())
                        first['official_parity']=parity
                        for v in ('agentview','eye_in_hand'):
                            cur=po['images'][v][0].numpy();old=z['rgb_'+v][0];first[v+'_mae']=float(np.abs(cur.astype(float)-old).mean())
                            Image.fromarray(np.concatenate([old,cur],axis=1)).save(a.out/f'state{m["state_id"]}_collision{int(collision)}_{v}.png')
                    for action in np.clip(plan,-1,1):
                        obs,flags=step_flags(env.step(action),env);n+=1;executed.append(action.copy());success=flags['success']
                        if success or flags['done'] or flags['terminated'] or flags['truncated'] or n>=220:break
                    ci+=1
                    if success or flags['done'] or flags['terminated'] or flags['truncated']:break
                row={'state_id':m['state_id'],'seed':m['seed'],'collision_mesh':collision,'visual_mesh':True,'success':success,'steps':n,'initial':first,
                     'old_success':m['success'],'old_steps':m['steps'],'trajectory_is_new_inference_under_old_source_state':True,'action_sha256':__import__('hashlib').sha256(np.array(executed).tobytes()).hexdigest()}
                rows.append(row);print(json.dumps(row),flush=True)
            finally:env.close()
    (a.out/'report.json').write_text(json.dumps({'cases':rows,'selection':'two fixed old source states, paired flag diagnostic only; not allowed to tune method or relabel official Clean','parameters_updated':False},indent=2)+'\n')
if __name__=='__main__':main()
