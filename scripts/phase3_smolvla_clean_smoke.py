"""10-game libero_spatial Clean smoke for frozen SmolVLA (E=8)."""
import json,sys,os,time
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.envs.utils import preprocess_observation
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from evaluate_libero import DUMMY_ACTION,is_success
CKPT='/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero'; MAX=300; E=8
def main():
    man=json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json')); cleans=[x for x in man['cases'] if x['factor']=='clean']; assert len(cleans)==10
    policy=SmolVLAPolicy.from_pretrained(CKPT).eval().cuda(); policy.config.n_action_steps=E
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=CKPT,preprocessor_overrides={'device_processor':{'device':'cuda'}}); envp=PolicyProcessorPipeline(steps=[LiberoProcessorStep()]); rows=[]; t_all=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for idx,c in enumerate(cleans):
        bddl=LIB/c['bddl']; init=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[0]; lang=grab_language_from_filename('libero_spatial',c['base_task']+'.bddl'); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); policy.reset(); env.noise=0; env.seed(0); env.reset(); obs=env.set_init_state(init)
        for _ in range(10): obs=env.step(DUMMY_ACTION)[0]
        steps=0; success=False; finite=True; t0=time.perf_counter()
        try:
            while steps<MAX:
                raw={'pixels':{'image':np.asarray(obs['agentview_image']),'image2':np.asarray(obs['robot0_eye_in_hand_image'])},'robot_state':{'eef':{'pos':np.asarray(obs['robot0_eef_pos'],dtype=np.float32)[None],'quat':np.asarray(obs['robot0_eef_quat'],dtype=np.float32)[None]},'gripper':{'qpos':np.asarray(obs['robot0_gripper_qpos'],dtype=np.float32)[None]}}}
                x=preprocess_observation(raw); x['task']=[lang]; x=envp(x); x=pre(x)
                with torch.inference_mode(): a=policy.select_action(x)
                a=post(a); an=a.cpu().numpy().reshape(-1); finite=finite and bool(np.isfinite(an).all())
                obs,rew,done,info=env.step(an); steps+=1; success=bool(is_success(env,info,rew,done))
                if done or success or steps>=MAX: break
        finally: env.close()
        row={'task':c['base_task'],'index':idx,'success':success,'steps':steps,'finite':finite,'seconds':time.perf_counter()-t0}; rows.append(row); print(json.dumps(row),flush=True)
    peak=torch.cuda.max_memory_allocated()/1024**2; summary={'rows':rows,'success':sum(r['success'] for r in rows),'n':len(rows),'n_action_steps':E,'max_steps':MAX,'peak_mem_mb':peak,'total_seconds':time.perf_counter()-t_all,'mean_steps':sum(r['steps'] for r in rows)/len(rows),'mean_seconds':sum(r['seconds'] for r in rows)/len(rows)}
    out=ROOT/'docs/data/phase3';out.mkdir(parents=True,exist_ok=True);(out/'smolvla_clean_smoke.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
