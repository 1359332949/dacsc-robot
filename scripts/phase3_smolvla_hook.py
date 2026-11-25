"""Load SmolVLA, one real forward, validate layer[2] hook."""
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
CKPT='/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero'
def main():
    man=json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json')); c=next(x for x in man['cases'] if x['factor']=='clean')
    policy=SmolVLAPolicy.from_pretrained(CKPT).eval().cuda(); policy.config.n_action_steps=8
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=CKPT,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    envp=PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    bddl=LIB/c['bddl']; init=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[0]; lang=grab_language_from_filename('libero_spatial',c['base_task']+'.bddl')
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); env.noise=0; env.seed(0); env.reset(); obs=env.set_init_state(init)
    for _ in range(10): obs=env.step(np.zeros(7))[0]
    raw={'pixels':{'image':np.asarray(obs['agentview_image']),'image2':np.asarray(obs['robot0_eye_in_hand_image'])},'robot_state':{'eef':{'pos':np.asarray(obs['robot0_eef_pos'],dtype=np.float32)[None],'quat':np.asarray(obs['robot0_eef_quat'],dtype=np.float32)[None]},'gripper':{'qpos':np.asarray(obs['robot0_gripper_qpos'],dtype=np.float32)[None]}}}
    x=preprocess_observation(raw); x['task']=[lang]; x=envp(x); x=pre(x)
    layer=policy.model.vlm_with_expert.vlm.model.vision_model.encoder.layers[2]; calls=[]
    def hook(m,a,out): calls.append(tuple(out.shape)); return None
    h=layer.register_forward_hook(hook)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    with torch.inference_mode(): a=policy.select_action(x)
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)*1000; peak=torch.cuda.max_memory_allocated()/1024**2; h.remove(); env.close()
    print(json.dumps({'hook_calls':calls,'hook_n':len(calls),'each_shape':calls[0] if calls else None,'finite':bool(np.isfinite(a.cpu().numpy()).all()),'action_shape':list(a.shape),'forward_ms':ms,'peak_mem_mb':peak,'n_action_steps':policy.config.n_action_steps}))
if __name__=='__main__':main()
