import sys,os,json,torch,numpy as np
sys.path[:0]=['/hy-tmp/LIBERO-plus','/hy-tmp/patch-diag','/hy-tmp/patch-diag/repos/CoRP','/hy-tmp/patch-diag/repos/CoRP/scripts']
os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
from libero.libero.envs import OffScreenRenderEnv
from dacsc.static_visual_v2 import sensor_transform,array_hash
import libero.libero.envs.env_wrapper as wrapper
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS
case=json.load(open('/hy-tmp/patch-diag/docs/data/static_visual_v2_suite/final/results/05a2e087b49114ba4a11204b.json'))
man=json.load(open('/hy-tmp/patch-diag/docs/data/static_visual_v2_suite/final/dev_manifest.json'))
c=next(x for x in man['cases'] if x['case_id']==case['case_id'])
initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
seeds=case['seeds']; print('seeds',seeds)
env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
env.seed(seeds['environment_seed']);env.reset();obs=env.set_init_state(initial)
for _ in range(10): obs,_,_,_=env.step(DUMMY_ACTION)
noise=int(c['runtime_params']['noise']); obs=sensor_transform(obs,noise,seeds['sensor_noise_seed'],0,wrapper)
print('hash',array_hash(np.asarray(obs['agentview_image'])),case['initial_observation']['agentview']['sha256'])
print('equal',array_hash(np.asarray(obs['agentview_image']))==case['initial_observation']['agentview']['sha256'])
env.close()
