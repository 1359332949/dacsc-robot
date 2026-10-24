"""探 RoboCasa gym env：obs 键、shape、style_ids、action dict 接口。"""
from __future__ import annotations
import sys
sys.path.insert(0, "/root/dacsc-robot/repos/robocasa")
sys.path.insert(0, "/root/dacsc-robot/repos/diffusion_policy")

import numpy as np

# 运行时 shim：robosuite 1.5.2 不接受 load_model_on_init；这里吸收该参数，
# 不改 vendored robocasa 源码。
import robosuite.environments.manipulation.manipulation_env as _me
_orig_init = _me.ManipulationEnv.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("load_model_on_init", None)
    return _orig_init(self, *args, **kwargs)
_me.ManipulationEnv.__init__ = _patched_init
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action

env = RoboCasaGymEnv(env_name="CloseFridge", split=None, obj_instance_split="pretrain",
                     style_ids=[11, 12], camera_widths=256, camera_heights=256)
obs, info = env.reset(seed=0)
print("obs type:", type(obs))
for k, v in obs.items():
    shape = getattr(v, "shape", None)
    dtype = getattr(v, "dtype", None)
    print(f"  {k}: {shape} {dtype}")
print("action_space:", env.action_space)
a = np.zeros(12, dtype=np.float32)
d = convert_action(a)
print("convert_action keys:", list(d.keys()))
obs2, reward, done, trunc, info2 = env.step(d)
print("step ok reward", reward, "done", done, "info", {k: info2[k] for k in list(info2)[:5]})
print("PROBE ENV OK")
env.close()
