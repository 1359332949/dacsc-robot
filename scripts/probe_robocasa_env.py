"""探 RoboCasa gym env（robosuite master）：obs 键、shape、style_ids、action dict。"""
from __future__ import annotations
import sys
sys.path.insert(0, "/root/dacsc-robot/repos/robocasa")
sys.path.insert(0, "/root/dacsc-robot/repos/diffusion_policy")

import numpy as np
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action

env = RoboCasaGymEnv(env_name="CloseFridge", split=None, obj_instance_split="pretrain",
                     style_ids=[11, 12], camera_widths=256, camera_heights=256)
obs, info = env.reset(seed=0)
print("obs type:", type(obs))
for k, v in obs.items():
    print(f"  {k}: {getattr(v,'shape',None)} {getattr(v,'dtype',None)}")
a = np.zeros(12, dtype=np.float32)
d = convert_action(a)
print("convert_action keys:", list(d.keys()))
obs2, reward, done, trunc, info2 = env.step(d)
print("step ok reward", reward, "done", done, "success", info2.get("success"))
print("PROBE ENV OK")
env.close()
