"""P2 数据探针：验证 LeRobot 真实 batch 能被读成 CoRP-RoboCasa 需要的形式。

只跑单任务，确认：
  * 3 视角图像 [T=1,3,256,256]
  * 9D 策略 state = eef_pos_rel(3)+eef_rot_rel(4)+gripper(2)
  * action [10,12]
  * raw lang 文本可从 ep_meta.json 取到
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/root/dacsc-robot")
sys.path.insert(0, str(REPO / "repos" / "diffusion_policy"))

from diffusion_policy.dataset.lerobot_dataset import LerobotDataset  # noqa: E402

TASK = "/hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot"

SHAPE_META = {
    "obs": {
        "robot0_agentview_right_image": {
            "shape": [3, 256, 256], "type": "rgb",
            "lerobot_keys": ["video.robot0_agentview_right"]},
        "robot0_agentview_left_image": {
            "shape": [3, 256, 256], "type": "rgb",
            "lerobot_keys": ["video.robot0_agentview_left"]},
        "robot0_eye_in_hand_image": {
            "shape": [3, 256, 256], "type": "rgb",
            "lerobot_keys": ["video.robot0_eye_in_hand"]},
        "robot0_base_to_eef_pos": {
            "shape": [3], "lerobot_keys": ["state.end_effector_position_relative"]},
        "robot0_base_to_eef_quat": {
            "shape": [4], "lerobot_keys": ["state.end_effector_rotation_relative"]},
        "robot0_gripper_qpos": {
            "shape": [2], "lerobot_keys": ["state.gripper_qpos"]},
    },
    "action": {
        "shape": [12],
        "lerobot_keys": [
            "action.end_effector_position", "action.end_effector_rotation",
            "action.gripper_close", "action.base_motion", "action.control_mode",
        ],
    },
}


def main() -> int:
    ds = LerobotDataset(
        shape_meta=SHAPE_META,
        dataset_path=TASK,
        horizon=10,
        n_obs_steps=1,
        pad_before=0,
        pad_after=9,
        use_cache=True,
        seed=42,
    )
    print(f"len={len(ds)}")
    sample = ds[0]
    obs = sample["obs"]
    print("\nobs keys:")
    for k, v in obs.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print(f"action: {tuple(sample['action'].shape)} {sample['action'].dtype}")

    # 拼 9D state
    state9 = torch.cat([
        obs["robot0_base_to_eef_pos"][0],
        obs["robot0_base_to_eef_quat"][0],
        obs["robot0_gripper_qpos"][0],
    ])
    assert state9.shape == (9,), state9.shape
    print(f"\nstate9: {tuple(state9.shape)}")
    print("  values:", [round(float(x), 3) for x in state9])

    # raw lang
    extras = Path(TASK) / "extras"
    ep_dirs = sorted(extras.glob("episode_*"))
    meta = json.loads((ep_dirs[0] / "ep_meta.json").read_text())
    print(f"\nepisodes={len(ep_dirs)} lang={meta.get('lang')!r} "
          f"style_id={meta.get('style_id')} layout_id={meta.get('layout_id')}")

    # 视频尺寸 + 3 视角键
    print("\nvideo keys:", [k for k in obs if k.endswith("_image")])
    print("PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
