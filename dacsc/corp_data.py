"""CoRP-RoboCasa 数据适配（P2）。

把 LeRobot atomic 数据集读成 CoRP 需要的 batch：
    images: 3 views, [3,224,224]（官方 CoRP 协议：256 -> bilinear resize 224，不 crop）
    state:  9D = eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)
    action: [horizon, 12]
    lang:   原始文本（extras/episode_*/ep_meta.json 的 `lang`）

严格 split：按原生 style_id 过滤（train 11-30 / val 31-40 / test 41-50）。
不使用 coarse3 / layout10 等合成标签。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DP_REPO = _REPO_ROOT / "repos" / "diffusion_policy"
if str(_DP_REPO) not in sys.path:
    sys.path.insert(0, str(_DP_REPO))

from diffusion_policy.dataset.lerobot_dataset import LerobotDataset  # noqa: E402

IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# shape_meta 与官方 RoboCasa DP 基线一致（只取 9D 策略 state）。
SHAPE_META: dict = {
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

# CoRP view 顺序 -> dataset rgb key（必须与 ROBOCASA_CORP_CONFIG["view_names"] 一致）。
VIEW_TO_KEY: dict[str, str] = {
    "robot0_agentview_left": "robot0_agentview_left_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
    "robot0_agentview_right": "robot0_agentview_right_image",
}


def read_episode_meta(dataset_path: str | Path) -> dict[int, dict]:
    """episode_index -> {style_id, layout_id, lang}。"""
    extras = Path(dataset_path) / "extras"
    out: dict[int, dict] = {}
    for ep_dir in sorted(extras.glob("episode_*")):
        try:
            idx = int(ep_dir.name.split("_")[-1])
            meta = json.loads((ep_dir / "ep_meta.json").read_text())
        except Exception:
            continue
        out[idx] = {
            "style_id": meta.get("style_id"),
            "layout_id": meta.get("layout_id"),
            "lang": meta.get("lang"),
        }
    return out


class CorpRoboCasaDataset(Dataset):
    """多任务 atomic 数据集 + 原生 style 过滤。"""

    def __init__(
        self,
        dataset_paths: list[str | Path],
        *,
        style_min: int,
        style_max: int,
        horizon: int = 10,
        image_size: int = IMAGE_SIZE,
        use_cache: bool = True,
        episode_residues: set[int] | None = None,
        episode_mod: int = 10,
    ) -> None:
        self.horizon = horizon
        self.image_size = image_size
        self.datasets: list[LerobotDataset] = []
        self.flat: list[tuple[int, int, int, str, str]] = []  # (ds_idx, frame_idx, style, lang, task)
        self.style_ids = set(range(style_min, style_max + 1))

        for ds_idx, path in enumerate(dataset_paths):
            task_name = Path(path).parent.parent.name
            ds = LerobotDataset(
                shape_meta=SHAPE_META,
                dataset_path=str(path),
                horizon=horizon,
                n_obs_steps=1,
                pad_before=0,
                pad_after=horizon - 1,
                use_cache=use_cache,
                seed=42,
            )
            ep_meta = read_episode_meta(path)
            n_kept = 0
            for frame_idx, (traj_id, _step) in enumerate(ds.all_steps):
                m = ep_meta.get(int(traj_id))
                if not m or m["style_id"] not in self.style_ids:
                    continue
                if episode_residues is not None and (int(traj_id) % episode_mod) not in episode_residues:
                    continue
                self.flat.append((ds_idx, frame_idx, int(m["style_id"]), m["lang"] or "", task_name))
                n_kept += 1
            self.datasets.append(ds)
            print(f"[CorpRoboCasaDataset] {Path(path).parent.parent.name}: "
                  f"kept {n_kept}/{len(ds.all_steps)} frames", flush=True)
        if not self.flat:
            raise RuntimeError("filtered dataset is empty; check style range")

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, i: int) -> dict:
        ds_idx, frame_idx, style_id, lang, task_name = self.flat[i]
        sample = self.datasets[ds_idx][frame_idx]
        obs = sample["obs"]
        images = {}
        for view, key in VIEW_TO_KEY.items():
            img = obs[key][0]  # [3,256,256], float [0,1]
            img = F.interpolate(
                img.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            ).squeeze(0)
            images[view] = (img - IMAGENET_MEAN) / IMAGENET_STD
        state = torch.cat([
            obs["robot0_base_to_eef_pos"][0],
            obs["robot0_base_to_eef_quat"][0],
            obs["robot0_gripper_qpos"][0],
        ])
        return {
            "images": images,
            "state": state,
            "action": sample["action"],
            "instruction": lang,
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "task": task_name,
        }


class LinearNormalizer:
    """train-only state z-score + action identity。

    action 用 identity 与官方 RoboCasa DP 基线一致：其 normalizer 把 action
    设为 scale=1/offset=0。实测部分 action 维（base_motion / control_mode）
    在若干任务是常量，min-max 会退化，因此不采用 min-max。
    """

    def __init__(self, state_mean, state_std):
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32).clamp_min(1e-6)

    @classmethod
    def fit(cls, dataset: CorpRoboCasaDataset, max_steps: int = 2048) -> "LinearNormalizer":
        stride = max(1, len(dataset) // max_steps)
        states = []
        for i in range(0, len(dataset), stride):
            states.append(dataset[i]["state"])
            if len(states) >= max_steps:
                break
        state = torch.stack(states)
        return cls(state.mean(0), state.std(0).clamp_min(1e-3))

    def to(self, device):
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return self

    def normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.to(device=s.device, dtype=s.dtype)
        std = self.state_std.to(device=s.device, dtype=s.dtype)
        return (s - mean) / std

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def unnormalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return a


def collate_corp(batch: list[dict]) -> dict:
    views = list(batch[0]["images"].keys())
    return {
        "images": {v: torch.stack([b["images"][v] for b in batch]) for v in views},
        "state": torch.stack([b["state"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "style_id": torch.stack([b["style_id"] for b in batch]),
        "task": [b["task"] for b in batch],
    }
