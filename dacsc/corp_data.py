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
import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from dacsc.corp_images import IMAGE_SIZE, prepare_image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DP_REPO = _REPO_ROOT / "repos" / "diffusion_policy"
if str(_DP_REPO) not in sys.path:
    sys.path.insert(0, str(_DP_REPO))

# shape_meta 与官方 RoboCasa DP 基线一致（只取 9D 策略 state）。
SHAPE_META: dict = {
    "obs": {
        "robot0_agentview_right_image": {
            "shape": [3, 256, 256],
            "type": "rgb",
            "lerobot_keys": ["video.robot0_agentview_right"],
        },
        "robot0_agentview_left_image": {
            "shape": [3, 256, 256],
            "type": "rgb",
            "lerobot_keys": ["video.robot0_agentview_left"],
        },
        "robot0_eye_in_hand_image": {
            "shape": [3, 256, 256],
            "type": "rgb",
            "lerobot_keys": ["video.robot0_eye_in_hand"],
        },
        "robot0_base_to_eef_pos": {
            "shape": [3],
            "lerobot_keys": ["state.end_effector_position_relative"],
        },
        "robot0_base_to_eef_quat": {
            "shape": [4],
            "lerobot_keys": ["state.end_effector_rotation_relative"],
        },
        "robot0_gripper_qpos": {"shape": [2], "lerobot_keys": ["state.gripper_qpos"]},
    },
    "action": {
        "shape": [12],
        "lerobot_keys": [
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
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
    """Validate extras against the upstream episodes.jsonl; never drop bad episodes."""
    dataset_path = Path(dataset_path)
    rows = [
        json.loads(line)
        for line in (dataset_path / "meta/episodes.jsonl").read_text().splitlines()
    ]
    extras = dataset_path / "extras"
    out: dict[int, dict] = {}
    for row in rows:
        idx = row["episode_index"]
        if (
            type(idx) is not int
            or idx < 0
            or idx in out
            or type(row["length"]) is not int
            or row["length"] <= 0
        ):
            raise ValueError(f"invalid/duplicate episode in {dataset_path}")
        meta = json.loads((extras / f"episode_{idx:06d}" / "ep_meta.json").read_text())
        if any(
            type(meta[key]) is not int or not 1 <= meta[key] <= 60
            for key in ("style_id", "layout_id")
        ):
            raise ValueError(f"invalid native style/layout for episode {idx}")
        if not isinstance(meta["lang"], str) or not meta["lang"].strip():
            raise ValueError(f"missing instruction for episode {idx}")
        out[idx] = {
            "style_id": meta["style_id"],
            "layout_id": meta["layout_id"],
            "lang": meta["lang"],
            "length": row["length"],
        }
    if not out or {p.name for p in extras.glob("episode_*")} != {
        f"episode_{i:06d}" for i in out
    }:
        raise ValueError(f"extras and episodes.jsonl differ: {dataset_path}")
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
        shift: str = "none",
    ) -> None:
        from dacsc.upstream import verify_upstream

        verify_upstream(["robocasa", "diffusion_policy"])
        from diffusion_policy.dataset.lerobot_dataset import LerobotDataset

        self.horizon = horizon
        self.image_size = image_size
        self.shift = str(shift)
        self.datasets: list[LerobotDataset] = []
        # dataset, reader index, native style, instruction, task, episode, frame, episode length
        self.flat: list[tuple[int, int, int, str, str, int, int, int]] = []
        self.style_ids = set(range(style_min, style_max + 1))
        self.metadata = []

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
            self.metadata.append({"task": task_name, "episodes": ep_meta})
            n_kept = 0
            if set(map(int, ds.trajectory_ids)) != set(ep_meta):
                raise ValueError("reader episode IDs differ from validated metadata")
            if any(
                ep_meta[int(i)]["length"] != int(n)
                for i, n in zip(ds.trajectory_ids, ds.trajectory_lengths, strict=True)
            ):
                raise ValueError(
                    "reader episode lengths differ from validated metadata"
                )
            for frame_idx, (traj_id, step) in enumerate(ds.all_steps):
                m = ep_meta[int(traj_id)]
                if m["style_id"] not in self.style_ids:
                    continue
                if (
                    episode_residues is not None
                    and (int(traj_id) % episode_mod) not in episode_residues
                ):
                    continue
                self.flat.append(
                    (
                        ds_idx,
                        frame_idx,
                        m["style_id"],
                        m["lang"],
                        task_name,
                        int(traj_id),
                        int(step),
                        m["length"],
                    )
                )
                n_kept += 1
            self.datasets.append(ds)
            print(
                f"[CorpRoboCasaDataset] {Path(path).parent.parent.name}: "
                f"kept {n_kept}/{len(ds.all_steps)} frames",
                flush=True,
            )
        if not self.flat:
            raise RuntimeError("filtered dataset is empty; check style range")

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, i: int) -> dict:
        ds_idx, frame_idx, style_id, lang, task_name, episode, step, length = self.flat[
            i
        ]
        sample = self.datasets[ds_idx][frame_idx]
        obs = sample["obs"]
        images = {}
        for view, key in VIEW_TO_KEY.items():
            images[view] = prepare_image(obs[key][0], self.shift, self.image_size)
        state = torch.cat(
            [
                obs["robot0_base_to_eef_pos"][0],
                obs["robot0_base_to_eef_quat"][0],
                obs["robot0_gripper_qpos"][0],
            ]
        )
        valid = torch.arange(self.horizon) < length - step
        action = sample["action"].clone()
        if action.shape != (self.horizon, 12) or not valid.any():
            raise ValueError("invalid action chunk or frame index")
        # CoRP Appendix A.3: repeat final action, exclude repeated positions.
        action[~valid] = action[int(valid.sum()) - 1].clone()
        return {
            "images": images,
            "state": state,
            "action": action,
            "action_mask": valid,
            "instruction": lang,
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "task": task_name,
            "episode": episode,
            "frame": step,
        }

    def support(self, indices=None) -> dict:
        """Independent episode and frame support, computed without decoding video."""
        rows = self.flat if indices is None else [self.flat[i] for i in indices]
        groups = {}
        for ds, _, style, _, task, episode, _, _ in rows:
            g = groups.setdefault((task, style), {"frames": 0, "episodes": set()})
            g["frames"] += 1
            g["episodes"].add((ds, episode))
        return {
            "frames": len(rows),
            "episodes": len({(r[0], r[5]) for r in rows}),
            "metadata_sha256": hashlib.sha256(
                json.dumps(self.metadata, sort_keys=True).encode()
            ).hexdigest(),
            "selection_sha256": hashlib.sha256(
                json.dumps(
                    [(r[4], r[5], r[6], r[2]) for r in rows], separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "per_task_style": [
                {
                    "task": task,
                    "style": style,
                    "frames": g["frames"],
                    "episodes": len(g["episodes"]),
                }
                for (task, style), g in sorted(groups.items())
            ],
        }


class LinearNormalizer:
    """train-only state z-score + action identity。

    action 用 identity 与官方 RoboCasa DP 基线一致：其 normalizer 把 action
    设为 scale=1/offset=0。实测部分 action 维（base_motion / control_mode）
    在若干任务是常量，min-max 会退化，因此不采用 min-max。
    """

    def __init__(self, state_mean, state_std):
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        if (
            self.state_mean.shape != (9,)
            or self.state_std.shape != (9,)
            or not torch.isfinite(self.state_mean).all()
            or not torch.isfinite(self.state_std).all()
            or not (self.state_std > 0).all()
        ):
            raise ValueError(
                "state normalizer must contain nine finite means and positive standard deviations"
            )

    @classmethod
    def fit(
        cls, dataset: CorpRoboCasaDataset, max_steps: int = 2048
    ) -> "LinearNormalizer":
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
        "action_mask": torch.stack([b["action_mask"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "style_id": torch.stack([b["style_id"] for b in batch]),
        "task": [b["task"] for b in batch],
        "episode": [b["episode"] for b in batch],
        "frame": [b["frame"] for b in batch],
    }
