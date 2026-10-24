"""CoRP adapted to RoboCasa (P1).

把官方 CoRP 架构（DINOv2 ViT-S/14 + T5-tiny + K=48 FiLM resampler + 1-layer DiT
flow matching）原样保留，只改接口：

    views       2  -> 3   (agentview_left / eye_in_hand / agentview_right)
    state_dim   8  -> 9   (eef_pos_rel3 + eef_rot_rel4 + gripper2，与 RoboCasa DP 基线 shape_meta 一致)
    action_dim  7  -> 12  (RoboCasa 12D action)
    horizon     12 -> 10

并补充官方发布仓库缺失的**训练路径**（flow-matching velocity loss），因为官方只有推理。

权重初始化策略（严格、可审计）：
    * DINOv2 / T5 / resampler 交叉注意力与 FiLM / DiT block / adaLN / time embedding
      -> 从 corp_libero.pt 加载（形状一致的共享层）。
    * 官方 2 个 view 的 learned query / view embedding -> 按 view 映射拷到前 2 个 view；
      第 3 个 view 随机初始化。
    * state projection (8->9) / action_projection (7->12) / velocity_projection (7->12)
      -> 随机初始化（接口适配层，语义不一一对应），必须在论文里明确标注。

命名必须写成 "CoRP adapted to RoboCasa / our reproduction"，不能写成官方 CoRP 在 RoboCasa 上。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_corp_repo() -> Path:
    """定位官方 CoRP 源码。

    优先级：环境变量 CORP_REPO > <repo_root>/repos/CoRP > /hy-tmp/models/corp/CoRP。
    本地开发用第二种，服务器用第三种（或显式设置 CORP_REPO）。
    """
    candidates = []
    env = os.environ.get("CORP_REPO")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(_REPO_ROOT / "repos" / "CoRP")
    candidates.append(Path("/hy-tmp/models/corp/CoRP"))
    for cand in candidates:
        if (cand / "corp" / "policy.py").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "找不到官方 CoRP 源码；请设置 CORP_REPO 指向含 corp/ 的仓库根目录"
    )


_CORP_REPO = _resolve_corp_repo()
if str(_CORP_REPO) not in sys.path:
    sys.path.insert(0, str(_CORP_REPO))

from corp.action_generator import DiTActionGenerator  # noqa: E402
from corp.representation.extractor import RepresentationExtractor  # noqa: E402


# RoboCasa 真实接口（来自数据集 meta/modality.json，61/61 atomic task 一致）：
#   raw observation.state 16D =
#       base_position(0:3) + base_rotation(3:7)
#       + end_effector_position_relative(7:10) + end_effector_rotation_relative(10:14)
#       + gripper_qpos(14:16)
#   但 RoboCasa DP 基线（diffusion_policy shape_meta）只把
#       end_effector_position_relative(7:10) + end_effector_rotation_relative(10:14)
#       + gripper_qpos(14:16) 作为策略输入 = 9D。
#   CoRP-RoboCasa 与基线对齐，因此 state_dim=9（不是 raw 的 16D）。
#   action 12D =
#       base_motion(0:4) + control_mode(4:5)
#       + end_effector_position(5:8) + end_effector_rotation(8:11) + gripper_close(11:12)
#   语言：extras/episode_*/ep_meta.json 的 `lang` 原始文本（同时 meta/tasks.jsonl 有任务文本）。
# 与官方 config.json 保持同一结构，只改接口字段。
# 注意 view_names 用 RoboCasa 的真实观测键；官方映射见 VIEW_TO_OFFICIAL。
ROBOCASA_CORP_CONFIG: dict = {
    "image_size": 224,
    "action_dim": 12,
    "action_horizon": 10,
    "state_dim": 9,
    "hidden_dim": 768,
    "view_names": [
        "robot0_agentview_left",
        "robot0_eye_in_hand",
        "robot0_agentview_right",
    ],
    "representation_extractor": {
        "visual_encoder": "dinov2_vits14",
        "language_encoder": {"name": "t5-efficient-tiny", "max_length": 32},
        "cross_attention_resampler": {
            "tokens_per_view": 48,
            "num_heads": 8,
            "film_rank": 64,
        },
    },
    "action_generator": {
        "num_layers": 1,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "max_horizon": 128,
        "adaln_rank": 64,
        "num_sample_steps": 10,
    },
}

# RoboCasa view -> 官方 CoRP view（用于迁移 learned query / view embedding）。
# 官方只有 agentview + eye_in_hand 两个 view。
VIEW_TO_OFFICIAL: dict[str, str | None] = {
    "robot0_agentview_left": "agentview",
    "robot0_eye_in_hand": "eye_in_hand",
    "robot0_agentview_right": None,  # 新 view，随机初始化
}

# 官方 CoRP 的 view 顺序（来自 checkpoints/config.json）。
OFFICIAL_VIEW_NAMES: list[str] = ["agentview", "eye_in_hand"]


class CorpRoboCasa(nn.Module):
    """两阶段 CoRP，接口适配到 RoboCasa。"""

    def __init__(self, config: dict = ROBOCASA_CORP_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.view_names = tuple(config["view_names"])
        self.state_dim = int(config["state_dim"])
        self.action_dim = int(config["action_dim"])
        self.action_horizon = int(config["action_horizon"])
        self.representation_extractor = RepresentationExtractor(config)
        self.action_generator = DiTActionGenerator(config)

    def extract(self, images: dict[str, torch.Tensor], state: torch.Tensor,
                instruction) -> object:
        """返回 Representation。images: {view: [B,3,224,224]}，state: [B,state_dim]。"""
        return self.representation_extractor(
            images=images, state=state, instruction=instruction
        )

    def flow_matching_loss(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Rectified-flow velocity loss（与官方 sample() 的积分方向严格一致）。

        官方 sample(): action=randn() 且 tau: 0->1，即 x(0)=eps(noise) -> x(1)=data。
        因此线性路径 x_tau=(1-tau)*eps+tau*x0，目标速度 v=x0-eps，
        loss=MSE(v_theta(x_tau,tau), v)。
        """
        representation = self.extract(images, state, instruction)
        action = action.to(dtype=representation.action_dtype)
        device = action.device
        b = action.shape[0]
        eps = torch.randn_like(action)
        # 论文: tau ~ Beta(1.0, 1.5)，不是 Uniform(0,1)
        tau = torch.distributions.Beta(1.0, 1.5).sample((b,)).to(device=device, dtype=action.dtype)
        # 必须与官方 DiTActionGenerator.sample 的 Euler 积分方向一致：
        #   x(0)=eps(noise) -> x(1)=action(data), dx/dtau = action - eps
        x_tau = eps * (1.0 - tau[:, None, None]) + action * tau[:, None, None]
        target = action - eps
        prefix = self.action_generator.prepare_prefix(representation)
        pred = self.action_generator(x_tau, tau, prefix)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample_action(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """返回归一化动作 [B, action_horizon, action_dim]（由调用方反归一化）。"""
        representation = self.extract(images, state, instruction)
        return self.action_generator.sample(representation, num_steps=num_steps)


def load_corp_robocasa(
    checkpoint: str | Path,
    *,
    config: dict = ROBOCASA_CORP_CONFIG,
    device: str | torch.device = "cpu",
) -> tuple[CorpRoboCasa, dict]:
    """从官方 corp_libero.pt 加载共享权重，接口层随机初始化，并返回可审计报告。"""
    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "corp_libero_v2":
        raise ValueError("expected a corp_libero_v2 checkpoint")
    libero_sd: dict[str, torch.Tensor] = payload["model"]

    model = CorpRoboCasa(config)
    model_sd = model.state_dict()

    copied: list[str] = []
    random_init: list[str] = []
    partial: list[str] = []

    for key, tensor in model_sd.items():
        if key not in libero_sd:
            random_init.append(key)
            continue
        src = libero_sd[key]
        if src.shape == tensor.shape:
            model_sd[key] = src.clone()
            copied.append(key)
        else:
            random_init.append(key)

    # 部分迁移：view query / view embedding（[3,...] vs [2,...]）。
    _merge_view_params(model_sd, libero_sd, config, partial)

    model.load_state_dict(model_sd)
    report = {
        "total": len(model_sd),
        "copied": len(copied) + len(partial),
        "random_init": len(random_init) - len(partial),
        "copied_keys": copied,
        "partial_keys": partial,
        "random_init_keys": [k for k in random_init if k not in partial],
    }
    return model.to(device).eval(), report


def _merge_view_params(
    model_sd: dict[str, torch.Tensor],
    libero_sd: dict[str, torch.Tensor],
    config: dict,
    partial: list[str],
) -> None:
    """把官方 2 个 view 的 query/view_embedding 拷到映射后的 view，第 3 个 view 保留随机。"""
    view_names = config["view_names"]
    prefix = "representation_extractor.cross_attention_resampler"
    q_key = f"{prefix}.queries"
    ve_key = f"{prefix}.view_embedding"
    for key, off_names in ((q_key, OFFICIAL_VIEW_NAMES), (ve_key, OFFICIAL_VIEW_NAMES)):
        if key not in libero_sd or key not in model_sd:
            continue
        src = libero_sd[key]  # [2, ...]
        dst = model_sd[key]   # [3, ...]
        for i, name in enumerate(view_names):
            off = VIEW_TO_OFFICIAL.get(name)
            if off is not None and off in off_names:
                j = off_names.index(off)
                dst[i] = src[j].clone()
        model_sd[key] = dst
        partial.append(key)
