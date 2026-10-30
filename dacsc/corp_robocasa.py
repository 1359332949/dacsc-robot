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
from dacsc.corp_images import image_contract

_REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FORMAT = "corp_robocasa_v3"
OBJECTIVE = "corp_masked_flow_repeat_last_v1"
CORP_STYLE_MODES = (
    "none",
    "random_stats",
    "cross_style_stats",
    "cross_style_stats_action_aware",
)


def masked_action_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """CoRP Eq.5: average over valid action positions and dimensions only."""
    if (
        pred.shape != target.shape
        or mask.dtype != torch.bool
        or mask.shape != target.shape[:2]
    ):
        raise ValueError(
            "action mask must be boolean [B,H] for matching [B,H,D] actions"
        )
    if not mask.any():
        raise ValueError("action batch has no valid positions")
    loss = F.mse_loss(pred[mask].float(), target[mask].float())
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "nonfinite flow loss; refusing to update model parameters"
        )
    return loss


def _resolve_corp_repo() -> Path:
    """Use the explicit checkout or the project checkout; invalid configuration fails."""
    env = os.environ.get("CORP_REPO")
    candidate = Path(env).expanduser() if env else _REPO_ROOT / "repos" / "CoRP"
    if (candidate / "corp" / "policy.py").is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        "找不到官方 CoRP 源码；请设置 CORP_REPO 指向含 corp/ 的仓库根目录"
    )


_CORP_REPO = _resolve_corp_repo()
if str(_CORP_REPO) not in sys.path:
    sys.path.insert(0, str(_CORP_REPO))

from corp.action_generator import DiTActionGenerator  # noqa: E402
from corp.representation.extractor import (
    Representation,
    RepresentationExtractor,
)  # noqa: E402


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

    def __init__(
        self, config: dict = ROBOCASA_CORP_CONFIG, token_style: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.view_names = tuple(config["view_names"])
        self.state_dim = int(config["state_dim"])
        self.action_dim = int(config["action_dim"])
        self.action_horizon = int(config["action_horizon"])
        self.representation_extractor = RepresentationExtractor(config)
        self.action_generator = DiTActionGenerator(config)
        # token-level style statistics modulation（patch tokens, pre-resampler）。
        # 用 forward hook 插入，避免复制官方 RepresentationExtractor.forward。
        self._current_domain_ids: torch.Tensor | None = None
        self._last_style_tokens: torch.Tensor | None = None
        self._token_importance: torch.Tensor | None = None
        self._style_handle = None
        self.token_style = None
        self.set_token_style(token_style)

    def set_token_style(self, token_style: nn.Module | None) -> None:
        """注册/替换 token_style hook；晚赋值必须走这里，不能直接改属性。"""
        if self._style_handle is not None:
            self._style_handle.remove()
            self._style_handle = None
        self.token_style = token_style
        if token_style is not None:
            self._style_handle = (
                self.representation_extractor.visual_encoder.register_forward_hook(
                    self._token_style_hook
                )
            )

    def _token_style_hook(self, module, inputs, output):
        """visual_encoder 输出 (B*V,T,D) -> (B,V,T,D) -> 调制 -> 还原。"""
        if self.token_style is None:
            return output
        if output.shape[0] % len(self.view_names):
            raise ValueError("visual batch must be divisible by the view count")
        b = output.shape[0] // len(self.view_names)
        # Official extractor concatenates whole view batches: [V, B, T, D].
        z = output.reshape(len(self.view_names), b, *output.shape[1:]).transpose(0, 1)
        if self.training and torch.is_grad_enabled() and z.requires_grad:
            z = z * 1.0
            z.retain_grad()
            self._last_style_tokens = z
        z = self.token_style(
            z, domain_ids=self._current_domain_ids, training=self.training
        )
        return z.transpose(0, 1).reshape(output.shape)

    def extract(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction,
        domain_ids: torch.Tensor | None = None,
    ) -> Representation:
        """返回 Representation。images: {view: [B,3,224,224]}，state: [B,state_dim]。"""
        if self.token_style is not None and self._style_handle is None:
            raise RuntimeError(
                "token_style is set but no forward hook is registered; "
                "use set_token_style() instead of assigning token_style directly"
            )
        self._current_domain_ids = domain_ids
        try:
            return self.representation_extractor(
                images=images, state=state, instruction=instruction
            )
        finally:
            self._current_domain_ids = None

    def update_action_importance(
        self, momentum: float = 0.9, return_summary: bool = False
    ) -> dict | None:
        """用 patch-token 梯度更新 action importance（供 *_action_aware 使用）。"""
        z = self._last_style_tokens
        self._last_style_tokens = None
        if self.token_style is None or z is None or z.grad is None:
            return None
        score = (z.detach().float() * z.grad.detach().float()).abs().mean(dim=(0, 1, 3))
        score = score / score.sum().clamp_min(1e-8)
        if self._token_importance is None:
            self._token_importance = score
        else:
            self._token_importance = (
                momentum * self._token_importance + (1.0 - momentum) * score
            )
        # The modulator expects protection strengths, not a probability mass.
        lo, hi = self._token_importance.amin(), self._token_importance.amax()
        protection = ((self._token_importance - lo) / (hi - lo).clamp_min(1e-12)).clamp(
            0, 1
        )
        self.token_style.set_action_importance(protection)
        return self.token_style.summary() if return_summary else None

    def training_runtime_state(self) -> dict:
        ts = self.token_style
        return {
            "token_importance": self._token_importance,
            "action_importance": None if ts is None else ts.action_importance,
            "donor_rng": (
                None
                if ts is None or ts._donor_gen is None
                else ts._donor_gen.get_state()
            ),
            "donor_device": (
                None
                if ts is None or ts._donor_gen is None
                else str(ts._donor_gen.device)
            ),
            "donor_contract": None if ts is None else ts.donor_contract(),
        }

    def load_training_runtime_state(
        self, state: dict, *, restore_donor_rng: bool = True
    ) -> None:
        device = next(self.parameters()).device
        value = state["token_importance"]
        self._token_importance = None if value is None else value.to(device)
        self._last_style_tokens = None
        if self.token_style is not None:
            if state["donor_contract"] != self.token_style.donor_contract():
                raise ValueError(
                    "checkpoint donor rule differs from the current implementation"
                )
            value = state["action_importance"]
            self.token_style.set_action_importance(
                None if value is None else value.to(device)
            )
            if restore_donor_rng and state["donor_rng"] is not None:
                if state["donor_device"] != str(device):
                    raise ValueError(
                        "donor RNG device differs; exact resume requires the same device"
                    )
                self.token_style._generator(device).set_state(state["donor_rng"].cpu())

    def flow_matching_loss(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rectified-flow velocity loss（与官方 sample() 的积分方向严格一致）。

        官方 sample(): action=randn() 且 tau: 0->1，即 x(0)=eps(noise) -> x(1)=data。
        因此线性路径 x_tau=(1-tau)*eps+tau*x0，目标速度 v=x0-eps，
        loss=MSE(v_theta(x_tau,tau), v)。
        """
        device = action.device
        b = action.shape[0]
        eps = torch.randn_like(action)
        # 论文: tau ~ Beta(1.0, 1.5)，不是 Uniform(0,1)
        tau = (
            torch.distributions.Beta(1.0, 1.5)
            .sample((b,))
            .to(device=device, dtype=action.dtype)
        )
        # 先生成 flow noise / tau，再 extract；token_style 的 RNG 消耗不会改变 flow noise。
        representation = self.extract(images, state, instruction, domain_ids=domain_ids)
        action = action.to(dtype=representation.action_dtype)
        eps = eps.to(dtype=representation.action_dtype)
        # 必须与官方 DiTActionGenerator.sample 的 Euler 积分方向一致：
        #   x(0)=eps(noise) -> x(1)=action(data), dx/dtau = action - eps
        x_tau = eps * (1.0 - tau[:, None, None]) + action * tau[:, None, None]
        target = action - eps
        prefix = self.action_generator.prepare_prefix(representation)
        pred = self.action_generator(x_tau, tau, prefix)
        return masked_action_mse(pred, target, action_mask)

    @torch.no_grad()
    def sample_action(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction,
        num_steps: int | None = None,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回归一化动作 [B, action_horizon, action_dim]（由调用方反归一化）。"""
        representation = self.extract(images, state, instruction, domain_ids=domain_ids)
        return self.action_generator.sample(representation, num_steps=num_steps)


def load_corp_robocasa(
    checkpoint: str | Path,
    *,
    config: dict = ROBOCASA_CORP_CONFIG,
    device: str | torch.device = "cpu",
    token_style: nn.Module | None = None,
) -> tuple[CorpRoboCasa, dict]:
    """从官方 corp_libero.pt 加载共享权重，接口层随机初始化，并返回可审计报告。"""
    checkpoint = Path(checkpoint).expanduser().resolve()
    from dacsc.upstream import verify_upstream

    verify_upstream(["CoRP"])
    # Use the release loader: validates config_sha256, normalizer and ALL source keys.
    from corp.checkpoint import load_policy

    source = load_policy(checkpoint, device="cpu")
    interface_fields = {"view_names", "state_dim", "action_dim", "action_horizon"}
    for key in set(source.config) | set(config):
        if key not in interface_fields and source.config.get(key) != config.get(key):
            raise ValueError(f"unsupported CoRP architecture change: {key}")
    if source.config["view_names"] != OFFICIAL_VIEW_NAMES:
        raise ValueError("source checkpoint has an unsupported camera-view contract")
    libero_sd = source.state_dict()

    model = CorpRoboCasa(config, token_style=token_style)
    if token_style is not None:
        assert model._style_handle is not None, "token_style hook not registered"
    model_sd = model.state_dict()

    copied: list[str] = []
    random_init: list[str] = []
    partial: list[str] = []
    interface_keys = {
        "representation_extractor.state_encoder.projection.weight",
        "action_generator.action_projection.weight",
        "action_generator.velocity_projection.weight",
        "action_generator.velocity_projection.bias",
    }
    view_keys = {
        "representation_extractor.cross_attention_resampler.queries",
        "representation_extractor.cross_attention_resampler.view_embedding",
    }
    if set(libero_sd) - set(model_sd):
        raise ValueError(f"unmapped source weights: {set(libero_sd) - set(model_sd)}")

    for key, tensor in model_sd.items():
        if key not in libero_sd:
            if not key.startswith("token_style."):
                raise ValueError(f"missing shared source weight: {key}")
            random_init.append(key)
            continue
        src = libero_sd[key]
        if key in view_keys:
            if src.shape[1:] != tensor.shape[1:]:
                raise ValueError(f"invalid view weight shape: {key}")
            continue
        if src.shape == tensor.shape:
            model_sd[key] = src.clone()
            copied.append(key)
        else:
            if key not in interface_keys:
                raise ValueError(f"unexpected shared weight shape mismatch: {key}")
            random_init.append(key)

    # 部分迁移：view query / view embedding（[3,...] vs [2,...]）。
    _merge_view_params(model_sd, libero_sd, config, partial)

    model.load_state_dict(model_sd)
    report = {
        "total": len(model_sd),
        "copied": len(copied) + len(partial),
        "random_init": len(random_init),
        "copied_keys": copied,
        "partial_keys": partial,
        "random_init_keys": random_init,
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
            raise ValueError(f"missing required view weight: {key}")
        src = libero_sd[key]  # [2, ...]
        dst = model_sd[key]  # [3, ...]
        for i, name in enumerate(view_names):
            if name not in VIEW_TO_OFFICIAL:
                raise ValueError(f"no declared view mapping: {name}")
            off = VIEW_TO_OFFICIAL.get(name)
            if off is not None and off in off_names:
                j = off_names.index(off)
                dst[i] = src[j].clone()
        model_sd[key] = dst
        partial.append(key)


def load_adapted_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CorpRoboCasa, tuple[torch.Tensor, torch.Tensor]]:
    """Load the current masked-loss/image contract; old experiment files are rejected."""
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    from dacsc.upstream import verify_upstream

    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("objective") != OBJECTIVE
        or payload.get("images") != image_contract()
        or payload.get("upstream") != verify_upstream()
    ):
        raise ValueError(
            "checkpoint training/image contract differs; legacy loading is unsupported"
        )
    token_style = None
    ts_cfg = payload.get("token_style")
    if ts_cfg:
        if ts_cfg.get("mode") not in CORP_STYLE_MODES[1:]:
            raise ValueError("token-style mode is outside the current CoRP protocol")
        from dacsc.token_style import TokenStyleModulator

        token_style = TokenStyleModulator(**ts_cfg)
        if "training_runtime" not in payload:
            raise ValueError(
                "token-style checkpoint lacks its donor contract; legacy loading is unsupported"
            )
    model = CorpRoboCasa(payload["config"], token_style=token_style)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    if "training_runtime" in payload:
        model.load_training_runtime_state(
            payload["training_runtime"], restore_donor_rng=False
        )
    norm = payload["normalizer"]
    order = payload["training_state"]["data_order"]
    model.evaluation_origin = {
        "seed": payload["run_args"]["seed"],
        "budget": payload["budget"],
        "intervention": payload["token_style"],
        "pixel_augmentation": payload["pixel_augmentation"],
        "optimization": payload["optimization"],
        "initialization_sha256": payload["initialization_sha256"],
        "data_metadata_sha256": order["metadata_sha256"],
        "data_length": order["dataset_length"],
    }
    from dacsc.eval_protocol import validate_training_origin

    validate_training_origin(model.evaluation_origin)
    return model.to(device).eval(), (norm["state_mean"], norm["state_std"])
