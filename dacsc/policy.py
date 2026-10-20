"""RoboCasa365 的 DINOv2 策略：把官方 DP 的视觉编码器换成 DINOv2 ViT-S/14。

设计原则（重要）：
  **不改动官方 repos/diffusion_policy 与 repos/robomimic 的任何文件。**
  做法是继承官方策略类，只重写 __init__（换编码器），
  其余方法（conditional_sample / predict_action / compute_loss / get_optimizer /
  set_normalizer）全部原样继承 —— 因为它们都通过 self.obs_encoder 泛化调用。

为什么只换编码器：
  - 官方 DP 的视觉编码器是 robomimic 的 ResNet18ConvFiLM
  - CoRP（arXiv 2610.08183）4.3 证明：DINOv2 预训练 ViT-S/14 比其它编码器高 ≥14.9 分
  - 我们的方法（通道级域敏感性门控）作用在视觉编码器上 → 先把它换掉，再插门控

v1 范围：只换编码器，**不加门控**。目的是先验证「CoRP 式编码器 + 官方评测管线」能跑通。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)

from dacsc.dinov2_encoder import DINOv2ObsEncoder
from dacsc.domain_stats import DomainActivationStats, make_domain_scorer
from dacsc.gating import DINOv2GateBank, install_gates, gate_summary, remove_gates
from dacsc.hard_gate import (HardMaskController, install_hard_gates,
                             remove_hard_gates, topk_ste_mask, gumbel_ste_mask,
                             wrs_mask, mask_iou_by_domain, score_separation_by_domain)
from dacsc.dg_methods import DgCDScorer, grad_reverse
from dacsc.losses import (ActionImportanceEMA, ActionImportanceRSC, choose_rho,
                          coral_from_domains, gate_regularizer)

log = logging.getLogger(__name__)


class RoboCasaDINOv2Policy(DiffusionTransformerHybridImagePolicy):
    """官方 DP 策略 + DINOv2 ViT-S/14 观测编码器。

    v3 会在这里注册通道门控（插在 DINOv2 的层/维度上）。
    """

    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 num_inference_steps: int | None = None,
                 # ---- 我们的编码器参数 ----
                 dino_backbone: str = "dinov2_vits14",
                 crop_size: int = 224,
                 view_feat_dim: int = 256,
                 low_dim_feat_dim: int = 128,
                 feature_dim: int = 256,
                 freeze_dino: bool = False,
                 freeze_obs_encoder: bool = False,
                 random_crop: bool = True,
                 pool: str = "mean",
                 dino_pretrained: bool = True,
                 dino_unfreeze_last_n_blocks: int | None = None,
                 use_resampler: bool = False,
                 resampler_num_queries: int = 48,
                 resampler_dim: int = 64,
                 resampler_heads: int = 6,
                 resampler_dropout: float = 0.0,
                 # ---- DiT 架构（默认与官方 bs192 配置一致）----
                 n_layer: int = 8,
                 n_cond_layers: int = 0,
                 n_head: int = 4,
                 n_emb: int = 256,
                 p_drop_emb: float = 0.0,
                 p_drop_attn: float = 0.3,
                 causal_attn: bool = True,
                 time_as_cond: bool = True,
                 obs_as_cond: bool = True,
                 pred_action_steps_only: bool = False,
                 flow_matching: bool = False,
                 flow_time_scale: float = 1000.0,
                 # ---- 官方配置里存在、但本策略不需要的参数（显式接收以免漏进 kwargs）----
                 # DINOv2 自己做裁剪与归一化，故忽略 crop_shape / vision_backbone 等
                 vision_backbone: str = "resnet18",
                 crop_shape=None,
                 obs_encoder_group_norm: bool = False,
                 eval_fixed_crop: bool = False,
                 # ---- ★ 通道门控（我们的方法）----
                 enable_gating: bool = False,
                 gate_layer_indices: list | None = None,
                 gate_init: float = 0.999,
                 num_domains: int = 50,
                 gate_rho_max: float = 0.3,
                 gate_rho_warmup_steps: int = 10000,
                 gate_lambda_d: float = 1.0,
                 gate_lambda_a: float = 1.0,
                 gate_lambda_rho: float = 1.0,
                 gate_stats_momentum: float = 0.99,
                 use_action_importance: bool = True,
                 gate_weight_decay: float = 0.0,
                 gate_learning_rate: float | None = None,
                 gate_diagnostics_every: int = 50,
                 # ---- v2 hard gate ----
                 gate_mode: str = "soft",
                 hard_layer_indices: list | None = None,
                 hard_drop_ratio: float = 0.33,
                 hard_mask_temperature: float = 0.1,
                 hard_mask_mode: str = "topk_ste",
                 hard_mask_domain_shared: bool = False,
                 hard_selection_mode: str = "linear",  # linear | two_stage
                 hard_candidate_ratio: float = 0.6,
                 hard_action_protect_ratio: float = 0.2,
                 hard_wrs_recover: bool = True,
                 hard_wrs_normalize: str = "per_sample",
                 hard_action_lambda: float = 1.0,
                 hard_domain_lambda: float = 1.0,
                 hard_domain_grl: float = 0.25,
                 hard_consistency_weight: float = 0.0,
                 hard_clean_importance_every: int = 4,
                 hard_train_only: bool = True,
                 hard_domain_mode: str = "classifier",  # classifier | dgcd
                 domain_id_min: int = 11,
                 domain_group_size: int = 1,
                 domain_head_lr: float | None = None,
                 # ★ 打分与重要性实现的选择（默认用移植自开源实现的版本）
                 domain_score_mode: str = "dgcd",   # "dgcd"(官方移植) | "kl_gauss"(自研消融)
                 dgcd_rho: float = 1e-3,
                 dgcd_group_smooth: float = 0.0,
                 dgcd_domain_center: bool = False,
                 layout_residual: bool = False,
                 layout_id_min: int = 11,
                 layout_num: int = 50,
                 layout_stats_momentum: float = 0.99,
                 token_score_pool: str = "mean",  # mean | meanmax
                 action_importance_impl: str = "taylor",  # "taylor" | "rsc"
                 rsc_drop_fraction: float = 0.3,
                 lambda_coral: float = 0.0,          # >0 时加入 Deep CORAL 项
                 **kwargs):
        # 跳过官方 __init__ 里的 robomimic 编码器构造，只初始化 nn.Module
        BaseImagePolicy.__init__(self)
        self._dacsc_version = "dinov2-v1"

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, f"只支持 1D action，收到 {action_shape}"
        action_dim = action_shape[0]

        # ---------- 观测编码器：DINOv2（唯一的实质改动）----------
        obs_encoder = DINOv2ObsEncoder(
            shape_meta=shape_meta,
            dino_backbone=dino_backbone,
            crop_size=crop_size,
            view_feat_dim=view_feat_dim,
            low_dim_feat_dim=low_dim_feat_dim,
            feature_dim=feature_dim,
            freeze_dino=freeze_dino,
            random_crop=random_crop,
            pool=pool,
            pretrained=dino_pretrained,
            unfreeze_last_n_blocks=dino_unfreeze_last_n_blocks,
            use_resampler=use_resampler,
            resampler_num_queries=resampler_num_queries,
            resampler_dim=resampler_dim,
            resampler_heads=resampler_heads,
            resampler_dropout=resampler_dropout,
        )
        self.flow_matching = bool(flow_matching)
        self.flow_time_scale = float(flow_time_scale)
        if use_resampler and not self.flow_matching:
            raise ValueError("use_resampler=True 当前要求 flow_matching=True")
        if freeze_obs_encoder:
            for p in obs_encoder.parameters():
                p.requires_grad_(False)
        log.info("DINOv2 obs encoder: backbone=%s views=%d feature_dim=%d "
                 "freeze_dino=%s freeze_obs_encoder=%s",
                 dino_backbone, len(obs_encoder.rgb_keys), feature_dim,
                 freeze_dino, freeze_obs_encoder)

        # ---------- 以下脚手架复制自官方 __init__（官方文件不动）----------
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.tokens_per_obs = int(obs_encoder.tokens_per_obs)
        model_obs_tokens = int(n_obs_steps) * self.tokens_per_obs
        if not obs_as_cond:
            assert self.tokens_per_obs == 1, "resampler token output 目前只支持 obs_as_cond=True"
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=model_obs_tokens,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.model_obs_tokens = model_obs_tokens
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # ---------- ★ 安装通道门控 ----------
        self.enable_gating = bool(enable_gating)
        self.gate_cfg = dict(
            rho_max=float(gate_rho_max), rho_warmup_steps=int(gate_rho_warmup_steps),
            lambda_d=float(gate_lambda_d), lambda_a=float(gate_lambda_a),
            lambda_rho=float(gate_lambda_rho),
            weight_decay=float(gate_weight_decay),
            learning_rate=gate_learning_rate)
        self.use_action_importance = bool(use_action_importance)
        self.gate_diagnostics_every = int(gate_diagnostics_every)
        self.gate_bank = None
        self.domain_stats = None
        self.action_importance = None
        self._gate_handles = []
        self._collect: dict = {}
        self._prev_post_gate: list = []
        self._last_cond_features = None
        self._last_coral = None
        self._cond_hook = None
        self._gate_step = 0
        self._last_gate_metrics: dict = {}
        self._current_mask_info: dict = {}
        self._current_domain_ids = None
        self._current_layout_ids = None
        self._stats_update_enabled = True
        self.layout_residual_enabled = bool(layout_residual)
        self.layout_id_min = int(layout_id_min)
        self.layout_num = int(layout_num)
        self.layout_stats_momentum = float(layout_stats_momentum)
        self.token_score_pool = str(token_score_pool)

        if self.enable_gating:
            assert gate_mode in ("soft", "hard"), f"未知 gate_mode={gate_mode}"
            self.gate_mode = gate_mode
            dino = obs_encoder.dino
            hidden = int(getattr(dino, "embed_dim"))
            n_layers = len(getattr(dino, "blocks"))

            if gate_mode == "soft":
                self.gate_bank = DINOv2GateBank(
                    num_layers=n_layers, hidden_dim=hidden,
                    layer_indices=gate_layer_indices, init_gate=gate_init)
                self.domain_stats = make_domain_scorer(
                    mode=domain_score_mode, layer_indices=self.gate_bank.layer_indices,
                    num_channels=hidden, num_domains=int(num_domains),
                    momentum=float(gate_stats_momentum), dgcd_rho=float(dgcd_rho))
                self.domain_score_mode = domain_score_mode
                self.action_importance_impl = action_importance_impl
                self.lambda_coral = float(lambda_coral)
                if self.use_action_importance:
                    if action_importance_impl == "rsc":
                        self.action_importance = ActionImportanceRSC(
                            layer_indices=self.gate_bank.layer_indices,
                            num_channels=hidden,
                            drop_fraction=float(rsc_drop_fraction))
                    else:
                        self.action_importance = ActionImportanceEMA(
                            layer_indices=self.gate_bank.layer_indices,
                            num_channels=hidden)
                self._gate_handles = install_gates(
                    dino, self.gate_bank, collect=self._collect,
                    retain_post_grad=self.use_action_importance)
                self._install_cond_hook()
                log.info("soft gate 启用：%d 层 × %d 维，初始 g=%.4f",
                         len(self.gate_bank), hidden, gate_init)
            else:
                self.gate_bank = None
                self.domain_stats = None
                self.domain_score_mode = "domain_head"
                self.action_importance_impl = action_importance_impl
                self.lambda_coral = 0.0
                self.hard_layer_indices = (list(range(n_layers))
                                           if hard_layer_indices is None
                                           else [int(x) for x in hard_layer_indices])
                assert self.hard_layer_indices, "hard_layer_indices 不能为空"
                self.hard_controller = HardMaskController(
                    self.hard_layer_indices, drop_ratio=hard_drop_ratio,
                    temperature=hard_mask_temperature, train_only=hard_train_only)
                self.hard_masks: dict[int, torch.Tensor] = {}
                self.hard_cfg = dict(
                    drop_ratio=float(hard_drop_ratio),
                    temperature=float(hard_mask_temperature),
                    mask_mode=str(hard_mask_mode),
                    domain_shared=bool(hard_mask_domain_shared),
                    selection_mode=str(hard_selection_mode),
                    candidate_ratio=float(hard_candidate_ratio),
                    protect_ratio=float(hard_action_protect_ratio),
                    wrs_recover=bool(hard_wrs_recover),
                    wrs_normalize=str(hard_wrs_normalize),
                    action_lambda=float(hard_action_lambda),
                    domain_lambda=float(hard_domain_lambda),
                    domain_grl=float(hard_domain_grl),
                    consistency=float(hard_consistency_weight),
                    clean_importance_every=int(hard_clean_importance_every),
                    train_only=bool(hard_train_only),
                    domain_mode=str(hard_domain_mode),
                    domain_id_min=int(domain_id_min),
                    domain_group_size=int(domain_group_size),
                    domain_head_lr=domain_head_lr,
                    group_smooth=float(dgcd_group_smooth),
                    domain_center=bool(dgcd_domain_center))
                self.hard_domain_mode = str(hard_domain_mode)
                self.hard_num_domains = int(num_domains)
                n_views = len(obs_encoder.rgb_keys)
                self.register_buffer(
                    "_layout_mean", torch.zeros(self.layout_num, n_views, hidden))
                self.register_buffer(
                    "_layout_seen", torch.zeros(self.layout_num, n_views, dtype=torch.bool))
                if self.hard_domain_mode == "classifier":
                    self.domain_head = nn.Linear(hidden, int(num_domains))
                    self.hard_dgcd = None
                elif self.hard_domain_mode == "dgcd":
                    self.domain_head = None
                    self.hard_dgcd = DgCDScorer(rho=float(dgcd_rho))
                elif self.hard_domain_mode == "random":
                    # 对照：per-sample 随机 33% mask，无任何域结构。
                    self.domain_head = None
                    self.hard_dgcd = None
                else:
                    raise ValueError(f"未知 hard_domain_mode={hard_domain_mode}")
                if self.use_action_importance:
                    if action_importance_impl == "rsc":
                        self.action_importance = ActionImportanceRSC(
                            layer_indices=self.hard_layer_indices,
                            num_channels=hidden,
                            drop_fraction=float(rsc_drop_fraction))
                    else:
                        self.action_importance = ActionImportanceEMA(
                            layer_indices=self.hard_layer_indices,
                            num_channels=hidden)
                # hook 在 forward 内用当前 batch 特征现场生成 mask。
                self.hard_controller.set_mask_builder(self._build_hard_mask)
                need_pre_grad = (self.use_action_importance or
                                 self.hard_domain_mode == "classifier")
                self._gate_handles = install_hard_gates(
                    dino, self.hard_controller, collect=self._collect,
                    retain_post_grad=self.use_action_importance,
                    retain_pre_grad=need_pre_grad)
                log.info("hard gate 启用：layers=%s drop=%.2f temp=%.2f "
                         "action_lambda=%.2f domain_grl=%.2f",
                         self.hard_layer_indices, hard_drop_ratio,
                         hard_mask_temperature, hard_action_lambda, hard_domain_grl)
        else:
            self.gate_mode = "off"

    # ---------- ★ 门控的对外接口 ----------
    def gate_modules(self):
        """返回所有通道门控模块（未启用时为空列表）。"""
        if self.gate_bank is None:
            return []
        return list(self.gate_bank.gates.values())

    def gate_parameters(self):
        """门控参数（用于单独设置 weight_decay / lr）。"""
        return [p for m in self.gate_modules() for p in m.parameters()]

    def gates(self):
        """当前所有门控值 (L, C)；未启用时返回 None。"""
        return None if self.gate_bank is None else self.gate_bank.gates_matrix()

    def _install_cond_hook(self) -> None:
        """用 DiT 的 forward_pre_hook 捕获 cond 特征（供 CORAL 用）。

        为什么用 hook：官方 `compute_loss` 内部自己算 cond，我们不便改官方文件；
        `TransformerForDiffusion.forward(x, timesteps, cond)` 的第三个位置参数就是 cond。
        """
        if self._cond_hook is not None:
            return

        def pre_hook(module, args, kwargs=None):
            cond = None
            if len(args) >= 3:
                cond = args[2]
            elif kwargs and "cond" in kwargs:
                cond = kwargs["cond"]
            if torch.is_tensor(cond) and not self.lambda_coral <= 0:
                self._last_cond_features = cond.mean(dim=1).detach()   # (B, Do)

        self._cond_hook = self.model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def gate_report(self) -> dict:
        if self.gate_mode == "hard":
            rep = {"enable_gating": True, "mode": "hard",
                   "hard_layer_indices": list(self.hard_layer_indices),
                   "hard_cfg": dict(self.hard_cfg),
                   "gate_step": self._gate_step}
            if self._last_gate_metrics:
                rep["metrics"] = dict(self._last_gate_metrics)
            if self.action_importance is not None:
                rep["action_importance"] = self.action_importance.summary()
            return rep
        if self.gate_bank is None:
            return {"enable_gating": False}
        rep = {"enable_gating": True, "mode": "soft", **gate_summary(self.gate_bank)}
        if self.domain_stats is not None:
            rep["domain_stats"] = self.domain_stats.summary()
        if self.action_importance is not None:
            rep["action_importance"] = self.action_importance.summary()
        rep["gate_step"] = self._gate_step
        return rep

    # ---------- ★ 关掉 hook（评测/导出时用）----------
    def disable_gating_hooks(self):
        """拆掉门控 hook（用于「装/拆」对照与导出）。"""
        remove_gates(self._gate_handles)
        self._gate_handles = []

    def reenable_gating_hooks(self):
        """重新装上门控 hook。"""
        if not self.enable_gating or self._gate_handles:
            return
        if self.gate_mode == "hard":
            need_pre_grad = (self.use_action_importance or
                             self.hard_domain_mode == "classifier")
            self._gate_handles = install_hard_gates(
                self.obs_encoder.dino, self.hard_controller, collect=self._collect,
                retain_post_grad=self.use_action_importance,
                retain_pre_grad=need_pre_grad)
        else:
            self._gate_handles = install_gates(
                self.obs_encoder.dino, self.gate_bank, collect=self._collect,
                retain_post_grad=self.use_action_importance)

    # ---------- ★ checkpoint：门控统计状态也持久化 ----------
    def state_dict(self, *args, **kwargs):
        """把 DgCD / KL 统计和动作重要性 EMA 一并写入 model state_dict。"""
        sd = super().state_dict(*args, **kwargs)
        if self.domain_stats is not None:
            for k, v in self.domain_stats.state_dict().items():
                sd[f"_dacsc_domain_stats.{k}"] = v
        if self.action_importance is not None:
            for k, v in self.action_importance.state_dict().items():
                sd[f"_dacsc_action_importance.{k}"] = v
        sd["_dacsc_gate_step"] = torch.tensor(self._gate_step, dtype=torch.long)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        sd = dict(state_dict)
        extras_dom, extras_imp = {}, {}
        for k in list(sd.keys()):
            if k.startswith("_dacsc_domain_stats."):
                extras_dom[k.split(".", 1)[1]] = sd.pop(k)
            elif k.startswith("_dacsc_action_importance."):
                extras_imp[k.split(".", 1)[1]] = sd.pop(k)
        gate_step = sd.pop("_dacsc_gate_step", None)
        result = super().load_state_dict(sd, strict=strict, **kwargs)
        if extras_dom and self.domain_stats is not None:
            self.domain_stats.load_state_dict(extras_dom)
        if extras_imp and self.action_importance is not None:
            self.action_importance.load_state_dict(extras_imp)
        if gate_step is not None:
            self._gate_step = int(gate_step.item())
        return result

    # ---------- ★ 优化器：必须把门控参数加进去 ----------
    def get_optimizer(self, transformer_weight_decay: float, obs_encoder_weight_decay: float,
                      learning_rate: float, betas: Tuple[float, float],
                      obs_encoder_learning_rate: float | None = None,
                      gate_weight_decay: float = 0.0,
                      gate_learning_rate: float | None = None,
                      domain_head_lr: float | None = None):
        """在官方实现基础上**多一组门控参数**。

        官方 `get_optimizer` 只返回 DiT（decay/no-decay 两组）+ obs_encoder，
        **不含门控参数** —— 若不覆盖，门控会算出梯度但永远不更新。
        （这个 bug 是被训练烟测抓出来的。）
        """
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        # DINOv2 主干用更小的 lr；view_proj/low_dim_net/fuse 跟随主 lr。
        dino_params, proj_params = [], []
        for name, p in self.obs_encoder.named_parameters():
            if not p.requires_grad:
                continue
            (dino_params if name.startswith("dino.") else proj_params).append(p)
        if dino_params:
            group = {"params": dino_params, "weight_decay": obs_encoder_weight_decay}
            if obs_encoder_learning_rate is not None:
                group["lr"] = float(obs_encoder_learning_rate)
            optim_groups.append(group)
        if proj_params:
            optim_groups.append({
                "params": proj_params,
                "weight_decay": obs_encoder_weight_decay,
            })
        if self.gate_mode == "hard" and self.domain_head is not None:
            dh = list(self.domain_head.parameters())
            if dh:
                group = {"params": dh, "weight_decay": 0.0}
                if domain_head_lr is not None:
                    group["lr"] = float(domain_head_lr)
                optim_groups.append(group)
        gp = self.gate_parameters()
        if gp:
            group = {"params": gp, "weight_decay": float(gate_weight_decay)}
            eff_lr = gate_learning_rate if gate_learning_rate is not None else \
                self.gate_cfg.get("learning_rate")
            if eff_lr is not None:
                group["lr"] = float(eff_lr)
            optim_groups.append(group)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def _maybe_clean_action_importance(self, batch):
        """Taylor pruning 的 clean-forward 动作重要性。

        每 N 步在**不加 hard mask**的 pre-gate 特征上做一次前向/反向，
        用 |f·∂L_DP/∂f| 更新 EMA；随后清掉这次诊断产生的参数梯度，
        避免影响正式训练 step。
        """
        if not self.use_action_importance:
            return
        # validation 在 torch.no_grad() 下运行，不能做 clean_loss.backward()
        if not torch.is_grad_enabled():
            return
        every = int(self.hard_cfg.get("clean_importance_every", 0))
        if every <= 0:
            return
        if not hasattr(self, "_clean_imp_counter"):
            self._clean_imp_counter = 0
        self._clean_imp_counter += 1
        if self._clean_imp_counter % every != 0:
            return
        old_mask = self.hard_controller.mask
        old_build = self.hard_controller.build_enabled
        self.hard_controller.mask = None
        self.hard_controller.build_enabled = False
        self._collect.clear()
        try:
            clean_loss = self._base_loss(batch)
            clean_loss.backward()
            pre = self._collect.get("pre_gate_live", [])
            if pre:
                self.action_importance.update_from_tensors(dict(pre))
        finally:
            self.hard_controller.mask = old_mask
            self.hard_controller.build_enabled = old_build
            self._prev_post_gate = []
            self.zero_grad(set_to_none=True)
            self._collect.clear()

    # ---------- ★ v2 hard gate：当前 batch 内生成 per-sample mask ----------
    def _expanded_domain_indices(self, n_actual: int, device):
        """把 batch 级 style_id 展开到 DINOv2 的 (B*To*V) 样本维，并映射成粗域索引。"""
        did = self._current_domain_ids
        if did is None:
            return None
        n_views = len(self.obs_encoder.rgb_keys)
        did_exp = did.repeat_interleave(int(self.n_obs_steps) * n_views)
        if did_exp.shape[0] < n_actual:
            did_exp = did_exp.repeat((n_actual + did_exp.shape[0] - 1)
                                     // did_exp.shape[0])
        did_exp = did_exp[:n_actual].to(device=device, dtype=torch.long)
        return ((did_exp - int(self.hard_cfg["domain_id_min"])) //
                max(int(self.hard_cfg["domain_group_size"]), 1)).clamp(
            0, int(self.hard_num_domains) - 1)

    def _expanded_layout_indices(self, n_actual: int, device):
        did = self._current_layout_ids
        if did is None:
            return None
        n_views = len(self.obs_encoder.rgb_keys)
        did_exp = did.repeat_interleave(int(self.n_obs_steps) * n_views)
        if did_exp.shape[0] < n_actual:
            did_exp = did_exp.repeat((n_actual + did_exp.shape[0] - 1)
                                     // did_exp.shape[0])
        did_exp = did_exp[:n_actual].to(device=device, dtype=torch.long)
        return (did_exp - int(self.layout_id_min)).clamp(0, int(self.layout_num) - 1)

    @torch.no_grad()
    def _layout_residual(self, pooled: torch.Tensor, layout_idx,
                         collect_metrics: bool = False):
        """用每个 layout 的在线通道均值去掉 geometry/task 共享分量。

        统计更新用 bincount + index_add 一次完成，避免每个 layout 一次
        Python 循环和多次 GPU kernel 启动；日志标量只在诊断步回读。
        """
        if (not self.layout_residual_enabled or layout_idx is None
                or not hasattr(self, "_layout_mean")):
            return pooled, 0.0, 0
        n_views = self._layout_mean.shape[1]
        idx = layout_idx.to(device=pooled.device, dtype=torch.long)
        view = torch.arange(pooled.shape[0], device=pooled.device) % n_views
        key = idx * n_views + view
        mean_flat = self._layout_mean.reshape(-1, self._layout_mean.shape[-1])
        seen_flat = self._layout_seen.reshape(-1)
        seen = seen_flat[key]
        resid = pooled.detach().clone()
        if seen.any():
            resid[seen] = resid[seen] - mean_flat[key[seen]].to(resid)
        # 只用于日志；非诊断步不做 GPU→CPU 同步。
        frac = float(seen.float().mean()) if collect_metrics else 0.0
        n_seen = int(seen_flat.sum()) if collect_metrics else 0
        if self.training and self._stats_update_enabled:
            p = pooled.detach()
            n_keys = mean_flat.shape[0]
            counts = torch.bincount(key, minlength=n_keys)
            ids = torch.nonzero(counts > 0, as_tuple=False).view(-1)
            if ids.numel():
                sums = p.new_zeros((n_keys, p.shape[1]))
                sums.index_add_(0, key, p)
                batch_mean = sums[ids] / counts[ids, None].to(p.dtype)
                m = float(self.layout_stats_momentum)
                was_seen = seen_flat[ids]
                updated = torch.where(
                    was_seen[:, None],
                    m * mean_flat[ids].to(p.dtype) + (1.0 - m) * batch_mean,
                    batch_mean)
                mean_flat[ids] = updated.to(mean_flat.dtype)
                seen_flat[ids] = True
        return resid, frac, n_seen

    def _build_hard_mask(self, feat: torch.Tensor, layer: int) -> torch.Tensor:
        """由**当前 batch 的 pre-gate 特征**生成当前 batch 的 mask。

        feat: (B*To*V, T, C)，已 detach；这里只产生扰动 mask，
        不把 DgCD/域分数反传到视觉主干。

        诊断相关计算（IoU / score cosine / 原型相似度 / 各类 sqrt 统计）
        默认只在 ``gate_diagnostics_every`` 步执行；普通 step 只保留轻量
        mask 生成，避免每步 O(N²) 的 pair 统计拖住 GPU。
        """
        every = int(getattr(self, "gate_diagnostics_every", 0) or 0)
        do_diag = bool(
            every > 0 and (self._gate_step % every == 0)
            and not getattr(self, "_diag_suppressed", False))
        if not hasattr(self, "_last_diag"):
            self._last_diag = {
                "step": -1,
                "mask_iou_same": 0.0, "mask_iou_diff": 0.0, "mask_iou_gap": 0.0,
                "mask_iou_sample_same": 0.0, "mask_iou_sample_diff": 0.0,
                "mask_iou_sample_gap": 0.0,
                "drop_iou_same": 0.0, "drop_iou_diff": 0.0, "drop_iou_gap": 0.0,
                "drop_iou_sample_same": 0.0, "drop_iou_sample_diff": 0.0,
                "drop_iou_sample_gap": 0.0,
                "score_cos_same": 0.0, "score_cos_diff": 0.0,
                "score_cos_gap": 0.0,
                "proto_cos_mean": 0.0, "proto_cos_max": 0.0,
            }

        did_idx = self._expanded_domain_indices(feat.shape[0], feat.device)
        if did_idx is None:
            return torch.ones((feat.shape[0], feat.shape[-1]), device=feat.device,
                              dtype=feat.dtype)
        if self.token_score_pool == "meanmax":
            pooled_raw = 0.5 * (feat.mean(dim=1) + feat.amax(dim=1))
        else:
            pooled_raw = feat.mean(dim=1)               # (N, C)
        layout_idx = self._expanded_layout_indices(feat.shape[0], feat.device)
        pooled, layout_resid_frac, layout_seen_count = self._layout_residual(
            pooled_raw, layout_idx, collect_metrics=do_diag)

        if self.hard_domain_mode == "random":
            u_raw = torch.randn_like(pooled)
        elif self.hard_domain_mode == "dgcd":
            u_raw = self.hard_dgcd.score(pooled.detach(), did_idx)
        else:
            # 只读域头权重形成通道分数；域损失在 _compute_hard_aux 中另算。
            u_raw = self.domain_head.weight[did_idx] * pooled.detach()
        u = (u_raw - u_raw.mean(dim=1, keepdim=True)) / u_raw.std(
            dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        # 可选：把逐样本 DgCD 分数向所属粗域原型平滑。域门控的目标是
        # “同域相近、异域可分”，而不是让同一 style 的 mask 随 instance 漂移。
        gs = float(self.hard_cfg.get("group_smooth", 0.0))
        nd = int(self.hard_num_domains)
        sums = torch.zeros((nd, u.shape[1]), device=u.device, dtype=u.dtype)
        counts = torch.zeros((nd, 1), device=u.device, dtype=u.dtype)
        sums.index_add_(0, did_idx, u)
        counts.index_add_(0, did_idx, torch.ones((u.shape[0], 1),
                                                 device=u.device, dtype=u.dtype))
        proto = (sums / counts.clamp_min(1.0)).detach()
        active = counts[:, 0] > 0
        if bool(self.hard_cfg.get("domain_center", False)):
            # 去掉所有 style 共有的公共通道成分，只保留跨域差异。
            # 用张量条件避免每个 step 都为 int(active.sum()) 同步。
            centered = proto[active] - proto[active].mean(dim=0, keepdim=True)
            proto[active] = torch.where(active.sum() >= 2, centered, proto[active])
        proto_cos_mean = proto_cos_max = 0.0
        if do_diag:
            pn = F.normalize(proto[active].float(), dim=1)
            if pn.shape[0] >= 2:
                cm = pn @ pn.t()
                tri = torch.triu(
                    torch.ones((pn.shape[0], pn.shape[0]), dtype=torch.bool,
                               device=pn.device), diagonal=1)
                pc = cm[tri]
                proto_cos_mean = float(pc.mean())
                proto_cos_max = float(pc.max())
        if gs > 0:
            u = (1.0 - gs) * u + gs * proto[did_idx]
            u = (u - u.mean(dim=1, keepdim=True)) / u.std(
                dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

        v_vec = torch.zeros(feat.shape[-1], device=feat.device, dtype=feat.dtype)
        v_raw_std = feat.new_zeros(())
        v_ready = False
        if self.action_importance is not None and self.action_importance.ready():
            ai = self.action_importance
            li = ai._l2i.get(layer)
            if li is not None:
                raw = ai.raw_value[li].to(device=feat.device, dtype=feat.dtype) \
                    if hasattr(ai, "raw_value") else ai.value[li].to(feat.device)
                v_vec = ai.value[li].to(device=feat.device, dtype=feat.dtype)
                v_vec = (v_vec - v_vec.mean()) / v_vec.std().clamp_min(1e-6)
                v_raw_std = raw.std()
                v_ready = True

        sel_mode = str(self.hard_cfg.get("selection_mode", "linear"))
        eligible_frac = u.new_ones(())
        if sel_mode == "two_stage" and v_ready:
            # 两阶段：先按域分数选候选，再在候选里排除动作关键通道。
            # 不再让 u 和 v 线性相减，避免动作项把域排序整体压平。
            c = u.shape[1]
            k_cand = max(1, min(c, int(round(
                float(self.hard_cfg.get("candidate_ratio", 0.6)) * c))))
            k_prot = max(0, min(c, int(round(
                float(self.hard_cfg.get("protect_ratio", 0.2)) * c))))
            cand_thr = torch.topk(u, k_cand, dim=1, largest=True).values[:, -1:]
            cand = u >= cand_thr
            prot_thr = (torch.topk(v_vec, k_prot, largest=True).values[-1]
                        if k_prot > 0 else torch.tensor(float("inf"), device=u.device))
            eligible = cand & (v_vec[None, :] < prot_thr)
            eligible_frac = eligible.float().mean()
            # 候选优先；候选不足 k 时再按 u 补齐（不会丢动作关键通道的硬保护）。
            scores = u + 10.0 * eligible.to(u.dtype) - 1e-3 * v_vec[None, :]
        else:
            scores = u - float(self.hard_cfg["action_lambda"]) * v_vec[None, :]
        mode = str(self.hard_cfg["mask_mode"])
        if mode == "gumbel_ste":
            mask = gumbel_ste_mask(
                scores, drop_ratio=float(self.hard_cfg["drop_ratio"]),
                temperature=float(self.hard_cfg["temperature"]),
                group_ids=did_idx,
                domain_shared=bool(self.hard_cfg.get("domain_shared", False)))
        elif mode in ("dgcd_wrs", "domaindrop_wrs"):
            mask = wrs_mask(
                scores, drop_ratio=float(self.hard_cfg["drop_ratio"]),
                recover=bool(self.hard_cfg.get("wrs_recover", True)),
                normalize=str(self.hard_cfg.get("wrs_normalize", "per_sample")))
        else:
            mask = topk_ste_mask(
                scores, drop_ratio=float(self.hard_cfg["drop_ratio"]),
                temperature=float(self.hard_cfg["temperature"]))

        # 确定性 mask 用于判断“分数本身是否真的有域选择性”；
        # 只有诊断步才需要，避免普通 step 再做一次 top-k。
        mask_det = None
        if do_diag:
            mask_det = topk_ste_mask(
                scores, drop_ratio=float(self.hard_cfg["drop_ratio"]),
                temperature=float(self.hard_cfg["temperature"]))
            iou_same, iou_diff, iou_gap = mask_iou_by_domain(mask_det, did_idx)
            iou_s_samp, iou_d_samp, iou_g_samp = mask_iou_by_domain(mask, did_idx)
            drop_same, drop_diff, drop_gap = mask_iou_by_domain(
                mask_det, did_idx, drop=True)
            drop_s_samp, drop_d_samp, drop_g_samp = mask_iou_by_domain(
                mask, did_idx, drop=True)
            cos_same, cos_diff, cos_gap = score_separation_by_domain(scores, did_idx)
            self._last_diag = {
                "step": int(self._gate_step),
                "mask_iou_same": float(iou_same),
                "mask_iou_diff": float(iou_diff),
                "mask_iou_gap": float(iou_gap),
                "mask_iou_sample_same": float(iou_s_samp),
                "mask_iou_sample_diff": float(iou_d_samp),
                "mask_iou_sample_gap": float(iou_g_samp),
                "drop_iou_same": float(drop_same),
                "drop_iou_diff": float(drop_diff),
                "drop_iou_gap": float(drop_gap),
                "drop_iou_sample_same": float(drop_s_samp),
                "drop_iou_sample_diff": float(drop_d_samp),
                "drop_iou_sample_gap": float(drop_g_samp),
                "score_cos_same": float(cos_same),
                "score_cos_diff": float(cos_diff),
                "score_cos_gap": float(cos_gap),
                "proto_cos_mean": float(proto_cos_mean),
                "proto_cos_max": float(proto_cos_max),
            }

        self.hard_masks[layer] = mask.detach()
        self._current_mask_info = {
            "layer": int(layer), "did_idx": did_idx.detach(),
            "mask": mask.detach(),
            "mask_det": mask_det.detach() if mask_det is not None else None,
            "scores": scores.detach(), "u": u.detach(), "v": v_vec.detach(),
            "v_raw_std_t": v_raw_std, "v_ready": bool(v_ready),
            "eligible_frac_t": eligible_frac,
            "layout_resid_frac": float(layout_resid_frac),
            "layout_seen_count": int(layout_seen_count),
            "diagnostics_every": int(every),
            "diagnostics_done": bool(do_diag),
            "diag": dict(self._last_diag),
        }
        return mask

    @torch.no_grad()
    def _probe_gate_effect(self, batch):
        """同 batch、同噪声/时间步，比较 gate on/off 的扩散动作 MSE。"""
        old_mask = self.hard_controller.mask
        old_build = self.hard_controller.build_enabled
        old_dom = self._current_domain_ids
        old_layout = self._current_layout_ids
        old_stats_update = self._stats_update_enabled
        old_info = self._current_mask_info
        old_collect = self._collect
        old_suppress = getattr(self, "_diag_suppressed", False)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed = 1234567 + int(self._gate_step)
        try:
            self._current_domain_ids = batch.get("domain_id", None)
            self._current_layout_ids = batch.get("layout_id", None)
            self._stats_update_enabled = False
            # 主训练前向已经记录过本 step 的诊断；probe 的额外两次
            # forward 不再重复 O(N²) 统计。
            self._diag_suppressed = True
            self._hard_probe_metrics = {}
            for name, build in (("on", True), ("off", False)):
                self._collect = {}
                self._current_mask_info = {}
                self.hard_controller.set_mask(None)
                self.hard_controller.build_enabled = build
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                loss = self._base_loss(batch)
                self._hard_probe_metrics[f"gate_{name}_mse"] = float(loss.detach())
            on = self._hard_probe_metrics["gate_on_mse"]
            off = self._hard_probe_metrics["gate_off_mse"]
            self._hard_probe_metrics["gate_delta"] = on - off
            self._hard_probe_metrics["gate_rel_delta"] = (on - off) / max(abs(off), 1e-8)
        finally:
            self.hard_controller.mask = old_mask
            self.hard_controller.build_enabled = old_build
            self._current_domain_ids = old_dom
            self._current_layout_ids = old_layout
            self._current_mask_info = old_info
            self._stats_update_enabled = old_stats_update
            self._collect = old_collect
            self._diag_suppressed = old_suppress
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _compute_hard_aux(self, batch, dp_loss):
        """域分类器消融项的损失；DgCD 主路径没有额外监督 loss。"""
        did = batch.get("domain_id", None)
        info = self._current_mask_info
        if did is None:
            log.warning("hard gate: batch 没有 domain_id，跳过域监督")
            return torch.zeros((), device=dp_loss.device, dtype=dp_loss.dtype)
        if not info:
            return torch.zeros((), device=dp_loss.device, dtype=dp_loss.dtype)
        layer = int(info["layer"])
        live = [f for l, f in self._collect.get("pre_gate_live", []) if l == layer]
        feat_live = live[-1] if live else None
        did_idx = info["did_idx"]
        domain_logits = None
        domain_loss = torch.zeros((), device=dp_loss.device, dtype=dp_loss.dtype)
        if self.hard_domain_mode == "classifier" and feat_live is not None:
            pooled = feat_live.mean(dim=1)
            domain_logits = self.domain_head(grad_reverse(
                pooled, float(self.hard_cfg["domain_grl"])))
            domain_loss = F.cross_entropy(domain_logits, did_idx)

        # 旧的 feature-mean MSE consistency 已移除；预测级一致性在具备双分支
        # 预算后再接入。这里显式返回 0，避免把错误定义重新打开。
        cons = torch.zeros((), device=dp_loss.device, dtype=dp_loss.dtype)
        aux = float(self.hard_cfg["domain_lambda"]) * domain_loss \
            + float(self.hard_cfg["consistency"]) * cons

        mask = info["mask"]
        step_now = self._gate_step
        self._gate_step += 1
        # 指标只在 step 0 / 20 的倍数记录。普通 step 不做任何 GPU→CPU
        # 回读，避免把训练循环拖成“等 Python 读标量”。
        should_log = (step_now == 0 or step_now % 20 == 0
                      or step_now % 50 == 0)
        if should_log:
            keep_frac = float((mask > 0).float().mean())
            ai_summary = (self.action_importance.summary()
                          if self.action_importance is not None else {})
            diag = info.get("diag", {})
            self._last_gate_metrics = {
                "step": int(step_now), "mode": "hard", "layer": layer,
                "drop_ratio": float(self.hard_cfg["drop_ratio"]),
                "mask_keep_fraction": keep_frac,
                "suppression": float(1.0 - keep_frac),
                "domain_loss": float(domain_loss.detach()),
                "domain_acc": (float((domain_logits.argmax(1) == did_idx).float().mean())
                               if domain_logits is not None else 0.0),
                "u_std": float(info["u"].std()),
                "v_ready": bool(info["v_ready"]),
                "v_std": float(info["v"].std()),
                "v_raw_std": float(info["v_raw_std_t"]),
                "v_raw_mean": float(ai_summary.get("raw_mean", 0.0)),
                "v_raw_rel_std": float(ai_summary.get("raw_rel_std", 0.0)),
                "selection_mode": str(self.hard_cfg.get("selection_mode", "linear")),
                "eligible_frac": float(info["eligible_frac_t"]),
                "layout_resid_frac": float(info.get("layout_resid_frac", 0.0)),
                "layout_seen_count": int(info.get("layout_seen_count", 0)),
                "consistency": float(cons.detach()),
                "mask_mode": str(self.hard_cfg["mask_mode"]),
                "diagnostics_every": int(info.get("diagnostics_every", 0)),
                "diagnostics_done": bool(info.get("diagnostics_done", False)),
                "diagnostics_step": int(diag.get("step", -1)),
                "mask_iou_same": float(diag.get("mask_iou_same", 0.0)),
                "mask_iou_diff": float(diag.get("mask_iou_diff", 0.0)),
                "mask_iou_gap": float(diag.get("mask_iou_gap", 0.0)),
                "mask_iou_sample_same": float(diag.get("mask_iou_sample_same", 0.0)),
                "mask_iou_sample_diff": float(diag.get("mask_iou_sample_diff", 0.0)),
                "mask_iou_sample_gap": float(diag.get("mask_iou_sample_gap", 0.0)),
                "drop_iou_same": float(diag.get("drop_iou_same", 0.0)),
                "drop_iou_diff": float(diag.get("drop_iou_diff", 0.0)),
                "drop_iou_gap": float(diag.get("drop_iou_gap", 0.0)),
                "drop_iou_sample_same": float(diag.get("drop_iou_sample_same", 0.0)),
                "drop_iou_sample_diff": float(diag.get("drop_iou_sample_diff", 0.0)),
                "drop_iou_sample_gap": float(diag.get("drop_iou_sample_gap", 0.0)),
                "score_cos_same": float(diag.get("score_cos_same", 0.0)),
                "score_cos_diff": float(diag.get("score_cos_diff", 0.0)),
                "score_cos_gap": float(diag.get("score_cos_gap", 0.0)),
                "proto_cos_mean": float(diag.get("proto_cos_mean", 0.0)),
                "proto_cos_max": float(diag.get("proto_cos_max", 0.0)),
                "dp_loss": float(dp_loss.detach()),
                "aux_loss": float(aux.detach()),
            }
        if step_now % 50 == 0:
            self._probe_gate_effect(batch)
            if self._last_gate_metrics:
                self._last_gate_metrics.update(
                    getattr(self, "_hard_probe_metrics", {}))
        if should_log:
            log.info("GATE_METRICS %s",
                     json.dumps(self._last_gate_metrics, sort_keys=True))
        return aux

    def predict_action(self, obs_dict):
        """Override only for token-conditioned resampler; scalar path uses base."""
        if self.tokens_per_obs == 1:
            return super().predict_action(obs_dict)
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, _ = value.shape[:2]
        To = int(self.n_obs_steps)
        T = int(self.horizon)
        Da = int(self.action_dim)
        this_nobs = {
            k: v[:, :To, ...].reshape(-1, *v.shape[2:])
            for k, v in nobs.items()
        }
        feat = self.obs_encoder(this_nobs)
        cond = feat.reshape(B, To * self.tokens_per_obs, -1)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            cond_data, cond_mask, cond=cond, **self.kwargs)
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    # ---------- ★ 动作头：DDPM 或 rectified-flow matching ----------
    def _base_loss(self, batch):
        if self.flow_matching:
            return self._flow_loss(batch)
        return super().compute_loss(batch)

    def _flow_loss(self, batch):
        """Rectified flow matching loss: predict velocity x1-x0."""
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B = int(nactions.shape[0])
        To = int(self.n_obs_steps)
        this_nobs = {
            k: v[:, :To, ...].reshape(-1, *v.shape[2:])
            for k, v in nobs.items()
        }
        feat = self.obs_encoder(this_nobs)
        cond = feat.reshape(B, To * self.tokens_per_obs, -1)
        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            trajectory = nactions[:, start:start + self.n_action_steps]
        x0 = torch.randn_like(trajectory)
        t = torch.rand((B, 1, 1), device=trajectory.device, dtype=trajectory.dtype)
        xt = (1.0 - t) * x0 + t * trajectory
        target = trajectory - x0
        ts = t.reshape(B) * self.flow_time_scale
        pred = self.model(xt, ts, cond)
        return F.mse_loss(pred, target)

    def conditional_sample(self, condition_data, condition_mask,
                           cond=None, generator=None, **kwargs):
        if not self.flow_matching:
            return super().conditional_sample(
                condition_data, condition_mask, cond=cond,
                generator=generator, **kwargs)
        B, T, Da = condition_data.shape
        x = torch.randn(condition_data.shape, dtype=condition_data.dtype,
                        device=condition_data.device, generator=generator)
        steps = max(1, int(self.num_inference_steps))
        for i in range(steps):
            t = torch.full((B,), i / float(steps),
                           dtype=x.dtype, device=x.device)
            v = self.model(x, t * self.flow_time_scale, cond)
            x = x + v / float(steps)
            x[condition_mask] = condition_data[condition_mask]
        return x

    # ---------- ★ 在官方 compute_loss 之上追加 L_gate ----------
    def compute_loss(self, batch):
        # 验证/导出在 no_grad 下运行：只算 DP loss，不生成 mask、不更新统计、
        # 不推进 gate_step，也不写 GATE_METRICS。
        if not torch.is_grad_enabled():
            return self._base_loss(batch)
        clean_every = (int(self.hard_cfg.get("clean_importance_every", 0))
                       if hasattr(self, "hard_cfg") else 0)
        # 1) 仅在关闭 clean-forward 时，才回退到上一轮梯度估计 Taylor v。
        if (self.enable_gating and self.use_action_importance
                and self.action_importance_impl == "taylor" and clean_every <= 0):
            if self._prev_post_gate:
                self.action_importance.update_from_tensors(dict(self._prev_post_gate))
                self._prev_post_gate = []
        # 2) 本轮前向。hard mask 由 hook 用当前 batch 特征现场生成。
        self._collect.clear()
        self._last_cond_features = None
        if self.enable_gating and self.gate_mode == "hard":
            self.hard_controller.select_layer()
            self.hard_controller.set_mask(None)
            self.hard_controller.build_enabled = True
            self._current_domain_ids = batch.get("domain_id", None)
            self._current_layout_ids = batch.get("layout_id", None)
            self._current_mask_info = {}
        loss = self._base_loss(batch)
        if not self.enable_gating:
            return loss
        if self.gate_mode == "hard":
            aux = self._compute_hard_aux(batch, loss)
            self._maybe_clean_action_importance(batch)
            return loss + aux

        # 3) 用门控前激活更新跨域分布统计 → u^disc
        did = batch.get("domain_id", None)
        if did is None:
            log.warning("batch 里没有 domain_id，跳过域统计更新（门控正则仍会应用，"
                        "但 u^disc 会保持为 0）")
        else:
            # 观测张量在 compute_loss 里被 reshape 成 (B*To, ...)，
            # 编码器又把视角维并进去 → (B*To*V)，域标签必须按同样顺序展开：
            #   flatten 索引 = (b * To + t) * V + v  ⇒  每个样本连续重复 To*V 次
            n_views = len(self.obs_encoder.rgb_keys)
            did_exp = did.repeat_interleave(int(self.n_obs_steps) * n_views)
            pre = self._collect.get("pre_gate", [])
            if pre:
                n_actual = pre[0][1].shape[0]
                if did_exp.shape[0] != n_actual:      # 兜底：按实际长度对齐
                    if did_exp.shape[0] < n_actual:
                        did_exp = did_exp.repeat(
                            (n_actual + did_exp.shape[0] - 1) // did_exp.shape[0])
                    did_exp = did_exp[:n_actual]
            for l, f in pre:
                self.domain_stats.update(l, f, did_exp)

        # 4) 计算 L_gate（渐进抑制预算）
        u = self.domain_stats.u_disc().to(loss.device)
        v = None
        if self.action_importance is not None and self.action_importance.ready():
            v = self.action_importance.value.to(loss.device)
        step_now = self._gate_step
        rho = choose_rho(step_now, self.gate_cfg["rho_warmup_steps"],
                         self.gate_cfg["rho_max"])
        self._gate_step += 1
        parts = gate_regularizer(
            gates=self.gate_bank.gates_matrix().to(loss.device),
            u_disc=u, v_action=v, rho=rho,
            lambda_d=self.gate_cfg["lambda_d"],
            lambda_a=self.gate_cfg["lambda_a"],
            lambda_rho=self.gate_cfg["lambda_rho"])
        self._last_gate_parts = parts

        # 4b) 诊断日志：每 50 步记录门控/分项/梯度尺度
        if self.enable_gating and step_now % 50 == 0:
            g = self.gate_bank.gates_matrix().detach()
            u_mean = float(u.mean())
            u_std = float(u.std())
            v_mean = float(v.mean()) if v is not None else 0.0
            v_std = float(v.std()) if v is not None else 0.0
            dp_g = reg_g = None
            try:
                g_live = self.gate_bank.gates_matrix()
                dp_g = torch.autograd.grad(loss, g_live, retain_graph=True,
                                           allow_unused=True)[0]
                reg_g = torch.autograd.grad(parts["loss"], g_live, retain_graph=True,
                                            allow_unused=True)[0]
            except Exception as exc:
                log.debug("gate grad diagnostics unavailable: %s", exc)
            self._last_gate_metrics = {
                "step": int(step_now),
                "rho": float(rho),
                "gate_mean": float(g.mean()),
                "gate_min": float(g.min()),
                "gate_max": float(g.max()),
                "suppression": float(1.0 - g.mean()),
                "domain_term": float(parts["domain_term"]),
                "action_term": float(parts["action_term"]),
                "budget_term": float(parts["budget_term"]),
                "gate_loss": float(parts["loss"]),
                "u_disc_mean": u_mean,
                "u_disc_std": u_std,
                "v_action_mean": v_mean,
                "v_action_std": v_std,
                "dp_grad_norm": float(dp_g.norm()) if dp_g is not None else 0.0,
                "reg_grad_norm": float(reg_g.norm()) if reg_g is not None else 0.0,
            }
            log.info("GATE_METRICS %s", json.dumps(self._last_gate_metrics, sort_keys=True))

        # 5) 可选：Deep CORAL（移植自 DomainBed，MIT）
        coral_term = None
        if self.lambda_coral > 0 and did is not None:
            cond_feats = self._last_cond_features
            if cond_feats is not None:
                coral_term = coral_from_domains(cond_feats, did)
                loss = loss + self.lambda_coral * coral_term
        self._last_coral = coral_term

        # 6) 保留本轮的门控后激活，供下一轮读梯度
        if self.use_action_importance:
            self._prev_post_gate = list(self._collect.get("post_gate_live", []))

        return loss + parts["loss"]
