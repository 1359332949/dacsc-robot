"""Official ResNet18ConvFiLM policy with faithful CNN channel dropout."""
from __future__ import annotations

import json
import logging

import torch

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from dacsc.cnn_dropout import install_cnn_dropout

log = logging.getLogger(__name__)


class CNNChannelDropoutPolicy(DiffusionTransformerHybridImagePolicy):
    """Adds PLACE / DomainDrop / DgCD hooks to the official CNN encoder."""

    def __init__(self,
                 *,
                 cnn_dropout_mode: str = "none",
                 cnn_layer_selection: str = "fixed",
                 cnn_layer_indices: list | None = None,
                 cnn_drop_ratio: float = 0.33,
                 cnn_recover: bool = True,
                 cnn_progressive: bool = False,
                 cnn_velocity: float = 50.0,
                 cnn_num_domains: int = 20,
                 cnn_domain_key: str = "style",
                 cnn_domain_id_min: int = 11,
                 cnn_domain_group_size: int = 1,
                 cnn_domain_lambda: float = 1.0,
                 cnn_domain_grl: float = 0.25,
                 cnn_domain_head_lr: float = 1.0e-3,
                 cnn_grl_warmup_steps: int = 10,
                 cnn_layer_wise_prob: float = 1.0,
                 cnn_domain_head_norm: bool = False,
                 cnn_freeze_backbone: bool = False,
                 cnn_diagnostics_every: int = 20,
                 cnn_train_only: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.cnn_dropout_mode = str(cnn_dropout_mode)
        self.cnn_num_domains = int(cnn_num_domains)
        self.cnn_domain_key = str(cnn_domain_key)
        self.cnn_domain_id_min = int(cnn_domain_id_min)
        self.cnn_domain_group_size = max(int(cnn_domain_group_size), 1)
        self.cnn_diagnostics_every = int(cnn_diagnostics_every)
        self.cnn_domain_head_lr = float(cnn_domain_head_lr)
        self._cnn_step = 0
        self._last_cnn_metrics = {}
        self._last_dp_loss_t = None
        self._last_aux_loss_t = None
        self.cnn_freeze_backbone = bool(cnn_freeze_backbone)
        self.cnn_controllers = install_cnn_dropout(
            self.obs_encoder,
            mode=self.cnn_dropout_mode,
            layer_selection=cnn_layer_selection,
            layer_indices=cnn_layer_indices,
            drop_ratio=cnn_drop_ratio,
            recover=cnn_recover,
            progressive=cnn_progressive,
            velocity=cnn_velocity,
            num_domains=self.cnn_num_domains,
            domain_lambda=cnn_domain_lambda,
            domain_grl=cnn_domain_grl,
            grl_warmup_steps=cnn_grl_warmup_steps,
            layer_wise_prob=cnn_layer_wise_prob,
            domain_head_norm=cnn_domain_head_norm,
            train_only=cnn_train_only,
        )
        if self.cnn_freeze_backbone:
            for p in self.obs_encoder.parameters():
                p.requires_grad_(False)
            for controller in self.cnn_controllers:
                heads = getattr(controller, "domain_heads", None)
                if heads is not None:
                    for p in heads.parameters():
                        p.requires_grad_(True)
        log.info("CNN dropout: mode=%s views=%d layers=%s selection=%s "
                 "ratio=%.3f progressive=%s",
                 self.cnn_dropout_mode, len(self.cnn_controllers),
                 cnn_layer_indices, cnn_layer_selection, cnn_drop_ratio,
                 cnn_progressive)

    def get_optimizer(self, transformer_weight_decay, obs_encoder_weight_decay,
                      learning_rate, betas):
        """Keep the official groups but give the DomainDrop heads their own LR."""
        domain_params = []
        for controller in self.cnn_controllers:
            heads = getattr(controller, "domain_heads", None)
            if heads is not None:
                domain_params.extend(list(heads.parameters()))
        domain_ids = {id(p) for p in domain_params}
        obs_params = [p for p in self.obs_encoder.parameters()
                      if id(p) not in domain_ids]
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({"params": obs_params,
                             "weight_decay": obs_encoder_weight_decay})
        if domain_params:
            optim_groups.append({"params": domain_params,
                                 "weight_decay": 0.0,
                                 "lr": self.cnn_domain_head_lr})
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        if not torch.is_grad_enabled():
            return super().compute_loss(batch)

        if self.cnn_domain_key == "layout":
            did = batch.get("layout_id", None)
        elif self.cnn_domain_key == "task":
            did = batch.get("task_id", None)
        else:
            did = batch.get("domain_id", None)
        did_idx = None
        if did is not None:
            did = did.repeat_interleave(int(self.n_obs_steps))
            did_idx = ((did - self.cnn_domain_id_min)
                       // self.cnn_domain_group_size).clamp(
                           0, self.cnn_num_domains - 1)

        for controller in self.cnn_controllers:
            controller.reset()
            controller.set_context(did_idx, self._cnn_step)

        loss = super().compute_loss(batch)
        self._last_dp_loss_t = loss.detach()
        aux = [c.aux_loss() for c in self.cnn_controllers]
        aux = [a for a in aux if a is not None]
        aux_mean = torch.stack(aux).mean() if aux else None
        self._last_aux_loss_t = (aux_mean.detach() if aux_mean is not None else None)
        if aux_mean is not None:
            loss = loss + aux_mean

        self._cnn_step += 1
        every = self.cnn_diagnostics_every
        if every > 0 and self._cnn_step % every == 0:
            metrics = {"step": self._cnn_step, "mode": self.cnn_dropout_mode,
                       "dp_loss": float(self._last_dp_loss_t),
                       "aux_loss": (float(self._last_aux_loss_t)
                                    if self._last_aux_loss_t is not None else 0.0)}
            for i, controller in enumerate(self.cnn_controllers):
                for k, v in controller.summary().items():
                    metrics[f"view{i}_{k}"] = v
            self._last_cnn_metrics = metrics
            log.info("CNN_DROPOUT %s", json.dumps(metrics, sort_keys=True))
        return loss

    def cnn_report(self):
        return {"mode": self.cnn_dropout_mode,
                "step": self._cnn_step,
                "metrics": dict(self._last_cnn_metrics)}
