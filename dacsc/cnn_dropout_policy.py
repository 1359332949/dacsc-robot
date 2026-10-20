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
                 cnn_num_domains: int = 10,
                 cnn_domain_id_min: int = 11,
                 cnn_domain_group_size: int = 5,
                 cnn_domain_lambda: float = 1.0,
                 cnn_domain_grl: float = 0.25,
                 cnn_diagnostics_every: int = 20,
                 cnn_train_only: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.cnn_dropout_mode = str(cnn_dropout_mode)
        self.cnn_num_domains = int(cnn_num_domains)
        self.cnn_domain_id_min = int(cnn_domain_id_min)
        self.cnn_domain_group_size = max(int(cnn_domain_group_size), 1)
        self.cnn_diagnostics_every = int(cnn_diagnostics_every)
        self._cnn_step = 0
        self._last_cnn_metrics = {}
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
            train_only=cnn_train_only,
        )
        log.info("CNN dropout: mode=%s views=%d layers=%s selection=%s "
                 "ratio=%.3f progressive=%s",
                 self.cnn_dropout_mode, len(self.cnn_controllers),
                 cnn_layer_indices, cnn_layer_selection, cnn_drop_ratio,
                 cnn_progressive)

    def compute_loss(self, batch):
        if not torch.is_grad_enabled():
            return super().compute_loss(batch)

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
        aux = [c.aux_loss() for c in self.cnn_controllers]
        aux = [a for a in aux if a is not None]
        if aux:
            loss = loss + torch.stack(aux).mean()

        self._cnn_step += 1
        every = self.cnn_diagnostics_every
        if every > 0 and self._cnn_step % every == 0:
            metrics = {"step": self._cnn_step, "mode": self.cnn_dropout_mode}
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
