"""Checkpoint and evaluation state management, independent of datasets/CUDA."""

from __future__ import annotations

from contextlib import contextmanager
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class ModelEMA:
    """Parameter EMA delegates all averaging to PyTorch's public SWA/EMA API.

    ParameterList avoids copying forward hooks and nonpersistent donor state.
    Persistent buffers are copied, as in AveragedModel(use_buffers=False).
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        parameters = list(model.named_parameters())
        self.names = [name for name, _ in parameters]
        canonical = {id(parameter): name for name, parameter in parameters}
        self.aliases = {
            name: canonical[id(parameter)]
            for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        self.averaged = AveragedModel(
            nn.ParameterList(model.parameters()),
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
        )
        # W_0 is already the initialization; include the first update in EMA.
        self.averaged.n_averaged.fill_(1)
        self.buffers = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k not in self.aliases
        }

    @property
    def shadow(self):
        parameters = dict(
            zip(self.names, self.averaged.module.parameters(), strict=True)
        )
        return {
            **self.buffers,
            **{
                alias: parameters[name].detach() for alias, name in self.aliases.items()
            },
        }

    def update(self, model):
        if [name for name, _ in model.named_parameters()] != self.names:
            raise ValueError("EMA parameter identity changed")
        self.averaged.update_parameters(nn.ParameterList(model.parameters()))
        current = model.state_dict()
        for key in self.buffers:
            self.buffers[key].copy_(current[key].detach())

    def state_dict(self):
        return {
            "decay": self.decay,
            "names": self.names,
            "aliases": self.aliases,
            "averaged": self.averaged.state_dict(),
            "buffers": self.buffers,
        }

    def load_state_dict(self, state):
        if (
            state["decay"] != self.decay
            or state["names"] != self.names
            or state["aliases"] != self.aliases
            or set(state["buffers"]) != set(self.buffers)
        ):
            raise ValueError("EMA configuration differs from checkpoint")
        self.averaged.load_state_dict(state["averaged"], strict=True)
        for key, value in state["buffers"].items():
            self.buffers[key].copy_(value)


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def evaluation_state(model, ema=None, seed: int = 0):
    """All evaluation paths use the same weights and leave training/RNG intact."""
    rng = capture_rng()
    training = model.training
    backup = None
    try:
        if ema is not None:
            backup = {
                k: value.detach().clone() for k, value in model.state_dict().items()
            }
            model.load_state_dict(ema.shadow, strict=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.eval()
        yield
    finally:
        if backup is not None:
            model.load_state_dict(backup, strict=True)
        model.train(training)
        restore_rng(rng)


def training_state(
    model, optimizer, scheduler, ema, *, step: int, data_order: dict
) -> dict:
    return {
        "version": 2,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "rng": capture_rng(),
        "data_order": data_order,
        "runtime": model.training_runtime_state(),
    }


def restore_training_state(payload: dict, model, optimizer, scheduler, ema) -> int:
    if "training_state" not in payload:
        raise ValueError(
            "checkpoint contains inference weights only; exact resume is unavailable"
        )
    state = payload["training_state"]
    if state["version"] != 2:
        raise ValueError("unsupported training checkpoint version")
    if (state["ema"] is None) != (ema is None):
        raise ValueError("EMA setting differs from the resume checkpoint")
    model.load_state_dict(payload["raw_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema.load_state_dict(state["ema"])
    model.load_training_runtime_state(state["runtime"])
    restore_rng(state["rng"])
    return state["step"]


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
