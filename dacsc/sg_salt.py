"""SG-SALT: SALT + action-subspace (A) protection at the CoRP write interface.

    L_SG-SALT = L_SALT + lambda * || A^T (z_phi - z_0) ||_2^2

- A: orthonormal action-sensitive basis (last-Euler token-mean write space, 768 x r)
- z_0: frozen (beta=0) write at the same observation and action-noise seed
- z_phi: adapted (current beta) write under the same sampling procedure

lambda == 0 takes an explicit branch returning the SALT loss alone, so the
protected version is bitwise identical to SALT (no extra RNG consumption).
"""

from __future__ import annotations

import torch

from dacsc.leftover_tta import sample_with_grad
from dacsc.salt_tta import flow_matching_leftover_loss


class WriteHook:
    """Capture the last-Euler token-mean write (keeps the graph when needed)."""

    def __init__(self, attention, num_steps: int, horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = int(num_steps)
        self.horizon = int(horizon)
        self.step = 0
        self.value = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.num_steps:
            self.value = out.reshape(-1, out.shape[-1]).mean(dim=0)
        return out

    def close(self) -> None:
        self.attention.attend = self.original  # type: ignore[assignment]


def _sample_write(policy, representation, seed: int, with_grad: bool) -> torch.Tensor:
    generator = policy.action_generator
    hook = WriteHook(generator.blocks[0].attention,
                     int(policy.config["action_generator"]["num_sample_steps"]),
                     int(policy.config["action_horizon"]))
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if with_grad:
                sample_with_grad(generator, representation,
                                 policy.config["action_generator"]["num_sample_steps"])
            else:
                with torch.no_grad():
                    sample_with_grad(generator, representation,
                                     policy.config["action_generator"]["num_sample_steps"])
        z = hook.value
    finally:
        hook.close()
    if z is None:
        raise RuntimeError("write hook did not capture a last-Euler write")
    return z


def _beta_owner(adapter):
    """Accept either a SaltKeyAdapter or a SaltTTA (which owns `.adapter`)."""
    return adapter if hasattr(adapter, "beta") else adapter.adapter


def frozen_write(policy, adapter, representation, seed: int = 1234) -> torch.Tensor:
    """beta=0 write, detached, same sampling seed (does not disturb global RNG)."""
    owner = _beta_owner(adapter)
    backup = owner.beta.detach().clone()
    try:
        with torch.no_grad():
            owner.beta.zero_()
        return _sample_write(policy, representation, seed=seed, with_grad=False).detach()
    finally:
        with torch.no_grad():
            owner.beta.copy_(backup)


def adapted_write(policy, representation, seed: int = 1234) -> torch.Tensor:
    return _sample_write(policy, representation, seed=seed, with_grad=True)


def protection_loss(z_phi: torch.Tensor, z0: torch.Tensor, a_basis: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    dz = (z_phi - z0.to(z_phi.device)).float()
    basis = a_basis.to(device=dz.device, dtype=dz.dtype)
    proj = dz @ basis
    return proj.pow(2).sum(), proj


def sg_salt_loss(
    policy,
    adapter,
    representation,
    target_norm: torch.Tensor,
    exec_len: int,
    *,
    draws: int = 8,
    mode: str = "leftover",
    rng: torch.Generator | None = None,
    a_basis: torch.Tensor | None = None,
    lam: float = 0.0,
    protect_seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (total_loss, salt_loss_detached, penalty_detached|None).

    lam == 0 -> explicit SALT branch: total == salt exactly, no extra sampling.
    """
    if lam == 0.0 or a_basis is None:
        salt = flow_matching_leftover_loss(policy.action_generator, representation,
                                           target_norm, exec_len, draws=draws, mode=mode, rng=rng)
        return salt, salt.detach(), None
    z0 = frozen_write(policy, adapter, representation, seed=protect_seed)
    salt = flow_matching_leftover_loss(policy.action_generator, representation,
                                       target_norm, exec_len, draws=draws, mode=mode, rng=rng)
    z_phi = adapted_write(policy, representation, seed=protect_seed)
    penalty, _ = protection_loss(z_phi, z0, a_basis)
    return salt + float(lam) * penalty, salt.detach(), penalty.detach()
