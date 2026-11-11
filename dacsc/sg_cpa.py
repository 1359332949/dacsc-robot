"""SG-CPA: Identity-Anchored Counterfactual Consensus + A-nullspace projected update.

Frozen in work order 4.1:
  K = 4 views (identity + 3 independent Gaussian-noise views, sigma = 0.05)
  aggregation: anchored mean  A_alpha = A_id + alpha * (mean(A_1..A_3) - A_id), alpha = 0.1
  loss: direct action-block distillation (normalised action space)
  parameterisation: K-projection singular-value scaling beta (SaltKeyAdapter)
  update: projected SGD  dbeta = -eta * P_null g
  J_A is evaluated on the ORIGINAL identity input write z(o_t, beta)  (not on an augmented view)

All images entering the policy must be ImageNet-normalised (as produced by
corp.preprocessing.prepare_observation); visual transforms are applied in pixel
space [0,1] and re-normalised (the 4.0 bug).
"""

from __future__ import annotations

import torch
from torch import nn

from dacsc.leftover_tta import sample_with_grad
from dacsc.salt_tta import SaltKeyAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mean_std(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def to_pixel(images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in images.items():
        mean, std = _mean_std(v.device, v.dtype)
        out[k] = (v * std + mean).clamp(0.0, 1.0)
    return out


def from_pixel(images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in images.items():
        mean, std = _mean_std(v.device, v.dtype)
        out[k] = (v - mean) / std
    return out


def apply_pixel_transform(images: dict[str, torch.Tensor], fn) -> dict[str, torch.Tensor]:
    return from_pixel(fn(to_pixel(images)))


def noise_views(images: dict[str, torch.Tensor], sigma: float, seeds: list[int],
                view_names: list[str]) -> list[dict[str, torch.Tensor]]:
    """sigma is the std in pixel space [0,1]; each camera gets its own generator."""
    views = []
    for idx, seed in enumerate(seeds):
        out = {}
        for ci, name in enumerate(view_names):
            v = images[name]
            g = torch.Generator(device=v.device).manual_seed(int(seed) + ci * 1000)
            out[name] = (v + sigma * torch.randn(v.shape, generator=g, device=v.device)).clamp(0, 1)
        views.append(out)
    return [from_pixel(v) for v in views]


class SGCpa:
    def __init__(self, policy, a_basis: torch.Tensor, *, alpha: float = 0.1,
                 sigma: float = 0.05, eta: float = 0.05, seed: int = 1234,
                 eta_param: float = 0.15, a_change_tol: float = 5e-3,
                 max_backtracks: int = 3):
        for p in policy.parameters():
            p.requires_grad_(False)
        self.policy = policy
        self.adapter = SaltKeyAdapter(policy.action_generator.blocks[0].attention.qkv,
                                      int(policy.config["hidden_dim"]), eta=eta_param)
        self.a_basis = a_basis.to(dtype=torch.float64)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.seed = int(seed)
        self.a_change_tol = float(a_change_tol)
        self.max_backtracks = int(max_backtracks)
        self.optimizer = torch.optim.SGD([self.adapter.beta], lr=1.0)  # step applied manually
        self.diagnostics: list[dict] = []

    # ------------------------------------------------------------------ utils
    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset()
        self.optimizer = torch.optim.SGD([self.adapter.beta], lr=1.0)
        if seed is not None:
            self.seed = int(seed)
        self.diagnostics = []

    def close(self) -> None:
        self.adapter.close()

    def _sample(self, rep, seed: int, grad: bool) -> torch.Tensor:
        torch.manual_seed(int(seed))
        if grad:
            return sample_with_grad(self.policy.action_generator, rep,
                                    self.policy.config["action_generator"]["num_sample_steps"])
        with torch.no_grad():
            return self.policy.action_generator.sample(rep)

    def _rep(self, images, state, instruction):
        return self.policy.representation_extractor(images, state, instruction)

    def build_target(self, prepared: dict, seed: int | None = None) -> dict:
        """Frozen teacher: identity action + 3 independent-noise views -> anchored target."""
        seed = self.seed if seed is None else int(seed)
        names = list(self.policy.config["view_names"])
        # teacher must be the FROZEN policy (beta = 0); restore beta afterwards
        with torch.no_grad():
            saved = self.adapter.beta.detach().clone()
            self.adapter.beta.zero_()
        try:
            base_rep = self._rep(prepared["images"], prepared["state"], prepared["instruction"])
            a_id = self._sample(base_rep, seed, grad=False)
            views = noise_views(prepared["images"], self.sigma,
                                [seed + 11, seed + 22, seed + 33], names)
            aug = []
            for v in views:
                with torch.no_grad():
                    rep = self._rep(v, prepared["state"], prepared["instruction"])
                    aug.append(self._sample(rep, seed, grad=False))
        finally:
            with torch.no_grad():
                self.adapter.beta.copy_(saved)
        aug_mean = torch.stack(aug).mean(dim=0)
        target = a_id + self.alpha * (aug_mean - a_id)
        return {"target": target.detach(), "a_id": a_id, "views": views,
                "aug": aug, "aug_mean": aug_mean.detach()}

    def _student_actions(self, prepared: dict, target_info: dict, seed: int) -> list[torch.Tensor]:
        acts = []
        for v in target_info["views"]:
            rep = self._rep(v, prepared["state"], prepared["instruction"])
            acts.append(self._sample(rep, seed, grad=True))
        return acts

    def loss(self, prepared: dict, target_info: dict, seed: int | None = None) -> torch.Tensor:
        seed = self.seed if seed is None else int(seed)
        acts = self._student_actions(prepared, target_info, seed)
        target = target_info["target"]
        return sum(((a - target) ** 2).mean() for a in acts) / len(acts)

    # ------------------------------------------------------------ Jacobian
    class _ZHook:
        def __init__(self, attention, steps, horizon):
            self.a, self.o = attention, attention.attend
            self.steps, self.h = steps, horizon
            self.step, self.z = 0, None
            attention.attend = self._f

        def _f(self, *a, **k):
            out = self.o(*a, **k)
            if out.shape[1] != self.h:
                self.step = 0
                return out
            self.step += 1
            if self.step == self.steps:
                self.z = out.reshape(-1, out.shape[-1]).mean(dim=0)
            return out

        def close(self):
            self.a.attend = self.o

    def write(self, prepared: dict, seed: int | None = None) -> torch.Tensor:
        """Differentiable identity-input write z(o_t, beta)."""
        seed = self.seed if seed is None else int(seed)
        rep = self._rep(prepared["images"], prepared["state"], prepared["instruction"])
        hook = self._ZHook(self.policy.action_generator.blocks[0].attention,
                           int(self.policy.config["action_generator"]["num_sample_steps"]),
                           int(self.policy.config["action_horizon"]))
        try:
            torch.manual_seed(int(seed))
            sample_with_grad(self.policy.action_generator, rep,
                             self.policy.config["action_generator"]["num_sample_steps"])
            z = hook.z
        finally:
            hook.close()
        return z

    def jacobian_a(self, prepared: dict, seed: int | None = None) -> torch.Tensor:
        z = self.write(prepared, seed)
        rows = []
        for i in range(self.a_basis.shape[1]):
            gi = torch.autograd.grad(z.double() @ self.a_basis[:, i].to(z.device).double(),
                                     self.adapter.beta,
                                     retain_graph=(i < self.a_basis.shape[1] - 1))[0]
            rows.append(gi.detach().cpu())
        return torch.stack(rows).double()

    @staticmethod
    def nullspace(j_a: torch.Tensor) -> torch.Tensor:
        d = j_a.shape[1]
        return torch.eye(d, dtype=torch.float64) - j_a.T @ torch.linalg.pinv(j_a @ j_a.T) @ j_a

    # --------------------------------------------------------------- update
    def step(self, prepared: dict, seed: int | None = None) -> dict:
        seed = self.seed if seed is None else int(seed)
        info = self.build_target(prepared, seed=seed)
        z0 = self.write(prepared, seed).detach()
        loss = self.loss(prepared, info, seed=seed)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = self.adapter.beta.grad.detach().double().cpu()
        if float(g.norm()) == 0.0:
            return {"skipped": True, "loss": float(loss.detach())}
        self.optimizer.zero_grad(set_to_none=True)
        j_a = self.jacobian_a(prepared, seed=seed)
        p_null = self.nullspace(j_a)
        d_beta = -(self.eta * (p_null @ g)).float().to(self.adapter.beta.device)
        with torch.no_grad():
            before = self.adapter.beta.detach().clone()
        accepted = False
        a_change = None
        for attempt in range(self.max_backtracks + 1):
            with torch.no_grad():
                self.adapter.beta.copy_(before + d_beta)
            z1 = self.write(prepared, seed).detach().double().cpu()
            a_change = float((self.a_basis.T @ (z1 - z0.double().cpu())).norm())
            if a_change <= self.a_change_tol:
                accepted = True
                break
            d_beta = d_beta * 0.5
        if not accepted:
            with torch.no_grad():
                self.adapter.beta.copy_(before)
        rec = {"skipped": False, "loss": float(loss.detach()),
               "g_norm": float(g.norm()), "d_beta_norm": float(d_beta.norm()),
               "a_change": a_change, "accepted": accepted,
               "backtracks": attempt, "beta_norm": float(self.adapter.beta.detach().norm()),
               "retained_fraction": float((p_null @ g).norm() / (g.norm() + 1e-12))}
        self.diagnostics.append(rec)
        return rec
