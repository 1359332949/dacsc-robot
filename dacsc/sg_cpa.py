"""SG-CPA: Identity-Anchored Counterfactual Consensus + A-nullspace projected update.

Final 4.0 protocol (revised after review):
  K = 4 views: Identity, mild Gaussian noise (sigma=0.05), mild Gaussian blur
  (k=3, sigma=0.5), photometric (brightness/contrast/saturation 1.05), applied
  in pixel space [0,1] then re-normalised.
  target = Medoid of the K teacher action blocks (shared action noise).  The
  anchored-mean variant (A_id + alpha*(mean - A_id)) is kept as an ablation.
  loss = mean_k || student_k(beta, same shared noise) - sg(target) ||^2
  update = projected SGD  dbeta = -eta * P_null g,  P_null = I - V_r V_r^T
  J_A is evaluated on the ORIGINAL identity input write z(o_t, beta) (r=5 VJPs).
  Beta is bounded by |beta| <= adapter.eta via a global step shrink; a
  finite-step check compares A^T z against the beta=0 write of the SAME
  observation and halves / rolls back the step when needed.

All images entering the policy must be ImageNet-normalised (as produced by
corp.preprocessing.prepare_observation); visual transforms run in pixel space
(the 4.0 bug was transforming normalised tensors directly).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dacsc.leftover_tta import sample_with_grad
from dacsc.salt_tta import SaltKeyAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_VIEWS = ("identity", "noise", "blur", "photo")


def _mean_std(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def to_pixel(images):
    out = {}
    for k, v in images.items():
        mean, std = _mean_std(v.device, v.dtype)
        out[k] = (v * std + mean).clamp(0.0, 1.0)
    return out


def from_pixel(images):
    out = {}
    for k, v in images.items():
        mean, std = _mean_std(v.device, v.dtype)
        out[k] = (v - mean) / std
    return out


def apply_pixel_transform(images, fn):
    return from_pixel(fn(to_pixel(images)))


def _noise_fn(sigma, seed):
    def fn(images):  # images are pixel-space [0,1], one generator per camera
        out = {}
        for ci, (k, v) in enumerate(images.items()):
            g = torch.Generator(device=v.device).manual_seed(int(seed) + ci * 1000)
            out[k] = (v + sigma * torch.randn(v.shape, generator=g, device=v.device)).clamp(0, 1)
        return out
    return fn


def _blur_fn(kernel, sigma):
    return lambda images: {k: TF.gaussian_blur(v, kernel_size=[kernel, kernel],
                                               sigma=[sigma, sigma]) for k, v in images.items()}


def _photo_fn(factor):
    def fn(images):
        out = {}
        for k, v in images.items():
            x = torch.clamp(v * factor, 0, 1)
            mean = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = torch.clamp((x - mean) * factor + mean, 0, 1)
            out[k] = TF.adjust_saturation(x, factor)
        return out
    return fn


def build_views(seed: int, sigma: float = 0.05, blur_sigma: float = 0.5,
                photo: float = 1.05):
    """Return [(name, pixel-space transform fn)] with the identity first."""
    return [
        ("identity", lambda im: im),
        ("noise", _noise_fn(sigma, seed + 11)),
        ("blur", _blur_fn(3, blur_sigma)),
        ("photo", _photo_fn(photo)),
    ]


def action_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute distance in normalised action space (7 dims)."""
    return float((a - b).abs().mean())


def group_distances(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Group decomposition: xyz / rotation / gripper (gripper native +/-1 units)."""
    d = (a - b).abs()
    return {"all": float(d.mean()),
            "xyz": float(d[..., :3].mean()),
            "rot": float(d[..., 3:6].mean()),
            "grip": float(d[..., 6].mean())}


def medoid_index(actions: list[torch.Tensor]) -> int:
    K = len(actions)
    return int(torch.argmin(torch.tensor(
        [sum(action_distance(actions[i], actions[j]) for j in range(K) if j != i)
         for i in range(K)])))


class SGCpa:
    def __init__(self, policy, a_basis: torch.Tensor, *, aggregation: str = "medoid",
                 alpha: float = 0.1, eta: float = 0.05, eta_param: float = 0.15,
                 seed: int = 1234, a_change_tol: float = 5e-3,
                 max_backtracks: int = 3, views: list | None = None,
                 sigma: float = 0.05, blur_sigma: float = 0.5, photo: float = 1.05,
                 use_projection: bool = True, medoid_include_identity: bool = True):
        for p in policy.parameters():
            p.requires_grad_(False)
        self.policy = policy
        self.adapter = SaltKeyAdapter(policy.action_generator.blocks[0].attention.qkv,
                                      int(policy.config["hidden_dim"]), eta=eta_param)
        self.a_basis = a_basis.to(dtype=torch.float64)
        self.aggregation = aggregation
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.seed = int(seed)
        self.a_change_tol = float(a_change_tol)
        self.max_backtracks = int(max_backtracks)
        self.use_projection = bool(use_projection)
        self.medoid_include_identity = bool(medoid_include_identity)
        self.view_specs = views if views is not None else build_views(seed, sigma, blur_sigma, photo)
        self.optimizer = torch.optim.SGD([self.adapter.beta], lr=1.0)
        self.diagnostics: list[dict] = []

    # ------------------------------------------------------------------ basics
    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset()
        self.optimizer = torch.optim.SGD([self.adapter.beta], lr=1.0)
        if seed is not None:
            self.seed = int(seed)
        self.diagnostics = []

    def close(self) -> None:
        self.adapter.close()

    @staticmethod
    def _fork_rng():
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        return torch.random.fork_rng(devices=devices)

    def _rep(self, images, state, instruction):
        return self.policy.representation_extractor(images, state, instruction)

    def _sample(self, rep, seed: int, grad: bool):
        with self._fork_rng():
            torch.manual_seed(int(seed))
            if grad:
                return sample_with_grad(self.policy.action_generator, rep,
                                        self.policy.config["action_generator"]["num_sample_steps"])
            with torch.no_grad():
                return self.policy.action_generator.sample(rep)

    def make_view_images(self, prepared: dict, seed: int | None = None) -> list:
        seed = self.seed if seed is None else int(seed)
        images = []
        for name, fn in self.view_specs:
            images.append(apply_pixel_transform(prepared["images"], fn) if name != "identity"
                          else prepared["images"])
        return images

    # --------------------------------------------------------------- target
    def build_target(self, prepared: dict, seed: int | None = None) -> dict:
        """Frozen teacher: K views with shared action noise -> medoid / anchored mean."""
        seed = self.seed if seed is None else int(seed)
        with torch.no_grad():
            saved = self.adapter.beta.detach().clone()
            self.adapter.beta.zero_()
        try:
            views = self.make_view_images(prepared, seed)
            acts = []
            for img in views:
                rep = self._rep(img, prepared["state"], prepared["instruction"])
                acts.append(self._sample(rep, seed, grad=False))
        finally:
            with torch.no_grad():
                self.adapter.beta.copy_(saved)
        if self.aggregation == "medoid":
            if self.medoid_include_identity or len(acts) == 1:
                pool, offset = acts, 0
            else:
                pool, offset = acts[1:], 1
            k = offset + medoid_index(pool)
            target = acts[k].detach()
        elif self.aggregation == "mean_anchored":
            aug_mean = torch.stack(acts[1:]).mean(dim=0) if len(acts) > 1 else acts[0]
            target = (acts[0] + self.alpha * (aug_mean - acts[0])).detach()
            k = -1
        else:
            raise ValueError(f"unknown aggregation {self.aggregation}")
        dists = [action_distance(acts[0], a) for a in acts]
        return {"target": target, "acts": acts, "views": views, "medoid_index": k,
                "view_distances": dists,
                "groups": {name: group_distances(acts[0], a) for name, a in zip(
                    [n for n, _ in self.view_specs], acts)}}

    def loss(self, prepared: dict, target_info: dict, seed: int | None = None) -> torch.Tensor:
        seed = self.seed if seed is None else int(seed)
        total = None
        for img in target_info["views"]:
            rep = self._rep(img, prepared["state"], prepared["instruction"])
            chunk = self._sample(rep, seed, grad=True)
            term = ((chunk - target_info["target"]) ** 2).mean()
            total = term if total is None else total + term
        return total / len(target_info["views"])

    # ------------------------------------------------------------- Jacobian
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
        seed = self.seed if seed is None else int(seed)
        rep = self._rep(prepared["images"], prepared["state"], prepared["instruction"])
        hook = self._ZHook(self.policy.action_generator.blocks[0].attention,
                           int(self.policy.config["action_generator"]["num_sample_steps"]),
                           int(self.policy.config["action_horizon"]))
        try:
            with self._fork_rng():
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
    def nullspace(j_a: torch.Tensor, rank_tol: float = 1e-8) -> torch.Tensor:
        d = j_a.shape[1]
        _, s, vh = torch.linalg.svd(j_a.double(), full_matrices=False)
        if float(s.max()) == 0.0:
            return torch.eye(d, dtype=torch.float64)
        rank = int((s > rank_tol * s.max()).sum())
        v = vh[:rank]
        return torch.eye(d, dtype=torch.float64) - v.T @ v

    # --------------------------------------------------------------- update
    def step(self, prepared: dict, seed: int | None = None) -> dict:
        seed = self.seed if seed is None else int(seed)
        info = self.build_target(prepared, seed=seed)
        z_before = self.write(prepared, seed).detach()
        with torch.no_grad():
            saved_ref = self.adapter.beta.detach().clone()
            self.adapter.beta.zero_()
        try:
            z_ref_cur = self.write(prepared, seed).detach().double().cpu()
        finally:
            with torch.no_grad():
                self.adapter.beta.copy_(saved_ref)
        loss = self.loss(prepared, info, seed=seed)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = self.adapter.beta.grad.detach().double().cpu()
        if float(g.norm()) == 0.0:
            rec = {"skipped": True, "attempted": True, "accepted": False, "loss": float(loss.detach())}
            self.diagnostics.append(rec)
            return rec
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_projection:
            j_a = self.jacobian_a(prepared, seed=seed)
            p_null = self.nullspace(j_a)
            update = p_null @ g
        else:
            j_a, p_null = None, None
            update = g
        d_beta = -(self.eta * update).float().to(self.adapter.beta.device)
        with torch.no_grad():
            before = self.adapter.beta.detach().clone()
            step_scale = 1.0
            for i in range(d_beta.numel()):
                target_i = float(before[i] + d_beta[i])
                if abs(target_i) > self.adapter.eta and abs(float(d_beta[i])) > 1e-12:
                    bound = self.adapter.eta if target_i > 0 else -self.adapter.eta
                    step_scale = min(step_scale, max(0.0, (bound - float(before[i])) / float(d_beta[i])))
            if step_scale < 1.0:
                d_beta = d_beta * step_scale
        accepted = False
        local_change = cum_change = None
        for attempt in range(self.max_backtracks + 1):
            with torch.no_grad():
                self.adapter.beta.copy_(before + d_beta)
            z1 = self.write(prepared, seed).detach().double().cpu()
            local_change = float((self.a_basis.T @ (z1 - z_before.double().cpu())).norm())
            cum_change = float((self.a_basis.T @ (z1 - z_ref_cur)).norm())
            if max(local_change, cum_change) <= self.a_change_tol:
                accepted = True
                break
            d_beta = d_beta * 0.5
        if not accepted:
            with torch.no_grad():
                self.adapter.beta.copy_(before)
        rec = {"skipped": False, "attempted": True, "accepted": accepted,
               "loss": float(loss.detach()), "g_norm": float(g.norm()),
               "d_beta_norm": float(d_beta.norm()),
               "local_a_change": local_change, "cumulative_a_change": cum_change,
               "backtracks": attempt,
               "beta_norm": float(self.adapter.beta.detach().norm()),
               "beta_abs_max": float(self.adapter.beta.detach().abs().max()),
               "retained_fraction": (float((p_null @ g).norm() / (g.norm() + 1e-12))
                                    if p_null is not None else 1.0),
               "medoid_index": info["medoid_index"],
               "view_distances": info["view_distances"],
               "step_scale": step_scale}
        self.diagnostics.append(rec)
        return rec
