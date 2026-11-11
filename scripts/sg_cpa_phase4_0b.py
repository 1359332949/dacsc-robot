"""工单 4.0b: weak single-transform consensus + identity anchoring (offline).

Reuses the 4.0 data/transform/sampling helpers from scripts.sg_cpa_phase4_0.
K = 4 (identity + 3 variants of ONE transform family); each transform tested alone.
Anchored consensus: A_alpha = A_id + alpha * (mean_{k>=2} A_k - A_id).
Reports D_view, D_anchor, D_fit(alpha), action-group decomposition (XYZ/rot/gripper),
and re-checks the projected update magnitude under the anchored-consensus gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sg_cpa_phase4_0 import (  # noqa: E402
    A_FILE, CKPT, DEV, collect_batches, sample_actions,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import SaltTTA  # noqa: E402

OUT = Path("/hy-tmp/validation/sg_cpa_4_0b")
ALPHAS = (0.0, 0.1, 0.3, 1.0)
SEEDS = (1234, 2345, 3456)

# IMPORTANT: prepare_observation() returns ImageNet-normalised images
# (corp/preprocessing.py:53-65). All visual transforms must be applied in pixel
# space [0,1], then re-normalised. Applying them on the normalised tensor (and
# clamping to [0,1]) destroys the input -- that was a bug in 4.0.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_pixel(img):
    return {k: ((v * _STD.to(v.device) + _MEAN.to(v.device)).clamp(0, 1)) for k, v in img.items()}


def from_pixel(img):
    return {k: ((v - _MEAN.to(v.device)) / _STD.to(v.device)) for k, v in img.items()}


def apply_pixel_transform(img_norm, fn):
    return from_pixel(fn(to_pixel(img_norm)))


# ---------------------------------------------------------------- weak transforms
def t_noise(img, sigma, seed):
    out = {}
    for k, v in img.items():
        g = torch.Generator(device=v.device).manual_seed(seed)
        out[k] = torch.clamp(v + sigma * torch.randn(v.shape, generator=g, device=v.device), 0, 1)
    return out


def t_blur(img, kernel, sigma):
    import torchvision.transforms.functional as TF
    return {k: TF.gaussian_blur(v, kernel_size=[kernel, kernel], sigma=[sigma, sigma]) for k, v in img.items()}


def t_bright(img, factor):
    import torchvision.transforms.functional as TF
    return {k: torch.clamp(v * factor, 0, 1) for k, v in img.items()}


def t_contrast(img, factor):
    import torchvision.transforms.functional as TF
    return {k: TF.adjust_contrast(v, factor) for k, v in img.items()}


def t_sat(img, factor):
    import torchvision.transforms.functional as TF
    return {k: TF.adjust_saturation(v, factor) for k, v in img.items()}


def t_photo_all(img, b, c, s):
    import torchvision.transforms.functional as TF
    out = {}
    for k, v in img.items():
        x = torch.clamp(v * b, 0, 1)
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = torch.clamp((x - mean) * c + mean, 0, 1)
        out[k] = TF.adjust_saturation(x, s)
    return out


def _tri(fn, vals):
    return [lambda im, v=v: fn(im, v) for v in vals]


FAMILIES = {
    "noise_0.01": [lambda im, sd=sd: t_noise(im, 0.01, sd) for sd in (0, 1, 2)],
    "noise_0.02": [lambda im, sd=sd: t_noise(im, 0.02, sd) for sd in (0, 1, 2)],
    "blur_k3": [lambda im, sg=sg: t_blur(im, 3, sg) for sg in (0.4, 0.5, 0.6)],
    "brightness_1.05": _tri(t_bright, (1.05, 1.06, 1.07)),
    "brightness_1.10": _tri(t_bright, (1.10, 1.11, 1.12)),
    "contrast_1.05": _tri(t_contrast, (1.05, 1.06, 1.07)),
    "contrast_1.10": _tri(t_contrast, (1.10, 1.11, 1.12)),
    "saturation_1.05": _tri(t_sat, (1.05, 1.06, 1.07)),
    "saturation_1.10": _tri(t_sat, (1.10, 1.11, 1.12)),
    # strong (4.0-equivalent) levels, with the CORRECT pixel-space application
    "strong_noise_0.05": [lambda im, sd=sd: t_noise(im, 0.05, sd) for sd in (0, 1, 2)],
    "strong_blur_k5": [lambda im, sg=sg: t_blur(im, 5, sg) for sg in (1.0, 1.1, 1.2)],
    "strong_photo_1.2": [lambda im, v=v: t_photo_all(im, v, v, v) for v in (1.20, 1.22, 1.24)],
}


def group_dist(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Normalised action space [H,7]: 0:3 xyz, 3:6 rot, 6 gripper."""
    d = (a - b).abs()
    return {
        "all": float(d.mean()),
        "xyz": float(d[..., :3].mean()),
        "rot": float(d[..., 3:6].mean()),
        "grip": float(d[..., 6].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class _Cfg: pass
    cfg = _Cfg()
    cfg.libero_src = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO")
    cfg.libero_config_path = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/.cache/libero")
    cfg.render_gpu = 0
    cfg.device = DEV
    from evaluate_libero import configure_libero
    configure_libero(cfg)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    t0 = time.time()
    batches = collect_batches(policy)
    print(f"collected {len(batches)} batches", flush=True)
    rows = []
    for bi, b in enumerate(batches):
        base = {"images": b["images"], "state": b["state"], "instruction": b["instruction"]}
        for fam, fns in FAMILIES.items():
            for seed in SEEDS:
                acts = {}
                acts["identity"] = sample_actions(policy, base, seed=seed)
                for j, fn in enumerate(fns):
                    img = apply_pixel_transform(b["images"], fn)
                    acts[f"aug{j}"] = sample_actions(policy,
                                                     {"images": img, "state": b["state"],
                                                      "instruction": b["instruction"]}, seed=seed)
                aug = [acts[f"aug{j}"] for j in range(3)]
                a_id = acts["identity"]
                a_aug_mean = torch.stack(aug).mean(dim=0)
                d_view = float(np.mean([float((aug[i] - aug[j]).abs().mean())
                                        for i in range(3) for j in range(i + 1, 3)]))
                aug_dev = group_dist(a_aug_mean, a_id)
                for alpha in ALPHAS:
                    target = a_id + alpha * (a_aug_mean - a_id)
                    d_anchor = group_dist(target, a_id)
                    d_fit = float(np.mean([float((aug[k] - target).abs().pow(2).mean()) for k in range(3)]))
                    rows.append({
                        "batch": bi, "family": fam, "seed": seed, "alpha": alpha,
                        "D_view": d_view, "aug_dev": aug_dev, "D_anchor": d_anchor,
                        "D_fit": d_fit,
                    })
    (args.out / "consensus_rows.json").write_text(json.dumps(rows))

    # ---- corrected mixed-view K sweep (checks the 4.0 K-conclusion under correct
    # pixel-space transforms; 4.0 applied transforms on normalised tensors by mistake)
    mixed_fns = [None] + [
        lambda im, sd=0: t_noise(im, 0.02, sd),
        lambda im: t_blur(im, 3, 0.5),
        lambda im: t_bright(im, 1.05),
        lambda im: t_contrast(im, 1.05),
        lambda im: t_sat(im, 1.05),
        lambda im, sd=1: t_noise(im, 0.02, sd),
        lambda im, sg=0.6: t_blur(im, 3, sg),
    ]
    mixed = {}
    for K in (2, 4, 8):
        dviews, anchors_mean, anchors_med = [], [], []
        for b in batches:
            acts = []
            for fn in mixed_fns[:K]:
                img = b["images"] if fn is None else apply_pixel_transform(b["images"], fn)
                acts.append(sample_actions(policy,
                                           {"images": img, "state": b["state"],
                                            "instruction": b["instruction"]}, seed=1234))
            ds = [float((acts[i] - acts[j]).abs().mean())
                  for i in range(K) for j in range(i + 1, K)]
            dviews.append(float(np.mean(ds)))
            mean = torch.stack(acts).mean(dim=0)
            anchors_mean.append(float((mean - acts[0]).abs().mean()))
            dist = [sum(float((acts[i] - acts[j]).abs().mean()) for j in range(K) if j != i)
                    for i in range(K)]
            km = int(np.argmin(dist))
            anchors_med.append(float((acts[km] - acts[0]).abs().mean()))
        mixed[f"K{K}"] = {"D_view": float(np.mean(dviews)),
                          "D_anchor_mean": float(np.mean(anchors_mean)),
                          "D_anchor_medoid": float(np.mean(anchors_med))}
    print("mixed corrected:", json.dumps(mixed, indent=2), flush=True)

    # aggregate per family x alpha
    agg = {}
    for fam in FAMILIES:
        for alpha in ALPHAS:
            sel = [r for r in rows if r["family"] == fam and r["alpha"] == alpha]
            agg[f"{fam}|alpha={alpha}"] = {
                "D_view": float(np.mean([r["D_view"] for r in sel])),
                "D_anchor_all": float(np.mean([r["D_anchor"]["all"] for r in sel])),
                "D_anchor_xyz": float(np.mean([r["D_anchor"]["xyz"] for r in sel])),
                "D_anchor_rot": float(np.mean([r["D_anchor"]["rot"] for r in sel])),
                "D_anchor_grip": float(np.mean([r["D_anchor"]["grip"] for r in sel])),
                "D_fit": float(np.mean([r["D_fit"] for r in sel])),
                "aug_dev_all": float(np.mean([r["aug_dev"]["all"] for r in sel])),
            }
    print(json.dumps(agg, indent=2), flush=True)

    # ---- K=1 no-op check (direct distillation, target = identity action)
    tta = SaltTTA(policy)
    b = batches[0]
    base = {"images": b["images"], "state": b["state"], "instruction": b["instruction"]}
    torch.manual_seed(1234)
    with torch.no_grad():
        rep = policy.representation_extractor(base["images"], base["state"], base["instruction"])
    torch.manual_seed(1234)
    own = sample_with_grad(policy.action_generator, rep,
                           policy.config["action_generator"]["num_sample_steps"])
    target = own.detach()
    tta.optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(own, target)
    loss.backward()
    noop_grad = float(tta.adapter.beta.grad.norm()) if tta.adapter.beta.grad is not None else 0.0

    # ---- projected update under anchored consensus gradient (family=noise, alpha=0.3)
    fam = "noise_0.01"
    alpha = 0.3
    img = apply_pixel_transform(b["images"], FAMILIES[fam][0])
    view = {"images": img, "state": b["state"], "instruction": b["instruction"]}
    with torch.no_grad():
        rep_aug = policy.representation_extractor(view["images"], view["state"], view["instruction"])
    torch.manual_seed(1)
    import torchvision.transforms.functional as TF  # noqa: F401
    aug_actions = sample_actions(policy, view, seed=1)
    base_actions = sample_actions(policy, base, seed=1)
    target_a = (base_actions + alpha * (aug_actions - base_actions))[0]

    # differentiable write + VJP for J_A (same convention as 4.0)
    class ZHook:
        def __init__(self, attention, steps, horizon):
            self.a, self.o = attention, attention.attend
            self.steps, self.h = steps, horizon
            self.step = 0; self.z = None
            attention.attend = self._f
        def _f(self, *a, **k):
            out = self.o(*a, **k)
            if out.shape[1] != self.h:
                self.step = 0; return out
            self.step += 1
            if self.step == self.steps:
                self.z = out.reshape(-1, out.shape[-1]).mean(dim=0)
            return out
        def close(self): self.a.attend = self.o

    def draw(**kw):
        hook = ZHook(policy.action_generator.blocks[0].attention,
                     int(policy.config["action_generator"]["num_sample_steps"]),
                     int(policy.config["action_horizon"]))
        torch.manual_seed(1234)
        chunk = sample_with_grad(policy.action_generator, rep_aug,
                                 policy.config["action_generator"]["num_sample_steps"])[0].float()
        z = hook.z
        hook.close()
        return chunk, z

    chunk, z = draw()
    # anchored-consensus distillation gradient FIRST (frees this graph), then J_A
    tta.optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(chunk, target_a.to(DEV))
    loss.backward()
    g = tta.adapter.beta.grad.detach().double().cpu()
    chunk2, z = draw()
    jrows = []
    for i in range(a_basis.shape[1]):
        gi = torch.autograd.grad(z.double() @ a_basis[:, i].to(DEV).double(), tta.adapter.beta,
                                 retain_graph=(i < a_basis.shape[1] - 1))[0]
        jrows.append(gi.detach().cpu())
    J_A = torch.stack(jrows).double()
    P = torch.eye(J_A.shape[1], dtype=torch.float64) - J_A.T @ torch.linalg.pinv(J_A @ J_A.T) @ J_A
    proj = P @ g
    projected = {
        "g_norm": float(g.norm()), "P_g_norm": float(proj.norm()),
        "retained_fraction": float(proj.norm() / (g.norm() + 1e-12)),
        "J_A_Pg_norm": float((J_A @ proj).norm()),
        "relative": float((J_A @ proj).norm() / (proj.norm() + 1e-12)),
    }
    # finite-step A change: raw vs projected (small step)
    def a_change(delta):
        with torch.no_grad():
            before = tta.adapter.beta.detach().clone()
            tta.adapter.beta.copy_(before + delta.to(DEV).float())
        hook = ZHook(policy.action_generator.blocks[0].attention,
                     int(policy.config["action_generator"]["num_sample_steps"]),
                     int(policy.config["action_horizon"]))
        torch.manual_seed(1234)
        with torch.no_grad():
            sample_with_grad(policy.action_generator, rep_aug,
                             policy.config["action_generator"]["num_sample_steps"])
            z_after = hook.z.detach().double().cpu()
        hook.close()
        with torch.no_grad():
            tta.adapter.beta.copy_(before)
        return float((a_basis.T @ (z_after - z.detach().double().cpu())).norm())

    step = 0.01 * g / (g.norm() + 1e-12)
    finite = {"raw": a_change(step), "projected": a_change(step * 0 + proj / (proj.norm() + 1e-12) * 0.01)}
    tta.close()

    results = {
        "consensus_aggregate": agg,
        "mixed_corrected": mixed,
        "K1_noop": {"distill_loss": 0.0, "distill_grad": noop_grad},
        "projected_update": projected,
        "finite_step_A_change": finite,
        "seconds": time.time() - t0,
    }
    (args.out / "sg_cpa_4_0b.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("K1_noop", "projected_update", "finite_step_A_change")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
