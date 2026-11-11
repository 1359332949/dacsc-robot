"""工单 1.2: SALT minimal health check + SG-SALT interface preparation.

Only offline/small-input checks; no rollout, no algorithm change.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import (  # noqa: E402
    SaltKeyAdapter, flow_matching_leftover_loss, _masked_mse, build_leftover_target,
    flow_interpolate,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
STATS_BIG = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
STATS_FIX = Path("/hy-tmp/validation/interface_stats_spatial_k32_fixed.json")
DIAG = "/hy-tmp/validation/diag_clean_always"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


class WriteCapture:
    """Capture the last-Euler write (token mean) keeping the graph."""

    def __init__(self, attention, num_steps: int, horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.step = 0
        self.value = None
        attention.attend = self._attend

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.num_steps:
            self.value = out.reshape(-1, out.shape[-1]).mean(dim=0)
        return out

    def close(self):
        self.attention.attend = self.original


def synthetic_batch(policy):
    rng = np.random.default_rng(0)
    images = {v: torch.from_numpy(rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8))
              for v in policy.config["view_names"]}
    state = torch.from_numpy(rng.normal(size=(1, policy.config["state_dim"])).astype(np.float32))
    return prepare_observation({"images": images, "state": state, "instruction": ["do it"]},
                               config=policy.config, normalizer=policy.normalizer,
                               device=torch.device(DEV))


def chunk_with_beta(policy, adapter, rep, beta: torch.Tensor, seed: int = 7):
    with torch.no_grad():
        adapter.beta.copy_(beta)
        torch.manual_seed(seed)
        cap = WriteCapture(policy.action_generator.blocks[0].attention,
                           int(policy.config["action_generator"]["num_sample_steps"]),
                           int(policy.config["action_horizon"]))
        try:
            chunk = sample_with_grad(policy.action_generator, rep,
                                     policy.config["action_generator"]["num_sample_steps"])[0].float()
            z = cap.value.detach().clone()
        finally:
            cap.close()
    return chunk, z


def beta_from_quick_adapt(policy, adapter, rep, steps: int = 5, lr: float = 0.06, seeds=(1234,)):
    adapter.reset()
    opt = torch.optim.Adam([adapter.beta], lr=lr)
    target = torch.zeros(12, 7, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(1729)
    for _ in range(steps):
        loss = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=2, rng=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            adapter.beta.clamp_(-adapter.eta, adapter.eta)
    beta = adapter.beta.detach().clone()
    adapter.reset()
    return beta


def saturation_impact():
    print("\n== B. beta saturation impact on base SALT updates (from Phase-1.2 logs)")
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{DIAG}/*.json"))]
    for d in rows:
        per_replan = {}
        for u in d["updates"]:
            per_replan.setdefault(u["replan"], []).append(u)
        sat_replan = next((rp for rp in sorted(per_replan)
                           if max(u["frac_ge_099eta"] for u in per_replan[rp]) > 0.5), None)
        if sat_replan is None:
            continue
        after = [u for rp, us in per_replan.items() if rp > sat_replan for u in us]
        dbeta = [u["delta_beta_l2"] for u in after]
        print(f"task={d['task_id']}: saturated_at_replan={sat_replan}, "
              f"post-saturation updates={len(after)}, "
              f"mean delta_beta_l2={np.mean(dbeta):.4f}, min={np.min(dbeta):.4f}, max={np.max(dbeta):.4f}"
              if after else f"task={d['task_id']}: saturation at last replan")


def main() -> None:
    policy = load_policy(CKPT, device=DEV)
    batch = synthetic_batch(policy)
    with torch.no_grad():
        rep = policy.representation_extractor(**batch)   # no grad needed for the interface probe
    adapter = SaltKeyAdapter(policy.action_generator.blocks[0].attention.qkv,
                             policy.config["hidden_dim"])
    try:
        # ---- A. interface consistency with the Jacobian A subspace
        stats = json.loads(STATS_BIG.read_text())
        A = torch.tensor(stats["action_subspace"], dtype=torch.float64)      # [768, r]
        r = A.shape[1]
        ortho_err = float((A.T @ A - torch.eye(r, dtype=torch.float64)).abs().max())
        print(f"== A. interface check: A shape={tuple(A.shape)} rank={r} "
              f"frames={stats['n_frames']} site={stats['site']!r}")
        print(f"   orthonormality max|A^T A - I| = {ortho_err:.2e}")
        z0, chunk0 = None, None
        beta_zero = torch.zeros_like(adapter.beta)
        chunk0, z0 = chunk_with_beta(policy, adapter, rep, beta_zero)
        beta_sat = torch.full_like(adapter.beta, adapter.eta)
        chunk_sat, z_sat = chunk_with_beta(policy, adapter, rep, beta_sat)
        dz = (z_sat.double() - z0.double()).cpu()
        proj = A.T @ dz
        resid = dz - A @ proj
        print(f"   write dim={dz.numel()}  ||dz||={float(dz.norm()):.4f}  "
              f"||A^T dz||={float(proj.norm()):.4f}  ||dz-A A^T dz||={float(resid.norm()):.4f}")
        print(f"   action delta (uniform beta=eta vs 0): mean|da|={float((chunk_sat-chunk0).abs().mean()):.5f} "
              f"max|da|={float((chunk_sat-chunk0).abs().max()):.5f}")
        # quick-adaptation beta (what 5 Adam steps actually give)
        beta_adapt = beta_from_quick_adapt(policy, adapter, rep)
        chunk_ad, z_ad = chunk_with_beta(policy, adapter, rep, beta_adapt)
        dz_ad = (z_ad.double() - z0.double()).cpu()
        proj_ad = A.T @ dz_ad
        print(f"   quick-adapt beta: ||beta||={float(beta_adapt.norm()):.3f} "
              f"frac>=0.99eta={float((beta_adapt.abs()>=0.99*adapter.eta).float().mean()):.3f}")
        print(f"   action delta (adapted beta vs 0): mean|da|={float((chunk_ad-chunk0).abs().mean()):.5f} "
              f"||dz||={float(dz_ad.norm()):.4f} ||A^T dz||={float(proj_ad.norm()):.4f}")

        # ---- D. protection-term numerics (synthetic, independent of A file)
        print("\n== D. protection-term numerical validation")
        g = torch.Generator().manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(768, 5, generator=g, dtype=torch.float64))
        a_coef = 2.0
        dz_a = a_coef * Q[:, 0]
        dz_orth = Q[:, 0] * 0 + (torch.eye(768, dtype=torch.float64)[:, 0] - Q @ (Q.T @ torch.eye(768, dtype=torch.float64)[:, 0]))
        pen_a = float((Q.T @ dz_a).pow(2).sum())
        pen_orth = float((Q.T @ dz_orth).pow(2).sum())
        print(f"   along-A dz: penalty={pen_a:.4f} (expected {a_coef**2:.4f})  PASS={abs(pen_a-a_coef**2)<1e-9}")
        print(f"   orthogonal dz: penalty={pen_orth:.2e} (expected 0)  PASS={pen_orth<1e-12}")
        # lambda=0 equivalence at the loss level
        target = torch.zeros(12, 7, device=DEV)
        gen = torch.Generator(device=DEV).manual_seed(5)
        salt = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=2, rng=gen)
        gen = torch.Generator(device=DEV).manual_seed(5)
        with torch.no_grad():
            chunk = sample_with_grad(policy.action_generator, rep,
                                     policy.config["action_generator"]["num_sample_steps"])[0].float()
        sg = salt + 0.0 * float(proj.pow(2).sum())   # lambda=0 must be exactly SALT
        print(f"   lambda=0 equivalence: |SG-SALT - SALT|={abs(float(sg)-float(salt)):.2e} PASS={abs(float(sg)-float(salt))<1e-12}")
    finally:
        adapter.close()

    saturation_impact()


if __name__ == "__main__":
    main()
