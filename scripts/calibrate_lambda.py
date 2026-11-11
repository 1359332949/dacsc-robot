"""Pre-register SG-SALT lambda from OFFLINE gradient magnitudes (no rollouts).

At a non-zero beta point (one SALT Adam step, since the quadratic protection has
zero gradient at beta=0) measure:
  g_salt = grad_beta L_SALT,  g_pen = grad_beta ||A^T(z_phi - z_0)||^2
lambda = mean(||g_salt|| / ||g_pen||) over 3 synthetic batches, snapped to a
coarse grid. The grid choice is frozen before the formal experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.salt_tta import SaltTTA, flow_matching_leftover_loss  # noqa: E402
from dacsc.sg_salt import adapted_write, frozen_write, protection_loss  # noqa: E402

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
GRID = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]


def make_batch(policy, seed: int):
    rng = np.random.default_rng(seed)
    images = {v: torch.from_numpy(rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8))
              for v in policy.config["view_names"]}
    state = torch.from_numpy(rng.normal(size=(1, policy.config["state_dim"])).astype(np.float32))
    return prepare_observation({"images": images, "state": state, "instruction": ["do it"]},
                               config=policy.config, normalizer=policy.normalizer,
                               device=torch.device(DEV))


def main() -> None:
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float32)
    tta = SaltTTA(policy)
    target = torch.zeros(12, 7, device=DEV)
    rows = []
    try:
        for seed in (0, 1, 2):
            batch = make_batch(policy, seed)
            with torch.no_grad():
                rep = policy.representation_extractor(**batch)
            tta.reset()
            gen = torch.Generator(device=DEV).manual_seed(7 + seed)
            salt0 = flow_matching_leftover_loss(policy.action_generator, rep, target, 6,
                                                draws=8, rng=gen)
            tta.optimizer.zero_grad(set_to_none=True)
            salt0.backward()
            tta.optimizer.step()
            with torch.no_grad():
                tta.adapter.beta.clamp_(-tta.eta, tta.eta)
            gen = torch.Generator(device=DEV).manual_seed(11 + seed)
            salt = flow_matching_leftover_loss(policy.action_generator, rep, target, 6,
                                               draws=8, rng=gen)
            tta.optimizer.zero_grad(set_to_none=True)
            salt.backward()
            g_salt = tta.adapter.beta.grad.detach().clone()
            z0 = frozen_write(policy, tta.adapter, rep)
            z = adapted_write(policy, rep)
            pen, _ = protection_loss(z, z0, a_basis)
            tta.optimizer.zero_grad(set_to_none=True)
            pen.backward()
            g_pen = tta.adapter.beta.grad.detach().clone()
            rows.append({"seed": seed, "salt_loss": float(salt.detach()),
                         "penalty": float(pen.detach()),
                         "grad_salt_norm": float(g_salt.norm()),
                         "grad_pen_norm": float(g_pen.norm()),
                         "ratio": float(g_salt.norm() / (g_pen.norm() + 1e-12))})
            tta.reset()
        mean_ratio = float(np.mean([r["ratio"] for r in rows]))
        lam = min(GRID, key=lambda g: abs(np.log10(g) - np.log10(max(mean_ratio, 1e-8))))
        print(json.dumps({"per_batch": rows, "mean_ratio": mean_ratio,
                          "lambda_grid": GRID, "lambda_chosen": lam}, indent=2))
    finally:
        tta.close()


if __name__ == "__main__":
    main()
