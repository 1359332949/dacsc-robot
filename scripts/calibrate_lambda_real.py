"""Pre-register SG-SALT lambda using REAL clean observations (no test-set results)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.salt_tta import SaltTTA, flow_matching_leftover_loss  # noqa: E402
from dacsc.sg_salt import adapted_write, frozen_write, protection_loss  # noqa: E402

CKPT = _CORP_REPO / "checkpoints/corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
GRID = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


def main() -> int:
    ap = type("A", (), {})()
    ap.libero_src = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO")
    ap.libero_config_path = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/.cache/libero")
    ap.render_gpu = 0
    ap.device = "cuda:0"
    configure_libero(ap)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(CKPT, device="cuda:0")
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float32)
    tta = SaltTTA(policy)
    target = torch.zeros(12, 7, device="cuda:0")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                        weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    np.random.seed(7); torch.manual_seed(7); env.seed(7); env.reset()
    obs = env.set_init_state(states[0])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    batches = []
    for k in range(8):
        batch = policy_observation(obs, task.language, policy)
        prepared = prepare_observation(batch, config=policy.config, normalizer=policy.normalizer,
                                       device=torch.device("cuda:0"))
        with torch.no_grad():
            rep = policy.representation_extractor(**prepared)
        batches.append(rep)
        with torch.no_grad():
            chunk = policy.action_generator.sample(rep)
        actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
        for a in actions[:6]:
            obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
            if done: break
        if done: break
    env.close()
    rows = []
    try:
        for idx, rep in enumerate(batches):
            tta.reset()
            gen = torch.Generator(device="cuda:0").manual_seed(7 + idx)
            salt0 = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=8, rng=gen)
            tta.optimizer.zero_grad(set_to_none=True)
            salt0.backward()
            tta.optimizer.step()
            with torch.no_grad():
                tta.adapter.beta.clamp_(-tta.eta, tta.eta)
            gen = torch.Generator(device="cuda:0").manual_seed(101 + idx)
            salt = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=8, rng=gen)
            tta.optimizer.zero_grad(set_to_none=True)
            salt.backward()
            g_salt = tta.adapter.beta.grad.detach().clone()
            z0 = frozen_write(policy, tta.adapter, rep)
            z = adapted_write(policy, rep)
            pen, _ = protection_loss(z, z0, a_basis)
            tta.optimizer.zero_grad(set_to_none=True)
            pen.backward()
            g_pen = tta.adapter.beta.grad.detach().clone()
            rows.append({"idx": idx, "salt_loss": float(salt.detach()), "penalty": float(pen.detach()),
                         "grad_salt_norm": float(g_salt.norm()), "grad_pen_norm": float(g_pen.norm()),
                         "ratio": float(g_salt.norm() / (g_pen.norm() + 1e-12))})
            tta.reset()
        mean_ratio = float(np.mean([r["ratio"] for r in rows]))
        lam = min(GRID, key=lambda g: abs(np.log10(g) - np.log10(max(mean_ratio, 1e-8))))
        print(json.dumps({"per_batch": rows, "mean_ratio": mean_ratio,
                          "lambda_grid": GRID, "lambda_chosen": lam}, indent=2))
    finally:
        tta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
