"""工单 4.0: offline design validation for SG-CPA (no robot rollouts / no success metrics).

Checks:
  A. consensus reliability (K=2/4/8, Mean vs Medoid) and D_view / D_anchor
  B. per-transform semantic preservation (action distance vs identity)
  C. action-block distillation vs flow-matching consistency at beta=0 (K=1 no-op)
  D. real A + real Jacobian nullspace projection validation
  E. compute cost and finite-step stability
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
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import SaltTTA, flow_matching_leftover_loss  # noqa: E402

CKPT = _CORP_REPO / "checkpoints/corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = Path("/hy-tmp/validation/sg_cpa_4_0")


# ---------------------------------------------------------------- transforms
def view_identity(images):
    return {k: v.clone() for k, v in images.items()}


def view_noise(images, sigma=0.05, seed=0):
    out = {}
    for k, v in images.items():
        g = torch.Generator(device=v.device).manual_seed(seed)
        out[k] = torch.clamp(v + sigma * torch.randn(v.shape, generator=g, device=v.device), 0, 1)
    return out


def view_blur(images, kernel=5, sigma=1.0):
    return {k: TF.gaussian_blur(v, kernel_size=[kernel, kernel], sigma=[sigma, sigma]) for k, v in images.items()}


def view_photo(images, b=1.2, c=1.2, s=1.2):
    out = {}
    for k, v in images.items():
        x = torch.clamp(v * b, 0, 1)
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = torch.clamp((x - mean) * c + mean, 0, 1)
        out[k] = TF.adjust_saturation(x, s)
    return out


VIEWS = [
    ("identity", view_identity),
    ("noise", lambda im: view_noise(im, 0.05, seed=0)),
    ("blur", view_blur),
    ("photo", view_photo),
    ("noise+blur", lambda im: view_blur(view_noise(im, 0.05, seed=1))),
    ("blur+photo", lambda im: view_photo(view_blur(im))),
    ("noise+photo", lambda im: view_photo(view_noise(im, 0.05, seed=2))),
    ("noise_strong", lambda im: view_noise(im, 0.10, seed=3)),
]


# ---------------------------------------------------------------- helpers
def sample_actions(policy, batch, seed: int):
    images = batch["images"]
    state = batch["state"]
    instruction = batch["instruction"]
    torch.manual_seed(seed)
    with torch.no_grad():
        rep = policy.representation_extractor(images, state, instruction)
        chunk = policy.action_generator.sample(rep)
    return chunk.float().cpu()   # normalized action space


def pairwise_dist(actions):
    K = len(actions)
    ds = []
    for i in range(K):
        for j in range(i + 1, K):
            ds.append(float((actions[i] - actions[j]).abs().mean()))
    return ds


def consensus(actions, mode: str):
    stack = torch.stack(actions)
    if mode == "mean":
        return stack.mean(dim=0)
    # medoid
    K = len(actions)
    dist = torch.zeros(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                dist[i] += float((actions[i] - actions[j]).abs().mean())
    k = int(torch.argmin(dist))
    return actions[k], k


def collect_batches(policy, n_tasks=3, n_chunks=6, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    batches = []
    for tid in (0, 1, 2)[:n_tasks]:
        task = suite.get_task(tid)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                            weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 render_gpu_device_id=0)
        np.random.seed(seed + tid * 1000); torch.manual_seed(seed + tid * 1000)
        env.seed(seed + tid * 1000); env.reset(); obs = env.set_init_state(states[0])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done: break
        for k in range(n_chunks):
            batch = policy_observation(obs, task.language, policy)
            prepared = prepare_observation(batch, config=policy.config, normalizer=policy.normalizer,
                                           device=torch.device(DEV))
            batches.append({"task_id": tid, "chunk": k, "images": prepared["images"],
                            "state": prepared["state"], "instruction": prepared["instruction"]})
            torch.manual_seed(1234 + tid * 100 + k)
            with torch.no_grad():
                rep = policy.representation_extractor(prepared["images"], prepared["state"], prepared["instruction"])
                chunk = policy.action_generator.sample(rep)
            actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
            for a in actions[:6]:
                obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                if done: break
            if done: break
        env.close()
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="2,4,8")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class _Cfg: pass
    cfg = _Cfg()
    cfg.libero_src = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO")
    cfg.libero_config_path = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/.cache/libero")
    cfg.render_gpu = 0
    cfg.device = DEV
    configure_libero(cfg)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    results = {}

    t0 = time.time()
    batches = collect_batches(policy)
    results["collection_seconds"] = time.time() - t0
    results["batches"] = len(batches)
    print(f"collected {len(batches)} observation batches in {results['collection_seconds']:.1f}s", flush=True)

    # ---- A/B: consensus + transform analytics
    per_batch = []
    ts = time.time()
    for bi, b in enumerate(batches):
        actions = {}
        for name, fn in VIEWS:
            view = {"images": fn(b["images"]), "state": b["state"], "instruction": b["instruction"]}
            actions[name] = sample_actions(policy, view, seed=1234)
        entry = {"batch": bi, "task_id": b["task_id"], "chunk": b["chunk"], "views": {}}
        for name in actions:
            entry["views"][name] = float((actions[name] - actions["identity"]).abs().mean())
        for K in [int(x) for x in args.K.split(",")]:
            keys = [v[0] for v in VIEWS[:K]]
            acts = [actions[k] for k in keys]
            ds = pairwise_dist(acts)
            mean = consensus(acts, "mean")
            med = consensus(acts, "medoid")
            entry[f"K{K}"] = {
                "views": keys,
                "D_view": float(np.mean(ds)),
                "D_anchor_mean": float((mean - actions["identity"]).abs().mean()),
                "D_anchor_medoid": float((med[0] - actions["identity"]).abs().mean()),
                "medoid_index": int(med[1]),
            }
        per_batch.append(entry)
    results["sample_seconds"] = time.time() - ts
    results["per_batch"] = per_batch

    # aggregate
    agg = {}
    for K in [int(x) for x in args.K.split(",")]:
        dviews = [e[f"K{K}"]["D_view"] for e in per_batch]
        da_m = [e[f"K{K}"]["D_anchor_mean"] for e in per_batch]
        da_med = [e[f"K{K}"]["D_anchor_medoid"] for e in per_batch]
        agg[f"K{K}"] = {"D_view_mean": float(np.mean(dviews)), "D_view_std": float(np.std(dviews)),
                        "D_anchor_mean": float(np.mean(da_m)), "D_anchor_medoid": float(np.mean(da_med))}
    results["consensus_aggregate"] = agg
    results["transform_means"] = {
        name: float(np.mean([e["views"][name] for e in per_batch])) for name, _ in VIEWS[1:]
    }
    print(json.dumps({"consensus": agg, "transforms": results["transform_means"]}, indent=2), flush=True)

    # ---- C: loss comparison at beta=0 (K=1 no-op) using real model
    tta = SaltTTA(policy)
    b = batches[0]
    consensus_action = None
    view_actions = []
    for name, fn in VIEWS[:4]:
        view = {"images": fn(b["images"]), "state": b["state"], "instruction": b["instruction"]}
        view_actions.append(sample_actions(policy, view, seed=1234))
    consensus_action = torch.stack(view_actions).mean(dim=0)   # [1,12,7]
    if consensus_action.dim() == 3:
        consensus_action = consensus_action[0]                 # [12,7] for losses

    def distill_loss(adapter_beta: torch.Tensor):
        tta.adapter.beta.data.copy_(adapter_beta)
        total = torch.zeros((), device=DEV)
        for name, fn in VIEWS[:4]:
            view = {"images": fn(b["images"]), "state": b["state"], "instruction": b["instruction"]}
            with torch.no_grad():
                rep = policy.representation_extractor(view["images"], view["state"], view["instruction"])
            torch.manual_seed(1234)
            chunk = sample_with_grad(policy.action_generator, rep,
                                     policy.config["action_generator"]["num_sample_steps"])[0].float()
            pred = chunk[0] if chunk.dim() == 3 else chunk
            target = consensus_action.to(DEV)
            if target.dim() == 1:
                target = target.reshape_as(pred)
            total = total + F.mse_loss(pred, target)
        return total / 4.0

    tta.adapter.beta.data.zero_()
    tta.optimizer.zero_grad(set_to_none=True)
    l_distill = distill_loss(tta.adapter.beta.detach())
    # flow-matching consistency at beta=0
    view0 = {"images": VIEWS[0][1](b["images"]), "state": b["state"], "instruction": b["instruction"]}
    with torch.no_grad():
        rep0 = policy.representation_extractor(view0["images"], view0["state"], view0["instruction"])
    gen = torch.Generator(device=DEV).manual_seed(7)
    l_flow = flow_matching_leftover_loss(policy.action_generator, rep0,
                                         consensus_action.to(DEV), 0, draws=4, mode="self", rng=gen)
    results["loss_at_beta0"] = {"distill": float(l_distill.detach()), "flow": float(l_flow.detach())}

    # K=1 no-op: target = identity action, beta=0
    tta.adapter.beta.data.zero_()
    with torch.no_grad():
        rep = policy.representation_extractor(view0["images"], view0["state"], view0["instruction"])
    torch.manual_seed(1234)
    own = sample_with_grad(policy.action_generator, rep,
                           policy.config["action_generator"]["num_sample_steps"])
    own_target = own.detach()
    l1_distill = F.mse_loss(own, own_target)
    l1_flow = flow_matching_leftover_loss(policy.action_generator, rep, own[0], 0, draws=4,
                                          mode="self", rng=torch.Generator(device=DEV).manual_seed(7))
    # gradients at beta=0
    def _own_loss():
        torch.manual_seed(1234)
        return F.mse_loss(sample_with_grad(policy.action_generator, rep,
                                           policy.config["action_generator"]["num_sample_steps"]),
                          own_target)

    def grad_norm(loss_fn):
        tta.adapter.beta.data.zero_()
        tta.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        g = tta.adapter.beta.grad
        return float(g.norm()) if g is not None else 0.0

    results["K1_noop"] = {
        "distill_loss": float(l1_distill.detach()),
        "flow_loss": float(l1_flow.detach()),
        "distill_grad": grad_norm(lambda: _own_loss()),
        "flow_grad": grad_norm(lambda: flow_matching_leftover_loss(
            policy.action_generator, rep, own_target[0], 0, draws=4, mode="self",
            rng=torch.Generator(device=DEV).manual_seed(7))),
    }
    results["K4_grads"] = {
        "distill_grad": grad_norm(lambda: distill_loss(tta.adapter.beta.detach())),
        "flow_grad": grad_norm(lambda: flow_matching_leftover_loss(
            policy.action_generator, rep0, consensus_action.to(DEV), 0, draws=4, mode="self",
            rng=torch.Generator(device=DEV).manual_seed(7))),
    }

    # ---- D: real Jacobian + projection
    beta0 = tta.adapter.beta.detach().clone()
    class ZHook:
        def __init__(self, attention, steps, horizon):
            self.a = attention; self.o = attention.attend; self.steps = steps; self.h = horizon
            self.step = 0; self.z = None; attention.attend = self._f
        def _f(self, *a, **k):
            out = self.o(*a, **k)
            if out.shape[1] != self.h:
                self.step = 0; return out
            self.step += 1
            if self.step == self.steps:
                self.z = out.reshape(-1, out.shape[-1]).mean(dim=0)
            return out
        def close(self): self.a.attend = self.o

    hook = ZHook(policy.action_generator.blocks[0].attention,
                 int(policy.config["action_generator"]["num_sample_steps"]),
                 int(policy.config["action_horizon"]))
    with torch.no_grad():
        rep = policy.representation_extractor(view0["images"], view0["state"], view0["instruction"])
    torch.manual_seed(1234)
    _ = sample_with_grad(policy.action_generator, rep, policy.config["action_generator"]["num_sample_steps"])
    z0 = hook.z.detach().clone()
    # simpler: one differentiable z, then VJP per A column
    hook = ZHook(policy.action_generator.blocks[0].attention,
                 int(policy.config["action_generator"]["num_sample_steps"]),
                 int(policy.config["action_horizon"]))
    torch.manual_seed(1234)
    _ = sample_with_grad(policy.action_generator, rep, policy.config["action_generator"]["num_sample_steps"])
    z = hook.z
    hook.close()
    t_jac = time.time()
    jrows = []
    for i in range(a_basis.shape[1]):
        gi = torch.autograd.grad(z.double() @ a_basis[:, i].to(DEV).double(), tta.adapter.beta,
                                 retain_graph=(i < a_basis.shape[1] - 1))[0]
        jrows.append(gi.detach().cpu())
    J_A = torch.stack(jrows).double()          # [r, d]
    results["jacobian_seconds"] = time.time() - t_jac
    results["J_A_shape"] = list(J_A.shape)
    results["J_A_norm"] = float(J_A.norm())
    # synthetic gradient + projection
    g = torch.randn(J_A.shape[1], dtype=torch.float64)
    JJt = J_A @ J_A.T
    pinv = torch.linalg.pinv(JJt)
    P = torch.eye(J_A.shape[1], dtype=torch.float64) - J_A.T @ pinv @ J_A
    d = P @ g
    results["projection"] = {
        "g_norm": float(g.norm()), "P_g_norm": float(d.norm()),
        "J_A_Pg_norm": float((J_A @ d).norm()),
        "relative": float((J_A @ d).norm() / (d.norm() + 1e-12)),
    }
    # Adam-style update then project (user requirement)
    adam = torch.optim.Adam([tta.adapter.beta], lr=0.06)
    tta.adapter.beta.grad = (-g.float().to(DEV)).reshape_as(tta.adapter.beta) * 0
    # simulate an Adam step from a real loss gradient
    tta.optimizer.zero_grad(set_to_none=True)
    loss = distill_loss(tta.adapter.beta.detach())
    loss.backward()
    before = tta.adapter.beta.detach().clone()
    tta.optimizer.step()
    delta_adam = (tta.adapter.beta.detach() - before)
    d_proj = (P @ delta_adam.double().cpu()).to(DEV)
    results["adam_projection"] = {
        "delta_adam_norm": float(delta_adam.norm()),
        "projected_norm": float(d_proj.norm()),
    }
    # finite-step stability: small projected update vs raw update
    for label, delta in (("raw", delta_adam), ("projected", d_proj)):
        with torch.no_grad():
            tta.adapter.beta.copy_(before + delta)
        torch.manual_seed(1234)
        with torch.no_grad():
            rep2 = policy.representation_extractor(view0["images"], view0["state"], view0["instruction"])
        hook2 = ZHook(policy.action_generator.blocks[0].attention,
                      int(policy.config["action_generator"]["num_sample_steps"]),
                      int(policy.config["action_horizon"]))
        with torch.no_grad():
            sample_with_grad(policy.action_generator, rep2, policy.config["action_generator"]["num_sample_steps"])
            z_after = hook2.z.detach().double().cpu()
        hook2.close()
        a_delta = float((a_basis.T @ (z_after - z0.double().cpu())).norm()) if z0 is not None else -1
        results.setdefault("finite_step", {})[label] = {"A_direction_change": a_delta}
        with torch.no_grad():
            tta.adapter.beta.copy_(before)
    tta.close()
    results["total_seconds"] = time.time() - t0
    (args.out / "sg_cpa_4_0.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("consensus_aggregate", "K1_noop", "projection", "total_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
