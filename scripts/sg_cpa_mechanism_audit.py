"""Action-sensitivity mechanism audit (offline, no robot rollouts).

A) A reliability: held-out source + target-domain observations; push along A vs A-perp
   vs random directions, fit action Jacobian row space, capture/overlap.
B) Projection audit: full vs A-nullspace-projected vs A-only updates -> dz_A, dz_Aperp,
   action changes (normalized + controller groups) under fixed and fresh latents.
C) Signal correctness: per-episode medoid-identity frequency / loss vs success (AUC)
   from the archived 4.3 diagnostics; fixed-latent vs fresh-latent action change correlation.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE  # noqa: E402

OUT = Path("/hy-tmp/validation/sg_cpa_audit")
DEV = "cuda:0"
EPS = (5e-3, 1e-2)
N_RANDOM = 16


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


class WriteHook:
    def __init__(self, attention, steps, horizon):
        self.a, self.o = attention, attention.attend
        self.steps, self.h = steps, horizon
        self.step, self.value = 0, None
        self.push, self.eps = None, 0.0
        attention.attend = self._f

    def _f(self, *args, **kwargs):
        out = self.o(*args, **kwargs)
        if out.shape[1] != self.h:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.steps:
            if self.push is not None:
                out = out + self.eps * self.push
            self.value = out.detach().reshape(-1, out.shape[-1]).mean(dim=0).float().cpu()
        return out

    def close(self):
        self.a.attend = self.o


def action(policy, rep, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def collect_obs(policy):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
    plan = [("source_heldout", "libero_object", t) for t in (0, 1, 2, 3)]
    plan += [("Camera", "libero_spatial", 705), ("Camera", "libero_spatial", 620)]
    plan += [("Background", "libero_spatial", 11), ("Background", "libero_spatial", 19)]
    plan += [("Noise", "libero_spatial", 1384), ("Noise", "libero_spatial", 1375)]
    out = []
    for domain, suite, tid in plan:
        task = suites[suite].get_task(tid)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                            weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
        np.random.seed(300 + tid); torch.manual_seed(300 + tid); env.seed(300 + tid); env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done: break
        for k in range(3):
            raw = policy_observation(obs, task.language, policy)
            prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                       device=torch.device(DEV))
            out.append({"domain": domain, "suite": suite, "task": tid, "chunk": k, "prep": prep})
            chunk = action(policy, policy.representation_extractor(**prep), 1234)
            acts = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
            for a in acts[:6]:
                obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                if done: break
            if done: break
        env.close()
    return out


def push_response(policy, prep, direction, eps, seed, hook):
    rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
    hook.push = direction.to(DEV).float()
    hook.eps = eps
    a1 = action(policy, rep, seed)
    hook.push = None
    a0 = a1  # placeholder; caller provides base
    return a1, hook.value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float32).to(DEV)
    A = a_basis.double()
    Acpu = A.cpu()
    obs = collect_obs(policy)
    print("observations", len(obs), flush=True)
    steps = int(policy.config["action_generator"]["num_sample_steps"])
    hook = WriteHook(policy.action_generator.blocks[0].attention, steps, int(policy.config["action_horizon"]))
    # ---------- Part A: A reliability
    rows_a = []
    rng = np.random.default_rng(0)
    # orthonormal A basis and orthogonal complement directions
    qa, _ = np.linalg.qr(A.cpu().numpy())
    for oi, o in enumerate(obs):
        prep = o["prep"]
        with torch.no_grad():
            rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
        base = action(policy, rep, 1234)
        # baseline write
        hook.push = None; hook.value = None
        action(policy, rep, 1234)
        w0 = hook.value.clone().to(DEV).double()
        resp = {"a_dir": [], "perp_dir": [], "rand_dir": []}
        for eps in EPS:
            sa, sp = [], []
            for j in range(A.shape[1]):
                d = A[:, j].float()
                hook.push = d; hook.eps = eps
                aj = action(policy, rep, 1234)
                sa.append(float((aj - base).abs().mean()))
            # A-perp random directions
            for j in range(5):
                r = torch.tensor(rng.normal(size=768), dtype=torch.float64)
                r = r - Acpu @ (Acpu.T @ r); r = r / r.norm()
                hook.push = r.float().to(DEV); hook.eps = eps
                aj = action(policy, rep, 1234)
                sp.append(float((aj - base).abs().mean()))
            resp["a_dir"].append((eps, float(np.mean(sa))))
            resp["perp_dir"].append((eps, float(np.mean(sp))))
            # random directions -> Jacobian fit
            R, Y = [], []
            for j in range(N_RANDOM):
                r = torch.tensor(rng.normal(size=768), dtype=torch.float64); r = r / r.norm()
                hook.push = r.float().to(DEV); hook.eps = eps
                aj = action(policy, rep, 1234)
                R.append(r.numpy()); Y.append((aj - base).reshape(-1).cpu().numpy())
            R = torch.tensor(np.stack(R), dtype=torch.float64)
            Y = torch.tensor(np.stack(Y), dtype=torch.float64)
            J = torch.linalg.lstsq(R, Y).solution.T.double()  # 84 x 768
            sv = torch.linalg.svdvals(J)
            cap = float((sv[:5] ** 2).sum() / (sv ** 2).sum())
            _, _, vh = torch.linalg.svd(J, full_matrices=False)
            V5 = vh[:5].T  # 768x5
            ov = float((V5.T @ Acpu).norm() / np.sqrt(5))
            resp["rand_dir"].append({"eps": eps, "capture_top5": cap, "overlap_A": ov})
        hook.push = None
        resp.update({"domain": o["domain"], "suite": o["suite"], "task": o["task"], "chunk": o["chunk"]})
        rows_a.append(resp)
        if oi % 5 == 0: print("A", oi, flush=True)
    # ---------- Part B: projection audit
    rows_b = []
    for oi, o in enumerate(obs):
        prep = o["prep"]
        obj = SGCpa(policy, a_basis.double(), seed=1234)
        try:
            info = obj.build_target(prep, seed=50000 + oi, view_seed=50000 + oi + 1)
            obj.adapter.beta.detach().zero_()
            loss = obj.loss(prep, info, seed=50000 + oi)
            obj.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            g = obj.adapter.beta.grad.detach().double().cpu().clone()
            j = obj.jacobian_a(prep, seed=50000 + oi)
            P = obj.nullspace(j)
            P_A = (A.cpu() @ A.cpu().T)
            with torch.no_grad():
                obj.adapter.beta.zero_()
            w0 = obj.write(prep, seed=50000 + oi).detach().double().cpu()
            updates = {"full": g, "proj": P @ g, "A_only": P_A @ g}
            with torch.no_grad():
                rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
            base_action = action(policy, rep, 50000 + oi)
            base_action_fresh = [action(policy, rep, s) for s in (777001, 777002)]
            for name, upd in updates.items():
                eta = 0.05
                dbeta = (eta * upd).float().to(obj.adapter.beta.device)
                with torch.no_grad():
                    obj.adapter.beta.zero_(); obj.adapter.beta.add_(dbeta)
                z = obj.write(prep, seed=50000 + oi).detach().double().cpu()
                dz = z - w0
                dzA = float((A.cpu().T @ dz).norm())
                dzperp = float((dz - A.cpu() @ (A.cpu().T @ dz)).norm())
                a1 = action(policy, rep, 50000 + oi)
                da = float((a1 - base_action).abs().mean())
                da_fresh = [float((action(policy, rep, s) - bf).abs().mean())
                            for s, bf in zip((777001, 777002), base_action_fresh)]
                with torch.no_grad():
                    obj.adapter.beta.zero_()
                rows_b.append({"domain": o["domain"], "suite": o["suite"], "task": o["task"],
                               "chunk": o["chunk"], "mode": name, "eta": eta,
                               "dz": float(dz.norm()), "dz_A": dzA, "dz_perp": dzperp,
                               "da_teacher": da, "da_fresh": float(np.mean(da_fresh)),
                               "g_norm": float(g.norm()), "retain": float(upd.norm() / (g.norm() + 1e-12))})
        finally:
            obj.close()
    # fixed-latent vs fresh-latent correlation over all (mode, obs)
    xs = [r["da_teacher"] for r in rows_b]; ys = [r["da_fresh"] for r in rows_b]
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 3 else None
    # ---------- Part C: signal vs success from archived 4.3 diag
    diag_files = list(Path("/hy-tmp/validation/p43/diag_sg_cpa").glob("*/*.json"))
    eps_rows = []
    for f in diag_files:
        d = json.loads(f.read_text())
        ups = d.get("updates", [])
        if not ups: continue
        succ = int(d.get("result", {}).get("success", 0))
        ident = float(np.mean([u["medoid_index"] == 0 for u in ups]))
        loss = float(np.mean([u["loss"] for u in ups]))
        eps_rows.append({"success": succ, "identity_frac": ident, "loss": loss,
                         "updates": len(ups)})
    def auc(score, labels):
        pos = [s for s, y in zip(score, labels) if y == 1]
        neg = [s for s, y in zip(score, labels) if y == 0]
        if not pos or not neg: return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return float(wins / (len(pos) * len(neg)))
    labels = [r["success"] for r in eps_rows]
    out = {"n_obs": len(obs), "n_random": N_RANDOM,
           "A_reliability": rows_a, "projection": rows_b,
           "latent_corr": corr,
           "signal_auc": {"n": len(eps_rows),
                          "identity_frac_auc": auc([r["identity_frac"] for r in eps_rows], labels),
                          "loss_auc": auc([r["loss"] for r in eps_rows], labels)}}
    (OUT / "audit.json").write_text(json.dumps(out, indent=2))
    # summary
    by_dom = {}
    for r in rows_a:
        by_dom.setdefault(r["domain"], {"A": [], "perp": [], "cap": [], "ov": []})
        by_dom[r["domain"]]["A"].append(r["a_dir"][-1][1])
        by_dom[r["domain"]]["perp"].append(r["perp_dir"][-1][1])
        by_dom[r["domain"]]["cap"].append(r["rand_dir"][-1]["capture_top5"])
        by_dom[r["domain"]]["ov"].append(r["rand_dir"][-1]["overlap_A"])
    print("A reliability (eps=1e-2):", {d: {k: round(float(np.mean(v)), 4) for k, v in kv.items()}
                                         for d, kv in by_dom.items()})
    print("projection ratios:", {m: {"dz_A/dz": round(float(np.mean([r["dz_A"] / (r["dz"] + 1e-12) for r in rows_b if r["mode"] == m])), 3),
                                     "da_teacher": round(float(np.mean([r["da_teacher"] for r in rows_b if r["mode"] == m])), 6),
                                     "da_fresh": round(float(np.mean([r["da_fresh"] for r in rows_b if r["mode"] == m])), 6)}
                                 for m in ("full", "proj", "A_only")})
    print("latent corr", corr, "signal auc", out["signal_auc"])
    print("seconds", time.time() - 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
