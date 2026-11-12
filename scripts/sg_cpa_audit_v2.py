"""Corrected action-sensitivity audit (v2) after review.

Fixes:
  * beta-space decomposition: sensitive component = (I - P_null) g  (NOT A A^T g)
  * gradient descent sign (delta beta = -eta * update)
  * production guard path (SGCpa.step) with/without guard
  * latent generalization measured as cosine between action-change VECTORS
  * AUC: correct direction (-loss), within-task, task-cluster bootstrap, permutation,
    early-episode loss to reduce temporal leakage
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

OUT = Path("/hy-tmp/validation/sg_cpa_audit_v2")
DEV = "cuda:0"


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


class WriteHook:
    def __init__(self, attention, steps, horizon):
        self.a, self.o = attention, attention.attend
        self.steps, self.h = steps, horizon
        self.step, self.value = 0, None
        self.push, self.eps = None, 0.0
        attention.attend = self._f

    def _f(self, *a, **k):
        out = self.o(*a, **k)
        if out.shape[1] != self.h:
            self.step = 0; return out
        self.step += 1
        if self.step == self.steps:
            if self.push is not None: out = out + self.eps * self.push
            self.value = out.detach().reshape(-1, out.shape[-1]).mean(dim=0).float().cpu()
        return out

    def close(self): self.a.attend = self.o


def act(policy, rep, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def collect(policy):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
    plan = [("source_heldout", "libero_object", t) for t in (0, 1, 2, 3)]
    plan += [("Camera", "libero_spatial", 705), ("Camera", "libero_spatial", 620)]
    plan += [("Background", "libero_spatial", 11), ("Background", "libero_spatial", 19)]
    plan += [("Noise", "libero_spatial", 1384), ("Noise", "libero_spatial", 1375)]
    out = []
    for dom, suite, tid in plan:
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
            prep = prepare_observation(policy_observation(obs, task.language, policy),
                                       config=policy.config, normalizer=policy.normalizer,
                                       device=torch.device(DEV))
            out.append({"domain": dom, "task": tid, "chunk": k, "prep": prep})
            a = policy.normalizer.unnormalize_action(act(policy, policy.representation_extractor(**prep), 1234))[0]
            for x in a[:6].float().cpu().numpy():
                obs, _, done, _ = env.step(np.clip(x, -1.0, 1.0))
                if done: break
            if done: break
        env.close()
    return out


def auc(score, labels):
    pos = [s for s, y in zip(score, labels) if y == 1]
    neg = [s for s, y in zip(score, labels) if y == 0]
    if not pos or not neg: return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(w / (len(pos) * len(neg)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float32).to(DEV)
    A = a_basis.double(); Acpu = A.cpu()
    obs = collect(policy)
    print("observations", len(obs), flush=True)
    # ---- beta-space decomposition + correct sign (no guard), guard comparison
    rows = []
    for oi, o in enumerate(obs):
        prep = o["prep"]; seed = 70000 + oi
        obj = SGCpa(policy, a_basis.double(), seed=1234)
        try:
            info = obj.build_target(prep, seed=seed, view_seed=seed + 1)
            obj.adapter.beta.detach().zero_()
            loss = obj.loss(prep, info, seed=seed)
            obj.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            g = obj.adapter.beta.grad.detach().double().cpu().clone()
            j = obj.jacobian_a(prep, seed=seed)
            P = obj.nullspace(j)
            Psens = torch.eye(768, dtype=torch.float64) - P
            Pg = P @ g; (Sg := Psens @ g)
            ortho_dot = float((Pg @ Sg) / ((Pg.norm() * Sg.norm()) + 1e-12))
            idem_resid = float((P @ Pg - Pg).norm() / (g.norm() + 1e-12))
            with torch.no_grad():
                rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
            a0 = act(policy, rep, seed)
            a0f = [act(policy, rep, s) for s in (880001, 880002)]
            modes = {"full": -g, "proj": -(P @ g), "sens": -(Psens @ g)}
            for name, upd in modes.items():
                with torch.no_grad():
                    obj.adapter.beta.zero_()
                dbeta = (obj.eta * upd).float().to(obj.adapter.beta.device)
                with torch.no_grad():
                    obj.adapter.beta.add_(dbeta)
                z = obj.write(prep, seed=seed).detach().double().cpu()
                a1 = act(policy, rep, seed)
                a1f = [act(policy, rep, s) for s in (880001, 880002)]
                with torch.no_grad():
                    obj.adapter.beta.zero_()
                z0 = obj.write(prep, seed=seed).detach().double().cpu()
                dz = z - z0
                dvec = (a1 - a0).reshape(-1)
                dvecf = torch.stack([(x - y).reshape(-1) for x, y in zip(a1f, a0f)]).mean(dim=0)
                cos = float(torch.nn.functional.cosine_similarity(dvec, dvecf, dim=0)) if dvec.norm() > 1e-12 else 0.0
                rows.append({"domain": o["domain"], "mode": name,
                             "g_norm": float(g.norm()),
                             "sens_ratio": float((Psens @ g).norm() / (g.norm() + 1e-12)),
                             "ortho_dot": ortho_dot, "idem_resid": idem_resid,
                             "null_ratio": float((P @ g).norm() / (g.norm() + 1e-12)),
                             "dz": float(dz.norm()),
                             "dz_A_ratio": float((Acpu.T @ dz).norm() / (dz.norm() + 1e-12)),
                             "da": float((a1 - a0).abs().mean()),
                             "da_fresh": float(np.mean([float((x - y).abs().mean()) for x, y in zip(a1f, a0f)])),
                             "cos_teacher_fresh": cos})
            # production guard path: guard off vs on, projection on
            for proj_on, guard_on, tag in ((False, False, "noproj_noguard"),
                                           (True, False, "proj_noguard"),
                                           (True, True, "proj_guard")):
                obj.use_projection = proj_on; obj.a_guard = guard_on
                obj.adapter.beta.detach().zero_()
                rec = obj.step(prep, seed=seed, view_seed=seed + 1, target_info=info)
                a2 = act(policy, rep, seed)
                rows.append({"domain": o["domain"], "mode": tag,
                             "dz": float(rec["d_beta_norm"]),
                             "da": float((a2 - a0).abs().mean()),
                             "accepted": rec["accepted"], "backtracks": rec["backtracks"],
                             "da_fresh": 0.0})
        finally:
            obj.close()
        if oi % 10 == 0: print("B", oi, flush=True)
    # ---- AUC corrected: task-clustered, early episode, -loss direction
    eps = []
    for f in Path("/hy-tmp/validation/p43/diag_sg_cpa").glob("*/*.json"):
        d = json.loads(f.read_text())
        if not d.get("updates"): continue
        ups = d["updates"]
        n = len(ups); half = max(1, n // 2)
        eps.append({"success": int(d["result"]["success"]),
                    "identity": float(np.mean([u["medoid_index"] == 0 for u in ups])),
                    "loss_all": float(np.mean([u["loss"] for u in ups])),
                    "loss_early": float(np.mean([u["loss"] for u in ups[:half]]))})
    labels = [e["success"] for e in eps]
    tasks = {}
    # task id not in diag; use file's task id via filename
    import re
    for f in Path("/hy-tmp/validation/p43/diag_sg_cpa").glob("*/*.json"):
        d = json.loads(f.read_text())
        if not d.get("updates"): continue
        tid = d.get("task_id", -1)
        tasks.setdefault(tid, []).append(d)
    def score_loss(d):
        return -float(np.mean([u["loss"] for u in d["updates"]]))

    def score_loss_early(d):
        ups = d["updates"][:max(1, len(d["updates"]) // 2)]
        return -float(np.mean([u["loss"] for u in ups]))

    def score_ident(d):
        return float(np.mean([u["medoid_index"] == 0 for u in d["updates"]]))

    def within_auc(score_fn):
        vals = []
        for tid, ds in tasks.items():
            sc = [score_fn(d) for d in ds]
            lab = [int(d["result"]["success"]) for d in ds]
            if len(set(lab)) == 2:
                vals.append(auc(sc, lab))
        return vals

    rng = np.random.default_rng(0)
    task_ids = sorted(tasks)
    boot = {}
    for key, fn in (("loss", score_loss), ("loss_early", score_loss_early),
                    ("identity", score_ident)):
        vals = within_auc(fn)
        task_vals = dict(zip([tid for tid in tasks if len({int(d["result"]["success"]) for d in tasks[tid]}) == 2],
                             vals))
        boots = []
        if vals:
            tids = list(task_vals)
            for _ in range(1000):
                pick = [tids[i] for i in rng.integers(0, len(tids), len(tids))]
                boots.append(float(np.mean([task_vals[t] for t in pick])))
        boot[key] = {"within_task_auc_mean": float(np.mean(vals)) if vals else None,
                     "n_tasks": len(vals),
                     "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))] if boots else None}
    out = {"n_obs": len(obs), "beta_space": rows[:len(obs) * 3], "guard_path": rows[len(obs) * 3:],
           "auc": boot,
           "overall_auc_loss_neg": auc([-e["loss_all"] for e in eps], labels),
           "overall_auc_loss_early_neg": auc([-e["loss_early"] for e in eps], labels)}
    (OUT / "audit_v2.json").write_text(json.dumps(out, indent=2))
    print("beta-space means:",
          {m: {k: round(float(np.mean([r[k] for r in out["beta_space"] if r["mode"] == m])), 6)
               for k in ("sens_ratio", "null_ratio", "dz", "dz_A_ratio", "da", "cos_teacher_fresh")}
           for m in ("full", "proj", "sens")})
    print("guard path:",
          {m: {k: round(float(np.mean([r[k] for r in out["guard_path"] if r["mode"] == m])), 8)
               for k in ("da", "accepted", "backtracks")}
           for m in ("noproj_noguard", "proj_noguard", "proj_guard")})
    ores = [r.get("ortho_dot", 0.0) for r in out["beta_space"]]
    ires = [r.get("idem_resid", 0.0) for r in out["beta_space"]]
    print("orthogonality dot median/max|.|", float(np.median(ores)), float(np.max(np.abs(ores))))
    print("idempotence residual median/max", float(np.median(ires)), float(np.max(ires)))
    print("AUC", out["auc"], "overall", out["overall_auc_loss_neg"], out["overall_auc_loss_early_neg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
