"""Group-wise action change decomposition for eta calibration (xyz/rot/grip)."""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import configure_libero  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE, collect  # noqa: E402

OUT = Path("/hy-tmp/validation/sg_cpa_eta_cal")
DEV = "cuda:0"


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def sample(policy, prep, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
            return policy.action_generator.sample(rep).detach()


def groups(policy, a0, a1):
    u0 = policy.normalizer.unnormalize_action(a0)[0].double()
    u1 = policy.normalizer.unnormalize_action(a1)[0].double()
    d = (u1 - u0)
    dn = (a1 - a0)[0].double()
    return {
        "ctrl_xyz_l2": float(d[:, :3].norm()),
        "ctrl_rot_l2": float(d[:, 3:6].norm()),
        "ctrl_grip_l2": float(d[:, 6].norm()),
        "ctrl_all_l2": float(d.norm()),
        "norm_xyz_mean": float(dn[:, :3].abs().mean()),
        "norm_rot_mean": float(dn[:, 3:6].abs().mean()),
        "norm_grip_mean": float(dn[:, 6].abs().mean()),
        "norm_all_mean": float(dn.abs().mean()),
    }


def main():
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    batches = collect(policy, n_tasks=1, n_chunks=2)
    etas = [20.0, 100.0]
    targets = []
    for bi, b in enumerate(batches):
        obj = SGCpa(policy, a_basis, seed=1234)
        try:
            targets.append(obj.build_target(b["prepared"], seed=90000 + bi, view_seed=90000 + bi + 1))
        finally:
            obj.close()
    rows = []
    for eta in etas:
        for bi, b in enumerate(batches):
            prep = b["prepared"]; seed = 90000 + bi; vseed = seed + 1
            a0 = sample(policy, prep, seed)
            obj = SGCpa(policy, a_basis, eta=eta, seed=1234)
            try:
                obj.adapter.beta.detach().zero_()
                for step in range(1, 11):
                    obj.step(prep, seed=seed, view_seed=vseed, target_info=targets[bi])
                a1 = sample(policy, prep, seed)
                rows.append({"eta": eta, "batch": bi, **groups(policy, a0, a1)})
            finally:
                obj.close()
        print("eta", eta, "done", flush=True)
    agg = {}
    for eta in etas:
        sel = [r for r in rows if r["eta"] == eta]
        agg[str(eta)] = {k: {"mean": float(np.mean([r[k] for r in sel])),
                             "std": float(np.std([r[k] for r in sel]))}
                         for k in sel[0] if k not in ("eta", "batch")}
    (OUT / "group_decomp.json").write_text(json.dumps({"aggregate": agg, "raw": rows}, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
