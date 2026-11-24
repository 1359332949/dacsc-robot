"""Clean source-trajectory residual diagnostic.

Runs Frozen CoRP on unprefixed source Clean episodes and records whether the
source conditional baseline explains visual features as the task progresses.
No target labels, target Clean images, or adaptation are used.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402
from closed_loop_tta_pilot import build_source  # noqa: E402
from target_only_action_validation import configure, sample_z, visual_from_obs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--validation-indices", default="0,1,2")
    ap.add_argument("--source-fit-start", type=int, default=10)
    ap.add_argument("--source-fit-count", type=int, default=10)
    ap.add_argument("--source-folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--content-rank", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == "libero_object")
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == 2222)
    base_bddl = variant["base_bddl"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    fit_ids, source, diag = build_source(args, policy, manifest, z, by_state, device, language)
    fit_context = np.stack([z["state8"][by_state[s]] for s in fit_ids])
    context_mean = fit_context.mean(axis=0)
    context_scale = fit_context.std(axis=0)
    context_scale[context_scale < 1e-8] = 1.0
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    val_ids = [int(i) for i in args.validation_indices.split(",")]
    if set(val_ids) & set(fit_ids):
        raise ValueError("validation indices overlap source fit IDs")
    if not set(val_ids) <= set(dev):
        raise ValueError("validation indices must be source dev indices")
    state_file = task["state_split"]["source_file"]
    initial_states = torch.load(state_file, weights_only=False)
    if any(i < 0 or i >= len(initial_states) for i in val_ids):
        raise ValueError("validation index out of range")
    records = []
    for init_index in val_ids:
        env = OffScreenRenderEnv(bddl_file_name=base_bddl, camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=args.render_gpu)
        seed = args.seed + init_index
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env.seed(seed); env.reset(); obs = env.set_init_state(initial_states[init_index])
        success, done, steps = False, False, 0
        for _ in range(SETTLING_STEPS):
            obs, reward, done, info = env.step(DUMMY_ACTION)
            success = is_success(env, info, reward, done)
            if success or done:
                break
        generator = torch.Generator(device=device); generator.manual_seed(seed)
        chunks = []
        chunk_index = 0
        while not success and not done and steps < args.max_steps:
            po = policy_observation(obs, language, policy)
            visual, rep = visual_from_obs(policy, po)
            pooled = visual.mean(dim=1)
            ctx = po["state"].detach().cpu().numpy()
            pred = torch.as_tensor(source.predict(ctx), device=device, dtype=pooled.dtype)
            residual = pooled - pred
            zctx = (ctx - context_mean) / context_scale
            ctx_dist = float(np.linalg.norm(zctx))
            chunks.append({
                "chunk": chunk_index,
                "steps": steps,
                "raw_residual_norm": float(residual.norm()),
                "visual_norm": float(pooled.norm()),
                "source_pred_norm": float(pred.norm()),
                "context_distance": ctx_dist,
                "support_radius": float(diag["confidence_gate"]["context_support_radius"]),
                "context": ctx[0].astype(float).tolist(),
            })
            z0 = torch.randn((1, policy.action_generator.action_horizon,
                              policy.action_generator.action_dim),
                             device=device, dtype=torch.float32, generator=generator)
            norm = sample_z(policy, rep, z0)
            controller = policy.normalizer.unnormalize_action(norm)[0].float().cpu().numpy()
            for k in range(args.executed_k):
                obs, reward, done, info = env.step(np.clip(controller[k], -1.0, 1.0))
                steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= args.max_steps:
                    break
            chunk_index += 1
        records.append({"init_index": init_index, "success": bool(success),
                        "steps": steps, "chunks": chunks,
                        "residual_norms": [c["raw_residual_norm"] for c in chunks],
                        "context_distances": [c["context_distance"] for c in chunks]})
        print(json.dumps({"init_index": init_index, "success": success,
                          "steps": steps, "max_residual": max(
                              [c["raw_residual_norm"] for c in chunks], default=0.0)}), flush=True)
        env.close()
    payload = {"schema": "source_trajectory_diagnostic_v1", "fit_ids": fit_ids,
               "validation_indices": val_ids, "source_diagnostics": diag,
               "records": records,
               "claim_scope": "Clean source-only diagnostic; no target labels"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
