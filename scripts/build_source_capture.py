"""Capture clean source features for a suite/lighting pair.

Only Clean source train+val states are materialized. The split manifest owns
which states are train/val; this script merely records their features.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from target_only_action_validation import configure, visual_from_obs  # noqa: E402
from dacsc.state_pairing import snapshot  # noqa: E402
from dacsc.tta_protocol import (load_split, role_ids, sha256_file as split_sha256_file,
                                 source_role_fingerprint, validate_split_identity)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--source-train-ids", default="")
    ap.add_argument("--source-val-ids", default="")
    ap.add_argument("--source-fit-ids", default="",
                    help="explicit estimator fit ids; must be subset of captured states")
    ap.add_argument("--split-json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2014)
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
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    split = load_split(args.split_json) if args.split_json else None
    if split is not None:
        validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    base_bddl = variant["base_bddl"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    if split is not None:
        train_ids = role_ids(split, "source_fit")
        val_ids = role_ids(split, "source_validation")
        ids = train_ids + val_ids
    else:
        train_ids = [int(x) for x in args.source_train_ids.split(",") if x.strip()]
        val_ids = [int(x) for x in args.source_val_ids.split(",") if x.strip()]
        if train_ids or val_ids:
            ids = train_ids + val_ids
        else:
            ids = [int(x) for x in args.state_ids.split(",") if x.strip()]
    if args.source_fit_ids:
        fit_ids = [int(x) for x in args.source_fit_ids.split(",") if x.strip()]
        if not set(fit_ids) <= set(ids):
            raise ValueError("source fit ids must be captured")
    else:
        fit_ids = train_ids
    features = {v: [] for v in task["variants"]["lighting"][0].get("policy_views", [])} if False else None
    arrays = {}
    records = []
    for row, state_id in enumerate(ids):
        env = OffScreenRenderEnv(bddl_file_name=base_bddl, camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=args.render_gpu)
        seed = args.seed + state_id
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        env.seed(seed); env.reset(); obs = env.set_init_state(states[state_id])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done: raise RuntimeError(f"episode ended during settling: {state_id}")
        po = policy_observation(obs, language, policy)
        _, _, patches = visual_from_obs(policy, po, return_patches=True)
        for view, p in patches.items():
            arrays.setdefault(f"clean_{view}", []).append(p[0].detach().cpu().numpy().astype(np.float32))
        arrays.setdefault("state8", []).append(po["state"].detach().cpu().numpy()[0].astype(np.float32))
        records.append({"index": state_id, "feature_row": row, "compare_pass": True})
        env.close()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", **{k: np.stack(v) for k, v in arrays.items()})
    views = list(patches.keys())
    manifest = {
        "condition": f"{args.suite}_clean", "policy_views": views,
        "dev_indices": ids, "heldout_indices": [], "records": records,
        "source_train_state_ids": train_ids,
        "source_val_state_ids": val_ids,
        "source_fit_state_ids": fit_ids,
        "num_states": len(ids), "split_seed": args.seed,
        "clean_bddl_sha256": hashlib.sha256(Path(base_bddl).read_bytes()).hexdigest(),
        "checkpoint_sha256": split_sha256_file(args.checkpoint),
        "matrix_sha256": split_sha256_file(args.matrix),
        "initial_state_file_sha256": split_sha256_file(Path(task["state_split"]["source_file"])),
        "split_sha256": split_sha256_file(args.split_json) if args.split_json else None,
        "source_role_fingerprint": source_role_fingerprint(split) if split is not None else None,
        "capture_script_sha256": split_sha256_file(Path(__file__)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
