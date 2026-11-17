"""Collect paired Clean/Shift first action chunks for source states."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure, visual_from_obs  # noqa: E402
from dacsc.tta_protocol import load_split, role_ids, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--roles", default="source_fit,source_validation")
    ap.add_argument("--seed", type=int, default=2014)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    ids = []
    for role in args.roles.split(","):
        ids.extend(role_ids(split, role.strip()))
    ids = list(dict.fromkeys(ids))
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    records = []
    for state_id in ids:
        rec = {"state_id": state_id, "seed": args.seed, "domains": {}}
        for domain, bddl in (("clean", variant["base_bddl"]), ("shift", variant["env_bddl_name"])):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.seed); torch.manual_seed(args.seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
                env.seed(args.seed); env.reset(); obs = env.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                po = policy_observation(obs, language, policy)
                state8 = po["state"][0].detach().cpu().numpy().astype(float)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    chunk = policy.predict(po)[0].float().cpu().numpy().astype(float)
                visual, _ = visual_from_obs(policy, po)
                rec["domains"][domain] = {
                    "actions": chunk.tolist(), "state8": state8.tolist(),
                    "visual_mean": visual.mean(dim=1)[0].detach().float().cpu().numpy().tolist(),
                }
            finally:
                env.close()
        records.append(rec)
        print(json.dumps({"state_id": state_id, "ok": True}), flush=True)
    payload = {"schema": "first_chunk_pair_v1", "suite": args.suite, "lighting_id": args.lighting_id,
               "seed": args.seed, "ids": ids, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(records)}))


if __name__ == "__main__":
    main()
