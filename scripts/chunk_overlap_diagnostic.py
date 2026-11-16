"""Kill-test diagnostic for predictor-free chunk-overlap TTA (no success claim)."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from dacsc.visual_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.tta_policy import TTAPolicy  # noqa: E402
from dacsc.chunk_overlap import ChunkOverlapUpdater  # noqa: E402


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    cp = cfg / "config.yaml"
    if not cp.is_file():
        cp.write_text(yaml.safe_dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
                                      "init_states": str(root / "init_files"), "assets": str(root / "assets"),
                                      "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--executed-k", type=int, default=6)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == "libero_object")
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == 2222)
    base_bddl = variant["base_bddl"]; shift_bddl = variant["env_bddl_name"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    init = torch.load(task["state_split"]["source_file"], weights_only=False)
    held = task["state_split"]["heldout_indices"][:args.states]
    policy = load_policy(args.checkpoint, device=args.device); policy.eval()
    env_c = OffScreenRenderEnv(bddl_file_name=base_bddl, camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    env_s = OffScreenRenderEnv(bddl_file_name=shift_bddl, camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    out = []
    try:
        for idx in held:
            for domain, env, reset in (("clean", env_c, True), ("shift", env_s, False)):
                if reset: env.seed(0); env.reset()
                obs = env.set_init_state(init[int(idx)])
                for _ in range(SETTLING_STEPS):
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    if done: raise RuntimeError("settling done")
                adapter = ResidualVisualAdapter(("agentview", "eye_in_hand"), dim=384,
                                                context_dim=0, hidden_dim=32)
                tta = TTAPolicy(policy, adapter)
                updater = ChunkOverlapUpdater(policy, tta.extractor, adapter, lr=1e-5)
                z0 = torch.randn(1, policy.action_generator.action_horizon,
                                 policy.action_generator.action_dim,
                                 device=next(policy.parameters()).device)
                ref = updater.set_reference(policy_observation(obs, language, policy), z0)
                actions = ref["executed"][0].detach().cpu().numpy()
                for a in actions[:args.executed_k]:
                    obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                    if done: break
                after = policy_observation(obs, language, policy)
                res = updater.update(after, args.executed_k, z0=z0)
                out.append({"init_index": int(idx), "domain": domain, **res})
                print(json.dumps(out[-1]), flush=True)
    finally:
        env_c.close(); env_s.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"records": len(out), "all_updated": all(r.get("updated") for r in out),
                      "mean_delta_loss": float(np.mean([r.get("delta_loss", 0) for r in out]))}, indent=2))


if __name__ == "__main__":
    main()
