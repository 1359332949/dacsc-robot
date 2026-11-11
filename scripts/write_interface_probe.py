"""Probe the CoRP cross-attention write (action-conditioning interface).

For each frame we capture the representation tokens and the write produced by
`action_generator.blocks[0].attention.attend` at the last Euler step. We then
re-run the policy while injecting a small push `eps * u` into that write at the
last Euler step, for a set of probe directions:

  * 12 random orthonormal directions  (random/action-agnostic)
  * 6 appearance directions           (source-write PCA basis, PSC stats)

Outputs: response magnitudes (||delta action|| / eps), the appearance/random
ratio, the rank of the action responses (action-sensitive subspace), and the
mean write norm. This re-measures the V4/V8 interface findings on the official
LIBERO checkpoint and LIBERO-Plus data (the earlier V-numbers were measured on
our own short-trained RoboCasa model and must not be reused).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION,
    SETTLING_STEPS,
    configure_libero,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


class WriteHook:
    """Capture the write and optionally push `eps * direction` at the last step."""

    def __init__(self, attention, num_steps: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.call = 0
        self.value = None
        self.push = None
        self.eps = 0.0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        is_last = (index + 1) % self.num_steps == 0
        if is_last and self.push is not None:
            output = output + self.eps * self.push
        if is_last:
            flat = output.detach().reshape(-1, output.shape[-1])
            self.value = flat.mean(dim=0).float().cpu()
        return output

    def reset(self):
        self.call = 0
        self.value = None

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--classification", type=Path,
                    default=Path("/hy-tmp/libero_plus/libero/libero/benchmark/task_classification.json"))
    ap.add_argument("--category", default="Camera Viewpoints")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--frames-per-task", type=int, default=2)
    ap.add_argument("--random-directions", type=int, default=12)
    ap.add_argument("--psc-stats", type=Path, default=Path("/hy-tmp/validation/libero_psc_stats.json"))
    ap.add_argument("--eps-scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config-path", type=Path, default=Path("/hy-tmp/libero_plus_config"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    hook = WriteHook(policy.action_generator.blocks[0].attention, num_steps)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    classification = json.loads(args.classification.read_text())[args.suite]
    chosen = [
        item for item in classification
        if item["category"] == args.category and (item.get("difficulty_level") or 0) == args.level
    ][: args.tasks]
    stats = json.loads(args.psc_stats.read_text())
    appearance = torch.tensor(stats["basis"], dtype=torch.float32, device=args.device)
    rng = np.random.default_rng(args.seed)
    random_dirs = torch.tensor(rng.normal(size=(args.random_directions, appearance.shape[1])), dtype=torch.float32, device=args.device)
    random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True)

    records = []
    try:
        for item in chosen:
            task_id = names.index(item["name"])
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                seed = args.seed + task_id * 1000
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                observation = env.set_init_state(states[0])
                for _ in range(SETTLING_STEPS):
                    observation, reward, done, info = env.step(DUMMY_ACTION)
                for frame in range(args.frames_per_task):
                    batch = policy_observation(observation, task.language, policy)
                    hook.reset()
                    with torch.no_grad():
                        base_action = policy.predict(batch)[0].float().cpu()
                    write = hook.value
                    if write is None:
                        break
                    eps = args.eps_scale * float(write.norm())
                    mean_s = torch.tensor(stats["mean"], dtype=torch.float32, device=args.device)
                    shift = (write.to(args.device) - mean_s)
                    shift_dir = shift / shift.norm().clamp_min(1e-8)
                    directions = torch.cat([appearance, random_dirs, shift_dir.unsqueeze(0)], dim=0)
                    responses = []
                    action_changes = []
                    for u in directions:
                        hook.reset()
                        hook.push = u
                        hook.eps = eps
                        with torch.no_grad():
                            pushed = policy.predict(batch)[0].float().cpu()
                        hook.push = None
                        delta = (pushed - base_action).reshape(-1).numpy()
                        responses.append(float(np.linalg.norm(delta) / eps))
                        action_changes.append(delta)
                    records.append({
                        "task": item["name"],
                        "frame": frame,
                        "eps": eps,
                        "write_norm": float(write.norm()),
                        "appearance_responses": responses[: appearance.shape[0]],
                        "random_responses": responses[appearance.shape[0]: -1],
                        "shift_direction_response": responses[-1],
                        "action_change_rank": int(np.linalg.matrix_rank(np.stack(action_changes), tol=1e-6)),
                    })
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                    for action in chunk:
                        observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                        if done:
                            break
                    if done:
                        break
            finally:
                env.close()
    finally:
        hook.close()

    app = np.concatenate([r["appearance_responses"] for r in records])
    rnd = np.concatenate([r["random_responses"] for r in records])
    shift_resp = np.array([r["shift_direction_response"] for r in records])
    payload = {
        "schema": "write_interface_probe_v1",
        "model": "official CoRP LIBERO checkpoint",
        "dataset": f"LIBERO-Plus {args.suite} {args.category} L{args.level}",
        "frames": len(records),
        "appearance_mean_response": float(app.mean()),
        "random_mean_response": float(rnd.mean()),
        "appearance_over_random": float(app.mean() / max(rnd.mean(), 1e-9)),
        "shift_direction_response_mean": float(shift_resp.mean()),
        "shift_over_random": float(shift_resp.mean() / max(rnd.mean(), 1e-9)),
        "action_change_rank_max": int(max(r["action_change_rank"] for r in records)) if records else 0,
        "records": records,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "frames", "appearance_mean_response", "random_mean_response",
        "appearance_over_random", "action_change_rank_max")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
