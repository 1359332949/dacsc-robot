"""Source statistics for PSC on the released CoRP LIBERO policy.

Statistics are captured from CLEAN LIBERO observations only, following the
official rollout protocol in CoRP/scripts/evaluate_libero.py (same settling
steps, same initial-state indices, same policy_observation conversion). The
captured quantity is the cross-attention write (SelfAttention.attend output),
averaged over action tokens at the LAST Euler step, matching the RoboCasa PSC
probes (V17/V18). Saved mean/basis/sigma_s/tau are consumed by
dacsc.psc.install.
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
    MAX_STEPS,
    SETTLING_STEPS,
    configure_libero,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream, verify_release_checkpoint  # noqa: E402


class CaptureAtLastStep:
    def __init__(self, attention_module, num_steps: int):
        self.attention = attention_module
        self.original = attention_module.attend
        self.num_steps = num_steps
        self.call = 0
        self.value = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if (index + 1) % self.num_steps == 0:
            flat = output.detach().reshape(-1, output.shape[-1])
            self.value = flat.mean(dim=0).float().cpu()
        return output

    def reset(self):
        self.call = 0
        self.value = None

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-2")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--chunks-per-episode", type=int, default=4)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--libero-src", type=Path, default=_CORP_REPO / "third_party" / "LIBERO")
    parser.add_argument("--libero-config-path", type=Path, default=_CORP_REPO / ".cache" / "libero")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    upstream = verify_upstream(["CoRP"])
    checkpoint_digest = verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    def parse_indices(spec: str) -> list[int]:
        out = []
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return out

    policy = load_policy(args.checkpoint, device=args.device)
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    capture = CaptureAtLastStep(policy.action_generator.blocks[0].attention, num_steps)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    features = []
    try:
        for task_id in parse_indices(args.tasks):
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=args.render_gpu,
            )
            try:
                for episode in range(args.episodes_per_task):
                    seed = args.seed + task_id * 1000 + episode
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    env.seed(seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[episode])
                    for _ in range(SETTLING_STEPS):
                        observation, reward, done, info = env.step(DUMMY_ACTION)
                    for _ in range(args.chunks_per_episode):
                        batch = policy_observation(observation, task.language, policy)
                        capture.reset()
                        with torch.no_grad():
                            chunk = policy.predict(batch)[0].float().cpu().numpy()
                        if capture.value is not None:
                            features.append(capture.value)
                        for action in chunk:
                            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                            if done:
                                break
                        if done:
                            break
            finally:
                env.close()
    finally:
        capture.close()

    matrix = torch.stack(features).float()
    mu = matrix.mean(dim=0)
    centred = matrix - mu
    _, s, vh = torch.linalg.svd(centred, full_matrices=False)
    rank = min(args.rank, vh.shape[0])
    basis = vh[:rank].float()
    coords = centred @ basis.T
    residual = centred - coords @ basis
    payload = {
        "schema": "dacsc_psc_source_stats_v1",
        "task": args.suite,
        "suite": args.suite,
        "rank": rank,
        "n_features": int(matrix.shape[0]),
        "mean": mu.tolist(),
        "basis": basis.tolist(),
        "sigma_s": coords.std(dim=0).tolist(),
        "tau": float(residual.pow(2).mean()),
        "singular_values": [float(v) for v in s[:rank]],
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_unchanged": file_sha256(args.checkpoint) == checkpoint_digest,
        "upstream": upstream,
        "source": "clean LIBERO observations (official rollout protocol)",
        "tasks": parse_indices(args.tasks),
        "episodes_per_task": args.episodes_per_task,
        "chunks_per_episode": args.chunks_per_episode,
        "seed": args.seed,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"suite": args.suite, "n": int(matrix.shape[0]), "rank": rank}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
