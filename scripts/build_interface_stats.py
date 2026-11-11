"""Build source statistics and the action-sensitive subspace for ISC-AP.

Site: the cross-attention write (`action_generator.blocks[0].attention.attend`
output at the last Euler step). Source data: clean LIBERO rollouts.

  * source write statistics: mean, PCA basis V (K x D), support covariance
  * action-sensitive subspace: finite-difference pushes `w <- w + eps*d` at the
    last Euler step for m random orthonormal directions d, recording action
    changes; the Jacobian row space is recovered by least squares and its
    energy-90% rank r gives the action-sensitive directions A (D x r).

Formulas follow the moment-matching / whitening-coloring family (CORAL,
AdaBN, SSA/PSC spectral alignment); the difference of ISC-AP is the site (write
interface) and the explicit action-subspace protection.
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
    def __init__(self, attention, num_steps: int, action_horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.action_horizon = action_horizon
        self.step = 0
        self.value = None
        self.push = None
        self.eps = 0.0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        # verified call pattern: call 0 = prefix (representation tokens),
        # Euler steps are calls 1..num_steps with T == action_horizon.
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        is_last = self.step == self.num_steps
        if is_last and self.push is not None:
            output = output + self.eps * self.push
        if is_last:
            self.value = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu()
        return output

    def reset(self):
        self.step = 0
        self.value = None

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def parse_tasks(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--chunks-per-episode", type=int, default=2)
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--push-directions", type=int, default=24)
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    hook = WriteHook(policy.action_generator.blocks[0].attention, num_steps,
                     int(policy.config["action_horizon"]))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    rng = np.random.default_rng(args.seed)
    writes: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    pushes: list[np.ndarray] = []
    try:
        for task_id in parse_tasks(args.tasks):
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                seed = args.seed + task_id * 1000
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                observation = env.set_init_state(initial_states[0])
                for _ in range(SETTLING_STEPS):
                    observation, reward, done, info = env.step(DUMMY_ACTION)
                for _ in range(args.chunks_per_episode):
                    batch = policy_observation(observation, task.language, policy)
                    hook.reset()
                    # fix the sampling noise: otherwise the finite-difference
                    # response is dominated by flow-sampling noise and the
                    # action subspace estimate is meaningless
                    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(1234)
                        with torch.no_grad():
                            base_action = policy.predict(batch)[0].float().cpu()
                    write = hook.value
                    if write is None:
                        break
                    writes.append(write.numpy())
                    directions = torch.tensor(
                        rng.normal(size=(args.push_directions, write.shape[0])), dtype=torch.float32, device=args.device
                    )
                    directions = directions / directions.norm(dim=1, keepdim=True)
                    eps = 0.05 * float(write.norm())
                    frame_responses = []
                    for direction in directions:
                        hook.reset()
                        hook.push = direction
                        hook.eps = eps
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(1234)
                            with torch.no_grad():
                                pushed = policy.predict(batch)[0].float().cpu()
                        hook.push = None
                        # normalise by the push scale so frames with different
                        # write norms contribute consistently to the Jacobian fit
                        frame_responses.append(((pushed - base_action).reshape(-1).numpy() / eps))
                    responses.append(np.stack(frame_responses))
                    pushes.append(directions.cpu().numpy())
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

    writes_arr = np.stack(writes).astype(np.float64)
    mean = writes_arr.mean(axis=0)
    centred = writes_arr - mean
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order][:, : args.rank].T  # [K, D]
    support_cov = basis @ covariance @ basis.T

    # action-sensitive subspace: stack pushes and responses, least-squares J, SVD
    push_mat = np.concatenate(pushes, axis=0)          # [N, D]
    resp_mat = np.concatenate(responses, axis=0)       # [N, d_a]
    jacobian = np.linalg.lstsq(push_mat, resp_mat, rcond=None)[0].T  # [d_a, D]
    # (resp_mat is already divided by eps: the least-squares solution is then a
    #  finite-difference estimate of d(action)/d(write), independent of push size)
    uj, sj, vhj = np.linalg.svd(jacobian, full_matrices=False)
    energy = np.cumsum(sj ** 2) / np.sum(sj ** 2)
    rank_action = int(np.searchsorted(energy, args.energy) + 1)
    action_dirs = vhj[:rank_action].T  # [D, r]

    payload = {
        "schema": "interface_stats_v1",
        "model": "official CoRP LIBERO checkpoint",
        "site": "cross-attention write (last Euler step)",
        "suite": args.suite,
        "tasks": parse_tasks(args.tasks),
        "n_frames": int(writes_arr.shape[0]),
        "rank": args.rank,
        "mean": mean.tolist(),
        "basis": basis.tolist(),                    # [K, D]
        "support_cov": support_cov.tolist(),        # [K, K]
        "action_subspace": action_dirs.tolist(),    # [D, r]
        "action_rank": rank_action,
        "action_singular_values": sj[:rank_action].tolist(),
        "push_directions": args.push_directions,
    }
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"frames": int(writes_arr.shape[0]), "rank": args.rank,
                      "action_rank": rank_action, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
