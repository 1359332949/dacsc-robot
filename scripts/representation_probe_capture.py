"""Capture CoRP features/actions on clean LIBERO or LIBERO-Plus observations.

For each frame the released policy produces:
  * representation tokens z_t (visual-language-state, [T, D]) -> mean over tokens
  * cross-attention write at the last Euler step (mean over action tokens)
  * predicted action chunk (mean over horizon)
The captures feed scripts/representation_probe_analyse.py, which measures the
geometry of appearance/domain shift vs task/action-relevant directions for the
official LIBERO checkpoint (dataset is recorded explicitly).
"""

from __future__ import annotations

import argparse
import json
import re
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
from dacsc.upstream import verify_release_checkpoint  # noqa: E402

BASE_TASK_PATTERNS = (
    "_light_",
    "_table_",
    "_view_",
    "_language_",
    "_add_",
    "_level",
    "_tb_",
    "_noise_",
    "_initstate_",
)


def base_task_name(name: str) -> str:
    for pattern in BASE_TASK_PATTERNS:
        if pattern in name:
            return name.split(pattern)[0]
    return name


class Capture:
    def __init__(self, policy, num_steps: int):
        self.policy = policy
        self.num_steps = num_steps
        self.call = 0
        self.rep = None
        self.write = None
        self.attention = policy.action_generator.blocks[0].attention
        self.original_attend = self.attention.attend
        attention = self.attention
        capture_self = self

        def attend(*args, **kwargs):
            output = capture_self.original_attend(*args, **kwargs)
            index = capture_self.call
            capture_self.call += 1
            if (index + 1) % capture_self.num_steps == 0:
                flat = output.detach().reshape(-1, output.shape[-1])
                capture_self.write = flat.mean(dim=0).float().cpu()
            return output

        attention.attend = attend  # type: ignore[assignment]
        self.hook = policy.representation_extractor.register_forward_hook(self._rep_hook)

    def _rep_hook(self, module, inputs, output):
        tokens = output.tokens.detach()
        self.rep = tokens.reshape(-1, tokens.shape[-1]).mean(dim=0).float().cpu()

    def reset(self):
        self.call = 0
        self.rep = None
        self.write = None

    def close(self):
        self.attention.attend = self.original_attend  # type: ignore[assignment]
        self.hook.remove()


def parse_indices(spec: str) -> list[int]:
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
    ap.add_argument("--dataset", choices=["libero", "liberoplus"], required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", default=None, help="clean LIBERO task indices")
    ap.add_argument("--classification", type=Path, default=None)
    ap.add_argument("--per-category", type=int, default=1)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--chunks-per-episode", type=int, default=2)
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
    capture = Capture(policy, int(policy.config["action_generator"]["num_sample_steps"]))
    suite = benchmark.get_benchmark_dict()[args.suite]()

    if args.dataset == "libero":
        task_ids = parse_indices(args.tasks) if args.tasks else list(range(suite.get_num_tasks()))
        selected = [(None, task_id) for task_id in task_ids]
    else:
        classification = json.loads(args.classification.read_text())[args.suite]
        groups: dict[str, list] = {}
        for item in classification:
            groups.setdefault(item["category"], []).append(item)
        names = suite.get_task_names()
        rng = np.random.default_rng(args.seed)
        selected = []
        for category in sorted(groups):
            items = groups[category]
            picks = rng.choice(len(items), size=min(args.per_category, len(items)), replace=False)
            for index in picks:
                name = items[int(index)]["name"]
                selected.append((category, names.index(name)))

    reps, writes, actions, meta = [], [], [], []
    try:
        for dimension, task_id in selected:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            )
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
                    env.seed(seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[episode])
                    for _ in range(SETTLING_STEPS):
                        observation, reward, done, info = env.step(DUMMY_ACTION)
                    for chunk_index in range(args.chunks_per_episode):
                        batch = policy_observation(observation, task.language, policy)
                        capture.reset()
                        with torch.no_grad():
                            chunk = policy.predict(batch)[0].float().cpu()
                        if capture.rep is None:
                            continue
                        reps.append(capture.rep.numpy())
                        writes.append(capture.write.numpy())
                        actions.append(chunk.mean(dim=0).numpy())
                        meta.append(
                            {
                                "dataset": args.dataset,
                                "suite": args.suite,
                                "dimension": dimension or "clean",
                                "task_id": task_id,
                                "task_name": task.name,
                                "base_task": base_task_name(task.name),
                                "language": task.language,
                                "episode": episode,
                                "chunk": chunk_index,
                            }
                        )
                        for action in chunk.numpy():
                            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                            if done:
                                break
                        if done:
                            break
            finally:
                env.close()
    finally:
        capture.close()

    np.savez(
        args.out,
        reps=np.stack(reps),
        writes=np.stack(writes),
        actions=np.stack(actions),
        meta=np.array(meta, dtype=object),
    )
    print(json.dumps({"frames": len(reps), "tasks": len(selected), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
