"""LIBERO-Plus photometric adapter around official CoRP evaluate_suite.

No duplicated model loading, observation conversion, rollout or success detector.
Standard LIBERO clean evaluation should use CoRP/scripts/evaluate_libero.py directly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO

sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import MAX_STEPS, configure_libero, evaluate_suite
from corp import load_policy
from dacsc.corp_images import SHIFTS, apply_corruption, image_contract
from dacsc.eval_protocol import file_sha256
from dacsc.run_output import entry_output
from dacsc.upstream import verify_upstream, verify_release_checkpoint


class PhotometricPolicy:
    """Change only resized, unnormalized RGB; delegate prediction to released CoRP."""

    def __init__(self, policy, shift, sigma, device):
        self.policy, self.shift, self.sigma, self.device = policy, shift, sigma, device
        self.config, self.normalizer = policy.config, policy.normalizer

    def predict(self, observation):
        if self.shift == "none" and self.sigma == 0:
            return self.policy.predict(observation)
        observation = dict(observation)
        observation["images"] = {
            name: apply_corruption(
                image.to(self.device).permute(0, 3, 1, 2).float() / 255,
                self.shift,
                self.sigma,
            ).permute(0, 2, 3, 1)
            for name, image in observation["images"].items()
        }
        return self.policy.predict(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt"
    )
    parser.add_argument("--suite", choices=list(MAX_STEPS), required=True)
    parser.add_argument("--libero-src", type=Path, required=True)
    parser.add_argument("--libero-config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument(
        "--shifts", nargs="+", choices=SHIFTS, default=["none", "color_jitter"]
    )
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--render-gpu", type=int, default=0)
    args = parser.parse_args()
    if min(args.per_category, args.episodes_per_task) <= 0:
        parser.error("task/episode support must be positive")
    if len(set(args.shifts)) != len(args.shifts) or len(set(args.blur_sigmas)) != len(
        args.blur_sigmas
    ):
        parser.error("conditions must be unique")
    # Validate image parameters before importing or creating a simulator.
    for sigma in args.blur_sigmas:
        apply_corruption(torch.zeros(3, 224, 224), "none", sigma)
    with entry_output(args.output_dir) as out:
        upstream = verify_upstream(["CoRP"])
        configure_libero(args)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.suite]()
        names = {name: i for i, name in enumerate(suite.get_task_names())}
        metadata = json.loads(
            (
                args.libero_src / "libero/libero/benchmark/task_classification.json"
            ).read_text()
        )[args.suite]
        groups = {}
        for item in metadata:
            groups.setdefault(item["category"], []).append(item)
        rng = random.Random(args.seed)
        selected = []
        for category, items in sorted(groups.items()):
            if len(items) < args.per_category:
                raise ValueError(f"insufficient tasks in {category}")
            selected.extend(rng.sample(items, args.per_category))
        task_ids = [names[item["name"]] for item in selected]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("selected tasks must be nonempty and unique")
        args.tasks = ",".join(map(str, task_ids))
        checkpoint_digest = verify_release_checkpoint(args.checkpoint)
        artifacts = {}
        for task_id in task_ids:
            task = suite.get_task(task_id)
            for kind, filename in (
                ("bddl_files", task.bddl_file),
                ("init_states", task.init_states_file),
            ):
                path = Path(get_libero_path(kind)) / task.problem_folder / filename
                artifacts[str(path)] = file_sha256(path)
        policy = load_policy(args.checkpoint, device=args.device)
        results = {}
        for shift in args.shifts:
            for sigma in args.blur_sigmas:
                condition = f"{shift}_blur_{sigma:g}"
                current = copy.copy(args)
                current.output_dir = out / condition
                current.output_dir.mkdir()
                wrapped = PhotometricPolicy(policy, shift, sigma, args.device)
                results[condition] = evaluate_suite(
                    current,
                    args.suite,
                    wrapped,
                    benchmark,
                    get_libero_path,
                    OffScreenRenderEnv,
                )
                results[condition].update(shift=shift, blur_sigma=sigma)
        if file_sha256(args.checkpoint) != checkpoint_digest:
            raise RuntimeError("checkpoint changed during evaluation")
        if any(file_sha256(Path(path)) != digest for path, digest in artifacts.items()):
            raise RuntimeError(
                "LIBERO task or initial-state files changed during evaluation"
            )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "suite": args.suite,
                    "paired": True,
                    "pairing": "official per-task seed and identical initial-state index",
                    "checkpoint_sha256": checkpoint_digest,
                    "images": image_contract(),
                    "upstream": upstream,
                    "selection": "seeded random tasks within category",
                    "tasks": selected,
                    "model_origin": "official CoRP LIBERO release",
                    "task_artifacts": artifacts,
                    "episodes_per_task": args.episodes_per_task,
                    "conditions": results,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
