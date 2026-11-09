"""Frozen DINOv2 block-feature linear style probe for source training styles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.run_output import entry_file  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.check_data_layout import default_root  # noqa: E402
from train_corp_robocasa import all_task_paths, build_dataset  # noqa: E402


def parse_ints(values) -> list[int]:
    if isinstance(values, (str, int)):
        values = [values]
    result = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo, hi = token.split("-", 1)
                result.extend(range(int(lo), int(hi) + 1))
            else:
                result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise ValueError("integer lists must be nonempty and unique")
    return result


def stable_bucket(key: str, buckets: int = 5) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "little") % buckets


def sample_source_frames(dataset, frames_per_episode: int, seed: int) -> list[dict]:
    grouped: dict[tuple[int, int], list[tuple[int, tuple]]] = {}
    for dataset_index, entry in enumerate(dataset.flat):
        ds_index, _, style, _, task, episode, step, _ = entry
        grouped.setdefault((int(ds_index), int(episode)), []).append(
            (dataset_index, (int(style), str(task), int(step)))
        )
    rows = []
    for (ds_index, episode), values in sorted(grouped.items()):
        values.sort(key=lambda item: item[1][2])
        count = min(frames_per_episode, len(values))
        positions = torch.linspace(0, len(values) - 1, count).round().long().tolist()
        for position in positions:
            dataset_index, (style, task, step) = values[position]
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "dataset": ds_index,
                    "episode": episode,
                    "style": style,
                    "task": task,
                    "step": step,
                    "frame_holdout": stable_bucket(
                        f"frame:{seed}:{ds_index}:{episode}:{step}"
                    )
                    == 0,
                    "episode_holdout": stable_bucket(
                        f"episode:{seed}:{ds_index}:{episode}"
                    )
                    == 0,
                }
            )
    return rows


@torch.no_grad()
def extract_block_features(
    policy,
    dataset,
    rows: list[dict],
    layers: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    backbone = policy.representation_extractor.visual_encoder.backbone
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output.detach()

        return hook

    for layer in layers:
        handles.append(backbone.blocks[layer].register_forward_hook(make_hook(layer)))
    outputs = {layer: [] for layer in layers}
    views = len(policy.view_names)
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            samples = [dataset[row["dataset_index"]] for row in batch_rows]
            flat_images = torch.cat(
                [
                    torch.stack([sample["images"][name] for sample in samples])
                    for name in policy.view_names
                ],
                dim=0,
            ).to(device)
            captured.clear()
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                backbone(flat_images)
            batch = len(samples)
            for layer in layers:
                if layer not in captured:
                    raise RuntimeError(f"DINO block {layer} was not captured")
                value = captured[layer]
                if value.ndim != 3 or value.shape[0] != views * batch:
                    raise ValueError("unexpected DINO block feature shape")
                outputs[layer].append(
                    value.float().reshape(views, batch, value.shape[1], value.shape[2])
                    .mean(dim=(0, 2))
                    .cpu()
                )
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(values, dim=0) for layer, values in outputs.items()}


def balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    recalls = []
    for class_id in target.unique(sorted=True).tolist():
        mask = target.eq(class_id)
        recalls.append(float(prediction[mask].eq(class_id).float().mean()))
    return sum(recalls) / len(recalls)


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    seed: int,
) -> dict:
    x_train = features[train_mask]
    y_train = labels[train_mask]
    x_test = features[test_mask]
    y_test = labels[test_mask]
    if not len(x_train) or not len(x_test):
        raise ValueError("probe split is empty")
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    torch.manual_seed(seed)
    classifier = nn.Linear(features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=0.03, weight_decay=1e-3
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(classifier(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = classifier(x_test).argmax(dim=1)
        return {
            "seed": seed,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "train_classes": int(y_train.unique().numel()),
            "test_classes": int(y_test.unique().numel()),
            "top1": float(prediction.eq(y_test).float().mean()),
            "balanced_accuracy": balanced_accuracy(prediction, y_test),
            "final_train_loss": float(loss),
        }


def summarize(metrics: list[dict]) -> dict:
    values = torch.tensor([item["balanced_accuracy"] for item in metrics])
    return {
        "seeds": metrics,
        "balanced_accuracy_mean": float(values.mean()),
        "balanced_accuracy_std": float(values.std(unbiased=False)),
        "balanced_accuracy_min": float(values.min()),
        "balanced_accuracy_max": float(values.max()),
    }


def run(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    tasks = all_task_paths(args.max_tasks, args.data_root, args.tasks)
    dataset = build_dataset(
        tasks,
        (11, 50),
        10,
        224,
        True,
        normalize_images=True,
        cache_videos=args.cache_videos,
        cache_video_tasks=set(Path(path).parent.parent.name for path in tasks)
        if args.cache_videos
        else None,
    )
    rows = sample_source_frames(dataset, args.frames_per_episode, args.seed)
    styles = sorted({row["style"] for row in rows})
    style_to_label = {style: index for index, style in enumerate(styles)}
    labels = torch.tensor([style_to_label[row["style"]] for row in rows])
    features = extract_block_features(
        model,
        dataset,
        rows,
        args.layers,
        args.batch_size,
        device,
    )
    chance = 1.0 / len(styles)
    result = {
        "schema": "corp_style_probe_v1",
        "model": str(Path(args.model).resolve()),
        "model_sha256": file_sha256(Path(args.model)),
        "upstream": verify_upstream(["CoRP"]),
        "tasks": [Path(path).parent.parent.name for path in tasks],
        "train_styles": [11, 50],
        "selected_frames": len(rows),
        "selected_episodes": len(
            {(row["dataset"], row["episode"]) for row in rows}
        ),
        "styles": styles,
        "chance_balanced_accuracy": chance,
        "probe": {
            "architecture": "Linear(384, 40)",
            "feature": "mean over views and block tokens including CLS; frozen FP32",
            "normalization": "train-split z-score per feature dimension",
            "optimizer": "AdamW",
            "lr": 0.03,
            "weight_decay": 1e-3,
            "steps": args.probe_steps,
            "seeds": args.probe_seeds,
        },
        "splits": {},
    }
    for split_name, key in (
        ("frame_holdout", "frame_holdout"),
        ("episode_holdout", "episode_holdout"),
    ):
        test_mask = torch.tensor([row[key] for row in rows], dtype=torch.bool)
        train_mask = ~test_mask
        split_result = {}
        for layer in args.layers:
            metrics = [
                fit_linear_probe(
                    features[layer],
                    labels,
                    train_mask,
                    test_mask,
                    num_classes=len(styles),
                    steps=args.probe_steps,
                    seed=seed,
                )
                for seed in args.probe_seeds
            ]
            split_result[str(layer)] = summarize(metrics)
        result["splits"][split_name] = split_result
    episode_best = max(
        item["balanced_accuracy_mean"]
        for item in result["splits"]["episode_holdout"].values()
    )
    result["gate"] = {
        "episode_holdout_best_balanced_accuracy": episode_best,
        "near_random_upper": 2.0 * chance,
        "usable_signal_lower": 5.0 * chance,
        "passes_near_random_gate": episode_best > 2.0 * chance,
        "passes_usable_signal_gate": episode_best >= 5.0 * chance,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["gate"], indent=2), flush=True)
    print("DONE ->", args.out, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", type=Path, default=default_root())
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--frames-per-episode", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layers", nargs="+", default=["0-11"])
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--probe-seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-videos", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.layers = parse_ints(args.layers)
    args.probe_seeds = parse_ints(args.probe_seeds)
    if min(args.frames_per_episode, args.batch_size, args.probe_steps) <= 0:
        ap.error("frame, batch and probe-step counts must be positive")
    with entry_file(Path(args.out)):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
