"""Standard NumPy/SciPy summaries and strict alignment of existing evaluations.

No model inference, training, custom robustness scores or automatic significance
verdicts. Confidence intervals use training seeds as the independent units.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from dacsc.action_metrics import MSE_SCHEMA, action_mse_summary
from dacsc.eval_protocol import require_digest, validate_training_origin

STYLE_SPLITS = {
    "source": list(range(11, 31)),
    "val": list(range(31, 41)),
    "test": list(range(41, 51)),
}


def seed_summary(values: dict[int, float], confidence: float = 0.95) -> dict:
    if (
        not 0 < confidence < 1
        or not values
        or any(type(seed) is not int for seed in values)
    ):
        raise ValueError(
            "unique integer training seeds and a valid confidence level are required"
        )
    seeds = sorted(values)
    data = np.asarray([values[s] for s in seeds], dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite seed estimate")
    out = {
        "n_training_seeds": len(data),
        "training_seeds": seeds,
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if len(data) > 1 else None,
        "confidence_level": confidence,
        "confidence_interval": None,
        "ci_method": "scipy.stats.t.interval + scipy.stats.sem(ddof=1)",
        "ci_unit": "training_seed",
        "assumption": "approximately normal independent seed estimates; finite evaluation set held fixed",
        "per_seed": [{"training_seed": s, "value": float(values[s])} for s in seeds],
    }
    if len(data) < 2:
        out["ci_status"] = "not_estimated_single_seed"
    elif np.ptp(data) <= 16 * np.finfo(np.float64).eps * max(
        np.finfo(np.float64).tiny, float(np.max(np.abs(data)))
    ):
        out["ci_status"] = "not_estimated_zero_observed_seed_variation"
    else:
        interval = stats.t.interval(
            confidence,
            df=len(data) - 1,
            loc=np.mean(data),
            scale=stats.sem(data, ddof=1),
        )
        out.update(
            confidence_interval=[float(x) for x in interval], ci_status="estimated"
        )
    return out


def _training_identity(origin: dict, *, include_intervention: bool) -> dict:
    validate_training_origin(origin)
    identity = copy.deepcopy(origin)
    identity.pop("seed")
    if include_intervention:
        if identity["intervention"] is not None:
            identity["intervention"].pop("donor_seed")
    else:
        identity.pop("intervention")
    return identity


def _sr_case(directory: Path, tasks: list[str], episodes: int, split: str):
    # Reuse the production validator/official success counts, not a second SR formula.
    from scripts.aggregate_sr import agg

    case = agg(directory, tasks, episodes)
    if any(t["styles"] != STYLE_SPLITS[split] for t in case["tasks"].values()):
        raise ValueError(
            "SR cases must use the same declared split; source/val are not paired scenes"
        )
    rows = {}
    for r in case["records"]:
        key = (r["task"], r["manifest_sha256"], r["policy_seed"])
        if key in rows:
            raise ValueError("duplicate frozen scene in an SR case")
        rows[key] = (r["actual_style"], r["actual_layout"], r["environment_seed"])
    meta = copy.deepcopy(case["protocol"])
    meta["task_horizons"] = {
        task: value["max_steps"] for task, value in case["tasks"].items()
    }
    return case["macro_task_success_rate"], meta, rows


def _mse_case(path: Path, split: str, condition: str):
    case = json.loads(path.read_text())
    if case.get("schema") != MSE_SCHEMA or "checkpoint_sha256" not in case:
        raise ValueError("unsupported MSE schema; rerun evaluation")
    require_digest(case["checkpoint_sha256"])
    validate_training_origin(case["training"])
    selected = case["splits"][split]
    metric = selected["conditions"][condition]
    if metric["schema"] != MSE_SCHEMA:
        raise ValueError("unsupported action record schema")
    if metric["action_space"] != "raw_unclipped_action_commands":
        raise ValueError(
            "MSE action command space differs; no mixing normalized/physical/clipped errors"
        )
    derived = action_mse_summary(metric["records"])
    for key in ("micro_action_mse", "macro_task_action_mse"):
        if not math.isclose(metric[key], derived[key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("MSE summary does not match its frame records")
    rows = {}
    for r in derived["records"]:
        if r["style_id"] not in STYLE_SPLITS[split] or (
            split == "source" and r["episode"] % 10 != 9
        ):
            raise ValueError("MSE frame violates the declared held-out split")
        if not 0 < r["valid_action_positions"] <= metric["prediction_horizon"]:
            raise ValueError("invalid forecast support")
        rows[(r["task"], r["episode"], r["frame"])] = (
            r["style_id"],
            r["valid_action_positions"],
        )
    support = selected["support"]
    if (
        support["frames"] != derived["samples"]
        or support["episodes"] != derived["episodes"]
    ):
        raise ValueError(
            "MSE support does not match its independent episode/frame records"
        )
    require_digest(support["metadata_sha256"])
    require_digest(support["selection_sha256"])
    meta = {
        k: case[k]
        for k in (
            "checkpoint_sha256",
            "training",
            "images",
            "upstream",
            "precision",
            "euler_steps",
            "eval_seed",
            "evaluation_batch_size",
        )
    }
    meta.update(
        condition=condition,
        prediction_horizon=metric["prediction_horizon"],
        metadata_sha256=support["metadata_sha256"],
        selection_sha256=support["selection_sha256"],
    )
    return derived["macro_task_action_mse"], meta, rows


def compare_evaluations(manifest: dict, *, confidence: float = 0.95) -> dict:
    """Manifest explicitly selects matching seed pairs; never infer seeds from paths."""
    kind = manifest.get("kind")
    comparison = manifest.get("comparison")
    split = manifest.get("split")
    if (
        kind not in ("sr", "mse")
        or comparison not in ("methods", "conditions")
        or split not in STYLE_SPLITS
    ):
        raise ValueError(
            "declare kind=sr/mse, comparison=methods/conditions and one common split"
        )
    expected = {"kind", "comparison", "split", "runs"} | (
        {"tasks", "episodes_per_task"}
        if kind == "sr"
        else {"left_condition", "right_condition"}
    )
    if set(manifest) != expected:
        raise ValueError(
            "comparison manifest accepts only declared standard metric fields; custom weights/scores are unsupported"
        )
    runs = manifest["runs"]
    if not runs:
        raise ValueError("no paired runs")
    left = {}
    right = {}
    paired = []
    reference_rows = None
    reference_identity = None
    for run in runs:
        if set(run) != {"training_seed", "left", "right"}:
            raise ValueError(
                "each run must explicitly declare one seed and its left/right evaluations"
            )
        seed = run["training_seed"]
        if type(seed) is not int or seed in left:
            raise ValueError("duplicate/noninteger training seed")
        if kind == "sr":
            a, ma, ka = _sr_case(
                Path(run["left"]),
                manifest["tasks"],
                manifest["episodes_per_task"],
                split,
            )
            b, mb, kb = _sr_case(
                Path(run["right"]),
                manifest["tasks"],
                manifest["episodes_per_task"],
                split,
            )
            condition_keys = {"shift", "blur_sigma"}
        else:
            a, ma, ka = _mse_case(Path(run["left"]), split, manifest["left_condition"])
            b, mb, kb = _mse_case(
                Path(run["right"]), split, manifest["right_condition"]
            )
            condition_keys = {"condition"}
        if ma["training"]["seed"] != seed or mb["training"]["seed"] != seed:
            raise ValueError(
                "manifest seed differs from actual checkpoint training seed"
            )
        if ka != kb:
            raise ValueError(
                "unpaired scene/frame identity, support or policy seed; no dropping samples"
            )
        if reference_rows is not None and ka != reference_rows:
            raise ValueError(
                "evaluation sample set must stay fixed across training seeds"
            )
        reference_rows = ka
        if _training_identity(
            ma["training"], include_intervention=False
        ) != _training_identity(mb["training"], include_intervention=False):
            raise ValueError("training budget, recipe, data or initialization differs")
        if comparison == "methods":
            if any(ma[k] != mb[k] for k in condition_keys):
                raise ValueError(
                    "method comparison requires identical evaluation conditions"
                )
            ignored = {"checkpoint_sha256", "training"}
        else:
            if (
                ma["checkpoint_sha256"] != mb["checkpoint_sha256"]
                or ma["training"] != mb["training"]
            ):
                raise ValueError("condition comparison requires the same checkpoint")
            if all(ma[k] == mb[k] for k in condition_keys):
                raise ValueError("condition comparison needs two distinct conditions")
            ignored = condition_keys
        if {k: v for k, v in ma.items() if k not in ignored} != {
            k: v for k, v in mb.items() if k not in ignored
        }:
            raise ValueError("evaluation runtime/inference protocol differs")
        identity = []
        for meta in (ma, mb):
            identity.append(
                {
                    **{
                        k: v
                        for k, v in meta.items()
                        if k not in ("checkpoint_sha256", "training")
                    },
                    "training": _training_identity(
                        meta["training"], include_intervention=True
                    ),
                }
            )
        if reference_identity is not None and identity != reference_identity:
            raise ValueError("method/configuration changed across training seeds")
        reference_identity = identity
        left[seed] = float(a)
        right[seed] = float(b)
        paired.append(
            {
                "training_seed": seed,
                "left_checkpoint_sha256": ma["checkpoint_sha256"],
                "right_checkpoint_sha256": mb["checkpoint_sha256"],
                "pairs": len(ka),
                "tasks": len({key[0] for key in ka}),
            }
        )
    delta = {seed: float(np.subtract(right[seed], left[seed])) for seed in left}
    output = {
        "schema": "corp_standard_comparison_v1",
        "kind": kind,
        "comparison": comparison,
        "split": split,
        "metric": (
            "macro_task_success_rate" if kind == "sr" else "macro_task_action_mse"
        ),
        "unit": "fraction" if kind == "sr" else "raw_action_command_squared",
        "difference_direction": "right_minus_left",
        "positive_difference_means": "right_better" if kind == "sr" else "right_worse",
        "left": seed_summary(left, confidence),
        "right": seed_summary(right, confidence),
        "paired_difference": seed_summary(delta, confidence),
        "pair_support": paired,
        "evaluation_uncertainty": "not an episode/frame-level CI; finite frozen evaluation set held fixed",
        "inputs": manifest,
    }
    if kind == "sr":
        output["paired_difference_percentage_points"] = seed_summary(
            {s: 100 * v for s, v in delta.items()}, confidence
        )
    else:
        output["relative_mse_change_percent_per_seed"] = [
            {
                "training_seed": s,
                "value": 100 * delta[s] / left[s] if left[s] != 0 else None,
                "status": "defined" if left[s] != 0 else "undefined_zero_reference_mse",
            }
            for s in sorted(left)
        ]
    return output
