"""Numerical/identity regressions against standard PyTorch, NumPy and SciPy APIs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.action_metrics import MSE_SCHEMA, action_mse_summary
from dacsc.evaluation_statistics import compare_evaluations, seed_summary
from dacsc.corp_data import LinearNormalizer
from dacsc.corp_images import image_contract
from dacsc.upstream import verify_upstream
from scripts.train_corp_robocasa import evaluate
from test_corp_regressions import (
    sr_payload,
    training_origin_fixture,
    tiny_corp,
    tiny_extractor,
)


def error_row(task="A", episode=0, frame=0, error=1.0, positions=1, style=31):
    return {
        "task": task,
        "episode": episode,
        "frame": frame,
        "style_id": style,
        "valid_action_positions": positions,
        "squared_error_sum": [float(error * positions)] * 12,
    }


def sr_case(
    directory,
    seed=42,
    successes=((True, False), (True, False)),
    *,
    checkpoint=None,
    shift="none",
    intervention=None,
):
    directory.mkdir(parents=True)
    digest = (
        checkpoint or hashlib.sha256(f"{directory.name}:{seed}".encode()).hexdigest()
    )
    for task, values in zip(("A", "B"), successes, strict=True):
        payload = sr_payload(task, values, "val")
        payload["protocol"]["checkpoint_sha256"] = digest
        payload["protocol"]["shift"] = shift
        payload["protocol"]["training"] = training_origin_fixture(seed)
        payload["protocol"]["training"]["intervention"] = intervention
        payload["summary"]["shift"] = shift
        for r in payload["records"]:
            r["shift"] = shift
        (directory / f"{task}.json").write_text(json.dumps(payload))
    return directory


def mse_case(
    path, seed=42, errors=(4.0, 0.0), *, checkpoint="a" * 64, intervention=None
):
    rows = [error_row(task=t, error=e) for t, e in zip(("A", "B"), errors, strict=True)]
    metric = action_mse_summary(rows)
    metric["prediction_horizon"] = 10
    origin = training_origin_fixture(seed)
    origin["intervention"] = intervention
    payload = {
        "schema": MSE_SCHEMA,
        "checkpoint_sha256": checkpoint,
        "training": origin,
        "images": image_contract(),
        "upstream": verify_upstream(),
        "precision": "fp32",
        "euler_steps": 10,
        "eval_seed": 123,
        "evaluation_batch_size": 2,
        "splits": {
            "val": {
                "support": {
                    "metadata_sha256": "e" * 64,
                    "selection_sha256": "f" * 64,
                    "frames": 2,
                    "episodes": 2,
                },
                "conditions": {"none": metric},
            }
        },
    }
    path.write_text(json.dumps(payload))
    return path


def sr_manifest(pairs, comparison="methods"):
    return {
        "kind": "sr",
        "comparison": comparison,
        "split": "val",
        "tasks": ["A", "B"],
        "episodes_per_task": 2,
        "runs": [
            {"training_seed": seed, "left": str(a), "right": str(b)}
            for seed, a, b in pairs
        ],
    }


class MSEDefinitionsTests(unittest.TestCase):
    def test_standalone_writer_records_episodes_seed_and_rejects_checkpoint_change_during_load(
        self,
    ):
        from contextlib import redirect_stdout
        import io
        from types import SimpleNamespace
        import scripts.eval_corp_robocasa as entry
        from test_corp_pipeline import metadata, Reader

        class Policy(nn.Module):
            action_horizon = 10
            evaluation_origin = training_origin_fixture()

            def sample_action(self, images, state, *args, **kwargs):
                return torch.zeros(len(state), 10, 12, device=state.device)

        for mutate in (False, True):
            with self.subTest(
                checkpoint_changed=mutate
            ), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                metadata(root / "v1.0/pretrain/atomic/Task/date/lerobot")
                checkpoint = root / "model.pt"
                checkpoint.write_bytes(b"original fixture weights")
                out = root / "evaluation.json"

                def load(*args, **kwargs):
                    if mutate:
                        checkpoint.write_bytes(b"changed fixture weights")
                    return Policy(), (torch.zeros(9), torch.ones(9))

                modules = {
                    "diffusion_policy.dataset.lerobot_dataset": SimpleNamespace(
                        LerobotDataset=Reader
                    )
                }
                argv = [
                    "eval",
                    "--model",
                    str(checkpoint),
                    "--data-root",
                    str(root),
                    "--per-task",
                    "2",
                    "--batch-size",
                    "2",
                    "--out",
                    str(out),
                ]
                with patch.dict(sys.modules, modules), patch.object(
                    entry, "load_adapted_checkpoint", side_effect=load
                ), patch.object(
                    torch.cuda, "is_available", return_value=False
                ), patch.object(
                    sys, "argv", argv
                ), redirect_stdout(
                    io.StringIO()
                ):
                    if mutate:
                        with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                            entry.main()
                    else:
                        self.assertEqual(entry.main(), 0)
                if mutate:
                    self.assertFalse(out.exists())
                else:
                    result = json.loads(out.read_text())
                    self.assertEqual(result["training"]["seed"], 42)
                    self.assertNotIn("test", result["splits"])
                    condition = result["splits"]["val"]["conditions"]["none"]
                    self.assertEqual(len(condition["per_action_dimension_mse"]), 12)
                    self.assertEqual(condition["episodes"], 1)
                    self.assertTrue(condition["records"])
                    self.assertEqual(result["schema"], MSE_SCHEMA)

    def test_micro_vs_task_macro_and_episode_support_are_distinct(self):
        rows = [error_row("A", error=100.0)] + [
            error_row("B", frame=i, error=0.0) for i in range(9)
        ]
        result = action_mse_summary(rows)
        self.assertEqual(result["micro_action_mse"], 10.0)
        self.assertEqual(result["macro_task_action_mse"], 50.0)
        self.assertEqual(result["samples"], 10)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["per_task_support"]["B"]["episodes"], 1)
        self.assertEqual(len(result["per_episode"]), 2)
        self.assertNotIn("overall", result)
        self.assertNotIn("per_dim", result)

    def test_twelve_axes_and_five_groups_match_numpy_means(self):
        row = error_row(positions=2)
        row["squared_error_sum"] = list(map(float, range(12)))
        result = action_mse_summary([row])
        np.testing.assert_allclose(
            list(result["per_action_dimension_mse"].values()), np.arange(12) / 2
        )
        self.assertEqual(len(result["per_action_group_mse"]), 5)
        self.assertEqual(len(result["per_action_dimension_mse"]), 12)
        self.assertEqual(result["per_action_group_mse"]["eef_position"], 0.5)
        self.assertEqual(result["micro_action_mse"], 2.75)

    def test_valid_positions_not_frame_count_define_the_mse_denominator(self):
        result = action_mse_summary(
            [error_row(error=1.0), error_row(episode=1, error=0.0, positions=3)]
        )
        self.assertEqual(result["micro_action_mse"], 0.25)
        self.assertEqual(result["valid_action_scalars"], 48)

    def test_bad_or_duplicate_records_raise_instead_of_silently_reweighting(self):
        for fault in ("duplicate", "style_change", "nan", "negative_support"):
            rows = [error_row(), error_row(frame=1)]
            if fault == "duplicate":
                rows[1]["frame"] = 0
            elif fault == "style_change":
                rows[1]["style_id"] = 32
            elif fault == "nan":
                rows[1]["squared_error_sum"][0] = float("nan")
            else:
                rows[1]["valid_action_positions"] = -1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                action_mse_summary(rows)

    def test_real_evaluator_uses_pytorch_errors_and_keeps_raw_unclipped_commands(self):
        class Policy(nn.Module):
            def sample_action(self, *args, **kwargs):
                return torch.full((1, 2, 12), 2.0)

        batch = {
            "images": {"v": torch.zeros(1, 3, 8, 8)},
            "state": torch.zeros(1, 9),
            "action": torch.ones(1, 2, 12),
            "action_mask": torch.tensor([[True, False]]),
            "image_space": "imagenet_normalized",
            "instruction": ["fixture"],
            "style_id": torch.tensor([31]),
            "task": ["A"],
            "episode": [0],
            "frame": [0],
        }
        with patch("torch.nn.functional.mse_loss", wraps=F.mse_loss) as kernel:
            result, _ = evaluate(
                Policy(),
                [batch],
                LinearNormalizer(torch.zeros(9), torch.ones(9)),
                torch.device("cpu"),
                1,
                10,
            )
        self.assertEqual(kernel.call_args.kwargs["reduction"], "none")
        self.assertEqual(result["micro_action_mse"], 1.0)
        self.assertEqual(result["records"][0]["valid_action_positions"], 1)
        self.assertEqual(result["action_space"], "raw_unclipped_action_commands")


class SeedStatisticsTests(unittest.TestCase):
    def test_small_but_real_variation_is_not_rounded_to_zero_by_an_absolute_tolerance(
        self,
    ):
        result = seed_summary({42: 1e-20, 43: 2e-20, 44: 3e-20})
        self.assertEqual(result["ci_status"], "estimated")
        self.assertIsNotNone(result["confidence_interval"])
        np.testing.assert_allclose(result["sample_std"], 1e-20, rtol=1e-14, atol=0)

    def test_training_origin_is_loaded_from_checkpoint_not_evaluation_seed(self):
        import dacsc.corp_robocasa as corp
        from dacsc.pixel_augmentation import (
            TrainingPixelAugmentation,
            pixel_augmentation_contract,
        )

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "adapted.pt"
            origin = training_origin_fixture(43)
            payload = {
                "format": corp.CHECKPOINT_FORMAT,
                "objective": corp.OBJECTIVE,
                "images": image_contract(),
                "upstream": verify_upstream(),
                "config": corp.ROBOCASA_CORP_CONFIG,
                "model": tiny_corp().state_dict(),
                "token_style": None,
                "task_aux": None,
                "initialization_policy": origin["initialization_policy"],
                "pixel_augmentation": None,
                "normalizer": {
                    "state_mean": torch.zeros(9),
                    "state_std": torch.ones(9),
                },
                "run_args": {"seed": 43},
                "budget": origin["budget"],
                "optimization": origin["optimization"],
                "initialization_sha256": origin["initialization_sha256"],
                "training_state": {
                    "data_order": {
                        "metadata_sha256": origin["data_metadata_sha256"],
                        "dataset_length": 100,
                    }
                },
            }
            for pixel in (None, pixel_augmentation_contract("color_jitter", 43)):
                payload["pixel_augmentation"] = pixel
                origin["pixel_augmentation"] = pixel
                torch.save(payload, path)
                with patch.object(
                    corp, "RepresentationExtractor", side_effect=tiny_extractor
                ), patch.object(
                    corp, "DiTActionGenerator", side_effect=lambda _: nn.Linear(2, 2)
                ), patch.object(
                    TrainingPixelAugmentation,
                    "__init__",
                    side_effect=AssertionError("inference constructed training jitter"),
                ):
                    model, _ = corp.load_adapted_checkpoint(path)
                self.assertEqual(model.evaluation_origin, origin)

    def test_standard_numpy_sample_sd_and_scipy_student_t_use_training_seed_df(self):
        with patch("scipy.stats.t.interval", wraps=stats.t.interval) as interval:
            result = seed_summary({42: 1.0, 43: 2.0, 44: 3.0})
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)
        self.assertEqual(result["n_training_seeds"], 3)
        self.assertEqual(interval.call_args.kwargs["df"], 2)
        expected = stats.t.interval(
            0.95, df=2, loc=2.0, scale=stats.sem([1.0, 2.0, 3.0], ddof=1)
        )
        np.testing.assert_allclose(result["confidence_interval"], expected)

    def test_single_seed_or_roundoff_only_variation_never_produces_spurious_certainty(
        self,
    ):
        one = seed_summary({42: 0.5})
        self.assertIsNone(one["sample_std"])
        self.assertIsNone(one["confidence_interval"])
        tiny = seed_summary({42: 0.5 - 0.4, 43: 0.6 - 0.5, 44: 0.7 - 0.6})
        self.assertIsNone(tiny["confidence_interval"])
        self.assertIn("zero_observed", tiny["ci_status"])
        with self.assertRaises(ValueError):
            seed_summary({42: float("nan")})


class PairedComparisonTests(unittest.TestCase):
    def test_custom_scores_and_hidden_weight_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = sr_case(root / "a")
            b = sr_case(root / "b")
            for key, value in (
                ("kind", "invented_robustness_score"),
                ("weights", {"A": 100, "B": 1}),
            ):
                manifest = sr_manifest([(42, a, b)])
                manifest[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    compare_evaluations(manifest)

    def test_mse_conditions_use_the_same_checkpoint_and_frame_set(self):
        with tempfile.TemporaryDirectory() as d:
            path = mse_case(Path(d) / "conditions.json", errors=(4.0, 0.0))
            payload = json.loads(path.read_text())
            rows = copy.deepcopy(
                payload["splits"]["val"]["conditions"]["none"]["records"]
            )
            for row in rows:
                row["squared_error_sum"] = [2 * x for x in row["squared_error_sum"]]
            metric = action_mse_summary(rows)
            metric["prediction_horizon"] = 10
            payload["splits"]["val"]["conditions"]["color_jitter"] = metric
            path.write_text(json.dumps(payload))
            manifest = {
                "kind": "mse",
                "comparison": "conditions",
                "split": "val",
                "left_condition": "none",
                "right_condition": "color_jitter",
                "runs": [{"training_seed": 42, "left": str(path), "right": str(path)}],
            }
            result = compare_evaluations(manifest)
            self.assertEqual(result["paired_difference"]["mean"], 2.0)
            self.assertEqual(result["pair_support"][0]["pairs"], 2)

    def test_sr_differences_are_paired_by_training_seed_and_expressed_in_percentage_points(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pairs = []
            patterns = [
                (((True, True), (False, False)), ((True, True), (True, False))),
                (((True, False), (False, False)), ((True, True), (True, False))),
                (((True, True), (True, False)), ((True, True), (False, False))),
            ]
            for seed, (a, b) in zip((42, 43, 44), patterns):
                pairs.append(
                    (
                        seed,
                        sr_case(root / f"left{seed}", seed, a),
                        sr_case(root / f"right{seed}", seed, b),
                    )
                )
            result = compare_evaluations(sr_manifest(pairs))
            delta = np.asarray([0.25, 0.5, -0.25])
            self.assertAlmostEqual(result["paired_difference"]["mean"], np.mean(delta))
            self.assertAlmostEqual(
                result["paired_difference"]["sample_std"], np.std(delta, ddof=1)
            )
            self.assertAlmostEqual(
                result["paired_difference_percentage_points"]["mean"],
                100 * np.mean(delta),
            )
            self.assertEqual(result["paired_difference"]["n_training_seeds"], 3)
            self.assertEqual(result["paired_difference"]["ci_unit"], "training_seed")
            self.assertEqual([p["pairs"] for p in result["pair_support"]], [4, 4, 4])

    def test_conditions_require_same_checkpoint_but_methods_can_change_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = sr_case(root / "a", checkpoint="a" * 64)
            b = sr_case(root / "b", checkpoint="a" * 64, shift="color_jitter")
            result = compare_evaluations(sr_manifest([(42, a, b)], "conditions"))
            self.assertIsNone(result["paired_difference"]["confidence_interval"])
            with self.assertRaisesRegex(ValueError, "identical evaluation conditions"):
                compare_evaluations(sr_manifest([(42, a, b)]))
            for path in b.glob("*.json"):
                payload = json.loads(path.read_text())
                payload["protocol"]["checkpoint_sha256"] = "b" * 64
                path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "same checkpoint"):
                compare_evaluations(sr_manifest([(42, a, b)], "conditions"))

    def test_changed_scene_wrong_seed_budget_or_split_is_rejected_without_intersection(
        self,
    ):
        for fault in (
            "scene",
            "seed",
            "budget",
            "source_val",
            "policy_seed",
            "horizon",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                a = sr_case(root / "a")
                b = sr_case(root / "b")
                for path in b.glob("*.json"):
                    payload = json.loads(path.read_text())
                    if fault == "scene":
                        payload["records"][0]["manifest_sha256"] = "e" * 64
                    elif fault == "seed":
                        payload["protocol"]["training"]["seed"] = 43
                    elif fault == "budget":
                        payload["protocol"]["training"]["budget"]["steps"] = 10000
                    elif fault == "source_val":
                        payload["summary"]["styles"] = list(range(11, 31))
                        for row in payload["records"]:
                            row.update(style_ids=list(range(11, 31)), actual_style=11)
                    elif fault == "horizon":
                        payload["summary"]["max_steps"] = 600
                    else:
                        payload["protocol"]["base_seed"] = 43
                        payload["summary"]["seed"] = 43
                        for row in payload["records"]:
                            row["environment_seed"] += 1
                            row["policy_seed"] += 1
                    path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    compare_evaluations(sr_manifest([(42, a, b)]))

    def test_evaluation_set_cannot_change_between_training_seeds_or_duplicate_a_seed(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pairs = []
            for seed in (42, 43):
                a = sr_case(root / f"a{seed}", seed)
                b = sr_case(root / f"b{seed}", seed)
                pairs.append((seed, a, b))
            for directory in pairs[1][1:]:
                for path in directory.glob("*.json"):
                    payload = json.loads(path.read_text())
                    payload["records"][0]["manifest_sha256"] = "f" * 64
                    path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "stay fixed"):
                compare_evaluations(sr_manifest(pairs))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                compare_evaluations(sr_manifest([pairs[0], pairs[0]]))

    def test_mse_comparison_uses_macro_task_and_a_zero_reference_is_not_divided_by_epsilon(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = mse_case(root / "a.json", errors=(0.0, 0.0))
            b = mse_case(root / "b.json", errors=(2.0, 0.0), checkpoint="b" * 64)
            manifest = {
                "kind": "mse",
                "comparison": "methods",
                "split": "val",
                "left_condition": "none",
                "right_condition": "none",
                "runs": [{"training_seed": 42, "left": str(a), "right": str(b)}],
            }
            result = compare_evaluations(manifest)
            self.assertEqual(result["paired_difference"]["mean"], 1.0)
            self.assertEqual(result["positive_difference_means"], "right_worse")
            self.assertIsNone(
                result["relative_mse_change_percent_per_seed"][0]["value"]
            )
            self.assertEqual(result["pair_support"][0]["pairs"], 2)

    def test_mse_record_summary_frame_set_and_inference_batch_must_match(self):
        for fault in ("summary", "frame", "batch", "support", "action_space"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                a = mse_case(root / "a.json")
                b = mse_case(root / "b.json", checkpoint="b" * 64)
                payload = json.loads(b.read_text())
                metric = payload["splits"]["val"]["conditions"]["none"]
                if fault == "summary":
                    metric["macro_task_action_mse"] += 1
                elif fault == "frame":
                    metric["records"][0]["frame"] = 2
                elif fault == "batch":
                    payload["evaluation_batch_size"] = 1
                elif fault == "action_space":
                    metric["action_space"] = "physical_centimetres"
                else:
                    payload["splits"]["val"]["support"]["episodes"] = 999
                b.write_text(json.dumps(payload))
                manifest = {
                    "kind": "mse",
                    "comparison": "methods",
                    "split": "val",
                    "left_condition": "none",
                    "right_condition": "none",
                    "runs": [{"training_seed": 42, "left": str(a), "right": str(b)}],
                }
                with self.assertRaises(ValueError):
                    compare_evaluations(manifest)

    def test_comparison_cli_resolves_relative_paths_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sr_case(root / "left")
            sr_case(root / "right")
            manifest = sr_manifest([(42, "left", "right")])
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            out = root / "result.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/compare_corp_evaluations.py"),
                "--manifest",
                str(path),
                "--out",
                str(out),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(out.read_text())["paired_difference"]["n_training_seeds"], 1
            )
            first = out.read_bytes()
            again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(out.read_bytes(), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
