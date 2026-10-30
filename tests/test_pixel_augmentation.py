"""CPU regressions for the actual Torchvision operator and train/eval boundary."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ColorJitter, functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_data import CorpRoboCasaDataset, LinearNormalizer, collate_corp
from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD, apply_image_shift
from dacsc.pixel_augmentation import (
    TrainingPixelAugmentation,
    pixel_augmentation_contract,
)
from dacsc.training_state import (
    atomic_torch_save,
    training_state,
    restore_training_state,
)
from dacsc.evaluation_statistics import compare_evaluations
from scripts import train_corp_robocasa as train
from test_corp_pipeline import metadata, Reader
from test_corp_regressions import TinyModel, seed_episode
from test_standard_metrics import sr_case, sr_manifest


def rgb_batch(batch=2, views=3):
    image = torch.linspace(0.2, 0.7, 3 * 8 * 8).reshape(3, 8, 8)
    return image.expand(batch, views, 3, 8, 8).clone()


class PixelAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_torchvision_forward_order_factors_and_normalization_are_unchanged(self):
        rgb = rgb_batch()
        aug = TrainingPixelAugmentation(42)
        self.assertIs(type(aug.transform), ColorJitter)
        self.assertEqual(aug.transform.brightness, (0.6, 1.4))
        self.assertIsNone(aug.transform.hue)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            expected = torch.stack(
                [ColorJitter(0.4, 0.4, 0.4, 0.0)(observation) for observation in rgb]
            )
            expected = TVF.normalize(
                expected,
                IMAGENET_MEAN.flatten().tolist(),
                IMAGENET_STD.flatten().tolist(),
            )
            expected_state = torch.get_rng_state()
        actual = aug(rgb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(aug.generator.get_state(), expected_state))
        # Fixed evaluation corruption is a different, deterministic API/recipe.
        fixed = TVF.normalize(
            apply_image_shift(rgb, "color_jitter"),
            IMAGENET_MEAN.flatten().tolist(),
            IMAGENET_STD.flatten().tolist(),
        )
        self.assertFalse(torch.allclose(actual, fixed))

    def test_each_observation_draws_fresh_parameters_and_shares_them_across_views(self):
        aug = TrainingPixelAugmentation(42)
        rgb = rgb_batch()
        first = aug(rgb)
        second = aug(rgb)
        for view in (1, 2):
            torch.testing.assert_close(first[:, 0], first[:, view], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first[0], first[1]))
        self.assertFalse(torch.allclose(first, second))
        other = TrainingPixelAugmentation(43)(rgb)
        self.assertFalse(torch.allclose(first, other))

    def test_constructor_and_calls_do_not_change_global_flow_rng_even_on_exception(
        self,
    ):
        before = torch.get_rng_state().clone()
        aug = TrainingPixelAugmentation(42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        aug(rgb_batch())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with patch.object(
            aug.transform, "forward", side_effect=RuntimeError("fixture failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                aug(rgb_batch())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rng_roundtrip_and_changed_seed_or_recipe_are_not_silently_reset(self):
        aug = TrainingPixelAugmentation(42)
        aug(rgb_batch())
        state = copy.deepcopy(aug.state_dict())
        expected = aug(rgb_batch())
        restored = TrainingPixelAugmentation(42)
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(rgb_batch()), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            TrainingPixelAugmentation(43).load_state_dict(state)
        bad = copy.deepcopy(state)
        bad["config"]["brightness"] = 0.5
        with self.assertRaises(ValueError):
            TrainingPixelAugmentation(42).load_state_dict(bad)

    def test_normalized_or_malformed_rgb_is_rejected(self):
        aug = TrainingPixelAugmentation(42)
        for bad in (
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 1, 8, 8),
            torch.full((2, 3, 3, 8, 8), -1.0),
            torch.full((2, 3, 3, 8, 8), float("nan")),
        ):
            with self.assertRaises(ValueError):
                aug(bad)

    def test_only_train_dataset_can_emit_raw_rgb_and_other_targets_do_not_change(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Path(d) / "Task/date/lerobot"
            metadata(ds)
            modules = {
                "diffusion_policy.dataset.lerobot_dataset": SimpleNamespace(
                    LerobotDataset=Reader
                )
            }
            with patch.dict(sys.modules, modules):
                args = dict(
                    style_min=11,
                    style_max=30,
                    horizon=4,
                    episode_residues=set(range(9)),
                )
                clean = CorpRoboCasaDataset([ds], **args)
                raw = CorpRoboCasaDataset([ds], **args, normalize_images=False)
                source = CorpRoboCasaDataset(
                    [ds], style_min=11, style_max=30, horizon=4, episode_residues={9}
                )
                val = CorpRoboCasaDataset([ds], style_min=31, style_max=40, horizon=4)
                for opts in (
                    dict(style_min=31, style_max=40),
                    dict(style_min=41, style_max=50),
                    dict(style_min=11, style_max=30, episode_residues={9}),
                    {**args, "shift": "color_jitter"},
                ):
                    with self.assertRaises(ValueError):
                        CorpRoboCasaDataset([ds], **opts, normalize_images=False)
            self.assertEqual(raw[0]["image_space"], "rgb01")
            self.assertEqual(source[0]["image_space"], "imagenet_normalized")
            self.assertEqual(val[0]["image_space"], "imagenet_normalized")
            a, b = clean[0], raw[0]
            for key in ("state", "action", "action_mask", "style_id"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for key in ("instruction", "task", "episode", "frame"):
                self.assertEqual(a[key], b[key])
            batch = collate_corp([raw[0], raw[1]])
            copied = copy.deepcopy(batch)
            aug = TrainingPixelAugmentation(42)
            moved = train.move_batch(
                batch,
                LinearNormalizer(torch.zeros(9), torch.ones(9)),
                torch.device("cpu"),
                pixel_augmentation=aug,
            )
            for key in ("state", "action", "action_mask", "style_id"):
                torch.testing.assert_close(batch[key], copied[key], rtol=0, atol=0)
            self.assertEqual(
                moved["images"][next(iter(moved["images"]))].shape, (2, 3, 224, 224)
            )
            with self.assertRaises(ValueError):
                train.move_batch(
                    batch,
                    LinearNormalizer(torch.zeros(9), torch.ones(9)),
                    torch.device("cpu"),
                )
            with self.assertRaises(ValueError):
                collate_corp([raw[0], clean[0]])
            with self.assertRaises(ValueError):
                train.move_batch(
                    collate_corp([clean[0]]),
                    LinearNormalizer(torch.zeros(9), torch.ones(9)),
                    torch.device("cpu"),
                    pixel_augmentation=aug,
                )

    def test_evaluation_never_constructs_or_calls_training_jitter(self):
        class Policy(nn.Module):
            def sample_action(self, images, state, *args, **kwargs):
                return torch.zeros(len(state), 2, 12)

        batch = {
            "images": {"v": torch.zeros(1, 3, 8, 8)},
            "image_space": "imagenet_normalized",
            "state": torch.zeros(1, 9),
            "action": torch.zeros(1, 2, 12),
            "action_mask": torch.ones(1, 2, dtype=torch.bool),
            "style_id": torch.tensor([31]),
            "task": ["Task"],
            "episode": [0],
            "frame": [0],
            "instruction": ["open"],
        }
        with patch.object(
            TrainingPixelAugmentation,
            "__call__",
            side_effect=AssertionError("train augmentation leaked"),
        ):
            for shift in ("none", "color_jitter"):
                result, _ = train.evaluate(
                    Policy(),
                    [batch],
                    LinearNormalizer(torch.zeros(9), torch.ones(9)),
                    torch.device("cpu"),
                    1,
                    10,
                    shift=shift,
                )
                self.assertEqual(result["micro_action_mse"], 0.0)

    def test_full_resume_preserves_next_batch_augmentation_flow_noise_and_update(self):
        seed_episode(7)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        aug = TrainingPixelAugmentation(42)
        generator = torch.Generator().manual_seed(42)
        epoch = generator.get_state()
        loader = DataLoader(
            TensorDataset(torch.arange(6)),
            batch_size=2,
            shuffle=True,
            generator=generator,
        )
        iterator = iter(loader)
        first = next(iterator)[0]

        def update(m, o, s, a, indices):
            image = rgb_batch() + indices[:, None, None, None, None] * 0.01
            augmented = a(image)
            noise = torch.randn(2, 2)
            x = augmented.mean(dim=(1, 3, 4))[:, :2] + noise
            o.zero_grad()
            m(x).square().sum().backward()
            o.step()
            s.step()
            m.runtime += 1
            return augmented, noise

        update(model, opt, sched, aug, first)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.pt"
            atomic_torch_save(
                {
                    "raw_model": model.state_dict(),
                    "training_state": training_state(
                        model,
                        opt,
                        sched,
                        None,
                        step=1,
                        data_order={"epoch_rng": epoch, "batches_in_epoch": 1},
                        augmentation=aug,
                    ),
                },
                path,
            )
            indices = next(iterator)[0]
            expected, expected_noise = update(model, opt, sched, aug, indices)
            expected_weights = copy.deepcopy(model.state_dict())
            saved = torch.load(path, weights_only=False)
            other = TinyModel()
            other_opt = torch.optim.AdamW(other.parameters(), lr=0.01)
            other_sched = torch.optim.lr_scheduler.LambdaLR(other_opt, lambda _: 1.0)
            other_aug = TrainingPixelAugmentation(42)
            g = torch.Generator()
            g.set_state(saved["training_state"]["data_order"]["epoch_rng"])
            it = iter(
                DataLoader(loader.dataset, batch_size=2, shuffle=True, generator=g)
            )
            torch.testing.assert_close(next(it)[0], first)
            restored_indices = next(it)[0]
            self.assertEqual(
                restore_training_state(
                    saved, other, other_opt, other_sched, None, augmentation=other_aug
                ),
                1,
            )
            actual, actual_noise = update(
                other, other_opt, other_sched, other_aug, restored_indices
            )
            torch.testing.assert_close(restored_indices, indices, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_noise, expected_noise, rtol=0, atol=0)
            for name, value in other.state_dict().items():
                torch.testing.assert_close(
                    value, expected_weights[name], rtol=0, atol=0
                )
            with self.assertRaisesRegex(ValueError, "augmentation setting"):
                restore_training_state(saved, other, other_opt, other_sched, None)

    def test_cli_keeps_training_augmentation_and_evaluation_shift_independent(self):
        with tempfile.TemporaryDirectory() as d:
            for mode, shift in (("none", "color_jitter"), ("color_jitter", "none")):
                argv = [
                    "train",
                    "--steps",
                    "1",
                    "--out-dir",
                    str(Path(d) / mode),
                    "--train-augmentation",
                    mode,
                    "--eval-shift",
                    shift,
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    train, "run", return_value=0
                ) as run:
                    self.assertEqual(train.main(), 0)
                    self.assertEqual(run.call_args.args[0].train_augmentation, mode)
                    self.assertEqual(run.call_args.args[0].eval_shift, shift)
            argv = [
                "train",
                "--steps",
                "1",
                "--out-dir",
                str(Path(d) / "invalid"),
                "--train-augmentation",
                "color_jitter",
                "--token-style-mode",
                "cross_style_stats",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                train, "run"
            ) as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertFalse((Path(d) / "invalid").exists())

    def test_runner_forwards_pixel_baseline_and_rejects_combined_modes(self):
        # Interpreter stub captures argv only; no training entry is executed.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            log = root / "argv.jsonl"
            stub = root / "capture"
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['ARGUMENT_LOG'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PY": str(stub),
                "ARGUMENT_LOG": str(log),
                "OUT": str(root / "runs"),
                "STEPS": "2000",
                "SEEDS": "42 43",
                "MODES": "none",
                "TRAIN_AUGMENTATION": "color_jitter",
                "SHIFT": "none",
            }
            command = ["bash", str(ROOT / "scripts/run_corp_experiment.sh")]
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            for argv, seed in zip(calls, (42, 43), strict=True):
                self.assertEqual(
                    argv[argv.index("--train-augmentation") + 1], "color_jitter"
                )
                self.assertEqual(argv[argv.index("--eval-shift") + 1], "none")
                self.assertEqual(
                    argv[argv.index("--out-dir") + 1],
                    str(root / f"runs/color_jitter/none/seed_{seed}"),
                )
            env["MODES"] = "cross_style_stats"
            failed = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(failed.returncode, 2)
            self.assertNotIn("ALL_DONE", failed.stdout)
            self.assertEqual(len(log.read_text().splitlines()), 2)

    def test_pixel_baseline_can_be_compared_as_a_method_but_not_as_an_eval_condition(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pairs = []
            for seed in (42, 43):
                left = sr_case(root / f"base{seed}", seed)
                right = sr_case(root / f"pixel{seed}", seed)
                for path in right.glob("*.json"):
                    payload = json.loads(path.read_text())
                    payload["protocol"]["training"]["pixel_augmentation"] = (
                        pixel_augmentation_contract("color_jitter", seed)
                    )
                    path.write_text(json.dumps(payload))
                pairs.append((seed, left, right))
            result = compare_evaluations(sr_manifest(pairs))
            self.assertEqual(result["paired_difference"]["n_training_seeds"], 2)
            with self.assertRaisesRegex(ValueError, "same checkpoint"):
                compare_evaluations(sr_manifest([pairs[0]], comparison="conditions"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
