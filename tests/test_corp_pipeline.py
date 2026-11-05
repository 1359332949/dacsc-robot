"""Regressions for project glue; official demos verify the released model/simulator."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import tarfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dacsc.corp_images import apply_image_shift, prepare_image, resize_rgb
from dacsc.corp_data import (
    CorpRoboCasaDataset,
    LinearNormalizer,
    SHAPE_META,
    collate_corp,
    read_episode_meta,
)
from dacsc.corp_robocasa import masked_action_mse
from dacsc.training_state import ModelEMA
from dacsc import upstream
from train_corp_robocasa import evaluate


def metadata(root, episodes=((0, 11, 3), (9, 12, 2), (1, 31, 3))):
    (root / "meta").mkdir(parents=True)
    rows = []
    for ep, style, length in episodes:
        rows.append({"episode_index": ep, "length": length})
        path = root / "extras" / f"episode_{ep:06d}"
        path.mkdir(parents=True)
        (path / "ep_meta.json").write_text(
            json.dumps({"style_id": style, "layout_id": 11, "lang": "open drawer"})
        )
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))


class Reader:
    """Only the upstream reader leaf is replaced; adapter and contracts execute."""

    def __init__(self, **kwargs):
        meta = read_episode_meta(kwargs["dataset_path"])
        self.trajectory_ids = list(meta)
        self.trajectory_lengths = [meta[e]["length"] for e in self.trajectory_ids]
        self.all_steps = [
            (e, t)
            for e, n in zip(self.trajectory_ids, self.trajectory_lengths)
            for t in range(n)
        ]
        self.horizon = kwargs["horizon"]
        assert kwargs["n_obs_steps"] == 1

    def __getitem__(self, idx):
        e, t = self.all_steps[idx]
        n = self.trajectory_lengths[self.trajectory_ids.index(e)]
        action = torch.zeros(self.horizon, 12)
        for j in range(min(n - t, self.horizon)):
            action[j].fill_(t + j + 1)
        obs = {}
        for key, info in SHAPE_META["obs"].items():
            obs[key] = torch.zeros(1, *info["shape"])
        return {"obs": obs, "action": action}


class DatasetTests(unittest.TestCase):
    def test_sr_style_ranges_fail_before_a_mixed_reverse_range_can_disappear(self):
        from scripts.eval_corp_robocasa_sr import parse_styles

        self.assertEqual(parse_styles("11-13, 20"), [11, 12, 13, 20])
        for value in ("20-11,30", "11-13,12", "1-10", "61", "", "11,,12"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_styles(value)

    def test_invalid_missing_duplicate_or_unaligned_metadata_never_silently_disappears(
        self,
    ):
        for fault in ("json", "missing", "field", "duplicate", "extra"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                metadata(root)
                path = root / "extras/episode_000000/ep_meta.json"
                if fault == "json":
                    path.write_text("{broken")
                elif fault == "missing":
                    path.unlink()
                elif fault == "field":
                    path.write_text(json.dumps({"style_id": 11, "lang": "open"}))
                elif fault == "duplicate":
                    p = root / "meta/episodes.jsonl"
                    p.write_text(p.read_text() + "\n" + p.read_text().splitlines()[0])
                else:
                    (root / "extras/episode_999999").mkdir()
                with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                    read_episode_meta(root)

    def test_mask_repeat_last_native_split_and_independent_support(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "Task/date/lerobot"
            metadata(root)
            modules = {
                "diffusion_policy.dataset.lerobot_dataset": SimpleNamespace(
                    LerobotDataset=Reader
                )
            }
            with patch.dict(sys.modules, modules):
                train = CorpRoboCasaDataset(
                    [root],
                    style_min=11,
                    style_max=30,
                    horizon=4,
                    episode_residues=set(range(9)),
                )
                source = CorpRoboCasaDataset(
                    [root], style_min=11, style_max=30, horizon=4, episode_residues={9}
                )
                val = CorpRoboCasaDataset([root], style_min=31, style_max=40, horizon=4)
            self.assertEqual({r[5] for r in train.flat}, {0})
            self.assertEqual({r[5] for r in source.flat}, {9})
            self.assertEqual({r[5] for r in val.flat}, {1})
            last = train[2]
            self.assertEqual(last["action_mask"].tolist(), [True, False, False, False])
            torch.testing.assert_close(last["action"], torch.full((4, 12), 3.0))
            batch = collate_corp([train[0], last])
            self.assertEqual(batch["action_mask"].shape, (2, 4))
            self.assertEqual(train.support()["episodes"], 1)
            self.assertEqual(train.support()["frames"], 3)
            self.assertNotEqual(
                train.support()["selection_sha256"],
                source.support()["selection_sha256"],
            )


class MaskedObjectiveTests(unittest.TestCase):
    def test_nonfinite_valid_loss_fails_before_an_optimizer_can_use_it(self):
        with self.assertRaises(FloatingPointError):
            masked_action_mse(
                torch.full((1, 1, 12), float("nan")),
                torch.zeros(1, 1, 12),
                torch.ones(1, 1, dtype=torch.bool),
            )

    def test_padding_errors_never_enter_loss_denominator_or_gradient(self):
        pred = torch.tensor([[[1.0, 3.0], [1000.0, 1000.0]]], requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.tensor([[True, False]])
        loss = masked_action_mse(pred, target, mask)
        self.assertEqual(float(loss.detach()), 5.0)
        loss.backward()
        self.assertEqual(float(pred.grad[0, 1].abs().sum()), 0)
        for bad in (
            torch.zeros_like(mask),
            mask.float(),
            torch.ones(1, 3, dtype=torch.bool),
        ):
            with self.assertRaises(ValueError):
                masked_action_mse(pred, target, bad)

    def test_offline_metrics_exclude_tail_for_every_action_group_and_task(self):
        class Policy(nn.Module):
            def sample_action(self, *args, **kwargs):
                return torch.tensor([[[1.0] * 12, [999.0] * 12]])

        batch = {
            "images": {"v": torch.zeros(1, 3, 4, 4)},
            "state": torch.zeros(1, 9),
            "instruction": ["open"],
            "action": torch.zeros(1, 2, 12),
            "style_id": torch.tensor([11]),
            "task": ["Task"],
            "action_mask": torch.tensor([[True, False]]),
            "image_space": "imagenet_normalized",
            "episode": [0],
            "frame": [0],
        }
        result, _ = evaluate(
            Policy(),
            [batch],
            LinearNormalizer(torch.zeros(9), torch.ones(9)),
            torch.device("cpu"),
            1,
            10,
        )
        self.assertEqual(result["micro_action_mse"], 1.0)
        self.assertEqual(result["per_task_action_mse"]["Task"], 1.0)
        self.assertTrue(all(x == 1 for x in result["per_action_group_mse"].values()))
        self.assertEqual(result["valid_action_scalars"], 12)


class OfficialImageTests(unittest.TestCase):
    def test_solid_red_contrast_and_hue_match_upstream_not_custom_rgb_math(self):
        red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1).expand(3, 8, 8)
        torch.testing.assert_close(
            apply_image_shift(red, "low_contrast")[:, 0, 0],
            torch.tensor([0.50923, 0.20923, 0.20923]),
        )
        torch.testing.assert_close(
            apply_image_shift(red, "hue_shift")[:, 0, 0], torch.tensor([0.0, 1.0, 1.0])
        )

    def test_batched_operators_match_torchvision_exactly(self):
        x = torch.rand(2, 3, 3, 8, 8)
        references = {
            "none": x,
            "grayscale": TVF.rgb_to_grayscale(x, 3),
            "low_contrast": TVF.adjust_contrast(x, 0.3),
            "hue_shift": TVF.adjust_hue(x, 0.5),
            "color_jitter": TVF.adjust_saturation(
                TVF.adjust_contrast(TVF.adjust_brightness(x, 1.6), 1.6), 2.2
            ),
        }
        for mode, expected in references.items():
            torch.testing.assert_close(
                apply_image_shift(x, mode), expected, rtol=0, atol=0
            )
        with self.assertRaises(ValueError):
            apply_image_shift(x, "typo")

    def test_dataset_float_and_simulator_uint8_share_official_pil_resize(self):
        arr = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        decoded = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
        expected = (
            torch.from_numpy(
                np.asarray(
                    Image.fromarray(arr).resize((224, 224), Image.Resampling.BILINEAR)
                ).copy()
            )
            .permute(2, 0, 1)
            .float()
            / 255
        )
        torch.testing.assert_close(resize_rgb(arr), expected, rtol=0, atol=0)
        for shift in ("none", "grayscale", "color_jitter", "low_contrast", "hue_shift"):
            torch.testing.assert_close(
                prepare_image(arr, shift), prepare_image(decoded, shift), rtol=0, atol=0
            )


class UpstreamInfrastructureTests(unittest.TestCase):
    def test_packaging_exports_committed_manifest_and_tests_and_rejects_dirty_sources(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repository"
            (root / "scripts").mkdir(parents=True)
            (root / "tests").mkdir()
            script = root / "scripts/pack_for_server.sh"
            script.write_text((ROOT / "scripts/pack_for_server.sh").read_text())
            (root / "README.md").write_text("committed fixture")
            (root / "upstream.lock.json").write_text("{}")
            (root / "tests/fixture.py").write_text("# fixture only")
            for args in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                )
            out = Path(d) / "package"
            result = subprocess.run(
                ["bash", str(script), str(out)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            archive = out / "dacsc_code.tgz"
            with tarfile.open(archive) as bundle:
                self.assertTrue(
                    {"README.md", "upstream.lock.json", "tests/fixture.py"}
                    <= set(bundle.getnames())
                )
                self.assertEqual(
                    bundle.extractfile("README.md").read(), b"committed fixture"
                )
            original = archive.read_bytes()
            again = subprocess.run(
                ["bash", str(script), str(out)], capture_output=True, text=True
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(archive.read_bytes(), original)
            (root / "README.md").write_text("uncommitted edit")
            dirty = subprocess.run(
                ["bash", str(script), str(Path(d) / "dirty")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(dirty.returncode, 0)
            self.assertFalse((Path(d) / "dirty/dacsc_code.tgz").exists())

    def test_completed_budget_does_not_resume_or_publish_pre_eval_metrics(self):
        from scripts import train_corp_robocasa as train
        from dacsc.corp_images import image_contract
        from dacsc.corp_robocasa import CHECKPOINT_FORMAT, OBJECTIVE

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model_step10.pt"
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "objective": OBJECTIVE,
                    "images": image_contract(),
                    "upstream": upstream.verify_upstream(),
                    "training_state": {"step": 10},
                },
                path,
            )
            args = SimpleNamespace(
                seed=42,
                max_tasks=0,
                data_root=Path(d),
                horizon=10,
                resume=path,
                steps=10,
                train_augmentation="none",
                task_aux_lambda=0.0,
                task_aux_depth=1,
            )
            with patch.object(train, "set_seed"), patch.object(
                torch.cuda, "is_available", return_value=True
            ), patch.object(
                train, "all_task_paths", return_value=["metadata-only fixture"]
            ), patch.object(
                train, "build_dataset", return_value=[0]
            ), self.assertRaisesRegex(
                ValueError, "budget is already complete"
            ):
                train.run(args, Path(d) / "output")
            self.assertFalse((Path(d) / "output/result.json").exists())

    def test_explicit_invalid_checkout_does_not_fall_back_to_another_model_source(self):
        from dacsc.corp_robocasa import _resolve_corp_repo

        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"CORP_REPO": d}
        ):
            with self.assertRaises(FileNotFoundError):
                _resolve_corp_repo()

    def test_bad_release_checkpoint_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "corp_libero.pt"
            path.write_bytes(b"incomplete checkpoint")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                upstream.verify_release_checkpoint(path)

    def test_shared_weights_keep_one_ema_value_for_all_state_dict_aliases(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.zeros(1))
        model.tied_weight = model.weight
        ema = ModelEMA(model, 0.9)
        model.weight.data.fill_(10)
        ema.update(model)
        model.load_state_dict(ema.shadow, strict=True)
        torch.testing.assert_close(model.weight, torch.ones(1))
        torch.testing.assert_close(ema.shadow["weight"], ema.shadow["tied_weight"])

    def test_normalizer_rejects_corrupt_or_wrong_dimension_statistics(self):
        for mean, std in (
            (torch.zeros(8), torch.ones(8)),
            (torch.zeros(9), torch.zeros(9)),
            (torch.full((9,), float("nan")), torch.ones(9)),
        ):
            with self.assertRaises(ValueError):
                LinearNormalizer(mean, std)

    def test_ema_uses_pytorch_and_dynamic_donor_buffer_is_not_averaged(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        model.register_buffer("donor", torch.zeros(0), persistent=False)
        ema = ModelEMA(model, 0.9)
        model.weight.data.fill_(10)
        model.donor = torch.ones(256)
        ema.update(model)
        torch.testing.assert_close(ema.shadow["weight"], torch.ones(1, 1))
        self.assertEqual(ema.averaged.__class__.__module__, "torch.optim.swa_utils")

    def test_source_hash_tampering_fails_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "repos/test"
            p.mkdir(parents=True)
            (p / "source.py").write_text("changed")
            (root / "upstream.lock.json").write_text(
                json.dumps(
                    {"test": {"revision": "pinned", "files": {"source.py": "0" * 64}}}
                )
            )
            with patch.object(upstream, "ROOT", root), self.assertRaisesRegex(
                RuntimeError, "source changed"
            ):
                upstream.verify_upstream()

    def test_runner_forwards_one_budget_and_stops_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stub = root / "python"
            captured = root / "args.json"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 23\n')
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PY": str(stub),
                "CAPTURE": str(captured),
                "OUT": str(root / "out"),
                "STEPS": "2000",
                "MODES": "none random_stats",
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_corp_experiment.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 23)
            args = captured.read_text().splitlines()
            self.assertEqual(args[args.index("--batch-size") + 1], "30")
            self.assertEqual(args[args.index("--steps") + 1], "2000")
            self.assertNotIn("ALL_DONE", result.stdout)


class ReleasedAdapterTests(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "repos/CoRP/checkpoints/corp_libero.pt").is_file(),
        "official checkpoint not available locally; run upstream smoke separately",
    )
    def test_real_release_adapter_hook_masked_loss_and_ema_state(self):
        import dacsc.corp_robocasa as corp_module
        from dacsc.corp_robocasa import load_corp_robocasa
        from dacsc.token_style import TokenStyleModulator

        checkpoint = ROOT / "repos/CoRP/checkpoints/corp_libero.pt"
        upstream.verify_release_checkpoint(checkpoint)
        threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            model, report = load_corp_robocasa(checkpoint)
            self.assertEqual(report["random_init"], 4)
            self.assertEqual(
                set(report["random_init_keys"]),
                {
                    "representation_extractor.state_encoder.projection.weight",
                    "action_generator.action_projection.weight",
                    "action_generator.velocity_projection.weight",
                    "action_generator.velocity_projection.bias",
                },
            )
            self.assertEqual(
                report["initialization_policy"],
                corp_module.INITIALIZATION_POLICY,
            )
            ema = ModelEMA(model, 0.99)
            self.assertEqual(set(ema.shadow), set(model.state_dict()))
            ema.update(model)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(ema.shadow[name], tensor)
            images = {view: torch.randn(2, 3, 224, 224) for view in model.view_names}
            state = torch.zeros(2, 9)
            instructions = ["open drawer", "close drawer"]
            labels = torch.tensor([11, 12])
            model.train()
            with torch.no_grad():
                torch.manual_seed(7)
                base = model.extract(images, state, instructions, labels).tokens
                style = TokenStyleModulator(
                    "cross_style_stats", alpha=0.5, donor_seed=42
                )
                model.set_token_style(style)
                torch.manual_seed(7)
                changed = model.extract(images, state, instructions, labels).tokens
                self.assertIsNotNone(model._style_handle)
                self.assertGreater(float(style.last_stats_t["mod_abs_mean"]), 0)
                self.assertFalse(torch.allclose(base, changed))
                loss = model.flow_matching_loss(
                    images,
                    state,
                    instructions,
                    torch.zeros(2, 10, 12),
                    torch.ones(2, 10, dtype=torch.bool),
                    labels,
                )
                self.assertTrue(torch.isfinite(loss))
                model.eval()
                actions = model.sample_action(images, state, instructions)
                self.assertEqual(actions.shape, (2, 10, 12))
                self.assertTrue(torch.isfinite(actions).all())
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
