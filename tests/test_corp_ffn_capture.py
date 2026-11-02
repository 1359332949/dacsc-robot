"""Local inference/fixture regressions for a read-only observer, never training."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_ffn_capture import (
    CoRPFFNCapture,
    load_registers_reference,
    rank_register_images,
)
from dacsc.corp_robocasa import load_corp_robocasa
from dacsc.upstream import verify_release_checkpoint
from test_corp_regressions import tiny_corp

REFERENCE = Path(
    os.environ.get(
        "DACSC_REGISTERS_REPO",
        ROOT
        / ".cache/analysis_sources/registers/860df43515c8d8e9e90952af25a46c26e4469570",
    )
)


class FixtureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(2, 4)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(4, 2)
        self.fail = False

    def forward(self, x):
        hidden = self.mlp.act(self.mlp.fc1(x))
        if self.fail:
            raise RuntimeError("fixture downstream failure")
        return x + self.mlp.fc2(hidden)


class FixtureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FixtureBlock(), FixtureBlock()])
        self.input_seen = None
        self.token_count = 257

    def forward(self, image):
        self.input_seen = image.detach().clone()
        # Distinct sample/view/position identities; not a scientific model.
        pos = torch.arange(self.token_count, dtype=image.dtype, device=image.device)
        x = (image[:, 0, 0, 0, None] + pos[None] / 100).unsqueeze(-1).expand(-1, -1, 2)
        for block in self.blocks:
            x = block(x)
        return x[:, 1:]


class FixtureVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FixtureBackbone()

    def forward(self, x):
        return self.backbone(x)


def fixture_policy():
    model = tiny_corp()
    model.representation_extractor.visual_encoder = FixtureVisual()
    return model.eval()


def images_for(model, batch=2):
    # Reverse insertion order so code relying on dict order fails.
    return {
        view: torch.stack(
            [torch.full((3, 224, 224), 1.0 + 3 * vi + b) for b in range(batch)]
        )
        for vi, view in reversed(list(enumerate(model.view_names)))
    }


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def hook_count(model):
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_registers_reference(REFERENCE)

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_reference_is_original_isolated_and_hash_checked(self):
        self.assertEqual(
            Path(self.ref.find_register_neurons.__code__.co_filename),
            REFERENCE.resolve() / "shared/algorithms.py",
        )
        self.assertEqual(
            Path(self.ref.log_internal.__code__.co_filename),
            REFERENCE.resolve() / "shared/hook_fn.py",
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                load_registers_reference(root)
            for rel in self.ref.contract["files"]:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REFERENCE / rel, path)
            (root / "shared/hook_fn.py").write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_registers_reference(root)

    def test_cls_patch_and_three_view_order_match_real_official_extractor(self):
        model = fixture_policy()
        images = images_for(model)
        backbone = model.representation_extractor.visual_encoder.backbone
        raw = {}
        handles = []
        for i, block in enumerate(backbone.blocks):
            handles.append(
                block.mlp.act.register_forward_hook(
                    lambda m, a, out, i=i: raw.__setitem__(
                        ("ffn", i), out.detach().clone()
                    )
                )
            )
            handles.append(
                block.register_forward_hook(
                    lambda m, a, out, i=i: raw.__setitem__(
                        ("blocks", i), out.detach().clone()
                    )
                )
            )
        with torch.inference_mode():
            baseline = model.extract(images, torch.zeros(2, 9), ["a", "b"])
        for handle in handles:
            handle.remove()
        before, rng = state_digest(model), torch.get_rng_state().clone()
        initial_hooks = hook_count(model)
        with patch.object(
            self.ref, "log_internal", wraps=self.ref.log_internal
        ) as logger:
            with CoRPFFNCapture(model, self.ref) as capture, torch.inference_mode():
                actual = model.extract(images, torch.zeros(2, 9), ["a", "b"])
                frame = capture.last
            self.assertEqual(logger.call_count, 4)
        self.assertEqual(hook_count(model), initial_hooks)
        self.assertEqual(state_digest(model), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(actual.tokens, baseline.tokens, rtol=0, atol=0)
        torch.testing.assert_close(
            backbone.input_seen,
            torch.cat([images[v] for v in model.view_names]),
            rtol=0,
            atol=0,
        )
        for sample in range(2):
            for vi, view in enumerate(model.view_names):
                ffn, blocks = frame.upstream_arrays(sample, view)
                for layer in range(2):
                    row = vi * 2 + sample
                    np.testing.assert_array_equal(
                        ffn[layer], raw["ffn", layer][row].numpy()
                    )
                    np.testing.assert_array_equal(
                        blocks[layer], raw["blocks", layer][row].numpy()
                    )
                    torch.testing.assert_close(
                        frame.ffn[layer][sample, vi, 0],
                        raw["ffn", layer][row, 0],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        frame.ffn[layer][sample, vi, 1:],
                        raw["ffn", layer][row, 1:],
                        rtol=0,
                        atol=0,
                    )
        self.assertEqual(frame.ffn[0].shape, (2, 3, 257, 4))
        self.assertFalse(frame.ffn[0].requires_grad)
        self.assertEqual(frame.blocks[0].dtype, torch.float32)

    def test_snapshots_do_not_alias_live_outputs_and_only_latest_frame_is_kept(self):
        model = fixture_policy()
        images = images_for(model)
        with CoRPFFNCapture(model, self.ref) as c, torch.inference_mode():
            first_output = model.extract(images, torch.zeros(2, 9), ["a", "b"]).tokens
            preserved = first_output.clone()
            first = c.last
            first.blocks[1].zero_()
            first.ffn[1].fill_(-999)
            torch.testing.assert_close(first_output, preserved, rtol=0, atol=0)
            model.extract(
                {k: v + 1 for k, v in images.items()}, torch.zeros(2, 9), ["a", "b"]
            )
            self.assertIsNot(c.last, first)
            self.assertTrue(
                all(len(x) == 1 for rows in c._logs.values() for x in rows.values())
            )
            self.assertFalse(torch.all(c.last.ffn[1] == -999))

    def test_eval_no_grad_and_no_style_are_required_without_changing_model_mode(self):
        model = fixture_policy()
        images = images_for(model)
        initial = hook_count(model)
        with self.assertRaisesRegex(RuntimeError, "requires eval"):
            with CoRPFFNCapture(model, self.ref):
                model.extract(images, torch.zeros(2, 9), ["a", "b"])
        self.assertFalse(model.training)
        model.train()
        with self.assertRaisesRegex(RuntimeError, "requires eval"):
            with CoRPFFNCapture(model, self.ref), torch.inference_mode():
                model.extract(images, torch.zeros(2, 9), ["a", "b"])
        self.assertTrue(model.training)
        self.assertEqual(hook_count(model), initial)
        model.eval()
        model.token_style = nn.Identity()
        with self.assertRaisesRegex(ValueError, "styled policy"):
            CoRPFFNCapture(model, self.ref)

    def test_exception_and_reentry_remove_observers_without_hiding_failure(self):
        model = fixture_policy()
        images = images_for(model)
        initial = hook_count(model)
        capture = CoRPFFNCapture(model, self.ref)
        with self.assertRaisesRegex(RuntimeError, "downstream failure"):
            with capture, torch.inference_mode():
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    capture.__enter__()
                model.representation_extractor.visual_encoder.backbone.blocks[
                    -1
                ].fail = True
                model.extract(images, torch.zeros(2, 9), ["a", "b"])
        self.assertEqual(hook_count(model), initial)
        self.assertIsNone(capture.last)
        model.representation_extractor.visual_encoder.backbone.blocks[-1].fail = False
        with capture, torch.inference_mode():
            model.extract(images, torch.zeros(2, 9), ["a", "b"])
            self.assertIsNotNone(capture.last)
        self.assertEqual(hook_count(model), initial)

    def test_invalid_shapes_views_and_missing_tokens_are_not_silently_reshaped(self):
        model = fixture_policy()
        with self.assertRaises(ValueError):
            CoRPFFNCapture(model, self.ref, view_names=("unknown",))
        backbone = model.representation_extractor.visual_encoder.backbone
        with CoRPFFNCapture(model, self.ref) as c, torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "divisible"):
                backbone(torch.zeros(5, 3, 224, 224))
            with self.assertRaisesRegex(ValueError, "224"):
                backbone(torch.zeros(6, 3, 14, 14))
            backbone.token_count = 256
            with self.assertRaisesRegex(ValueError, "CLS/patch"):
                backbone(torch.zeros(6, 3, 224, 224))
            self.assertIsNone(c.last)

    def test_cpu_bf16_log_conversion_preserves_forward_and_records_original_dtype(self):
        model = fixture_policy()
        x = images_for(model)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            base = model.extract(x, torch.zeros(2, 9), ["a", "b"]).tokens
            with CoRPFFNCapture(model, self.ref) as c:
                actual = model.extract(x, torch.zeros(2, 9), ["a", "b"]).tokens
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
        self.assertEqual(c.last.source_dtypes["ffn.0"], "torch.bfloat16")
        self.assertEqual(c.last.ffn[0].dtype, torch.float32)
        self.assertTrue(
            np.isfinite(c.last.upstream_arrays(0, model.view_names[0])[0]).all()
        )

    def test_calibration_and_changed_view_order_cannot_silently_mislabel_samples(self):
        model = fixture_policy()
        images = images_for(model)
        with CoRPFFNCapture(
            model, self.ref, view_names=(model.view_names[2],)
        ), torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "cannot label"):
                model.extract(images, torch.zeros(2, 9), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "full policy order"):
            CoRPFFNCapture(model, self.ref, view_names=model.view_names[:2])
        capture = CoRPFFNCapture(model, self.ref)
        old = model.representation_extractor.view_names
        model.representation_extractor.view_names = tuple(reversed(old))
        with capture, torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "camera order changed"):
                model.extract(images, torch.zeros(2, 9), ["a", "b"])
        self.assertEqual(hook_count(model), 0)

    def test_original_register_detector_runs_with_single_view_adapter_and_no_fallback(
        self,
    ):
        model = fixture_policy()
        before, initial = state_digest(model), hook_count(model)
        original_rng = random.getstate()
        with tempfile.TemporaryDirectory() as d:
            for i in (1, 2):
                Image.fromarray(np.full((224, 224, 3), i * 100, np.uint8)).save(
                    Path(d) / f"{i}.png"
                )
            with patch.object(
                self.ref, "find_register_neurons", wraps=self.ref.find_register_neurons
            ) as detector, redirect_stdout(io.StringIO()), redirect_stderr(
                io.StringIO()
            ):
                ranking = rank_register_images(
                    model,
                    self.ref,
                    d,
                    view_name=model.view_names[2],
                    register_norm_threshold=0.01,
                    detect_outliers_layer=-1,
                    processed_image_cnt=2,
                )
                self.assertEqual(detector.call_count, 1)
                self.assertEqual(detector.call_args.kwargs["device"], "cpu")
                self.assertNotIn("apply_sparsity_filter", detector.call_args.kwargs)
                self.assertEqual(len(ranking), 8)
                self.assertEqual({int(r[0]) for r in ranking}, {0, 1})
                self.assertTrue(
                    all(ranking[i][2] >= ranking[i + 1][2] for i in range(7))
                )
                with self.assertRaisesRegex(AssertionError, "No images processed"):
                    rank_register_images(
                        model,
                        self.ref,
                        d,
                        view_name=model.view_names[0],
                        register_norm_threshold=1e20,
                        detect_outliers_layer=-1,
                        processed_image_cnt=2,
                    )
            self.assertEqual(hook_count(model), initial)
            self.assertEqual(state_digest(model), before)
            self.assertEqual(random.getstate(), original_rng)
            for layer in (-3, 2, 0.5):
                with self.assertRaises(ValueError):
                    rank_register_images(
                        model,
                        self.ref,
                        d,
                        view_name=model.view_names[0],
                        register_norm_threshold=1,
                        detect_outliers_layer=layer,
                    )

    def test_real_release_all_layers_representation_actions_rng_and_weights_are_identical(
        self,
    ):
        path = ROOT / "repos/CoRP/checkpoints/corp_libero.pt"
        verify_release_checkpoint(
            path
        )  # Required here: never silently skip the real-weight check.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model, _ = load_corp_robocasa(path)
            images = {v: value * 0.03 for v, value in images_for(model).items()}
            state = torch.zeros(2, 9)
            instructions = ["open the drawer", "close the drawer"]
            baseline_state = state_digest(model)
            original_hooks = hook_count(model)
            raw = {}
            handles = [
                model.representation_extractor.visual_encoder.backbone.blocks[
                    0
                ].mlp.act.register_forward_hook(
                    lambda m, a, o: raw.update(ffn=o.detach().clone())
                )
            ]
            with torch.inference_mode():
                before = model.extract(images, state, instructions)
            for h in handles:
                h.remove()
            torch.manual_seed(123)
            base_action = model.sample_action(images, state, instructions)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(123)
            with CoRPFFNCapture(model, self.ref) as capture, torch.inference_mode():
                before_extract_rng = torch.get_rng_state().clone()
                after = model.extract(images, state, instructions)
                self.assertTrue(torch.equal(torch.get_rng_state(), before_extract_rng))
                for sample in range(2):
                    for view in range(3):
                        torch.testing.assert_close(
                            capture.last.ffn[0][sample, view],
                            raw["ffn"][view * 2 + sample],
                            rtol=0,
                            atol=0,
                        )
                actual_action = model.sample_action(images, state, instructions)
                self.assertEqual(len(capture.last.ffn), 12)
                self.assertEqual(capture.last.ffn[11].shape, (2, 3, 257, 1536))
                self.assertEqual(capture.last.blocks[11].shape, (2, 3, 257, 384))
            torch.testing.assert_close(after.tokens, before.tokens, rtol=0, atol=0)
            torch.testing.assert_close(
                after.pooled_language, before.pooled_language, rtol=0, atol=0
            )
            self.assertEqual(after.action_dtype, before.action_dtype)
            torch.testing.assert_close(actual_action, base_action, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual_action).all())
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(hook_count(model), original_hooks)
            self.assertEqual(state_digest(model), baseline_state)
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
