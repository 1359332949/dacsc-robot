"""CPU contract regressions, with expensive dependency leaves replaced by fixtures."""

from __future__ import annotations

import copy
import ast
from collections import OrderedDict
from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from enum import IntEnum
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dacsc.corp_robocasa as corp
from corp.representation.extractor import RepresentationExtractor
from dacsc.corp_data import LinearNormalizer
from dacsc.rollout_protocol import prepare_episode, rollout_episode, seed_episode
from dacsc.token_style import TokenStyleModulator
from dacsc.training_state import (
    atomic_torch_save,
    evaluation_state,
    restore_training_state,
    training_state,
)
from scripts.aggregate_sr import agg, aggregate
from scripts.train_corp_robocasa import ModelEMA, evaluate
import scripts.eval_corp_robocasa_sr as sr
from dacsc.eval_protocol import PROTOCOL_VERSION
from dacsc.corp_images import image_contract
from dacsc.upstream import verify_upstream


def official_registry_modules():
    """Execute the checked-in upstream registry/helper, without importing MuJoCo."""
    registry_path = ROOT / "repos/robocasa/robocasa/utils/dataset_registry.py"
    wanted = {"ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"}
    tree = ast.parse(registry_path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
    ]
    scope = {"OrderedDict": OrderedDict}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(registry_path), "exec"), scope)
    registry = SimpleNamespace(**{key: scope[key] for key in wanted})
    helper_path = ROOT / "repos/robocasa/robocasa/utils/dataset_registry_utils.py"
    nodes = [
        n
        for n in ast.parse(helper_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_task_horizon"
    ]
    helper_scope = {}
    exec(
        compile(ast.Module(nodes, type_ignores=[]), str(helper_path), "exec"),
        helper_scope,
    )
    scene_path = ROOT / "repos/robocasa/robocasa/models/scenes/scene_registry.py"
    nodes = [
        n
        for n in ast.parse(scene_path.read_text()).body
        if (isinstance(n, ast.ClassDef) and n.name == "LayoutType")
        or (isinstance(n, ast.FunctionDef) and n.name == "unpack_layout_ids")
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "LAYOUT_GROUPS_TO_IDS"
                for t in n.targets
            )
        )
    ]
    scene_scope = {"IntEnum": IntEnum}
    exec(
        compile(ast.Module(nodes, type_ignores=[]), str(scene_path), "exec"),
        scene_scope,
    )
    return {
        "robocasa.utils.dataset_registry": registry,
        "robocasa.utils.dataset_registry_utils": SimpleNamespace(
            get_task_horizon=helper_scope["get_task_horizon"]
        ),
        "robocasa.models.scenes.scene_registry": SimpleNamespace(
            LayoutType=scene_scope["LayoutType"],
            unpack_layout_ids=scene_scope["unpack_layout_ids"],
        ),
    }


def training_origin_fixture(seed=42):
    return {
        "seed": seed,
        "budget": {"steps": 2000, "batch_size": 30, "lr": 5e-5},
        "optimization": {"optimizer": "torch.optim.AdamW"},
        "initialization_sha256": "c" * 64,
        "data_metadata_sha256": "d" * 64,
        "data_length": 100,
        "intervention": None,
        "pixel_augmentation": None,
    }


def sr_payload(task, successes, split):
    styles = list(range(31, 41)) if split == "val" else list(range(11, 31))
    protocol = {
        "version": PROTOCOL_VERSION,
        "checkpoint_sha256": "a" * 64,
        "shift": "none",
        "blur_sigma": 0.0,
        "layouts": list(range(11, 61)),
        "base_seed": 42,
        "action_steps": 8,
        "action_horizon": 10,
        "euler_steps": 10,
        "image_size": 224,
        "precision": "fp32",
        "horizon_source": "robocasa.utils.dataset_registry_utils.get_task_horizon",
        "images": image_contract(),
        "upstream": verify_upstream(),
        "training": training_origin_fixture(),
    }
    return {
        "protocol": protocol,
        "summary": {
            "task": task,
            "episodes": len(successes),
            "successes": sum(successes),
            "success_rate": sum(successes) / len(successes),
            "styles": styles,
            "shift": "none",
            "blur_sigma": 0.0,
            "seed": 42,
            "max_steps": 900,
        },
        "records": [
            {
                "episode": i,
                "success": value,
                "shift": "none",
                "blur_sigma": 0.0,
                "environment_seed": 42 + i,
                "policy_seed": 42 + i,
                "actual_style": styles[0],
                "actual_layout": 11,
                "style_ids": styles,
                "manifest_sha256": f"{i:064x}",
                "steps": 8,
            }
            for i, value in enumerate(successes)
        ],
    }


class AggregationTests(unittest.TestCase):
    def write_task(self, directory, task="A", successes=(True, False)):
        directory.mkdir(parents=True, exist_ok=True)
        payload = sr_payload(task, successes, directory.name)
        (directory / f"{task}.json").write_text(json.dumps(payload))
        return payload

    def test_nested_json_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_task(root / "source")
            self.write_task(root / "val", successes=(False, False))
            result = aggregate(root, ["source", "val"], ["A"], 2)
            self.assertEqual(result["source"]["success_rate"], 0.5)
            self.assertEqual(result["source"]["per_style"][11]["episodes"], 2)
            self.assertEqual(len(result["source"]["per_style"]), 1)
            self.assertEqual(result["gap_source_minus_val"], 0.5)
            self.assertNotIn("test", result)

    def test_missing_extra_truncated_and_corrupt_results_fail(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            with self.assertRaises(ValueError):
                agg(directory, ["A"], 2)
            payload = self.write_task(directory)
            with self.assertRaises(ValueError):
                agg(directory, ["A", "B"], 2)
            with self.assertRaises(ValueError):
                agg(directory, ["A"], 3)
            payload["summary"]["successes"] = 0
            (directory / "A.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                agg(directory, ["A"], 2)
            (directory / "A.json").write_text('{"summary":')
            with self.assertRaises(json.JSONDecodeError):
                agg(directory, ["A"], 2)

    def test_cli_failure_does_not_publish_summary(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/aggregate_sr.py"),
                    d,
                    "--tasks",
                    "A",
                    "--episodes",
                    "2",
                ],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(d) / "summary.json").exists())

    def test_split_runner_defaults_to_source_val_and_auto_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("run_corp_sr_split.sh", "aggregate_sr.py"):
                (root / name).write_text((ROOT / "scripts" / name).read_text())
            # Replace only the expensive env runner; execute the actual split shell/aggregator.
            (root / "run_corp_sr_tasks.sh").write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\ntest -f "$OUT/A.json"\n'
            )
            for split in ("source", "val"):
                for task in ("A", "B"):
                    self.write_task(root / "results" / split, task, (True,))
            env = {
                **os.environ,
                "MODEL": "unused.pt",
                "OUT": str(root / "results"),
                "TASKS": "A B",
                "EPISODES": "1",
                "MANIFEST_ROOT": str(root / "manifest"),
                "PY": sys.executable,
                "PYTHONPATH": str(ROOT),
            }
            env.pop("SPLITS", None)
            result = subprocess.run(
                ["bash", str(root / "run_corp_sr_split.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "results/summary.json").read_text())
            self.assertEqual(summary["source"]["episodes"], 2)
            self.assertEqual(summary["val"]["success_rate"], 1.0)
            self.assertNotIn("test", summary)
            self.assertFalse((root / "results/test").exists())


class Language(nn.Module):
    def forward(self, instructions):
        return torch.zeros(len(instructions), 1, 2), torch.zeros(len(instructions), 2)


class Resampler(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(views, 2, 2))
        self.view_embedding = nn.Parameter(torch.zeros(views, 1, 2))

    def forward(self, patches, language):
        return torch.cat(list(patches.values()), dim=1)


class StateEncoder(nn.Module):
    def forward(self, state):
        return torch.zeros(len(state), 1, 2)


def tiny_extractor(config):
    # Execute the REAL official forward, replacing only expensive leaf modules.
    module = RepresentationExtractor.__new__(RepresentationExtractor)
    nn.Module.__init__(module)
    module.view_names = tuple(config["view_names"])
    module.visual_encoder = nn.Identity()
    module.language_encoder = Language()
    module.cross_attention_resampler = Resampler(len(config["view_names"]))
    module.state_encoder = StateEncoder()
    return module


def tiny_corp(style=None, config=corp.ROBOCASA_CORP_CONFIG):
    with patch.object(
        corp, "RepresentationExtractor", side_effect=tiny_extractor
    ), patch.object(corp, "DiTActionGenerator", side_effect=lambda _: nn.Linear(2, 2)):
        return corp.CorpRoboCasa(config, token_style=style)


def official_fixture(path):
    from corp.checkpoint import config_digest

    config = copy.deepcopy(corp.ROBOCASA_CORP_CONFIG)
    config.update(
        view_names=["agentview", "eye_in_hand"],
        state_dim=8,
        action_dim=7,
        action_horizon=12,
    )
    source = tiny_corp(config=config)
    payload = {
        "format": "corp_libero_v2",
        "config_sha256": config_digest(config),
        "model": source.state_dict(),
        "normalizer": {
            "format": "corp_libero_normalizer_v1",
            "state_mean": [0.0] * 8,
            "state_std": [1.0] * 8,
            "action_min": [-1.0] * 7,
            "action_max": [1.0] * 7,
        },
    }
    path.with_name("config.json").write_text(json.dumps(config))
    torch.save(payload, path)
    return payload


class ViewOrderTests(unittest.TestCase):
    def test_official_view_major_order_and_label_alignment(self):
        class Stamp(nn.Module):
            def forward(self, tokens, domain_ids, training):
                self.seen = tokens.detach().clone()
                return tokens + domain_ids[:, None, None, None] * 1000

        style = Stamp()
        model = tiny_corp(style)
        batch, views = 2, model.view_names
        images = {
            v: torch.tensor([100 * vi + b for b in range(batch)], dtype=torch.float32)
            .view(batch, 1, 1)
            .expand(batch, 4, 2)
            .clone()
            .requires_grad_()
            for vi, v in enumerate(views)
        }
        labels = torch.tensor([1, 2])
        rep = model.extract(images, torch.zeros(batch, 9), ["a", "b"], labels)
        for b in range(batch):
            for vi in range(len(views)):
                self.assertEqual(style.seen[b, vi, 0, 0], 100 * vi + b)
                self.assertEqual(
                    rep.tokens[b, vi * 4, 0], 100 * vi + b + labels[b] * 1000
                )
        rep.tokens.sum().backward()
        for value in images.values():
            self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_noop_style_preserves_official_output_and_no_grad_train_is_safe(self):
        images = {
            v: torch.randn(2, 4, 2) for v in corp.ROBOCASA_CORP_CONFIG["view_names"]
        }
        model = tiny_corp(TokenStyleModulator(mode="none"))
        with torch.no_grad():
            actual = model.extract(images, torch.zeros(2, 9), ["a", "b"])
        model.set_token_style(None)
        expected = model.extract(images, torch.zeros(2, 9), ["a", "b"])
        torch.testing.assert_close(actual.tokens, expected.tokens, rtol=0, atol=0)

    def test_importance_changes_modulation_not_just_nonzero(self):
        style = TokenStyleModulator(
            mode="cross_style_stats_action_aware", alpha=1, dim=2
        )
        model = tiny_corp(style)
        z = torch.ones(2, 3, 256, 2, requires_grad=True)
        weights = torch.linspace(0, 1, 256)
        (z * weights[None, None, :, None]).sum().backward()
        model._last_style_tokens = z
        model.update_action_importance()
        imp = style.action_importance
        self.assertAlmostEqual(float(imp[-1]), 1.0)
        self.assertAlmostEqual(float(imp[0]), 0.0)
        self.assertGreater(float(imp.mean()), 0.4)  # Old 1/256 mass fails this.
        tokens = torch.randn_like(z) + torch.tensor([0.0, 4.0])[:, None, None, None]
        out = style(tokens, torch.tensor([0, 1]))
        torch.testing.assert_close(out[:, :, -1], tokens[:, :, -1])
        self.assertGreater(float((out[:, :, 0] - tokens[:, :, 0]).abs().mean()), 0.1)

    def test_load_path_registers_hook_and_checkpoint_load_is_strict(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "official.pt"
            official_fixture(path)
            with patch.object(
                corp, "RepresentationExtractor", side_effect=tiny_extractor
            ), patch.object(
                corp, "DiTActionGenerator", side_effect=lambda _: nn.Linear(2, 2)
            ), patch(
                "corp.checkpoint.CoRP",
                side_effect=lambda config, norm: corp.CorpRoboCasa(config),
            ):
                loaded, _ = corp.load_corp_robocasa(
                    path, token_style=TokenStyleModulator(mode="cross_style_stats")
                )
                self.assertIsNotNone(loaded._style_handle)
                images = {v: torch.randn(2, 4, 2) for v in loaded.view_names}
                loaded.train()
                loaded.extract(
                    images, torch.zeros(2, 9), ["a", "b"], torch.tensor([11, 12])
                )
                self.assertGreater(
                    float(loaded.token_style.last_stats_t["mod_abs_mean"]), 0
                )
                bad = {
                    "format": corp.CHECKPOINT_FORMAT,
                    "objective": corp.OBJECTIVE,
                    "images": image_contract(),
                    "upstream": verify_upstream(),
                    "config": loaded.config,
                    "token_style": None,
                    "model": {},
                    "normalizer": {},
                }
                torch.save(bad, path)
                with self.assertRaises(RuntimeError):
                    corp.load_adapted_checkpoint(path)

    def test_donor_rng_and_protection_resume_exactly(self):
        style = TokenStyleModulator(mode="cross_style_stats_action_aware", dim=2)
        model = tiny_corp(style)
        tokens = torch.randn(2, 3, 4, 2)
        labels = torch.tensor([11, 12])
        style.set_action_importance(torch.linspace(0, 1, 4))
        style(tokens, labels)
        saved = copy.deepcopy(model.training_runtime_state())
        expected = style(tokens, labels)
        donors = style.last_donor.clone()
        other = tiny_corp(
            TokenStyleModulator(mode="cross_style_stats_action_aware", dim=2)
        )
        other.load_training_runtime_state(saved)
        actual = other.token_style(tokens, labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(donors, other.token_style.last_donor))


class FakeEnv:
    def __init__(self):
        self.env = self
        self.rng = np.random.default_rng(5)
        self.state = np.array([1.0, 2.0, 3.0])
        self.meta = {
            "style_id": np.int64(11),
            "layout_id": np.int64(11),
            "lang": "open",
        }
        self.sim = SimpleNamespace(
            model=SimpleNamespace(get_xml=lambda: "<recorded/>"),
            get_state=lambda: self.state.copy(),
        )
        self.steps = 0

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(size=3)
        return {}, {}

    def get_ep_meta(self):
        return self.meta

    def _get_observations(self, force_update):
        return {"state": self.state.copy()}

    def get_observation(self, obs):
        return obs

    def step(self, action):
        self.steps += 1
        self.last_action = action
        return {}, 0.0, True, False, {"success": False}


def fake_reset_to(env, payload):
    if payload["model"] != "<recorded/>":
        raise ValueError("model XML missing")
    env.state = payload["states"].copy()
    env.meta = json.loads(payload["ep_meta"])
    torch.rand(7)  # Restoration must not shift the policy noise sequence.


class RolloutTests(unittest.TestCase):
    def test_invalid_policy_chunks_fail_instead_of_hanging_or_clipping_nan(self):
        for chunk in (
            torch.empty(1, 0, 12),
            torch.zeros(1, 10, 7),
            torch.full((1, 10, 12), float("nan")),
        ):
            policy = SimpleNamespace(sample_action=lambda *args: chunk)
            with self.subTest(shape=chunk.shape), self.assertRaises(ValueError):
                rollout_episode(
                    FakeEnv(),
                    policy,
                    None,
                    "cpu",
                    {},
                    "open",
                    policy_inputs=lambda *args: ({}, torch.zeros(1, 9), ["open"]),
                    convert_action=lambda a: a,
                    max_steps=900,
                    action_steps=8,
                    blur_sigma=0.0,
                    shift="none",
                )

    def test_manifest_replays_state_labels_and_policy_noise(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.json.gz"
            kwargs = dict(
                task="A",
                styles=[11, 12],
                layouts=list(range(11, 61)),
                episode=0,
                seed=42,
                reset_to=fake_reset_to,
            )
            obs, meta, digest = prepare_episode(
                FakeEnv(), path, mode="create", **kwargs
            )
            noise = torch.randn(5)
            random.random()
            np.random.rand(10)
            torch.rand(100)
            replay, meta2, digest2 = prepare_episode(
                FakeEnv(), path, mode="replay", **kwargs
            )
            np.testing.assert_array_equal(obs["state"], replay["state"])
            torch.testing.assert_close(noise, torch.randn(5), rtol=0, atol=0)
            self.assertEqual(digest, digest2)
            self.assertIs(type(meta2["actual_style"]), int)
            with self.assertRaises(FileExistsError):
                prepare_episode(FakeEnv(), path, mode="create", **kwargs)
            with self.assertRaises(ValueError):
                prepare_episode(
                    FakeEnv(), path, mode="replay", **{**kwargs, "styles": [31]}
                )
            with gzip.open(path, "rt") as stream:
                self.assertEqual(json.load(stream)["model_xml"], "<recorded/>")

    def test_done_exits_outer_loop_and_stochastic_policy_is_paired(self):
        class Policy:
            def sample_action(self, *args):
                return torch.randn(1, 10, 12)

        actions = []
        for _ in range(2):
            seed_episode(12)
            env = FakeEnv()
            result = rollout_episode(
                env,
                Policy(),
                None,
                "cpu",
                {},
                "open",
                policy_inputs=lambda *args: ({}, torch.zeros(1, 9), ["open"]),
                convert_action=lambda a: a,
                max_steps=600,
                action_steps=8,
                blur_sigma=0.0,
                shift="none",
            )
            self.assertEqual(result["steps"], 1)
            actions.append(env.last_action)
        np.testing.assert_array_equal(*actions)

    def test_sr_entrypoint_writes_native_json_and_reuses_manifest(self):
        class Policy:
            action_horizon = 10
            config = corp.ROBOCASA_CORP_CONFIG
            evaluation_origin = training_origin_fixture()

            def eval(self):
                return self

            def sample_action(self, *args):
                return torch.randn(1, 10, 12)

        envs = []

        def factory(**kwargs):
            self.assertEqual(int(kwargs["layout_ids"]), -2)
            env = FakeEnv()
            env.close = lambda: None
            envs.append(env)
            return env

        modules = {
            "robocasa.wrappers.gym_wrapper": SimpleNamespace(RoboCasaGymEnv=factory),
            "robocasa.utils.env_utils": SimpleNamespace(convert_action=lambda a: a),
            "robocasa.scripts.dataset_scripts.playback_dataset": SimpleNamespace(
                reset_to=fake_reset_to
            ),
        }
        modules.update(official_registry_modules())
        with tempfile.TemporaryDirectory() as d, patch.dict(
            sys.modules, modules
        ), patch.object(
            sr,
            "load_adapted_checkpoint",
            return_value=(Policy(), (torch.zeros(9), torch.ones(9))),
        ), patch.object(
            sr, "policy_inputs", return_value=({}, torch.zeros(1, 9), ["open"])
        ):
            root = Path(d)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fixed synthetic weights")
            outputs = []
            for mode, shift in (("create", "none"), ("replay", "color_jitter")):
                output = root / f"{shift}.json"
                argv = [
                    "sr",
                    "--model",
                    str(checkpoint),
                    "--task",
                    "CloseFridge",
                    "--styles",
                    "11-12",
                    "--episodes",
                    "1",
                    "--device",
                    "cpu",
                    "--out",
                    str(output),
                    "--manifest-dir",
                    str(root / "manifest"),
                    "--manifest-mode",
                    mode,
                    "--shift",
                    shift,
                ]
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(sr.main(), 0)
                outputs.append(json.loads(output.read_text()))
            for output in outputs:
                self.assertEqual(output["summary"]["episodes"], 1)
                self.assertEqual(output["summary"]["max_steps"], 900)
                self.assertEqual(output["protocol"]["layouts"], list(range(11, 61)))
                self.assertIs(type(output["records"][0]["actual_style"]), int)
            self.assertEqual(
                outputs[0]["records"][0]["manifest_sha256"],
                outputs[1]["records"][0]["manifest_sha256"],
            )
            np.testing.assert_array_equal(envs[0].last_action, envs[1].last_action)


class TinyModel(nn.Linear):
    def __init__(self):
        super().__init__(2, 1)
        self.runtime = torch.tensor([0.0])

    def training_runtime_state(self):
        return {"value": self.runtime.clone()}

    def load_training_runtime_state(self, state):
        self.runtime = state["value"].clone()


class TrainingStateTests(unittest.TestCase):
    def test_ema_weights_and_rng_restore_even_after_exception(self):
        model = TinyModel().train()
        ema = ModelEMA(model, 0.9)
        original = copy.deepcopy(model.state_dict())
        ema.shadow["weight"].fill_(8)
        torch.manual_seed(42)
        expected = torch.rand(4)
        torch.manual_seed(42)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with evaluation_state(model, ema, seed=5):
                self.assertTrue(torch.all(model.weight == 8))
                self.assertFalse(model.training)
                torch.rand(70)
                raise RuntimeError("injected")
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, original[k], rtol=0, atol=0)

    def test_clean_shift_and_test_eval_all_use_same_ema_and_noise(self):
        class Policy(TinyModel):
            def sample_action(self, images, state, instructions, **kwargs):
                self.noises.append(torch.randn(len(state), 10, 12))
                self.weights.append(self.weight.detach().clone())
                return self.noises[-1] + self.weight[0, 0]

        model = Policy()
        model.noises, model.weights = [], []
        ema = ModelEMA(model, 0.9)
        ema.shadow["weight"].fill_(2)
        batch = {
            "images": {"v": torch.zeros(2, 3, 4, 4)},
            "state": torch.zeros(2, 9),
            "action": torch.zeros(2, 10, 12),
            "instruction": ["a", "b"],
            "action_mask": torch.ones(2, 10, dtype=torch.bool),
            "image_space": "imagenet_normalized",
            "episode": [0, 0],
            "frame": [0, 0],
            "style_id": torch.tensor([11, 12]),
            "task": ["A", "B"],
        }
        norm = LinearNormalizer(torch.zeros(9), torch.ones(9))
        for shift in ("none", "color_jitter", "none"):
            evaluate(model, [batch], norm, torch.device("cpu"), 1, 10, shift, ema=ema)
        for weight in model.weights:
            self.assertTrue(torch.all(weight == 2))
        torch.testing.assert_close(model.noises[0], model.noises[1], rtol=0, atol=0)
        torch.testing.assert_close(model.noises[0], model.noises[2], rtol=0, atol=0)

    def test_checkpoint_resume_matches_uninterrupted_next_update_and_data(self):
        seed_episode(5)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1 / (n + 1))
        ema = ModelEMA(model, 0.9)
        generator = torch.Generator().manual_seed(9)
        epoch_rng = generator.get_state()
        loader = DataLoader(
            TensorDataset(torch.arange(12)),
            batch_size=2,
            shuffle=True,
            generator=generator,
        )
        it = iter(loader)
        first = next(it)

        def update(m, o, sc, e):
            x = torch.randn(2, 2) + random.random() + float(np.random.rand())
            o.zero_grad()
            m(x).square().sum().backward()
            o.step()
            sc.step()
            e.update(m)
            m.runtime += 1

        update(model, opt, sched, ema)
        payload = {
            "raw_model": model.state_dict(),
            "training_state": training_state(
                model,
                opt,
                sched,
                ema,
                step=1,
                data_order={"epoch_rng": epoch_rng, "batches_in_epoch": 1},
            ),
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.pt"
            atomic_torch_save(payload, path)
            second_batch = next(it)
            update(model, opt, sched, ema)
            expected = copy.deepcopy(model.state_dict())
            expected_ema = copy.deepcopy(ema.shadow)

            resumed = TinyModel()
            opt2 = torch.optim.AdamW(resumed.parameters(), lr=0.01)
            sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda n: 1 / (n + 1))
            ema2 = ModelEMA(resumed, 0.9)
            saved = torch.load(path, weights_only=False)
            g2 = torch.Generator()
            g2.set_state(saved["training_state"]["data_order"]["epoch_rng"])
            it2 = iter(
                DataLoader(loader.dataset, batch_size=2, shuffle=True, generator=g2)
            )
            torch.testing.assert_close(next(it2)[0], first[0])
            for _ in range(
                saved["training_state"]["data_order"]["batches_in_epoch"] - 1
            ):
                next(it2)
            step = restore_training_state(saved, resumed, opt2, sched2, ema2)
            self.assertEqual(step, 1)
            torch.testing.assert_close(next(it2)[0], second_batch[0])
            update(resumed, opt2, sched2, ema2)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                torch.testing.assert_close(
                    ema2.shadow[key], expected_ema[key], rtol=0, atol=0
                )
            torch.testing.assert_close(resumed.runtime, model.runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
