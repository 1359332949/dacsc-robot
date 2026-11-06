"""Upstream contract, donor and output ownership regressions; no training entry is run."""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dacsc.corp_robocasa as corp
from dacsc.rollout_protocol import prepare_episode
from dacsc.run_output import entry_output
from dacsc.token_style import TokenStyleModulator
from scripts.aggregate_sr import agg, aggregate
from test_corp_regressions import (
    FakeEnv,
    fake_reset_to,
    official_fixture,
    official_registry_modules,
    sr_payload,
    tiny_extractor,
)


class OfficialRegistryTests(unittest.TestCase):
    def test_upstream_task_horizons_and_pretrain_layout_group(self):
        modules = official_registry_modules()
        with patch.dict(sys.modules, modules):
            from robocasa.utils.dataset_registry_utils import get_task_horizon
            from robocasa.models.scenes.scene_registry import (
                LayoutType,
                unpack_layout_ids,
            )

            self.assertEqual(unpack_layout_ids(LayoutType.TRAIN), list(range(11, 61)))
            self.assertEqual(get_task_horizon("CloseFridge"), 900)
            self.assertEqual(get_task_horizon("OpenCabinet"), 1050)
            self.assertEqual(get_task_horizon("CloseDrawer"), 450)
            with self.assertRaises(ValueError):
                get_task_horizon("nonexistent-task")

    def test_out_of_pretrain_layout_never_enters_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.json.gz"
            env = FakeEnv()
            env.meta["layout_id"] = 3
            with self.assertRaisesRegex(ValueError, "layout"):
                prepare_episode(
                    env,
                    path,
                    task="A",
                    scene_split="pretrain",
                    styles=[11],
                    layouts=list(range(11, 61)),
                    episode=0,
                    seed=42,
                    mode="create",
                    reset_to=fake_reset_to,
                )
            self.assertFalse(path.exists())

    def test_replay_rejects_changed_layout_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.json.gz"
            kwargs = dict(
                task="A",
                scene_split="pretrain",
                styles=[11],
                layouts=list(range(11, 61)),
                episode=0,
                seed=42,
                reset_to=fake_reset_to,
            )
            prepare_episode(FakeEnv(), path, mode="create", **kwargs)
            with self.assertRaises(ValueError):
                prepare_episode(
                    FakeEnv(), path, mode="replay", **{**kwargs, "layouts": [11]}
                )

    def test_target_manifest_records_scene_split_and_target_pair(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "target.json.gz"
            env = FakeEnv()
            env.meta = {"style_id": 1, "layout_id": 1, "lang": "open"}
            prepare_episode(
                env,
                path,
                task="OpenDrawer",
                scene_split="target",
                styles=list(range(1, 11)),
                layouts=list(range(1, 11)),
                episode=0,
                seed=42,
                mode="create",
                reset_to=fake_reset_to,
                requested_style=1,
                requested_layout=1,
            )
            with gzip.open(path, "rt") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["scene_split"], "target")
            self.assertEqual((manifest["actual_layout"], manifest["actual_style"]), (1, 1))

    def test_create_clears_stale_episode_meta_and_honors_requested_scene(self):
        class SceneEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self._ep_meta = {"layout_id": 15, "style_id": 20}
                self.reset_calls = 0

            def reset(self, seed):
                self.reset_calls += 1
                if self._ep_meta:
                    self.meta = dict(self._ep_meta)
                else:
                    self.meta = {"layout_id": 11, "style_id": 11}
                return super().reset(seed)

            def get_ep_meta(self):
                return self.meta

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.json.gz"
            kwargs = dict(
                task="A",
                scene_split="pretrain",
                styles=list(range(11, 31)),
                layouts=list(range(11, 61)),
                episode=0,
                seed=42,
                mode="create",
                reset_to=fake_reset_to,
            )
            stale = SceneEnv()
            prepare_episode(stale, path, **kwargs)
            with gzip.open(path, "rt") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["actual_style"], 11)
            self.assertEqual(manifest["actual_layout"], 11)

            path.unlink()
            forced = SceneEnv()
            prepare_episode(
                forced,
                path,
                **kwargs,
                requested_style=17,
                requested_layout=23,
            )
            with gzip.open(path, "rt") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["actual_style"], 17)
            self.assertEqual(manifest["actual_layout"], 23)
            self.assertEqual(manifest["requested_style"], 17)
            self.assertEqual(manifest["requested_layout"], 23)


class ExperimentIdentityTests(unittest.TestCase):
    def test_mixed_checkpoint_shift_seed_or_inference_budget_is_rejected(self):
        mutations = {
            "checkpoint_sha256": "b" * 64,
            "shift": "color_jitter",
            "base_seed": 43,
            "euler_steps": 20,
            "precision": "bf16",
            "layouts": [11],
        }
        for key, value in mutations.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                directory = Path(d)
                source_ranges = {"source": list(range(11, 51))}
                a = sr_payload("A", (True,), "source")
                b = sr_payload("B", (True,), "source")
                b["protocol"][key] = value
                (directory / "A.json").write_text(json.dumps(a))
                (directory / "B.json").write_text(json.dumps(b))
                with self.assertRaises(ValueError):
                    agg(directory, ["A", "B"], 1, source_ranges)

    def test_cross_split_identity_and_native_styles_are_validated(self):
        for change in ("checkpoint", "styles", "horizon"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                for split in ("source", "val"):
                    (root / split).mkdir()
                    payload = sr_payload("A", (True,), split)
                    if split == "val":
                        if change == "checkpoint":
                            payload["protocol"]["checkpoint_sha256"] = "b" * 64
                        elif change == "styles":
                            payload["summary"]["styles"] = list(range(11, 31))
                            payload["records"][0]["style_ids"] = list(range(11, 31))
                            payload["records"][0]["actual_style"] = 11
                        else:
                            payload["summary"]["max_steps"] = 600
                    (root / split / "A.json").write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    aggregate(root, ["source", "val"], ["A"], 1)

    def test_old_schema_and_record_condition_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source_ranges = {"source": list(range(11, 51))}
            payload = sr_payload("A", (True,), "source")
            payload["records"][0]["shift"] = "color_jitter"
            (directory / "A.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                agg(directory, ["A"], 1, source_ranges)
            del payload["protocol"]
            (directory / "A.json").write_text(json.dumps(payload))
            with self.assertRaises(KeyError):
                agg(directory, ["A"], 1, source_ranges)

    def test_style_schedule_and_fixed_layout_mismatch_fail(self):
        mutations = (
            lambda p: p["summary"].__setitem__("style_order", [12]),
            lambda p: p["records"][0].__setitem__("actual_style", 12),
            lambda p: p["records"][0].__setitem__("requested_layout", 12),
            lambda p: p["summary"].__setitem__("layout_id", 12),
        )
        for mutate in mutations:
            with self.subTest(), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                payload = sr_payload("A", (True,), "source")
                mutate(payload)
                (root / "A.json").write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    agg(root, ["A"], 1, {"source": list(range(11, 51))})

    def test_target_schedule_is_validated_against_official_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = sr_payload("A", (True, False), "target")
            (root / "A.json").write_text(json.dumps(payload))
            result = agg(root, ["A"], 2, {"target": list(range(1, 11))})
            self.assertEqual(result["macro_task_success_rate"], 0.5)
            payload["summary"]["layout_order"] = [1, 1]
            (root / "A.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                agg(root, ["A"], 2, {"target": list(range(1, 11))})

    def test_training_origin_data_contract_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = sr_payload("A", (True,), "source")
            payload["protocol"]["training"]["data_contract"]["val_styles"] = [50, 60]
            (root / "A.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                agg(root, ["A"], 1, {"source": list(range(11, 51))})


class OfficialCheckpointTests(unittest.TestCase):
    def test_coherent_checkpoint_metadata_cannot_enable_retired_normalization_modes(
        self,
    ):
        from dacsc.corp_images import image_contract
        from dacsc.upstream import verify_upstream

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "unsupported_mode.pt"
            for mode in ("normalize", "normalize_action_aware", "none", "unknown"):
                torch.save(
                    {
                        "format": corp.CHECKPOINT_FORMAT,
                        "objective": corp.OBJECTIVE,
                        "images": image_contract(),
                        "upstream": verify_upstream(),
                        "token_style": {"mode": mode},
                    },
                    path,
                )
                with self.subTest(mode=mode), self.assertRaisesRegex(
                    ValueError, "outside the current CoRP protocol"
                ):
                    corp.load_adapted_checkpoint(path)

    def test_release_loader_rejects_missing_extra_shape_and_digest_errors(self):
        for change in ("missing", "unexpected", "shape", "digest"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "corp_libero.pt"
                payload = official_fixture(path)
                key = "action_generator.weight"
                if change == "missing":
                    del payload["model"][key]
                elif change == "unexpected":
                    payload["model"]["unknown.weight"] = torch.ones(1)
                elif change == "shape":
                    payload["model"][key] = torch.ones(1, 1)
                else:
                    payload["config_sha256"] = "0" * 64
                torch.save(payload, path)
                with patch.object(
                    corp, "RepresentationExtractor", side_effect=tiny_extractor
                ), patch.object(
                    corp, "DiTActionGenerator", side_effect=lambda _: nn.Linear(2, 2)
                ), patch(
                    "corp.checkpoint.CoRP",
                    side_effect=lambda config, norm: corp.CorpRoboCasa(config),
                ):
                    with self.assertRaises((RuntimeError, ValueError)):
                        corp.load_corp_robocasa(path)


class OutputOwnershipTests(unittest.TestCase):
    def wrapper(self, out, script, *args, file=False):
        command = [sys.executable, "-m", "dacsc.run_output", "--out", str(out)]
        if file:
            command += ["--file", "--log", "A.log"]
        return subprocess.run(
            command + ["--", str(script), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_other_process_cannot_write_or_truncate_locked_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "run"
            script = root / "must_not_run.py"
            script.write_text("raise AssertionError('entry must not execute')\n")
            with FileLock(str(out) + ".lock", timeout=0):
                result = self.wrapper(out, script)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((out / "run.log").exists())
            self.assertIn("Timeout", result.stderr)

    def test_existing_logs_and_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "run"
            out.mkdir()
            log = out / "run.log"
            log.write_text("keep-this-log")
            script = root / "entry.py"
            script.write_text("print('new')\n")
            result = self.wrapper(out, script)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(log.read_text(), "keep-this-log")
            with self.assertRaises(FileExistsError):
                with entry_output(out):
                    pass

    def test_same_process_entry_shares_lock_and_child_failure_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "run"
            script = root / "entry.py"
            script.write_text(
                "from pathlib import Path\nimport sys\nfrom dacsc.run_output import entry_output\n"
                "with entry_output(Path(sys.argv[1])):\n print('fixture-only; no training')\n"
                "raise SystemExit(23)\n"
            )
            result = self.wrapper(out, script, str(out))
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertIn("fixture-only", (out / "run.log").read_text())

    def test_result_file_lock_precedes_log_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "A.json"
            script = root / "entry.py"
            script.write_text("print('must not run')\n")
            with FileLock(str(target) + ".lock", timeout=0):
                result = self.wrapper(target, script, file=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "A.log").exists())

    def test_matrix_runner_stops_on_first_failed_interpreter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            interpreter = root / "failure_stub"
            interpreter.write_text("#!/bin/sh\nexit 23\n")
            interpreter.chmod(0o755)
            for name in ("run_corp_experiment.sh",):
                with self.subTest(name=name):
                    env = {
                        **os.environ,
                        "PY": str(interpreter),
                        "OUT": str(root / name),
                        "STEPS": "51",
                        "SEEDS": "42 43",
                        "MODES": "none",
                    }
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts" / name)],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 23)
                    self.assertNotIn("ALL_DONE", result.stdout)
                    self.assertNotIn("seed=43", result.stdout)


class DonorRulesTests(unittest.TestCase):
    def test_wrong_action_importance_cannot_silently_disable_protection(self):
        module = TokenStyleModulator("cross_style_stats_action_aware")
        module.set_action_importance(torch.ones(3))
        with self.assertRaisesRegex(ValueError, "patch-token count"):
            module(torch.randn(2, 3, 4, 8), torch.tensor([11, 12]))
        with self.assertRaises(ValueError):
            module.set_action_importance(torch.ones(2, 2))

    def test_topk_action_protection_changes_actual_modulation(self):
        tokens = torch.randn(2, 3, 4, 8)
        module = TokenStyleModulator(
            "cross_style_stats_action_aware",
            alpha=1.0,
            action_weight=0.0,
            protect_ratio=0.25,
        )
        module.set_action_importance(torch.tensor([0.0, 0.2, 1.0, 0.4]))
        changed = module(tokens, torch.tensor([11, 12]))
        torch.testing.assert_close(changed[:, :, 2], tokens[:, :, 2], rtol=0, atol=0)
        self.assertGreater(float((changed[:, :, 0] - tokens[:, :, 0]).abs().mean()), 0)

    def test_changed_or_missing_donor_contract_cannot_be_loaded(self):
        from test_corp_regressions import tiny_corp

        model = tiny_corp(TokenStyleModulator("cross_style_stats"))
        saved = copy.deepcopy(model.training_runtime_state())
        saved["donor_contract"]["task_conditioned"] = True
        with self.assertRaisesRegex(ValueError, "donor rule"):
            model.load_training_runtime_state(saved)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "legacy_style.pt"
            from dacsc.corp_images import image_contract
            from dacsc.upstream import verify_upstream

            torch.save(
                {
                    "format": corp.CHECKPOINT_FORMAT,
                    "objective": corp.OBJECTIVE,
                    "images": image_contract(),
                    "upstream": verify_upstream(),
                    "token_style": {"mode": "cross_style_stats"},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "donor contract"):
                corp.load_adapted_checkpoint(path)

    def test_random_donor_matches_upstream_sample_permutation_and_preserves_view(self):
        batch, views = 5, 3
        tokens = torch.randn(batch, views, 8, 4)
        module = TokenStyleModulator("random_stats", alpha=1.0, donor_seed=42)
        out = module(tokens)
        expected = torch.randperm(batch, generator=torch.Generator().manual_seed(42))
        row = expected[:, None] * views + torch.arange(views)
        self.assertTrue(torch.equal(module.last_donor, row.flatten()))
        torch.testing.assert_close(
            out.mean(2), tokens[expected].mean(2), atol=1e-5, rtol=1e-5
        )

    def test_cross_style_is_exact_even_under_29_to_1_imbalance(self):
        tokens = torch.randn(30, 3, 8, 4)
        labels = torch.tensor([11] * 29 + [12])
        module = TokenStyleModulator("cross_style_stats", donor_seed=42)
        rng = torch.get_rng_state()
        module(tokens, labels)
        donors = module.last_donor.reshape(30, 3)
        self.assertTrue(torch.all(donors % 3 == torch.arange(3)))
        self.assertTrue(torch.all(labels[donors[:, 0] // 3] != labels))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_missing_malformed_or_single_style_labels_raise_without_fallback(self):
        module = TokenStyleModulator("cross_style_stats")
        tokens = torch.randn(4, 3, 8, 4)
        for labels in (
            None,
            torch.ones(4, dtype=torch.long),
            torch.tensor([11, 12]),
            torch.ones(4),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                module(tokens, labels)

    def test_donor_statistics_are_detached_as_in_upstream(self):
        tokens = torch.randn(2, 3, 8, 4, requires_grad=True)
        module = TokenStyleModulator("cross_style_stats", alpha=1.0)
        module(tokens, torch.tensor([11, 12]))[0].sum().backward()
        self.assertGreater(float(tokens.grad[0].abs().sum()), 0)
        self.assertEqual(float(tokens.grad[1].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
