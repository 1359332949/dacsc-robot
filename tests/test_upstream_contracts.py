"""Upstream contract and output ownership regressions; no training entry is run."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
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
from scripts.aggregate_sr import agg, aggregate
from test_corp_regressions import (
    FakeEnv,
    fake_reset_to,
    official_fixture,
    official_registry_modules,
    sr_payload,
    training_origin_fixture,
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

    def test_evaluation_range_must_match_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = sr_payload("A", (True,), "val")
            payload["protocol"]["training"]["data_contract"]["train_styles"] = [11, 55]
            payload["protocol"]["training"]["data_contract"]["val_styles"] = [56, 60]
            (root / "A.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                agg(root, ["A"], 1, {"val": list(range(51, 61))})

    def test_offline_eval_rejects_style_range_outside_checkpoint_contract(self):
        from scripts import eval_corp_robocasa as entry

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fixture")
            origin = training_origin_fixture()
            policy = SimpleNamespace(
                evaluation_origin=origin,
                action_horizon=10,
            )
            args = SimpleNamespace(
                model=str(checkpoint),
                data_root=root,
                max_tasks=0,
                tasks=None,
                per_task=1,
                batch_size=1,
                euler_steps=1,
                horizon=10,
                train_style_min=11,
                train_style_max=55,
                val_style_min=51,
                val_style_max=60,
                source_holdout_residue=9,
                shifts=["none"],
                out=str(root / "out.json"),
            )
            with patch.object(
                entry,
                "load_adapted_checkpoint",
                return_value=(policy, (torch.zeros(9), torch.ones(9))),
            ), self.assertRaisesRegex(ValueError, "source style range"):
                entry.run(args)

            args.train_style_max = 50
            args.val_style_min = 51
            origin["data_contract"]["val_styles"] = [56, 60]
            with patch.object(
                entry,
                "load_adapted_checkpoint",
                return_value=(policy, (torch.zeros(9), torch.ones(9))),
            ), self.assertRaisesRegex(ValueError, "val style range"):
                entry.run(args)


class OfficialCheckpointTests(unittest.TestCase):
    def test_retired_v7_checkpoint_format_is_rejected(self):
        from dacsc.corp_images import image_contract
        from dacsc.upstream import verify_upstream

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "retired_v7.pt"
            torch.save(
                {
                    "format": "corp_robocasa_v7",
                    "objective": corp.OBJECTIVE,
                    "images": image_contract(),
                    "upstream": verify_upstream(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "contract differs"):
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
