"""Acceptance glue regressions; fixture videos/GPU calls are NOT real evidence."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc import readonly_validation as rv
from test_corp_pipeline import metadata, Reader
from test_corp_ffn_capture import fixture_policy, images_for, REFERENCE


class CountingReader(Reader):
    calls = []
    video_backend = "fixture_not_real_video"
    video_backend_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(kwargs["dataset_path"])
        assert kwargs["use_cache"] is False

    def __getitem__(self, index):
        self.calls.append(index)
        sample = super().__getitem__(index)
        for vi, key in enumerate(rv.VIEW_TO_KEY.values()):
            sample["obs"][key] = torch.linspace(
                0.1 + vi * 0.05, 0.7, 3 * 256 * 256
            ).reshape(1, 3, 256, 256)
        return sample

    def get_video_path(self, episode, key):
        path = self.root / f"{key}_{episode}.mp4"
        if not path.exists():
            path.write_bytes(b"fixture_not_a_decodable_video")
        return path


def dataset(root, task="Task", episodes=((0, 11, 3), (9, 12, 2), (1, 31, 3))):
    path = root / f"v1.0/pretrain/atomic/{task}/date/lerobot"
    metadata(path, episodes)
    for name in ("info", "modality"):
        (path / f"meta/{name}.json").write_text("{}")
    return path


def receipt(path, stage="video", identity=None):
    data = {
        "schema": rv.SCHEMA,
        "stage": stage,
        "status": "passed",
        "code": identity or {},
        "environment": rv.environment(),
        "details": {"observations": 3},
        "artifacts": {},
    }
    names = (
        ("observations.pt", "sample_manifest.jsonl", "selection.json")
        if stage == "video"
        else ("official_fp32.log", "official_bf16.log")
    )
    for name in names:
        (path / name).write_text("fixture only")
        data["artifacts"][name] = rv.file_sha256(path / name)
    rv.write_json(path / "report.json", data)
    return data


class ReadOnlyValidationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_selection_is_bounded_native_source_episode_based_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(4):
                dataset(
                    root,
                    f"Task{i}",
                    tuple((e, 11, 8) for e in range(12)) + ((12, 41, 8),),
                )
            first = rv.select_observations(root)
            self.assertEqual(first, rv.select_observations(root))
            self.assertEqual(len(first), 27)
            self.assertEqual(
                len({(r["dataset_relpath"], r["episode"]) for r in first}), 9
            )
            self.assertTrue(
                all(r["style_id"] == 11 and r["episode"] % 10 != 9 for r in first)
            )
            self.assertTrue(all(r["frame"] in (0, 4, 7) for r in first))
            self.assertEqual(
                len({(r["dataset_relpath"], r["episode"], r["frame"]) for r in first}),
                27,
            )

    def test_empty_source_and_short_episode_never_use_targets_or_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root, episodes=((9, 11, 2), (1, 41, 2)))
            with self.assertRaisesRegex(ValueError, "no source"):
                rv.select_observations(root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset(root, episodes=((0, 11, 1),))
            self.assertEqual([r["frame"] for r in rv.select_observations(root)], [0])

    def fixture_video(self, root, out):
        dataset(root)
        modules = {
            "diffusion_policy.dataset.lerobot_dataset": SimpleNamespace(
                LerobotDataset=CountingReader
            ),
            "robocasa.utils.groot_utils.groot_dataset": SimpleNamespace(),
        }
        CountingReader.calls = []
        with patch.dict(sys.modules, modules), patch.object(
            rv, "source_module", return_value={"fixture": True}
        ):
            return rv.video_stage(SimpleNamespace(data_root=root), out)

    def test_real_reader_boundary_is_called_once_and_preserves_tail_labels_and_pixels(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "data", Path(temp) / "output"
            out.mkdir()
            details = self.fixture_video(root, out)
            self.assertEqual(len(CountingReader.calls), 3)
            self.assertEqual(details["observations"], 3)
            self.assertEqual(details["episodes"], 1)
            rows = torch.load(out / "observations.pt", weights_only=True)
            self.assertEqual(
                rows[-1]["sample"]["action_mask"].tolist(), [True] + [False] * 9
            )
            self.assertEqual(rows[0]["identity"]["layout_id"], 11)
            self.assertEqual(list(rows[0]["raw_images"]), list(rv.VIEWS))
            self.assertEqual(details["training_augmentation"], "not_instantiated")
            self.assertEqual(len(list(out.glob("raw_*.png"))), 3)

    def test_bad_preprocessing_and_mask_are_rejected_not_written_as_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "data", Path(temp) / "output"
            out.mkdir()
            self.fixture_video(root, out)
            item = torch.load(out / "observations.pt", weights_only=True)[0]
            raw = {
                "obs": {
                    rv.VIEW_TO_KEY[v]: item["raw_images"][v].unsqueeze(0)
                    for v in rv.VIEWS
                }
            }
            item["sample"]["action_mask"][0] = False
            with self.assertRaisesRegex(AssertionError, "mask"):
                rv.check_sample(item["sample"], item["identity"], raw)

    def test_cached_capture_entry_preserves_pairing_and_marks_action_not_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root, video, out = (
                Path(temp) / "data",
                Path(temp) / "video",
                Path(temp) / "capture",
            )
            video.mkdir()
            out.mkdir()
            details = self.fixture_video(root, video)
            rv.write_json(
                video / "report.json",
                {
                    "schema": rv.SCHEMA,
                    "stage": "video",
                    "status": "passed",
                    "code": {},
                    "environment": rv.environment(),
                    "details": details,
                    "artifacts": {
                        p.name: rv.file_sha256(p)
                        for p in video.iterdir()
                        if p.is_file()
                    },
                },
            )
            args = SimpleNamespace(
                video_artifacts=video,
                device="cpu",
                precision="fp32",
                checkpoint=Path("fixture.pt"),
                registers_source=REFERENCE,
                official_cuda_artifacts=None,
            )
            with patch(
                "dacsc.corp_robocasa.load_corp_robocasa",
                return_value=(fixture_policy(), {"fixture": True}),
            ), patch.object(
                rv, "verify_release_checkpoint", return_value="a" * 64
            ), patch.object(
                rv, "observe_batch", wraps=rv.observe_batch
            ) as observer:
                result = rv.capture_stage(args, out, {})
            originals = torch.load(video / "observations.pt", weights_only=True)
            for call, (start, shift) in zip(
                observer.call_args_list,
                ((0, "none"), (0, "color_jitter"), (2, "none"), (2, "color_jitter")),
                strict=True,
            ):
                for view in rv.VIEWS:
                    expected = torch.stack(
                        [
                            rv.prepare_image(x["raw_images"][view], shift)
                            for x in originals[start : start + 2]
                        ]
                    )
                    rv.exact(
                        call.args[2][view],
                        expected,
                        "eval input never training augmentation",
                    )
            self.assertEqual(result["observations"], 3)
            self.assertEqual(result["full_policy"]["status"], "not_run")
            checks = json.loads((out / "identity_checks.json").read_text())
            self.assertEqual(
                [(x["start"], x["count"], x["shift"]) for x in checks["visual"]],
                [
                    (0, 2, "none"),
                    (0, 2, "color_jitter"),
                    (2, 1, "none"),
                    (2, 1, "color_jitter"),
                ],
            )
            snapshots = torch.load(
                out / "first_batch_color_jitter.pt", weights_only=True
            )
            self.assertEqual(len(snapshots["identities"]), 2)
            self.assertEqual(
                {
                    r["shift"]
                    for r in json.loads((out / "activation_summary.json").read_text())
                },
                {"none", "color_jitter"},
            )
            self.assertTrue(checks["state_dict_unchanged"])

    def test_receipts_reject_failed_stage_missing_files_hash_tampering_and_escaping(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            original = receipt(path)
            self.assertEqual(rv.passed_artifacts(path, "video", {}), original)
            (path / "observations.pt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "digest"):
                rv.passed_artifacts(path, "video", {})
            data = receipt(path)
            for change in (
                {"status": "failed"},
                {"artifacts": {}},
                {"code": {"other_commit": True}},
                {"environment": {}},
                {"artifacts": {"../escape": "0" * 64}},
            ):
                rv.write_json(path / "report.json", {**data, **change})
                with self.assertRaises(ValueError):
                    rv.passed_artifacts(path, "video", {})

    def test_separate_isolated_envs_require_matching_runtime_stack(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            data = receipt(path)
            data["environment"]["executable"] = "/different/isolated/python"
            rv.write_json(path / "report.json", data)
            self.assertEqual(
                rv.passed_artifacts(path, "video", {})["environment"]["executable"],
                "/different/isolated/python",
            )
            data["environment"]["torch"] = "different-runtime"
            rv.write_json(path / "report.json", data)
            with self.assertRaises(ValueError):
                rv.passed_artifacts(path, "video", {})

    def test_cpu_cannot_be_reported_as_cuda_and_demo_calls_are_official(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA required"):
            rv.require_cuda("cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, out = root / "video", root / "out"
            video.mkdir()
            out.mkdir()
            receipt(video)
            digest = "a" * 64

            def official(command, **kwargs):
                self.assertEqual(Path(command[1]).name, "smoke_test.py")
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertTrue(kwargs["check"])
                kwargs["stdout"].write(
                    f"parameters=48689159\nsha256={digest}\nsmoke_ok\n"
                )

            with patch.object(
                rv, "require_cuda", return_value={"fixture": True}
            ), patch.object(
                rv, "verify_release_checkpoint", return_value=digest
            ), patch.object(
                rv.subprocess, "run", side_effect=official
            ) as runner:
                result = rv.official_cuda_stage(
                    SimpleNamespace(video_artifacts=video, device="cuda:0"), out, {}
                )
            self.assertEqual(runner.call_count, 2)
            self.assertEqual([c[-1] for c in result["commands"]], ["fp32", "bf16"])

    def test_observer_checks_all_views_tokens_rng_weights_and_hook_cleanup(self):
        model = fixture_policy()
        before = rv.state_digest(model)
        reference = rv.load_registers_reference(REFERENCE)
        with torch.inference_mode():
            frame, summary, checks = rv.observe_batch(
                model, reference, images_for(model)
            )
        self.assertTrue(checks["exact_output"])
        self.assertEqual(before, rv.state_digest(model))
        self.assertEqual(frame.ffn[0].shape[:3], (2, 3, 257))
        self.assertEqual({r["tokens"] for r in summary}, {"cls", "patch"})
        self.assertEqual({r["view"] for r in summary}, set(rv.VIEWS))
        # State digests must support scalar buffers and BF16 (not only numpy floats).
        self.assertEqual(
            rv.tensor_digest(torch.tensor(1.0)), rv.tensor_digest(torch.tensor(1.0))
        )
        self.assertIsInstance(
            rv.tensor_digest(torch.ones(2, dtype=torch.bfloat16)), str
        )

    def test_training_augmentation_is_upstream_rng_isolated_and_not_used_for_eval(self):
        rgb = torch.linspace(0.05, 0.95, 2 * 3 * 3 * 16 * 16).reshape(2, 3, 3, 16, 16)
        before = torch.get_rng_state().clone()
        checks = rv.augmentation_checks(rgb)
        self.assertTrue(checks["torchvision_exact"])
        self.assertFalse(checks["used_in_model_forward"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cli_records_failure_and_refuses_reusing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "output"
            command = ["video", "--data-root", temp, "--out", str(out)]
            with patch.object(
                rv, "code_identity", side_effect=RuntimeError("dirty tree")
            ):
                with self.assertRaisesRegex(RuntimeError, "dirty"):
                    rv.main(command)
            data = json.loads((out / "report.json").read_text())
            self.assertEqual(data["status"], "failed")
            self.assertIn("error.txt", data["artifacts"])
            original = (out / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                rv.main(command)
            self.assertEqual((out / "report.json").read_bytes(), original)

    def test_replay_plan_uses_exact_official_episode_zero_and_no_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = dataset(root)
            ep = path / "extras/episode_000000"
            np.savez(ep / "states.npz", states=np.zeros((3, 5)))
            with gzip.open(ep / "model.xml.gz", "wb") as stream:
                stream.write(b"<mujoco/>")
            args = SimpleNamespace(
                dataset=path,
                sim_python=Path(sys.executable),
                replay_output=root / "replay",
            )
            with patch.object(
                rv.subprocess, "run", side_effect=AssertionError("no execution")
            ):
                plan = rv.replay_plan_stage(args, root)
            self.assertEqual(plan["simulation_execution"], "not_run")
            self.assertIn("--use-actions", plan["commands"]["action"])
            self.assertNotIn("--use-actions", plan["commands"]["state"])
            self.assertNotIn("--first", plan["commands"]["state"])
            meta = json.loads((ep / "ep_meta.json").read_text())
            (ep / "ep_meta.json").write_text(json.dumps({**meta, "style_id": 41}))
            with self.assertRaisesRegex(ValueError, "not source"):
                rv.replay_plan_stage(args, root)

    def test_bundle_preserves_commit_original_sources_and_rejects_dirty_trees(self):
        spec = importlib.util.spec_from_file_location(
            "pack_readonly", ROOT / "scripts/pack_readonly_validation.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            (root / "docs").mkdir()
            (root / "docs/READONLY_SERVER_VALIDATION.md").write_text("fixture runbook")
            (root / "analysis_upstream.lock.json").write_bytes(
                (ROOT / "analysis_upstream.lock.json").read_bytes()
            )
            (root / "upstream.lock.json").write_text("{}")
            for command in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=f@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", "-C", str(root), *command], check=True, capture_output=True
                )
            out = Path(temp) / "package"
            with patch.object(module, "ROOT", root):
                module.pack(out, REFERENCE)
                with self.assertRaises(FileExistsError):
                    module.pack(out, REFERENCE)
                clone = Path(temp) / "clone"
                subprocess.run(
                    ["git", "clone", str(out / "code.bundle"), str(clone)],
                    check=True,
                    capture_output=True,
                )
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "-C", str(clone), "rev-parse", "HEAD"]
                    ),
                    subprocess.check_output(
                        ["git", "-C", str(root), "rev-parse", "HEAD"]
                    ),
                )
                with tarfile.open(out / "registers_analysis.tgz") as archive:
                    for rel in module.json.loads(
                        (root / "analysis_upstream.lock.json").read_text()
                    )["Registers"]["files"]:
                        self.assertEqual(
                            archive.extractfile("registers/" + rel).read(),
                            (REFERENCE / rel).read_bytes(),
                        )
                (root / "uncommitted").write_text("dirty")
                with self.assertRaisesRegex(RuntimeError, "commit"):
                    module.pack(Path(temp) / "dirty", REFERENCE)


if __name__ == "__main__":
    unittest.main()
