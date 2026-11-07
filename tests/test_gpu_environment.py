"""Static checks for the portable GPU environment contract."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_gpu_wheelhouse import (
    DirectoryLock,
    chunk_path,
    chunk_specs,
    compatible,
    compatible_artifact,
    source_archive,
    download,
    parse_lock,
)


class RangeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self):
        self.server.head_requests += 1
        self._send_headers(200, len(self.server.payload), total=None)

    def do_GET(self):
        self.server.get_requests += 1
        payload = self.server.payload
        range_header = self.headers.get("Range")
        if not range_header:
            self._send_headers(200, len(payload), total=None)
            self.wfile.write(payload)
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
        if not match:
            self.send_error(416)
            return
        start, end = (int(value) for value in match.groups())
        if start < 0 or start > end or end >= len(payload):
            self.send_error(416)
            return
        body = payload[start : end + 1]
        self._send_headers(206, len(body), total=len(payload), start=start, end=end)
        self.wfile.write(body)

    def _send_headers(
        self,
        status: int,
        length: int,
        *,
        total: int | None,
        start: int | None = None,
        end: int | None = None,
    ):
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()

    def log_message(self, *_args):
        pass


def lock_pins(name: str) -> dict[str, str]:
    pins = {}
    for raw in (ROOT / "environment" / name).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--") or line.startswith(" "):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)==([^ ;\\]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


class GPUEnvironmentContractTests(unittest.TestCase):
    def test_closed_loop_episode_defaults_match_v10_protocol(self):
        config = json.loads(
            (ROOT / "configs/v10_atomic_target_pilot.json").read_text()
        )
        self.assertEqual(config["evaluation"]["internal_method_episodes"], 20)
        self.assertEqual(config["evaluation"]["target_episodes_external"], 20)
        for name in (
            "run_corp_sr_split.sh",
            "run_corp_sr_tasks.sh",
            "run_v10_target_sr.sh",
        ):
            self.assertIn(
                'EPISODES="${EPISODES:-20}"',
                (ROOT / "scripts" / name).read_text(),
            )
        self.assertIn(
            "closed-loop rollout episodes per task",
            (ROOT / "scripts/eval_corp_robocasa_sr.py").read_text(),
        )
        self.assertIn(
            "not closed-loop episodes",
            (ROOT / "scripts/train_corp_robocasa.py").read_text(),
        )

    def test_locks_isolate_upstream_numpy_requirements(self):
        corp = lock_pins("corp.lock.txt")
        robocasa = lock_pins("robocasa.lock.txt")
        self.assertEqual(corp["numpy"], "1.26.4")
        self.assertEqual(robocasa["numpy"], "2.2.5")
        for pins in (corp, robocasa):
            self.assertEqual(pins["torch"].split("+")[0], "2.11.0")
            self.assertEqual(pins["torchvision"].split("+")[0], "0.26.0")
            self.assertIn("+cu128", pins["torch"])
            self.assertIn("+cu128", pins["torchvision"])
            self.assertNotIn("robosuite", pins)

    def test_robocasa_lock_includes_robosuite_runtime_requirements(self):
        robocasa = lock_pins("robocasa.lock.txt")
        robosuite_requires = {
            "numpy",
            "numba",
            "scipy",
            "mujoco",
            "qpsolvers",
            "pillow",
            "opencv-python",
            "pynput",
            "termcolor",
            "transformers",
            "pytest",
            "tqdm",
        }
        self.assertTrue(
            robosuite_requires.issubset(robocasa),
            robosuite_requires - robocasa.keys(),
        )
        self.assertEqual(robocasa["evdev"], "2.0.0")
        self.assertNotIn("mink", robocasa)

    def test_robocasa_lock_includes_robocasa_declared_runtime_requirements(self):
        robocasa = lock_pins("robocasa.lock.txt")
        self.assertIn("pygame", robocasa)
        self.assertIn("hidapi", robocasa)
        self.assertIn("datasets", robocasa)
        self.assertIn("jsonlines", robocasa)
        self.assertNotIn("lerobot", robocasa)
        self.assertNotIn("tianshou", robocasa)
        lerobot = lock_pins("lerobot.lock.txt")
        self.assertEqual(lerobot, {"lerobot": "0.3.3"})

    def test_corp_lock_includes_register_analysis_requirements(self):
        corp = lock_pins("corp.lock.txt")
        self.assertEqual(corp["matplotlib"], "3.11.2")

    def test_every_locked_requirement_has_integrity_hashes(self):
        for name in ("corp.lock.txt", "robocasa.lock.txt"):
            lines = (ROOT / "environment" / name).read_text().splitlines()
            requirements = 0
            for index, line in enumerate(lines):
                if "==" not in line or line[:1].isspace():
                    continue
                requirements += 1
                block = []
                for following in lines[index + 1 :]:
                    if following and not following[:1].isspace():
                        break
                    block.append(following)
                self.assertTrue(
                    any("--hash=sha256:" in item for item in block),
                    f"{name}: {line}",
                )
            self.assertGreater(requirements, 0)

    def test_robosuite_revision_and_fallback_scope_are_explicit(self):
        lock = json.loads((ROOT / "environment/upstream.lock.json").read_text())
        robosuite = lock["robosuite"]
        self.assertRegex(robosuite["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(robosuite["fallback_wheel"], "robosuite==1.5.2")
        self.assertIn("before MuJoCo replay", robosuite["fallback_scope"])

    def test_shell_scripts_parse_and_verifier_compiles(self):
        for name in (
            "build_gpu_env_locks.sh",
            "download_gpu_wheelhouse.sh",
            "install_gpu_system_deps.sh",
            "bootstrap_gpu_envs.sh",
            "verify_gpu_envs.sh",
            "verify_gpu_envs_local_docker.sh",
        ):
            path = ROOT / "scripts" / name
            subprocess.run(["bash", "-n", str(path)], check=True)
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "py_compile",
                str(ROOT / "scripts/verify_gpu_envs.py"),
            ],
            check=True,
        )
        system_deps = (ROOT / "scripts/install_gpu_system_deps.sh").read_text()
        self.assertIn("build-essential", system_deps)
        self.assertIn("libgl1", system_deps)
        self.assertIn("libegl1", system_deps)
        bootstrap = (ROOT / "scripts/bootstrap_gpu_envs.sh").read_text()
        self.assertIn("venv --seed --python 3.12", bootstrap)

    def test_wheel_filter_selects_linux_x86_64_python312(self):
        self.assertTrue(
            compatible("torch-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl")
        )
        self.assertTrue(compatible("huggingface_hub-0.36.2-py3-none-any.whl"))
        self.assertTrue(
            compatible("hf_xet-1.8.0-cp38-abi3-manylinux2014_x86_64.whl")
        )
        self.assertFalse(
            compatible("torch-2.11.0-cp312-cp312-macosx_14_0_arm64.whl")
        )
        self.assertFalse(compatible("torch-2.11.0-cp312-cp312-win_amd64.whl"))
        self.assertFalse(
            compatible("numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl")
        )
        self.assertTrue(source_archive("antlr4-python3-runtime-4.9.3.tar.gz"))
        self.assertTrue(compatible_artifact("package-1.0.zip"))

    def test_lock_parser_preserves_exact_versions_and_hashes(self):
        entries = parse_lock(ROOT / "environment/corp.lock.txt")
        by_name = {name: (version, hashes) for name, version, hashes in entries}
        self.assertEqual(by_name["numpy"][0], "1.26.4")
        self.assertEqual(by_name["torch"][0], "2.11.0+cu128")
        self.assertTrue(by_name["torch"][1])


class WheelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeRequestHandler)
        cls.server.payload = b""
        cls.server.head_requests = 0
        cls.server.get_requests = 0
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.server.head_requests = 0
        self.server.get_requests = 0
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.out = self.root / "wheelhouse"
        self.out.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/pkg.whl"

    def digest(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def test_parallel_ranges_are_assembled_and_hashed(self):
        self.server.payload = bytes(range(256)) * 1000
        target = self.out / "pkg.whl"
        download(
            self.url,
            self.digest(self.server.payload),
            self.out,
            "curl",
            chunk_size=32_000,
            range_threshold=1,
            workers=4,
        )
        self.assertEqual(target.read_bytes(), self.server.payload)
        self.assertFalse((self.out / ".pkg.whl.part").exists())
        self.assertFalse((self.out / ".pkg.whl.chunks").exists())
        self.assertGreater(self.server.get_requests, 2)

    def test_completed_chunks_are_reused_after_interruption(self):
        self.server.payload = bytes(range(251)) * 800
        target = self.out / "pkg.whl"
        specs = chunk_specs(len(self.server.payload), 40_000)
        scratch = self.out / ".pkg.whl.chunks"
        scratch.mkdir()
        first_start, first_end = specs[0]
        chunk_path(scratch, first_start, first_end).write_bytes(
            self.server.payload[first_start : first_end + 1]
        )
        download(
            self.url,
            self.digest(self.server.payload),
            self.out,
            "curl",
            chunk_size=40_000,
            range_threshold=1,
            workers=3,
        )
        self.assertEqual(target.read_bytes(), self.server.payload)
        self.assertEqual(self.server.get_requests, len(specs))

    def test_hash_mismatch_is_rejected_and_chunks_are_removed(self):
        self.server.payload = b"x" * 100_000
        target = self.out / "pkg.whl"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            download(
                self.url,
                self.digest(b"different content"),
                self.out,
                "curl",
                chunk_size=20_000,
                range_threshold=1,
                workers=3,
            )
        self.assertFalse(target.exists())
        self.assertFalse((self.out / ".pkg.whl.chunks").exists())

    def test_wheelhouse_ownership_lock_rejects_duplicate_writer(self):
        with DirectoryLock(self.out):
            with self.assertRaisesRegex(RuntimeError, "another wheelhouse download"):
                with DirectoryLock(self.out):
                    self.fail("duplicate writer acquired the wheelhouse")


if __name__ == "__main__":
    unittest.main()
