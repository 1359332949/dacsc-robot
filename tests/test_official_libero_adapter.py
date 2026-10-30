"""Exercise the actual official rollout loop with fixture environments and no simulator."""

from pathlib import Path
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_corp_libero_plus import PhotometricPolicy, evaluate_suite


class ReleasedLoopTests(unittest.TestCase):
    def test_official_loop_keeps_state_indices_and_policy_noise_paired_across_shifts(
        self,
    ):
        class Policy:
            config = {"image_size": 224}
            normalizer = SimpleNamespace(state_angle_center=torch.zeros(8))

            def __init__(self):
                self.noise = []
                self.observations = []

            def predict(self, obs):
                self.observations.append(obs)
                noise = torch.randn(1, 12, 7)
                self.noise.append(noise.clone())
                return noise

        class Environment:
            def __init__(self, **kwargs):
                self.steps = 0

            def seed(self, seed):
                self.seed_value = seed

            def reset(self):
                self.steps = 0

            def obs(self):
                image = np.full((256, 256, 3), [120, 60, 20], dtype=np.uint8)
                return {
                    "agentview_image": image,
                    "robot0_eye_in_hand_image": image,
                    "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                    "robot0_eef_pos": np.zeros(3),
                    "robot0_gripper_qpos": np.zeros(2),
                }

            def set_init_state(self, state):
                return self.obs()

            def step(self, action):
                self.steps += 1
                success = self.steps > 10
                return self.obs(), float(success), success, {"success": success}

            def close(self):
                pass

        task = SimpleNamespace(
            problem_folder="fixture",
            bddl_file="task.bddl",
            init_states_file="task.pruned_init",
            language="open drawer",
        )
        suite = SimpleNamespace(n_tasks=1, get_task=lambda _: task)
        benchmark = SimpleNamespace(
            get_benchmark_dict=lambda: {"libero_spatial": lambda: suite}
        )
        policy = Policy()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "init_states/fixture").mkdir(parents=True)
            torch.save(
                [np.zeros(4), np.ones(4)], root / "init_states/fixture/task.pruned_init"
            )
            for shift in ("none", "color_jitter"):
                out = root / shift
                out.mkdir()
                args = SimpleNamespace(
                    device="cpu",
                    precision="fp32",
                    tasks="0",
                    seed=7,
                    episodes_per_task=2,
                    render_gpu=0,
                    output_dir=out,
                )
                with redirect_stdout(io.StringIO()):
                    result = evaluate_suite(
                        args,
                        "libero_spatial",
                        PhotometricPolicy(policy, shift, 0.0, "cpu"),
                        benchmark,
                        lambda name: root / name,
                        Environment,
                    )
                self.assertEqual(result["successes"], 2)
                self.assertEqual(result["episodes"], 2)
        torch.testing.assert_close(policy.noise[0], policy.noise[2], rtol=0, atol=0)
        torch.testing.assert_close(policy.noise[1], policy.noise[3], rtol=0, atol=0)
        self.assertEqual(
            policy.observations[0]["images"]["agentview"].dtype, torch.uint8
        )
        self.assertEqual(
            policy.observations[2]["images"]["agentview"].dtype, torch.float32
        )
        self.assertEqual(evaluate_suite.__module__, "evaluate_libero")


if __name__ == "__main__":
    unittest.main(verbosity=2)
