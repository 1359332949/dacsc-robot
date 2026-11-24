import json, unittest
from pathlib import Path
import torch
from dacsc.failure_monitor import CausalFailureMonitor
from dacsc.monitor_ensemble import MonitorEnsemble

ROOT = Path(__file__).resolve().parents[1]


class MonitorArtifactTest(unittest.TestCase):
    def test_artifact_loads_with_checkpoint_structure(self):
        artifact = json.loads((ROOT / "docs/data/monitor_artifact_v1.json").read_text())
        ck = torch.load(ROOT / artifact["checkpoint"], map_location="cpu")
        self.assertFalse(artifact["use_visual"])
        self.assertFalse(ck["use_visual"])
        model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                     use_visual=ck["use_visual"])
        model.load_state_dict(ck["model"])
        self.assertIsNone(model.visual_proj)
        self.assertGreater(artifact["persistence"], 0)
        self.assertGreater(artifact["stride"], 0)

    def test_v2_ensemble_loads_and_has_calibration(self):
        a = json.loads((ROOT / "docs/data/monitor_artifact_v2.json").read_text())
        self.assertEqual(a["schema"], "monitor_artifact_v2")
        self.assertGreaterEqual(len(a["members"]), 2)
        self.assertIsNotNone(a["threshold"])
        ens = MonitorEnsemble(ROOT / "docs/data/monitor_artifact_v2.json")
        self.assertGreaterEqual(len(ens.members), 2)

    def test_artifact_roles_and_hash_fields(self):
        a = json.loads((ROOT / "docs/data/monitor_artifact_v1.json").read_text())
        self.assertEqual(a["schema"], "monitor_artifact_v1")
        self.assertEqual(a["role"], "target_dev")
        self.assertEqual(len(a["checkpoint_sha256"]), 64)
        self.assertEqual(len(a["split_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
