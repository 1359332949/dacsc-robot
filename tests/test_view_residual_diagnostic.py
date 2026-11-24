import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TestViewResidualDiagnostic(unittest.TestCase):
    def test_feature_rows_align_states(self):
        with tempfile.TemporaryDirectory() as td:
            cap = Path(td) / "cap"; cap.mkdir()
            # non-monotonic feature_row mapping: index 2 -> row 2, index 3 -> row 1
            records = [
                {"index": 0, "compare_pass": True, "feature_row": 0},
                {"index": 1, "compare_pass": False, "feature_row": None},
                {"index": 2, "compare_pass": True, "feature_row": 2},
                {"index": 3, "compare_pass": True, "feature_row": 1},
            ]
            # two patches, three feature dims
            clean = np.zeros((3, 2, 3), dtype=np.float32)
            shift = np.array([
                [[0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [2, 2, 2]],
                [[4, 4, 4], [4, 4, 4]],
            ], dtype=np.float32)
            np.savez_compressed(cap / "features.npz", clean_agentview=clean,
                                shift_agentview=shift, state8=np.zeros((3, 4), dtype=np.float32))
            manifest = {
                "records": records, "policy_views": ["agentview"],
                "dev_indices": [0, 2, 3], "heldout_indices": [],
            }
            (cap / "manifest.json").write_text(json.dumps(manifest))
            out = Path(td) / "out.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/view_residual_diagnostic.py"),
                            "--capture", str(cap), "--out", str(out),
                            "--fit-start", "0", "--fit-count", "2"],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            data = json.loads(out.read_text())
            # Valid rows in state order [0,2,3] are feature rows [0,2,1],
            # so residual norms are [0, 4*sqrt(3), 2*sqrt(3)].
            self.assertEqual(data["fit_ids"], [0, 2])
            self.assertEqual(data["analysis_ids"], [0, 2, 3])
            self.assertAlmostEqual(data["views"]["agentview"]["residual_norm_mean"],
                                   float(np.mean([0.0, 4.0 * np.sqrt(3), 2.0 * np.sqrt(3)])), places=4)


if __name__ == "__main__":
    unittest.main()
