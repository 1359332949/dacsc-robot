import json, tempfile, unittest
from pathlib import Path
from dacsc.tta_protocol import (assert_target_test_confirmatory, load_split, resolve_indices,
                                 role_for_state, seeds_for_role)


def valid_split():
    return {
        "schema": "tta_split_v2", "status": "frozen", "task_suite": "s",
        "task": "t", "lighting_id": 1,
        "source_train_state_ids": [0, 1], "source_val_state_ids": [2, 3],
        "target_dev_state_ids": [4, 5], "target_test_state_ids": [6, 7],
        "screening_seeds": [1, 2], "evaluation_seeds": [3, 4],
        "confirmatory_test": False,
    }


class TtaProtocolTest(unittest.TestCase):
    def write(self, d):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, f); f.close(); return Path(f.name)

    def test_valid_and_roles(self):
        s = load_split(self.write(valid_split()))
        self.assertEqual(resolve_indices(s, "source_fit", ""), [0, 1])
        self.assertEqual(role_for_state(s, 6), "target_test")

    def test_overlap_rejected(self):
        d = valid_split(); d["target_test_state_ids"] = [1]
        with self.assertRaises(ValueError):
            load_split(self.write(d))

    def test_outside_role_rejected(self):
        s = load_split(self.write(valid_split()))
        with self.assertRaises(ValueError):
            resolve_indices(s, "target_dev", "6")

    def test_target_dev_uses_screening_seeds(self):
        s = load_split(self.write(valid_split()))
        self.assertEqual(seeds_for_role(s, "target_dev"), [1, 2])
        self.assertEqual(seeds_for_role(s, "target_test"), [3, 4])

    def test_nonconfirmatory_target_test_rejected(self):
        s = load_split(self.write(valid_split()))
        with self.assertRaises(ValueError):
            assert_target_test_confirmatory(s)

    def test_used_target_test_status(self):
        d = valid_split(); d["target_test_used_for"] = ["pilot"]
        s = load_split(self.write(d))
        self.assertEqual(s["target_test_status"], "exploratory_accessed")

    def test_empty_seed_rejected(self):
        d = valid_split(); d["evaluation_seeds"] = []
        with self.assertRaises(ValueError):
            load_split(self.write(d))


if __name__ == "__main__":
    unittest.main()
