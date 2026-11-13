import tempfile, unittest
from pathlib import Path

from dacsc.task_pairing import canonical_base_name, parse_objects, task_pair_compatible


BDDL = """(define (problem test)
  (:objects
    black_bowl - bowl
    table - table
  )
  (:init (on black_bowl table))
  (:goal (and (in black_bowl table)))
)
"""


class TestTaskPairing(unittest.TestCase):
    def test_canonical_suffix(self):
        self.assertEqual(canonical_base_name("picking_up_black_bowl_view_1"),
                         canonical_base_name("picking_up_black_bowl"))
        self.assertEqual(canonical_base_name("task_background_2"), "task")

    def test_parse_objects(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bddl"; p.write_text(BDDL)
        self.assertEqual(parse_objects(BDDL), ["black_bowl", "table"])

    def test_pair_compatible(self):
        from dacsc.task_pairing import task_fingerprint
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bddl"; p.write_text(BDDL)
            fp = task_fingerprint(type("T", (), {"name": "task", "language": "put bowl"})(), p)
        self.assertTrue(task_pair_compatible(fp, fp)["pass"])
        other = dict(fp); other["language"] = "other"
        self.assertFalse(task_pair_compatible(fp, other)["pass"])


if __name__ == "__main__":
    unittest.main()
