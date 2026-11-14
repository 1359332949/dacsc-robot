import json, tempfile, unittest
from pathlib import Path

from dacsc.task_pairing import (
    canonical_base_name, canonical_instruction, classification_entries,
    load_task_map, parse_objects, resolve_task_name, task_fingerprint,
    task_pair_compatible, validate_classification,
)

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
    def test_real_camera_suffix(self):
        base = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
        camera = base + "_view_46_0_100_0_0_initstate_0"
        self.assertEqual(canonical_base_name(camera), base)
        self.assertEqual(canonical_instruction(camera), base)

    def test_space_separated_language_suffix(self):
        text = ("pick up the black bowl on the cookie box and place it on the plate "
                "view 46 0 100 0 0 initstate 0")
        self.assertEqual(
            canonical_instruction(text),
            "pick up the black bowl on the cookie box and place it on the plate")

    def test_no_broad_digit_strip(self):
        self.assertEqual(canonical_base_name("task_2024"), "task_2024")

    def test_parse_objects_keeps_types(self):
        objs = parse_objects(BDDL)
        self.assertIn({"name": "black_bowl", "type": "bowl"}, objs)
        self.assertIn({"name": "table", "type": "table"}, objs)
        # same name, different type must be distinguishable
        alt = parse_objects("(:objects black_bowl - plate)")
        self.assertNotEqual(objs, alt)

    def test_grouped_object_declaration(self):
        grouped = ("(:objects akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl "
                   "cookies_1 - cookies plate_1 - plate)")
        objs = parse_objects(grouped)
        self.assertIn({"name": "akita_black_bowl_1", "type": "akita_black_bowl"}, objs)
        self.assertIn({"name": "akita_black_bowl_2", "type": "akita_black_bowl"}, objs)

    def test_pair_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bddl"; p.write_text(BDDL)
            fp = task_fingerprint(type("T", (), {"name": "task", "language": "put bowl"})(), p)
        self.assertTrue(task_pair_compatible(fp, fp)["pass"])
        other = dict(fp); other["canonical_language"] = "other"
        self.assertFalse(task_pair_compatible(fp, other)["pass"])

    def test_task_map_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "map.json"
            p.write_text(json.dumps({"libero_spatial": ["a_view_1_initstate_0", "b"]}))
            m = load_task_map(p)
        self.assertEqual(resolve_task_name(m, "libero_spatial", 0)[1], "a")
        with self.assertRaises(IndexError):
            resolve_task_name(m, "libero_spatial", 2)

    def test_official_python_task_map(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "libero_suite_task_map.py"
            p.write_text("libero_task_map = {'libero_spatial': ['a_view_1_initstate_0', 'b'],"
                         " 'libero_object': ['c']}\n")
            m = load_task_map(p)
        self.assertIn("libero_spatial", m)
        self.assertEqual(resolve_task_name(m, "libero_spatial", 0)[1], "a")

    def test_classification_suite_scoped(self):
        data = {"libero_spatial": [{"task_id": 705, "category": "Camera Viewpoints"}],
                "libero_object": [{"task_id": 705, "category": "Lighting"}]}
        e = classification_entries(data, "libero_spatial", 705)
        self.assertEqual(len(e), 1)
        self.assertTrue(validate_classification(e, {"camera viewpoints"})["pass"])
        self.assertFalse(validate_classification(e, {"light conditions"})["pass"])
        self.assertEqual(classification_entries(data, "libero_spatial", 999), [])

    def test_classification_full_name_required(self):
        e = [{"task_id": 705, "name": "task_view_46_initstate_0", "category": "Camera Viewpoints"}]
        self.assertTrue(validate_classification(e, {"camera viewpoints"},
                                                expected_name="task_view_46_initstate_0")["pass"])
        self.assertFalse(validate_classification(e, {"camera viewpoints"},
                                                 expected_name="task_view_47_initstate_0")["pass"])
        self.assertFalse(validate_classification([{"task_id": 705, "category": "Camera Viewpoints"}],
                                                 {"camera viewpoints"},
                                                 expected_name="task_view_46_initstate_0")["pass"])


if __name__ == "__main__":
    unittest.main()
