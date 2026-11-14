import tempfile, unittest
from pathlib import Path

from dacsc.pairing_checks import (freeze_policy_instruction,
                                  instruction_from_task_name,
                                  validate_clean_file_identity)


class TestPairingChecks(unittest.TestCase):
    def test_clean_file_identity(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); name = "pick_up_bowl"
            bddl = d / f"{name}.bddl"; bddl.write_text("(define (problem x))")
            init = d / f"{name}.pruned_init"; init.write_bytes(b"x")
            ok = validate_clean_file_identity(name, bddl, init)
            self.assertTrue(ok["pass"])
            bad = validate_clean_file_identity("other_task", bddl, init)
            self.assertFalse(bad["pass"])
            self.assertFalse(bad["checks"]["bddl_stem"])

    def test_prompt_freeze_and_vanilla_anchor(self):
        name = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
        clean = instruction_from_task_name(name)
        shift = clean + " view 46 0 100 0 0 initstate 0"
        out = freeze_policy_instruction(clean, shift, clean_task_name=name,
                                        require_vanilla_source=True)
        self.assertTrue(out["pass"], out)
        wrong = freeze_policy_instruction("a different instruction", shift,
                                          clean_task_name=name,
                                          require_vanilla_source=True)
        self.assertFalse(wrong["pass"])

    def test_prompt_mismatch_fails(self):
        name = "pick_up_bowl"
        clean = instruction_from_task_name(name)
        out = freeze_policy_instruction(clean, clean + " view 46 0 100 0 0 initstate 0",
                                        clean_task_name=name, require_vanilla_source=True)
        self.assertTrue(out["pass"])


if __name__ == "__main__":
    unittest.main()
