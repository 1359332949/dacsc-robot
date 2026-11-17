import unittest

from dacsc.pilot_validation import (
    parse_init_indices, validate_executed_k, validate_init_indices, validate_methods,
)


class TestPilotValidation(unittest.TestCase):
    def test_methods_reject_unknown_and_duplicate(self):
        self.assertEqual(validate_methods("frozen,first_observation"),
                         ["frozen", "first_observation"])
        with self.assertRaises(ValueError):
            validate_methods("frozen,typo")
        with self.assertRaises(ValueError):
            validate_methods("frozen,frozen")

    def test_init_indices_unique_and_bounded(self):
        self.assertEqual(parse_init_indices("5,7"), [5, 7])
        with self.assertRaises(ValueError):
            parse_init_indices("5,5")
        with self.assertRaises(ValueError):
            parse_init_indices("")

    def test_executed_k_bounds(self):
        self.assertEqual(validate_executed_k(12, 12), 12)
        for value in (0, 13, -1):
            with self.assertRaises(ValueError):
                validate_executed_k(value, 12)

    def test_init_indices_must_be_heldout_and_disjoint(self):
        validate_init_indices([5, 7], 50, {5, 7, 9}, [0, 1, 3])
        with self.assertRaises(ValueError):
            validate_init_indices([5, 5], 50, {5}, [])
        with self.assertRaises(ValueError):
            validate_init_indices([50], 50, {50}, [])
        with self.assertRaises(ValueError):
            validate_init_indices([5], 50, {5}, [5])


if __name__ == "__main__":
    unittest.main()
