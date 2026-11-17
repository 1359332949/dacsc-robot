import importlib
import unittest


class TestImportSmoke(unittest.TestCase):
    def test_tta_modules_import(self):
        for name in (
            "dacsc.state_pairing",
            "dacsc.observation_render",
            "dacsc.libero_noise",
            "dacsc.action_guard",
            "dacsc.target_only_mean",
        ):
            with self.subTest(name=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
