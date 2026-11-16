import importlib.util
import unittest
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts" / "v12_aug_scores.py"
spec = importlib.util.spec_from_file_location("v12_aug", path)
v12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v12)


class TestV12Aug(unittest.TestCase):
    def test_photo_shape(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        for kind in ("brightness", "contrast", "saturation"):
            out = v12.photo_aug(img, kind, 1.2)
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_geo_shape(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        for kind, value in (("translate", 4), ("crop", 0.95), ("zoom", 1.05)):
            out = v12.geo_aug(img, kind, value)
            self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
