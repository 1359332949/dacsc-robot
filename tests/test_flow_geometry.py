import unittest
import numpy as np
from dacsc.flow_geometry import geometry,auc

class GeometryTests(unittest.TestCase):
    def test_constant_field(self):
        d=geometry(np.ones((10,12,7)))
        self.assertEqual(d['velocity_variance'],0);self.assertAlmostEqual(d['turning'],0)
        self.assertAlmostEqual(d['path_excess'],0)
    def test_scaling_does_not_improve_relative_straightness(self):
        v=np.arange(10*12*7).reshape(10,12,7)+1
        a=geometry(v);b=geometry(v*.01)
        self.assertAlmostEqual(a['relative_variance'],b['relative_variance'])
        self.assertAlmostEqual(a['velocity_variance']*.0001,b['velocity_variance'])
    def test_zero_is_not_proof_of_straight_motion(self):
        d=geometry(np.zeros((10,12,7)))
        self.assertTrue(d['zero_energy']);self.assertIsNone(d['turning']);self.assertIsNone(d['path_excess'])
    def test_auc_ties_and_direction(self):
        self.assertEqual(auc([1,0],[2,1]),1)
        self.assertEqual(auc([1,0],[1,1]),.5)
        self.assertIsNone(auc([1,1],[1,2]))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):geometry(np.full((10,12,7),np.nan))

if __name__=='__main__':unittest.main()
