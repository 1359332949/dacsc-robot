import unittest
import numpy as np
from dacsc.action_metrics import action_error, execution_clip, rotation_geodesic

class TestActionMetrics(unittest.TestCase):
    def test_identity(self):
        a=np.zeros((12,7)); a[:,6]=0.3
        e=action_error(a,a)
        self.assertEqual(e['xyz_mean'],0.0); self.assertEqual(e['rotation_geodesic_mean'],0.0); self.assertEqual(e['gripper_abs_mean'],0.0)

    def test_clip_masks_saturated_difference(self):
        a=np.zeros((12,7)); b=np.zeros((12,7)); a[:,0]=2.0; b[:,0]=3.0
        raw=action_error(a,b,executed=False); ex=action_error(a,b,executed=True)
        self.assertGreater(raw['xyz_mean'],0.0); self.assertEqual(ex['xyz_mean'],0.0)

    def test_rotation_geodesic(self):
        self.assertAlmostEqual(rotation_geodesic(np.zeros(3),np.array([0,0,np.pi/2])),np.pi/2,places=6)

    def test_clip_all_dims(self):
        a=np.full((1,7),2.0); np.testing.assert_allclose(execution_clip(a),np.ones((1,7)))

if __name__=='__main__': unittest.main()
