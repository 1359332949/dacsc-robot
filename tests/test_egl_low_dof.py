import unittest
import numpy as np
from scripts.egl_low_dof import correct,moments,rgb_fold
from scripts.summarize_egl_low_dof import transitions
from dacsc.flow_value_trust import accept_update,applied_log_trust

class Controls(unittest.TestCase):
    def test_rgb_matches_formula_and_handles_flat_channel(self):
        x=np.zeros((2,2,3),dtype=np.uint8);x[1,:,0]=40
        src=np.array([[50.,10.,30.],[2600.,125.,1000.]])
        y=correct(x,src);self.assertEqual(y.dtype,np.uint8);self.assertTrue((y[:,:,1]==5).all());np.testing.assert_array_equal(y[:,:,0],[[20,20],[50,50]])
        np.testing.assert_array_equal(correct(x,moments(x)),x)
    def test_fold_excludes_own_task(self):
        r=[dict(factor='clean',base_task=str(i),agentview=np.full((2,3),i),eye_in_hand=np.full((2,3),i)) for i in range(10)]
        x=rgb_fold(r,r[0]);r[0]['agentview'][:]=200;np.testing.assert_equal(rgb_fold(r,r[0])['agentview'],x['agentview'])
        with self.assertRaises(ValueError):rgb_fold(r[:-1],r[0])
    def test_gate_boundary(self):
        self.assertTrue(accept_update(1,.8));self.assertFalse(accept_update(1,.8001));self.assertFalse(accept_update(0,0));self.assertEqual(applied_log_trust('flow_gated',{'accepted':False}),0.)
    def test_discordant_counts(self):
        orig={i:{'success':i<2} for i in range(6)};meth={i:{'success':i!=1} for i in orig};r=transitions(orig,meth,list(orig));self.assertEqual((r['rescued'],r['harmed'],r['net'],r['exact_p']),(4,1,3,.375))
