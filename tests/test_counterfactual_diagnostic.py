import unittest
import numpy as np
from dacsc.counterfactual_diagnostic import require_current_pair,command_difference,separation,interaction

class CounterfactualTests(unittest.TestCase):
    def test_current_proprio_not_initial_anchor(self):
        a={'state8':np.arange(24).reshape(3,8).astype(float),'boundary_step':np.array([0,12,24]),'probe_anchor_state8':np.zeros((3,8))}
        q=require_current_pair(a,a,2)
        np.testing.assert_array_equal(q,a['state8'][2]);self.assertFalse(np.array_equal(q,a['probe_anchor_state8'][2]))
        q[0]=-1;self.assertNotEqual(a['state8'][2,0],-1)
    def test_reject_pair_state_mismatch(self):
        a={'state8':np.zeros((2,8)),'boundary_step':np.array([0,12])};b={k:v.copy() for k,v in a.items()};b['state8'][1,0]=.01
        with self.assertRaises(ValueError):require_current_pair(a,b,1)
    def test_clip_all_dimensions(self):
        a=np.ones((12,7))*3;b=np.ones((12,7))*2
        m=command_difference(a,b);self.assertEqual(m['postclip']['all_rms'],0);self.assertEqual(m['preclip']['all_rms'],1)
    def test_constant_representation_not_perfect_separation(self):
        m=separation(np.zeros(8),np.zeros(8),np.zeros(8),np.zeros(8))
        self.assertFalse(m['clean_distinguishable_numeric']);self.assertFalse(m['shift_distinguishable_numeric']);self.assertIsNone(m['shift_over_clean'])
    def test_nonadditivity_zero_only_for_additive_response(self):
        x=np.zeros(7);a=np.ones(7);b=np.ones(7)*2
        self.assertEqual(interaction(x,a,b,a+b),0)
        self.assertGreater(interaction(x,a,b,a+b+1),0)
    def test_shared_z0_does_not_imply_shared_later_x(self):
        z=np.ones(7);x1=z.copy();x2=z.copy()
        for _ in range(5):x1=x1+.1*(x1+1);x2=x2+.1*(x2+2)
        self.assertFalse(np.array_equal(x1,x2))
if __name__=='__main__':unittest.main()
