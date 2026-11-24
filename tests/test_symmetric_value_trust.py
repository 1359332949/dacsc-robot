from dataclasses import dataclass
import unittest
import numpy as np
import torch
from dacsc.symmetric_value_trust import gate_values,project_two,perturb,solve
from dacsc.static_visual_v2 import sensor_transform

@dataclass
class Cache:
    key:torch.Tensor
    value:torch.Tensor

class SymmetryTests(unittest.TestCase):
    def test_projection_swaps_and_nearest(self):
        for raw in [[.2,.1],[-.2,-.9],[-10,-9],[.2,-.1],[-.1,-.1]]:
            u=torch.tensor(raw,dtype=torch.float32); p=project_two(u);self.assertTrue(torch.equal(p.flip(0),project_two(u.flip(0))));self.assertEqual(float(p.max()),0); self.assertGreaterEqual(float(p.min()),-4.60518)
            if raw[0]!=raw[1]:
                y=u.clamp(-4.605170186,0);self.assertAlmostEqual(float((p-u).square().sum()),min(float((u-y*u.new_tensor(m)).square().sum()) for m in [[1,0],[0,1]]))
    def test_value_order_and_gradient(self):
        v=torch.tensor([[[[2.],[3.],[5.]]]]);c=Cache(v.clone(),v.clone());s=torch.tensor([0.,np.log(.5)],dtype=torch.float32,requires_grad=True);out=gate_values([c],s,1)[0];torch.testing.assert_close(out.value.flatten(),torch.tensor([2.,1.5,5.]));self.assertIs(out.key,c.key);out.value.sum().backward();torch.testing.assert_close(s.grad,torch.tensor([2.,1.5]));torch.testing.assert_close(gate_values([c],s.detach()*0,1)[0].value,v)
    def test_matches_original_agent_only_integration(self):
        from types import SimpleNamespace
        from dacsc.flow_repair_kv import integrate
        v=torch.arange(1.,7.).reshape(1,1,3,2);cache=[Cache(v.clone(),v.clone())]
        class Gen:
            num_sample_steps=10
            def prepare_prefix(self,rep):return cache
            def __call__(self,x,t,c):return x*.1+t[:,None,None]+c[0].value.sum()
        gen=Gen();p=SimpleNamespace(action_generator=gen);rep=SimpleNamespace(agentview_token_count=1,view_log_trust=torch.tensor(-1.));z=torch.zeros(1,12,7)
        a,v1=integrate(p,rep,z);b,v2=solve(gen,cache,z,torch.tensor([-1.,0.]),1)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(v1,v2))

    def test_same_kernel_only_destination_changes(self):
        class W:
            @staticmethod
            def motion_blur(image,severity):return np.asarray(image)+np.random.randint(1,10)
        w=W();w.gaussian_blur=w.zoom_blur=w.fog=w.glass_blur=w.motion_blur
        a=np.zeros((8,8,3),np.uint8);b=a+32;o={'agentview_image':a,'robot0_eye_in_hand_image':b};r=perturb(o,1,65,12,w,True)
        swapped=sensor_transform(dict(o,agentview_image=b),1,65,12,w)['agentview_image'];np.testing.assert_array_equal(r['robot0_eye_in_hand_image'],swapped);np.testing.assert_array_equal(o['robot0_eye_in_hand_image'],a+32);self.assertIs(r['agentview_image'],a)
        reg=perturb(o,1,65,12,w,False);self.assertIs(reg['robot0_eye_in_hand_image'],b);np.testing.assert_array_equal(reg['agentview_image']+32,r['robot0_eye_in_hand_image'])
