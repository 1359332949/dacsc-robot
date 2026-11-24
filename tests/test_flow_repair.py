import unittest
import torch
from dacsc.flow_repair import energy,project_,random_null

class RepairTests(unittest.TestCase):
    def test_relative_scale_invariance_and_constant_zero(self):
        v=torch.randn(10,4,12,7)
        self.assertAlmostEqual(float(energy(v)[0]),float(energy(v*.1)[0]),places=6)
        self.assertEqual(float(energy(torch.zeros_like(v))[0]),0)
    def test_projection_and_identity_scalar(self):
        d=torch.tensor([[1.,-10.,-.5]]);s=torch.tensor(2.)
        project_(d,s)
        self.assertEqual(float(d.max()),0);self.assertGreaterEqual(float(d.min()),-8)
        self.assertEqual(float(s.exp()),1)
    def test_null_matches_norm_bounds_and_scalar(self):
        d=-torch.rand(1,256)*7;d[0,0]=0;s=torch.tensor(-.7)
        n,ns=random_null(d,s,torch.Generator().manual_seed(1))
        self.assertAlmostEqual(float(n.norm()),float(d.norm()),places=4)
        self.assertLessEqual(float(n.max()),0);self.assertGreaterEqual(float(n.min()),-8)
        self.assertEqual(float(ns),float(s))
    def test_view_null_exactly_degenerate(self):
        d=torch.zeros(1,256);s=torch.tensor(-1.)
        n,ns=random_null(d,s,torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(n,d));self.assertTrue(torch.equal(ns,s))
    def test_anchor_has_no_external_reference(self):
        v=torch.randn(10,4,12,7)
        with self.assertRaises(ValueError):energy(v,'anchored')
        _,e=energy(v)
        a,_=energy(v,'anchored',e['mean_speed_norm'])
        b,_=energy(v)
        self.assertAlmostEqual(float(a),float(b),places=6)
    def test_uniform_logit_shift_is_not_view_trust(self):
        x=torch.randn(4,16)
        self.assertTrue(torch.allclose(x.softmax(-1),(x-3).softmax(-1),atol=1e-6))


class PostNormValueTests(unittest.TestCase):
    def test_value_gate_changes_only_agent_contribution_and_has_gradient(self):
        from dataclasses import dataclass
        from types import SimpleNamespace
        from dacsc.flow_repair_kv import integrate
        @dataclass
        class Cache:
            key: torch.Tensor
            value: torch.Tensor
        class Gen:
            num_sample_steps=2
            def prepare_prefix(self,rep):
                return [Cache(torch.tensor([[[[9.],[11.]]]]),torch.tensor([[[[2.],[5.]]]]))]
            def __call__(self,x,t,prefix):
                self.last=prefix[0]
                return torch.ones_like(x)*prefix[0].value.sum()
        gen=Gen();policy=SimpleNamespace(action_generator=gen)
        s=torch.tensor(0.,requires_grad=True)
        rep=SimpleNamespace(agentview_token_count=1,view_log_trust=s)
        x,_=integrate(policy,rep,torch.zeros(1,1,1))
        self.assertEqual(float(x.detach()),7.)
        x.sum().backward();self.assertEqual(float(s.grad),2.)
        rep.view_log_trust=torch.tensor(float(torch.log(torch.tensor(.5))))
        y,_=integrate(policy,rep,torch.zeros(1,1,1))
        self.assertEqual(float(y),6.)
        self.assertEqual(float(gen.last.value[0,0,1,0]),5.)
        self.assertTrue(torch.equal(gen.last.key,torch.tensor([[[[9.],[11.]]]])))

if __name__=='__main__':unittest.main()
