"""Analytical fixtures only; never load a checkpoint, simulator, or V2 examples."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from dacsc.fsfc import (Config, EpisodeGate, Prior, accepts, fit_priors, frobenius,
                        propose, source_ids, transform)
from scripts.fsfc_unified_probe_v1 import perturbed_images


class FSFCTest(unittest.TestCase):
    def setUp(self):
        self.p=torch.tensor([[[4.,0.],[4.,.1]],[[0.,0.],[.2,.1]]]);m=torch.zeros(2,2)
        self.prior=Prior(m,torch.ones_like(m),torch.ones_like(m))
        self.train=torch.arange(4.).reshape(4,1);self.val=self.train+10
        self.curve=lambda q,n:q.square().mean()

    def test_jacobian_does_not_cancel_opposite_actions(self):
        def f(p,z):
            a=z.item()*p[0,:,0].sum()+3*p[1,0,1]
            return torch.stack([a,-a])
        j=frobenius(f,self.p,torch.tensor([[1.],[3.]]));want=torch.tensor([[4.,0.],[0.,18.**.5]])
        torch.testing.assert_close(j,want)

    def test_fold_by_task_not_filename(self):
        cases=[dict(case_id=str(i),role='dev_clean',factor='clean',base_task=str(i)) for i in range(10)]
        cases+= [dict(cases[0],case_id='other-own'),dict(cases[3],role='target_blind',case_id='protected')]
        self.assertEqual(source_ids(cases,'0'),list('123456789'))
        with self.assertRaises(ValueError):source_ids(cases+[cases[1]],'0')

    def test_prior_hand_computation(self):
        c=torch.stack([self.p,self.p+1]);d=torch.ones(2,2,2);i=d.clone();i[:,:,0]=10
        r=fit_priors(c,d,i);self.assertEqual(set(r),{'jacobian','uniform','base_std'})
        for v in r.values():self.assertAlmostEqual(float(v.rho.mean()),1.,places=5)
        self.assertTrue((r['jacobian'].rho[:,0]<r['jacobian'].rho[:,1]).all())
        self.assertTrue(torch.equal(r['uniform'].rho,torch.ones(2,2)))
        self.assertTrue((fit_priors(c,d*0,i)['jacobian'].rho==0).all())

    def test_identity_dead_zone_and_accepted_coordinate(self):
        g,i=propose(self.p,self.prior,self.train,self.val,self.curve,Config(hard_mask=True));self.assertTrue(i['accepted']);self.assertGreater(float(g[0,0]),0.)
        self.assertTrue(((g>=0)&(g<=1)).all());q=transform(self.p,g,self.prior)
        self.assertTrue(torch.equal(q[1],self.p[1]));self.assertTrue(torch.equal(q[:,:,1],self.p[:,:,1]))
        untouched,info=propose(self.p*0,self.prior,self.train,self.val,lambda *a:self.fail('forward inside dead zone'),Config(hard_mask=True))
        self.assertEqual(info['reason'],'dead_zone');self.assertFalse(untouched.any())

    def test_patch_vs_image_scales(self):
        grid=torch.tensor([-100.,100.]).reshape(1,1,2,1)
        c=(grid+torch.arange(9.).reshape(9,1,1,1)).expand(-1,2,-1,3)
        d=torch.ones(9,2,3); prior=fit_priors(c,d,d)['jacobian']
        self.assertTrue(torch.all(prior.patch_scale>99))
        self.assertTrue(torch.allclose(prior.mean_scale,torch.full((2,3), (60/9)**.5+1e-6)))

    def test_soft_allows_flow_at_zero_mean_deviation(self):
        p=torch.tensor([[[2.,1.],[-2.,-1.]],[[3.,2.],[-3.,-2.]]]);g,info=propose(p,self.prior,self.train,self.val,self.curve)
        self.assertTrue(info['accepted']);self.assertTrue(g.any())
        h,_=propose(p,self.prior,self.train,self.val,self.curve,Config(hard_mask=True));self.assertFalse(h.any())

    def test_rejection_restores_exact_frozen(self):
        def opposite(q,z):return q.square().mean() if float(z.min())<10 else 100.-q.square().mean()
        g,info=propose(self.p,self.prior,self.train,self.val,opposite);self.assertFalse(info['accepted'])
        self.assertTrue(torch.equal(transform(self.p,g,self.prior),self.p))
        with self.assertRaises(ValueError):propose(self.p,self.prior,self.train,self.train,self.curve)
        self.assertFalse(accepts(1,1,.9,1.051));self.assertFalse(accepts(1,1,.951,.99));self.assertTrue(accepts(1,1,.95,1.05));self.assertFalse(accepts(1,1,float('nan'),1))

    def test_episode_reset_and_refresh(self):
        e=EpisodeGate();g,_=e.step(0,self.p,self.prior,self.train,self.val,self.curve);self.assertTrue(g.any())
        g,_=e.step(1,self.p*0,self.prior,self.train,self.val,self.curve,Config(hard_mask=True));self.assertFalse(g.any())
        g,_=e.step(2,self.p,self.prior,self.train,self.val,self.curve);self.assertTrue(g.any())
        with self.assertRaises(ValueError): e.step(0,self.p,self.prior,self.train,self.val,self.curve)
        self.assertIsNone(EpisodeGate().g)

    def test_source_augmentation_families_and_both_views(self):
        rgb=torch.zeros(2,9,9,3,dtype=torch.uint8);rgb[:,4,4]=255; orig=rgb.clone()
        out=dict(perturbed_images(rgb));self.assertEqual(len(out),7);self.assertTrue(torch.equal(rgb,orig))
        self.assertTrue(torch.equal(out['motion'][:,3],torch.zeros_like(out['motion'][:,3])))
        for v in out.values():self.assertEqual(v.shape,rgb.shape);self.assertEqual(v.dtype,torch.uint8);self.assertTrue(torch.equal(v[0],v[1]))

    def test_cli_default_no_assets(self):
        p=Path(__file__).resolve().parents[1]/'scripts/fsfc_unified_probe_v1.py'
        r=subprocess.run([sys.executable,str(p),'--checkpoint','/nonexistent'],check=True,text=True,capture_output=True)
        self.assertFalse(json.loads(r.stdout)['experiment_started'])
