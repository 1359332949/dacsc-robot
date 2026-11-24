import unittest
import torch
from dacsc.single_block_probe import align,moments,reference,fit,score
class Operator(unittest.TestCase):
    def test_identity_and_dormant(self):
        x=torch.arange(36,dtype=torch.float32).reshape(2,6,3);m=x[:,1:].mean(1);s=x[:,1:].std(1,unbiased=False);r=(m+2,s*3,torch.tensor([[0.,1e10,1e10]]).expand_as(m));self.assertIs(align(x,r,0.,'uniform'),x);y=align(x,r,.75,'dead');self.assertTrue(torch.equal(y[:,:1],x[:,:1]));self.assertTrue(torch.equal(y[:,:,1:],x[:,:,1:]));self.assertTrue((y[:,2:,0]-y[:,1:-1,0]>0).all());self.assertTrue(torch.equal(y.flip(0),align(x.flip(0),tuple(v.flip(0) for v in r),.75,'dead')))
    def test_low_variance_finite_and_exact_threshold(self):
        x=torch.zeros(2,257,4);ref=(torch.ones(2,4),torch.zeros(2,4),torch.full((2,4),float('inf')));self.assertTrue(torch.equal(align(x,ref,.25,'dead'),x));self.assertTrue(torch.isfinite(align(x,ref,.25,'uniform')).all())
    def test_nine_scene_quantile_inner_loo(self):
        x=torch.arange(9*2*4*3,dtype=torch.float64).reshape(9,2,4,3);raw=torch.stack([torch.stack(moments(z)) for z in x]);mu,ss,tau=reference(raw);torch.testing.assert_close(mu,x.mean((0,2)).float());torch.testing.assert_close(ss,x.permute(1,0,2,3).reshape(2,-1,3).std(1,unbiased=False).float()); self.assertTrue((tau>0).all());
        with self.assertRaises(ValueError):reference(raw[:8])
