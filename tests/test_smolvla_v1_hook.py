import unittest
import torch
from types import SimpleNamespace
from torch import nn
from dacsc.smolvla_v1_hook import align_tokens, fit_loto, LAYERS, SmolVLAV1Hook


class SmolVLAHookMath(unittest.TestCase):
    def test_bf16_bypass_and_inactive_bits(self):
        x = torch.randn(1, 1024, 768).bfloat16()
        ref = (torch.zeros(1,768), torch.ones(1,768), torch.full((1,768), float('inf')))
        self.assertIs(align_tokens(x, ref, 0), x)
        y = align_tokens(x, ref, .75)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(y.view(torch.int16), x.view(torch.int16)))

    def test_all_patches_and_view_symmetric_formula(self):
        x = torch.zeros(1, 1024, 3, dtype=torch.bfloat16)
        x[:, 0] = 10  # first patch must participate: there is no CLS
        mu = torch.full((1,3), 2.)
        ss = torch.ones(1,3)
        tau = torch.zeros(1,3)
        y = align_tokens(x,(mu,ss,tau),.75)
        p=x.float();m=p.mean(1);sd=p.std(1,unbiased=False)
        gain=(((1-.75)*sd+.75*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2.)
        expected=(gain[:,None]*(p-m[:,None])+((1-.75)*m+.75*mu)[:,None]).bfloat16()
        self.assertTrue(torch.equal(y,expected))
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.equal(y[:,0],y[:,1]))

    def test_outer_and_inner_task_exclusion(self):
        torch.manual_seed(7)
        raw={i:{'mean':torch.randn(i%3+1,4,2,3), 'second':None} for i in range(10)}
        for r in raw.values(): r['second']=r['mean'].square()+1
        refs,audit=fit_loto(raw)
        changed={i:{k:v.clone() for k,v in r.items()} for i,r in raw.items()}
        changed[0]['mean']+=100
        changed[0]['second']=changed[0]['mean'].square()+1
        other,_=fit_loto(changed)
        for l in LAYERS:
            for x,y in zip(refs[0,l],other[0,l]):torch.testing.assert_close(x,y,atol=0,rtol=0)
        for t,a in audit.items():
            self.assertEqual(set(a['outer']),set(range(10))-{t})
            for h,names in a['inner'].items():self.assertEqual(set(names),set(range(10))-{t,h})


class CameraProtocol(unittest.TestCase):
    def make(self):
        layers=nn.ModuleList([nn.Identity() for _ in range(12)])
        vision=SimpleNamespace(encoder=SimpleNamespace(layers=layers))
        policy=SimpleNamespace(model=SimpleNamespace(vlm_with_expert=SimpleNamespace(vlm=SimpleNamespace(model=SimpleNamespace(vision_model=vision)))),
                               config=SimpleNamespace(image_features={'observation.images.image':None,'observation.images.image2':None}))
        refs={3:(torch.stack((torch.ones(768)*2,torch.ones(768)*-2)),torch.ones(2,768),torch.zeros(2,768))}
        hook=SmolVLAV1Hook(policy,refs,layers=(3,),active_layer=3)
        return layers[2],hook,refs

    def test_camera_references_do_not_mix(self):
        layer,hook,ref=self.make()
        try:
            a=torch.ones(1,1024,768,dtype=torch.bfloat16)
            w=a*10
            hook.begin();ya=layer(a);yw=layer(w);hook.end()
            for i,(x,y) in enumerate(((a,ya),(w,yw))):
                expected=align_tokens(x,tuple(v[i][None] for v in ref[3]))
                self.assertTrue(torch.equal(y,expected))
        finally:hook.close()

    def test_missing_or_extra_camera_call_rejected(self):
        layer,hook,_=self.make();x=torch.ones(1,1024,768,dtype=torch.bfloat16)
        try:
            hook.begin();layer(x)
            with self.assertRaises(RuntimeError):hook.end()
        finally:hook.close()
        layer,hook,_=self.make()
        try:
            hook.begin();layer(x);layer(x)
            with self.assertRaises(RuntimeError):layer(x)
        finally:hook.close()

    def test_wrong_dtype_rejected(self):
        layer,hook,_=self.make()
        try:
            hook.begin()
            with self.assertRaises(ValueError):layer(torch.ones(1,1024,768))
        finally:hook.close()


if __name__=='__main__':unittest.main()
