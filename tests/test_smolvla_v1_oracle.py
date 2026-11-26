"""Independent port oracle: execute unchanged frozen CoRP V1 math from its source."""
import ast
import os
from pathlib import Path
import unittest
import torch
from dacsc.smolvla_v1_hook import align_tokens,fit_loto,LAYERS


def original_math():
    default=Path(__file__).resolve().parents[1]/'dacsc/single_block_probe.py'
    path=Path(os.environ.get('V1_ORACLE_SOURCE',str(default)))
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('align','fit','score')]
    namespace={'torch':torch,'ALPHAS':(.25,.5,.75)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class OriginalV1Oracle(unittest.TestCase):
    def test_bf16_serial_views_equal_original_fp32_formula(self):
        old=original_math()['align']
        torch.manual_seed(57)
        x=torch.randn(2,1024,768).bfloat16()
        x[:,:,0]=0  # zero-variance edge, still not a CLS token
        ref=(torch.randn(2,768)*.1,torch.rand(2,768)+.5,torch.rand(2,768)*.05)
        for alpha in (0,.25,.5,.75):
            # Original DINO function ignores its CLS; add one only for the oracle.
            with_cls=torch.cat((torch.zeros(2,1,768),x.float()),dim=1)
            expected=old(with_cls,ref,alpha,'dead')[:,1:].bfloat16()
            actual=torch.cat([align_tokens(x[v:v+1],tuple(r[v] for r in ref),alpha) for v in range(2)])
            self.assertTrue(torch.equal(expected.view(torch.int16),actual.view(torch.int16)))

    def test_full_trajectory_loto_matches_original_fit_and_score(self):
        old=original_math();torch.manual_seed(9)
        raw={}
        for t in range(10):
            m=torch.randn(t%3+1,4,2,5)*.1
            raw[t]={'mean':m,'second':m.square()+torch.rand_like(m)+.5}
        refs,_=fit_loto(raw)
        for li,l in enumerate(LAYERS):
            def stats(names):
                return old['fit'](torch.cat([torch.stack((raw[t]['mean'][:,li],raw[t]['second'][:,li]),dim=1) for t in names]))
            outer=list(range(1,10));mu,ss=stats(outer);scores=[]
            for held in outer:
                mi,si=stats([t for t in outer if t!=held]);m=raw[held]['mean'][:,li];q=raw[held]['second'][:,li]
                scores.append(old['score'](m,(q-m.square()).clamp_min(0).sqrt(),mi,si))
            expected=(mu,ss,torch.cat(scores).quantile(.95,dim=0))
            for x,y in zip(expected,refs[0,l]):torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
