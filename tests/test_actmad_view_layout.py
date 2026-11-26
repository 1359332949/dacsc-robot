"""Check camera ownership, gradients, and episode isolation for our VLA port."""
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from dacsc.baselines_tta import ActMADInspired, viewwise_patch_moments


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(12)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PortContract(unittest.TestCase):
    def test_views_remain_separate_for_all_buffer_lengths(self):
        for n in (1, 2, 3, 4):
            a = torch.arange(n * 6, dtype=torch.float64).reshape(n, 2, 3)
            w = a + 100
            p = torch.cat((a, w)).requires_grad_()
            m, sd = viewwise_patch_moments(p, n, 2)
            expected_m = torch.stack((a.mean((0, 1)), w.mean((0, 1))))
            expected_s = torch.stack((a.std((0, 1), unbiased=False), w.std((0, 1), unbiased=False)))
            torch.testing.assert_close(m, expected_m)
            torch.testing.assert_close(sd, expected_s)
            m.sum().backward()
            torch.testing.assert_close(p.grad, torch.full_like(p, 1 / (n * 2)))

    def test_update_reset_and_close(self):
        torch.manual_seed(4)
        backbone = TinyBackbone().eval().requires_grad_(False)
        ext = SimpleNamespace(view_names=('agentview', 'eye_in_hand'),
                              visual_encoder=SimpleNamespace(backbone=backbone))
        # Wrapper is callable, as in CoRP, and owns the same backbone.
        class Visual(nn.Module):
            def __init__(self, b):
                super().__init__(); self.backbone = b
            def forward(self, x):
                return self.backbone(x)
        ext.visual_encoder = Visual(backbone)
        policy = SimpleNamespace(representation_extractor=ext)
        ref = (torch.zeros(2, 3), torch.ones(2, 3), torch.ones(2, 3))
        cal = SimpleNamespace(ref=lambda i: ref)
        sample = torch.randn(1, 5, 3)
        batch = {'images': {'agentview': sample, 'eye_in_hand': sample + 2},
                 'state': torch.zeros(1, 8)}
        initial = {k: v.clone() for k, v in backbone.state_dict().items()}
        baseline = backbone(sample).detach()
        adapter = ActMADInspired(policy, cal)
        with torch.no_grad():
            for _ in range(5):
                loss = adapter.update(batch)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertGreater(adapter.last_param_l2, 0)
        self.assertEqual(len(adapter.frames), 4)
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in backbone.state_dict().items()))
        adapter.reset()
        self.assertEqual(adapter.frames, [])
        self.assertEqual(len(adapter.opt.state), 0)
        for k, v in backbone.state_dict().items():
            torch.testing.assert_close(v, initial[k], atol=0, rtol=0)
        adapter.update(batch)
        adapter.close()
        torch.testing.assert_close(backbone(sample), baseline, atol=0, rtol=0)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
        self.assertTrue(all(not b._forward_hooks for b in backbone.blocks))

    def test_rejects_ambiguous_counts(self):
        with self.assertRaises(ValueError):
            viewwise_patch_moments(torch.zeros(3, 2, 3), 2, 2)


if __name__ == '__main__':
    unittest.main()
