import unittest
import torch
from dacsc.visual_adapter import ResidualVisualAdapter, AdaptedRepresentationExtractor
from dacsc.action_conditioned_predictor import ActionConditionedPredictor
from dacsc.online_tta import OnlineTTAUpdater


class TestResidualVisualAdapter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.views = ("agentview", "eye_in_hand")
        self.adapter = ResidualVisualAdapter(self.views, dim=32, context_dim=4, hidden_dim=16)
        self.ctx = torch.randn(2, 4)

    def test_identity_is_exact(self):
        x = torch.randn(2, 5, 32)
        for v in self.views:
            y = self.adapter(x, v, self.ctx)
            self.assertTrue(torch.equal(x, y), v)

    def test_shape_dtype_and_view_validation(self):
        x = torch.randn(2, 5, 32)
        y = self.adapter(x, "agentview", self.ctx)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        with self.assertRaises(ValueError):
            self.adapter(x, "agentview", torch.randn(2, 3))
        with self.assertRaises(KeyError):
            self.adapter(x, "unknown", self.ctx)

    def test_nonzero_output_changes_tokens(self):
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            self.adapter.fc2.weight.normal_()
        y = self.adapter(x, "agentview", self.ctx)
        self.assertFalse(torch.equal(x, y))


class TestActionConditionedPredictor(unittest.TestCase):
    def test_forward_and_freeze(self):
        p = ActionConditionedPredictor(z_dim=32, state_dim=8, action_dim=7,
                                       horizon=12, lang_dim=16, hidden_dim=32)
        z = torch.randn(2, 32); q = torch.randn(2, 8)
        a = torch.randn(2, 12, 7); l = torch.randn(2, 16)
        y = p(z, q, a, l)
        self.assertEqual(y.shape, (2, 32))
        p.freeze()
        for param in p.parameters():
            self.assertFalse(param.requires_grad)


class TestOnlineUpdater(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.views = ("agentview", "eye_in_hand")
        self.adapter = ResidualVisualAdapter(self.views, dim=32, context_dim=0, hidden_dim=16)
        self.predictor = ActionConditionedPredictor(z_dim=32, state_dim=8, action_dim=7,
                                                    horizon=12, lang_dim=16, hidden_dim=32)
        self.predictor.freeze()

        def encode(adapted):
            return torch.stack([adapted[v].mean(dim=1) for v in self.views], dim=1).mean(dim=1)

        self.updater = OnlineTTAUpdater(self.adapter, self.predictor, encode, lr=1e-3)
        self.patches = {v: torch.randn(2, 5, 32) for v in self.views}
        self.z_next = torch.randn(2, 32)
        self.q = torch.randn(2, 8)
        self.a = torch.randn(2, 12, 7)
        self.lang = torch.randn(2, 16)

    def test_only_adapter_updates_and_predictor_is_frozen(self):
        before_p = {k: v.detach().clone() for k, v in self.predictor.state_dict().items()}
        before_a = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
        out = self.updater.update(self.patches, None, self.z_next, self.q, self.a, self.lang)
        self.assertTrue(out["updated"])
        for k, v in self.predictor.state_dict().items():
            self.assertTrue(torch.equal(v, before_p[k]), k)
        changed = any(not torch.equal(v, before_a[k]) for k, v in self.adapter.state_dict().items())
        self.assertTrue(changed)

    def test_episode_reset_restores_identity(self):
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            self.adapter.fc2.weight.normal_()
        self.assertFalse(torch.equal(x, self.adapter(x, "agentview", None)))
        self.updater.reset_episode()
        for v in self.views:
            self.assertTrue(torch.equal(x, self.adapter(x, v, None)))

    def test_nonfinite_target_rejected(self):
        z = self.z_next.clone(); z[0, 0] = float("nan")
        before = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
        out = self.updater.update(self.patches, None, z, self.q, self.a, self.lang)
        self.assertFalse(out["updated"])
        for k, v in self.adapter.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))


if __name__ == "__main__":
    unittest.main()
