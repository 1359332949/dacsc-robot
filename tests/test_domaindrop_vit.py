import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dacsc.domaindrop_vit import (
    ChannelDropoutConfig,
    ChannelDropoutController,
    LayerDomainDrop,
    _random_channel_mask,
)
from scripts.train_corp_robocasa import channel_training_loss, paired_flow_sample


class FakeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(channels, channels)

    def forward(self, tokens):
        return tokens


class FakeVisualEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([FakeBlock(channels) for _ in range(4)])


class FakeRepresentationExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.visual_encoder = FakeVisualEncoder(channels)


class FakePolicy(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.view_names = (
            "robot0_agentview_left",
            "robot0_eye_in_hand",
            "robot0_agentview_right",
        )
        self.representation_extractor = FakeRepresentationExtractor(channels)


class FakeFlowPolicy(FakePolicy):
    def __init__(self, channels=8):
        super().__init__(channels)
        self.seen_eps = None
        self.seen_tau = None

    def training_loss_with_flow(
        self,
        images,
        state,
        instruction,
        action,
        action_mask,
        *,
        task_labels,
        eps=None,
        tau=None,
    ):
        self.seen_eps = eps
        self.seen_tau = tau
        rows = torch.cat([images[name] for name in self.view_names], dim=0)
        self.representation_extractor.visual_encoder.backbone.blocks[0](rows)
        velocity = action * 0 + state[:, :1, None]
        flow = torch.zeros((), device=action.device)
        return SimpleNamespace(total=flow, flow=flow, task=None), velocity


class DomainDropVITTest(unittest.TestCase):
    def test_random_mask_is_exact_budget_and_recoverable(self):
        mask = _random_channel_mask(
            4,
            10,
            0.3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(mask.shape, (4, 10))
        self.assertEqual((mask == 0).sum(dim=1).tolist(), [3, 3, 3, 3])

    def test_domain_drop_logits_have_encoder_gradient(self):
        discriminator = LayerDomainDrop(
            6,
            num_domains=3,
            grl_lambda=0.25,
            wrs=True,
            recover=True,
        )
        feature = torch.randn(2, 6, 3, 4, requires_grad=True)
        labels = torch.tensor([0, 2])
        logits, mask = discriminator(feature, labels, 0.33)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(mask.shape, (2, 6))
        logits.sum().backward()
        self.assertIsNotNone(feature.grad)
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_controller_shares_one_mask_over_three_views(self):
        policy = FakePolicy(channels=8).train()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="random",
                layers=(0,),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0, 12: 1},
        ).train()
        controller.begin_step(torch.tensor([11, 12]))
        tokens = torch.arange(6 * 4 * 8, dtype=torch.float32).reshape(6, 4, 8)
        output = policy.representation_extractor.visual_encoder.backbone.blocks[0](
            tokens
        )
        self.assertEqual(len(controller.masks), 1)
        _, mask = controller.masks[0]
        self.assertEqual(mask.shape, (2, 8))
        masked = tokens.reshape(3, 2, 4, 8) * mask.reshape(1, 2, 1, 8)
        torch.testing.assert_close(output, masked.reshape(6, 4, 8))

    def test_random_control_keeps_same_domain_head_and_logs_diagnostics(self):
        policy = FakePolicy(channels=8).train()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="random",
                layers=(0, 1),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0, 12: 1},
        ).train()
        controller.begin_step(torch.tensor([11, 12]))
        controller._active_layer = 0
        tokens = torch.randn(6, 4, 8)
        policy.representation_extractor.visual_encoder.backbone.blocks[0](tokens)
        diagnostics = controller.diagnostics()
        self.assertEqual(len(controller.domain_logits), 1)
        self.assertIsNotNone(diagnostics["domain_accuracy"])
        self.assertIsNotNone(diagnostics["domain_balanced_accuracy"])
        self.assertEqual(diagnostics["active_layer"], 0)
        self.assertEqual(len(diagnostics["mask_events"]), 1)
        self.assertAlmostEqual(
            diagnostics["mask_events"][0]["mean_actual_drop_fraction"],
            0.5,
        )

    def test_only_active_layer_changes_output(self):
        policy = FakePolicy(channels=8).train()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="domaindrop_vit",
                layers=(0, 1),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0, 12: 1},
        ).train()
        controller.begin_step(torch.tensor([11, 12]))
        controller._active_layer = 0
        tokens = torch.randn(6, 4, 8)
        active = policy.representation_extractor.visual_encoder.backbone.blocks[0](
            tokens
        )
        self.assertFalse(torch.equal(active, tokens))
        inactive = policy.representation_extractor.visual_encoder.backbone.blocks[1](
            tokens
        )
        torch.testing.assert_close(inactive, tokens, rtol=0, atol=0)

    def test_duplicated_halves_receive_different_domain_masks(self):
        torch.manual_seed(7)
        policy = FakePolicy(channels=8).train()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="domaindrop_vit",
                layers=(0,),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0, 12: 1},
        ).train()
        controller.begin_step(torch.tensor([11, 12, 11, 12]))
        tokens = torch.randn(12, 4, 8)
        policy.representation_extractor.visual_encoder.backbone.blocks[0](tokens)
        _, mask = controller.masks[0]
        self.assertEqual(mask.shape, (4, 8))
        self.assertFalse(torch.equal(mask[0], mask[1]))

    def test_checkpoint_restores_discriminator_state_exactly(self):
        config = ChannelDropoutConfig(
            method="domaindrop_vit",
            layers=(0,),
            drop_rate=0.5,
            layer_probability=1.0,
        )
        first = ChannelDropoutController(FakePolicy(8).train(), config, {11: 0, 12: 1})
        with torch.no_grad():
            first.discriminators["0"].model.weight.add_(0.25)
            first.discriminators["0"].model.bias.sub_(0.1)
        state = first.state_dict()
        first.close()

        second = ChannelDropoutController(FakePolicy(8).train(), config, {11: 0, 12: 1})
        second.load_state_dict(state)
        torch.testing.assert_close(
            second.discriminators["0"].model.weight,
            first.discriminators["0"].model.weight,
        )
        torch.testing.assert_close(
            second.discriminators["0"].model.bias,
            first.discriminators["0"].model.bias,
        )
        second.close()

    def test_paired_flow_sample_has_two_identical_correctly_sized_halves(self):
        torch.manual_seed(13)
        eps, tau = paired_flow_sample(torch.randn(6, 4, 12))
        self.assertEqual(eps.shape, (6, 4, 12))
        self.assertEqual(tau.shape, (6,))
        torch.testing.assert_close(eps[:3], eps[3:])
        torch.testing.assert_close(tau[:3], tau[3:])

    def test_channel_training_loss_pairs_noise_and_masks_end_to_end(self):
        policy = FakeFlowPolicy(channels=8).train()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="domaindrop_vit",
                layers=(0,),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0, 12: 1},
        ).train()
        batch = {
            "images": {
                name: torch.randn(2, 4, 8) for name in policy.view_names
            },
            "state": torch.randn(2, 9),
            "instruction": ["open drawer", "open drawer"],
            "action": torch.randn(2, 4, 12),
            "action_mask": torch.ones(2, 4, dtype=torch.bool),
            "style_id": torch.tensor([11, 12]),
        }
        total, losses, consistency, domain_loss, diagnostics = channel_training_loss(
            policy,
            controller,
            batch,
            task_labels=None,
            consistency_weight=1.5,
            domain_loss_weight=1.0,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(losses.flow))
        self.assertTrue(torch.isfinite(consistency))
        self.assertTrue(torch.isfinite(domain_loss))
        self.assertEqual(policy.seen_eps.shape, (4, 4, 12))
        self.assertEqual(policy.seen_tau.shape, (4,))
        torch.testing.assert_close(policy.seen_eps[:2], policy.seen_eps[2:])
        torch.testing.assert_close(policy.seen_tau[:2], policy.seen_tau[2:])
        self.assertEqual(diagnostics["active_layer"], 0)
        self.assertEqual(diagnostics["mask_events"][0]["samples"], 4)

    def test_eval_mode_is_strict_noop_even_with_active_controller(self):
        policy = FakePolicy(channels=8).eval()
        controller = ChannelDropoutController(
            policy,
            ChannelDropoutConfig(
                method="domaindrop_vit",
                layers=(0,),
                drop_rate=0.5,
                layer_probability=1.0,
            ),
            {11: 0},
        ).train()
        controller.begin_step(torch.tensor([11]))
        tokens = torch.randn(6, 4, 8)
        output = policy.representation_extractor.visual_encoder.backbone.blocks[0](
            tokens
        )
        torch.testing.assert_close(output, tokens, rtol=0, atol=0)

    def test_domaindrop_config_rejects_per_view_variant_for_first_phase(self):
        with self.assertRaisesRegex(ValueError, "shared-view"):
            ChannelDropoutConfig(method="domaindrop_vit", views_shared=False)


if __name__ == "__main__":
    unittest.main()
