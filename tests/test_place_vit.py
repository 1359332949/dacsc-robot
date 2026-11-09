import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from dacsc.place_vit import PlaceDropoutConfig, PlaceDropoutController
from scripts.train_corp_robocasa import place_training_loss


class FakeBlock(nn.Module):
    def forward(self, tokens):
        return tokens


class FakeVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([FakeBlock() for _ in range(4)])


class FakeRepresentationExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = FakeVisualEncoder()


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.view_names = (
            "robot0_agentview_left",
            "robot0_eye_in_hand",
            "robot0_agentview_right",
        )
        self.representation_extractor = FakeRepresentationExtractor()


class FakeFlowPolicy(FakePolicy):
    def __init__(self):
        super().__init__()
        self.seen_batch_size = None

    def training_loss(
        self,
        images,
        state,
        instruction,
        action,
        action_mask,
        *,
        task_labels,
    ):
        self.seen_batch_size = action.shape[0]
        rows = torch.cat([images[name] for name in self.view_names], dim=0)
        self.representation_extractor.visual_encoder.backbone.blocks[0](rows)
        flow = torch.zeros((), device=action.device)
        return SimpleNamespace(total=flow, flow=flow, task=None)


class PlaceVITTest(unittest.TestCase):
    def test_progressive_formula_is_monotonic_and_matches_official(self):
        config = PlaceDropoutConfig(
            method="place",
            drop_rate=0.33,
            velocity=4.0,
            reference_epochs=30,
        )
        fractions = [config.drop_fraction(step, 500) for step in (0, 1, 100, 499, 500)]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions, sorted(fractions))
        expected = 0.33 * 2 / math.pi * math.atan((30 * 250 / 500) / 4)
        self.assertAlmostEqual(config.drop_fraction(250, 500), expected)

    def test_fixed_random_uses_constant_fraction(self):
        config = PlaceDropoutConfig(method="place_random", drop_rate=0.33)
        self.assertEqual(config.drop_fraction(0, 500), 0.33)
        self.assertEqual(config.drop_fraction(500, 500), 0.33)

    def test_mask_has_exact_budget_shared_across_views_and_tokens(self):
        policy = FakePolicy().train()
        controller = PlaceDropoutController(
            policy,
            PlaceDropoutConfig(
                method="place_random",
                layers=(0,),
                drop_rate=0.33,
                seed=7,
            ),
        ).train()
        controller.begin_step(3, 500)
        tokens = torch.ones(9, 5, 8)
        output = policy.representation_extractor.visual_encoder.backbone.blocks[0](
            tokens
        )
        _, mask = controller.masks[0]
        self.assertEqual(mask.shape, (3, 8))
        self.assertEqual((mask == 0).sum(dim=1).tolist(), [2, 2, 2])
        expected = tokens.reshape(3, 3, 5, 8) * (
            mask.reshape(1, 3, 1, 8) / (1 - 0.33)
        )
        torch.testing.assert_close(output, expected.reshape_as(output))

    def test_only_active_layer_changes_and_eval_is_noop(self):
        policy = FakePolicy().train()
        controller = PlaceDropoutController(
            policy,
            PlaceDropoutConfig(
                method="place_random",
                layers=(0, 1),
                drop_rate=0.5,
                seed=4,
            ),
        ).train()
        controller.begin_step(0, 10)
        controller._active_layer = 0
        tokens = torch.ones(9, 5, 8)
        output = policy.representation_extractor.visual_encoder.backbone.blocks[0](
            tokens
        )
        self.assertFalse(torch.equal(output, tokens))
        inactive = policy.representation_extractor.visual_encoder.backbone.blocks[1](
            tokens
        )
        torch.testing.assert_close(inactive, tokens)
        policy.eval()
        eval_output = policy.representation_extractor.visual_encoder.backbone.blocks[
            0
        ](tokens)
        torch.testing.assert_close(eval_output, tokens, rtol=0, atol=0)

    def test_fixed_seed_reproduces_same_layer_and_mask(self):
        policy = FakePolicy().train()
        controller = PlaceDropoutController(
            policy,
            PlaceDropoutConfig(
                method="place",
                layers=(0, 1, 2, 3),
                seed=13,
            ),
        ).train()
        controller.begin_step(123, 500)
        tokens = torch.ones(9, 5, 8)
        policy.representation_extractor.visual_encoder.backbone.blocks[
            controller.active_layer
        ](tokens)
        first = controller.diagnostics()
        first_mask = controller.masks[0][1].clone()
        controller.begin_step(123, 500)
        policy.representation_extractor.visual_encoder.backbone.blocks[
            controller.active_layer
        ](tokens)
        second = controller.diagnostics()
        self.assertEqual(first["active_layer"], second["active_layer"])
        torch.testing.assert_close(first_mask, controller.masks[0][1])

    def test_place_training_loss_uses_one_ordinary_batch(self):
        policy = FakeFlowPolicy().train()
        controller = PlaceDropoutController(
            policy,
            PlaceDropoutConfig(
                method="place",
                layers=(0,),
                drop_rate=0.33,
                velocity=4,
                reference_epochs=30,
                seed=42,
            ),
        ).train()
        batch = {
            "images": {name: torch.randn(2, 5, 8) for name in policy.view_names},
            "state": torch.randn(2, 9),
            "instruction": ["open drawer", "open drawer"],
            "action": torch.randn(2, 4, 12),
            "action_mask": torch.ones(2, 4, dtype=torch.bool),
            "style_id": torch.tensor([11, 12]),
        }
        total, losses, diagnostics = place_training_loss(
            policy,
            controller,
            batch,
            step=250,
            total_steps=500,
            task_labels=None,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(losses.flow))
        self.assertEqual(policy.seen_batch_size, 2)
        self.assertEqual(diagnostics["active_layer"], 0)
        self.assertAlmostEqual(diagnostics["drop_fraction"], 0.275251470346829)


if __name__ == "__main__":
    unittest.main()
