import unittest
from collections import Counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from dacsc.dgcd_vit import DgCDConfig, DgCDController, dgcd_scores
from scripts.train_corp_robocasa import (
    EnvironmentBalancedBatchSampler,
    dgcd_training_loss,
)


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
        for layer in range(len(self.representation_extractor.visual_encoder.backbone.blocks)):
            self.representation_extractor.visual_encoder.backbone.blocks[layer](rows)
        flow = torch.zeros((), device=action.device)
        return SimpleNamespace(total=flow, flow=flow, task=None)


def reference_dgcd_scores(feature_map, *, rho, group_size):
    values = feature_map.detach().float().mean(dim=(2, 3))
    batch_size, channels = values.shape
    num_groups = batch_size // group_size
    scaler_t_mean = torch.zeros(
        (batch_size, channels), requires_grad=True
    )
    scaler_t_var = torch.ones_like(scaler_t_mean).requires_grad_(True)
    scaler_e = []
    for _ in range(num_groups):
        scaler_e.append(
            (
                torch.zeros((group_size, channels), requires_grad=True),
                torch.ones((group_size, channels), requires_grad=True),
            )
        )

    def alignment():
        t_m = values.mean(dim=0, keepdim=True)
        t_v = values.var(dim=0, keepdim=True, unbiased=False)
        total = (values - t_m) / torch.sqrt(t_v + 1e-5)
        total = total * scaler_t_var + scaler_t_mean
        env = []
        for index in range(num_groups):
            sample = values[index * group_size : (index + 1) * group_size]
            s_m = sample.mean(dim=0, keepdim=True)
            s_v = sample.var(dim=0, keepdim=True, unbiased=False)
            aligned = (sample - s_m) / torch.sqrt(s_v + 1e-5)
            aligned = aligned * scaler_e[index][1] + scaler_e[index][0]
            env.append(aligned)
        return torch.cat(env), total

    env, total = alignment()
    dist = F.kl_div(
        F.log_softmax(total, dim=1),
        F.log_softmax(env, dim=1),
        reduction="batchmean",
        log_target=True,
    )
    dist.backward()
    params = [scaler_t_mean, scaler_t_var]
    for pair in scaler_e:
        params.extend(pair)
    for parameter in params:
        grad = parameter.grad
        parameter.data += rho * grad.div(
            torch.sqrt(torch.norm(grad, dim=-1, keepdim=True) + 1e-12)
        )

    with torch.no_grad():
        env, total = alignment()
        gram = torch.mm(total.t() + 1e-7, env + 1e-7).diag()

        def scaling(value):
            return (value - value.min(dim=-1, keepdim=True)[0]) / (
                value.max(dim=-1, keepdim=True)[0]
                - value.min(dim=-1, keepdim=True)[0]
            )

        total = scaling(total / gram)
        env = scaling(env / gram)
        return scaling((total - env).pow(2))


class DgCDVITTest(unittest.TestCase):
    def test_environment_sampler_produces_pure_task_balanced_style_groups(self):
        flat = []
        for style in (11, 12, 13):
            for task in ("TaskA", "TaskB"):
                for frame in range(12):
                    flat.append((0, len(flat), style, "instruction", task, 0, frame, 12))
        dataset = SimpleNamespace(flat=flat)
        sampler = EnvironmentBalancedBatchSampler(
            dataset,
            group_size=4,
            groups_per_batch=2,
            seed=17,
        )
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 8)
        for start in (0, 4):
            group = [flat[index] for index in batch[start : start + 4]]
            self.assertEqual(len({entry[2] for entry in group}), 1)
            self.assertEqual(sorted(Counter(entry[4] for entry in group).values()), [2, 2])

    def test_scores_match_official_formula_reference(self):
        torch.manual_seed(11)
        feature = torch.randn(8, 8, 3, 5)
        actual = dgcd_scores(feature, rho=0.05, group_size=4)
        expected = reference_dgcd_scores(feature, rho=0.05, group_size=4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0)
        self.assertLessEqual(float(actual.max()), 1)

    def test_default_batch_is_three_groups_of_32(self):
        config = DgCDConfig()
        self.assertEqual(config.group_size, 32)
        self.assertEqual(config.groups_per_batch, 3)
        self.assertEqual(config.batch_size, 96)

    def test_controller_drops_two_active_layers_and_shares_views(self):
        policy = FakePolicy().train()
        controller = DgCDController(
            policy,
            DgCDConfig(
                method="dgcd",
                layers=(0, 1, 2, 3),
                group_size=4,
                groups_per_batch=2,
                drop_rate=0.33,
                rho=0.05,
                seed=5,
            ),
        ).train()
        controller.begin_step(17)
        self.assertIn(3, controller.active_layers)
        self.assertEqual(len(controller.active_layers), 2)
        tokens = torch.randn(24, 5, 8)
        for layer in controller.active_layers:
            policy.representation_extractor.visual_encoder.backbone.blocks[layer](
                tokens
            )
        self.assertEqual(len(controller.masks), 2)
        for _, mask in controller.masks:
            self.assertEqual(mask.shape, (8, 8))
            self.assertEqual((mask == 0).sum(dim=1).tolist(), [2] * 8)
        diagnostics = controller.diagnostics()
        self.assertEqual(len(diagnostics["score_events"]), 2)

    def test_random_control_matches_layer_schedule_and_drop_counts(self):
        torch.manual_seed(3)
        deterministic = DgCDController(
            FakePolicy().train(),
            DgCDConfig(
                method="dgcd",
                layers=(0, 1, 2, 3),
                group_size=4,
                groups_per_batch=2,
                seed=9,
            ),
        ).train()
        random_control = DgCDController(
            FakePolicy().train(),
            DgCDConfig(
                method="dgcd_random",
                layers=(0, 1, 2, 3),
                group_size=4,
                groups_per_batch=2,
                seed=9,
            ),
        ).train()
        deterministic.begin_step(31)
        random_control.begin_step(31)
        self.assertEqual(
            deterministic.active_layers, random_control.active_layers
        )
        tokens = torch.randn(24, 5, 8)
        for controller in (deterministic, random_control):
            for layer in controller.active_layers:
                controller.policy.representation_extractor.visual_encoder.backbone.blocks[
                    layer
                ](tokens)
        for (_, first), (_, second) in zip(
            deterministic.masks, random_control.masks, strict=True
        ):
            self.assertEqual(
                (first == 0).sum(dim=1).tolist(),
                (second == 0).sum(dim=1).tolist(),
            )

    def test_eval_mode_is_noop(self):
        policy = FakePolicy().eval()
        controller = DgCDController(
            policy,
            DgCDConfig(
                method="dgcd",
                layers=(0, 1),
                group_size=4,
                groups_per_batch=2,
            ),
        ).train()
        controller.begin_step(0)
        tokens = torch.randn(24, 5, 8)
        for layer in controller.active_layers:
            output = policy.representation_extractor.visual_encoder.backbone.blocks[
                layer
            ](tokens)
            torch.testing.assert_close(output, tokens, rtol=0, atol=0)

    def test_dgcd_training_loss_uses_one_environment_batch(self):
        policy = FakeFlowPolicy().train()
        controller = DgCDController(
            policy,
            DgCDConfig(
                method="dgcd_random",
                layers=(0, 1, 2, 3),
                group_size=4,
                groups_per_batch=2,
                seed=19,
            ),
        ).train()
        batch = {
            "images": {name: torch.randn(8, 5, 8) for name in policy.view_names},
            "state": torch.randn(8, 9),
            "instruction": ["instruction"] * 8,
            "action": torch.randn(8, 4, 12),
            "action_mask": torch.ones(8, 4, dtype=torch.bool),
            "style_id": torch.tensor([11] * 4 + [12] * 4),
        }
        total, losses, diagnostics = dgcd_training_loss(
            policy,
            controller,
            batch,
            step=7,
            task_labels=None,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(losses.flow))
        self.assertEqual(policy.seen_batch_size, 8)
        self.assertEqual(len(diagnostics["active_layers"]), 2)


if __name__ == "__main__":
    unittest.main()
