#!/usr/bin/env python3
"""Token-level causal occlusion on the K-token flow/resampler policy."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dacsc.compat  # noqa: E402


def load_policy_and_loader(ckpt: Path, batch_size: int, device: str):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = ROOT / "repos/diffusion_policy/diffusion_policy/config"
    with hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = hydra.compose(
            config_name="train_dinov2_flow",
            overrides=["task=robocasa/pretrain_atomic61"])
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    policy = hydra.utils.instantiate(cfg.policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy.set_normalizer(dataset.get_normalizer())
    payload = torch.load(ckpt.open("rb"), map_location="cpu", pickle_module=dill)
    state = payload["state_dicts"]["model"]
    missing, unexpected = policy.load_state_dict(state, strict=False)
    print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    policy.to(device)
    sampler = dataset.get_dataset_sampler()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=False)
    return policy, loader


def clean_features(policy, batch):
    nobs = policy.normalizer.normalize(batch["obs"])
    To = int(policy.n_obs_steps)
    this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:])
                 for k, v in nobs.items()}
    return policy.obs_encoder(this_nobs)


def style_and_action_scores(policy, batch):
    policy.eval()
    policy.disable_gating_hooks()
    # Style score: between-style / within-style variance per token position.
    with torch.no_grad():
        feat = clean_features(policy, batch)
    B, To = int(batch["domain_id"].shape[0]), int(policy.n_obs_steps)
    P, D = int(feat.shape[1]), int(feat.shape[2])
    x = feat.reshape(B, To, P, D).mean(dim=1)       # (B,P,D)
    did = batch["domain_id"]
    uniq = did.unique(sorted=True)
    protos = torch.stack([x[did == u].mean(0) for u in uniq], dim=0)  # (S,P,D)
    between = (protos - protos.mean(0, keepdim=True)).pow(2).mean(0).mean(-1)
    within = torch.stack([
        (x[did == u] - x[did == u].mean(0, keepdim=True)).pow(2).mean(0).mean(-1)
        for u in uniq
    ], dim=0).mean(0)
    style_score = between / (within + 1e-8)

    # Action score: Taylor importance of each token from flow velocity loss.
    captured = {}
    def capture(module, inputs, output):
        output.retain_grad()
        captured["feat"] = output
        return output
    h = policy.obs_encoder.register_forward_hook(capture)
    try:
        policy.zero_grad(set_to_none=True)
        loss = policy._base_loss(batch)
        loss.backward()
    finally:
        h.remove()
    action_feat = captured["feat"].detach()
    action_grad = captured["feat"].grad.detach()
    action_score = (action_feat * action_grad).abs().mean(dim=(0, 2))
    policy.zero_grad(set_to_none=True)
    return P, style_score.detach(), action_score.detach(), float(loss.detach()), uniq


def eval_with_token_occlusion(policy, batch, selected, seed=12345):
    selected = selected.to(torch.long)
    def hook(module, inputs, output):
        out = output.clone()
        out[:, selected, :] = 0.0
        return out
    h = policy.obs_encoder.register_forward_hook(hook)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            loss = policy._base_loss(batch)
        return float(loss.detach())
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/runs/token_occlusion.json"))
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--drop-ratio", type=float, default=0.33)
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    policy, loader = load_policy_and_loader(args.ckpt, args.batch_size, args.device)

    def move(obj):
        if torch.is_tensor(obj):
            return obj.to(args.device)
        if isinstance(obj, dict):
            return {k: move(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(v) for v in obj)
        return obj

    records = []
    for batch in itertools.islice(loader, max(1, args.num_batches)):
        batch = move(batch)
        P, style_score, action_score, score_loss, uniq = style_and_action_scores(policy, batch)
        # Exclude token 0 (low-dim/language token); only visual tokens are occluded.
        visual = torch.arange(1, P, device=style_score.device)
        k = max(1, int(round(args.drop_ratio * len(visual))))
        s_vis, a_vis = style_score[1:], action_score[1:]
        style_top = visual[torch.topk(s_vis, k, largest=True).indices]
        action_top = visual[torch.topk(a_vis, k, largest=True).indices]
        gen = torch.Generator(device=style_score.device)
        gen.manual_seed(42 + len(records))
        rand = visual[torch.randperm(len(visual), device=visual.device, generator=gen)[:k]]
        losses = {
            "full": eval_with_token_occlusion(policy, batch, torch.empty(0, dtype=torch.long,
                                                                         device=visual.device)),
            "top_style": eval_with_token_occlusion(policy, batch, style_top),
            "top_action": eval_with_token_occlusion(policy, batch, action_top),
            "random": eval_with_token_occlusion(policy, batch, rand),
        }
        records.append({
            "tokens_per_obs": P,
            "visual_tokens": int(len(visual)),
            "drop_tokens": int(k),
            "domains": [int(x) for x in uniq.tolist()],
            "score_loss": score_loss,
            "loss": losses,
            "delta_vs_full": {k2: losses[k2] - losses["full"] for k2 in losses if k2 != "full"},
            "delta_vs_random": {"top_style": losses["top_style"] - losses["random"],
                                 "top_action": losses["top_action"] - losses["random"]},
            "top_style_indices": [int(x) for x in style_top.tolist()],
            "top_action_indices": [int(x) for x in action_top.tolist()],
            "random_indices": [int(x) for x in rand.tolist()],
            "style_score_mean": float(style_score.mean()),
            "action_score_mean": float(action_score.mean()),
        })
    out = {"records": records,
           "aggregate": {
               k2: sum(r["loss"][k2] for r in records) / len(records)
               for k2 in ["full", "top_style", "top_action", "random"]
           }}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"OUTPUT: {args.out}")


if __name__ == "__main__":
    main()
