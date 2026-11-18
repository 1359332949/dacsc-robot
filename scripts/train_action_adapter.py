"""Train a zero-initialized action-consistent adapter on frozen CoRP pairs."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402


def build_rep(adapter, rec, device):
    visual = torch.tensor(rec["shift_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec.get("shift_state_token", rec["state_token"]), dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    adapted = adapter(visual)
    tokens = torch.cat([adapted, lang, state], dim=1)
    return Representation(tokens=tokens, pooled_language=pooled, action_dtype=torch.float32)


def clean_rep(adapter, rec, device):
    visual = torch.tensor(rec["clean_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec.get("clean_state_token", rec["state_token"]), dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    adapted = adapter(visual)
    return Representation(tokens=torch.cat([adapted, lang, state], dim=1),
                          pooled_language=pooled, action_dtype=torch.float32)


def clean_frozen_rep(rec, device):
    visual = torch.tensor(rec["clean_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec.get("clean_state_token", rec["state_token"]), dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    return Representation(tokens=torch.cat([visual, lang, state], dim=1),
                          pooled_language=pooled, action_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, action="append", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--id-weight", type=float, default=1.0)
    ap.add_argument("--grip-weight", type=float, default=1.0)
    ap.add_argument("--z0-augment", type=int, default=1)
    ap.add_argument("--res-weight", type=float, default=0.0)
    ap.add_argument("--clip-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z0-epoch-resample", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    data = []
    for path in args.pairs:
        recs = json.loads(path.read_text())["records"]
        data.extend(recs)
    data = [r for r in data if r.get("pairing_pass")]
    if not data:
        raise ValueError("no pairing-pass records")
    roles = {r.get("role") for r in data}
    if "source_fit" in roles and "source_validation" in roles:
        train = [r for r in data if r.get("role") == "source_fit"]
        val = [r for r in data if r.get("role") == "source_validation"]
    else:
        states = sorted({int(r["state_id"]) for r in data})
        if len(states) < 2:
            raise ValueError("need at least two source states for grouped train/val")
        val_states_fallback = {states[-1]}
        train = [r for r in data if int(r["state_id"]) not in val_states_fallback]
        val = [r for r in data if int(r["state_id"]) in val_states_fallback]
    train_states = sorted({int(r["state_id"]) for r in train})
    val_states = sorted({int(r["state_id"]) for r in val})
    if not train or not val:
        raise ValueError("empty role split")
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    # Parity: recomputed clean frozen action must match the stored clean action.
    parity_max = 0.0
    for rec in data:
        z0 = torch.tensor(rec["z0"], dtype=torch.float32, device=device).unsqueeze(0)
        target = torch.tensor(rec["clean_normalized_actions"], dtype=torch.float32, device=device)
        with torch.no_grad():
            rec_action = sample_from_z0(policy, clean_frozen_rep(rec, device), z0)
        parity_max = max(parity_max, float((rec_action - target).abs().max()))
    if parity_max > 1e-4:
        raise RuntimeError(f"clean-action parity failed: {parity_max}")
    adapter = ResidualVisualAdapter(768, args.rank).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    lossf = nn.HuberLoss(reduction="none")
    weight = torch.ones(7, device=device)
    weight[6] = args.grip_weight
    def one(rec, epoch=0, resample=False):
        rep = build_rep(adapter, rec, device)
        c_rep = clean_rep(adapter, rec, device)
        frozen_rep = clean_frozen_rep(rec, device)
        base_z0 = torch.tensor(rec["z0"], dtype=torch.float32, device=device).unsqueeze(0)
        pair_total = 0.0; ident_total = 0.0; clip_total = 0.0; n = 0
        for aug in range(max(1, args.z0_augment)):
            if aug == 0:
                z0 = base_z0
            else:
                seed_off = epoch * 100003 if resample else 0
                torch.manual_seed(900001 * int(rec["state_id"]) + 97 * int(rec["chunk_index"]) + aug + seed_off)
                z0 = torch.randn((1, policy.action_generator.action_horizon,
                                  policy.action_generator.action_dim), device=device, dtype=torch.float32)
            with torch.no_grad():
                target = sample_from_z0(policy, frozen_rep, z0)
            pred = sample_from_z0(policy, rep, z0)
            pair_total = pair_total + (lossf(pred, target) * weight).mean()
            c_pred = sample_from_z0(policy, c_rep, z0)
            ident_total = ident_total + (lossf(c_pred, target) * weight).mean()
            if args.clip_weight > 0:
                p_ctrl = policy.normalizer.unnormalize_action(pred).clamp(-1.0, 1.0)
                t_ctrl = policy.normalizer.unnormalize_action(target).clamp(-1.0, 1.0)
                clip_total = clip_total + (lossf(p_ctrl, t_ctrl) * weight).mean()
            n += 1
        pair = pair_total / n; ident = ident_total / n
        loss = pair + args.id_weight * ident
        if args.clip_weight > 0:
            loss = loss + args.clip_weight * (clip_total / n)
        if args.res_weight > 0:
            visual = torch.tensor(rec["shift_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
            adapted_visual = adapter(visual)
            res = (adapted_visual - visual).square().mean()
            loss = loss + args.res_weight * res
        return loss, pair, ident
    best_val = float("inf"); best_state = None; best_epoch = -1
    for epoch in range(args.epochs):
        adapter.train(); tr = 0.0
        for rec in train:
            opt.zero_grad(); loss, pair, ident = one(rec, epoch=epoch, resample=args.z0_epoch_resample); loss.backward(); opt.step()
            tr += float(loss.detach())
        adapter.eval(); va = 0.0
        with torch.no_grad():
            for rec in val:
                loss, _, _ = one(rec, epoch=0, resample=False); va += float(loss)
        va = va / len(val)
        if va < best_val:
            best_val = va; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        print(json.dumps({"epoch": epoch, "train": tr / len(train), "val": va, "best_val": best_val}))
    if best_state is not None:
        adapter.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "rank": args.rank, "dim": 768,
                "val_states": val_states, "train_states": train_states,
                "best_epoch": best_epoch, "best_val": best_val,
                "id_weight": args.id_weight, "grip_weight": args.grip_weight,
                "z0_augment": args.z0_augment, "res_weight": args.res_weight,
                "clip_weight": args.clip_weight,
                "training_seed": args.seed, "z0_epoch_resample": bool(args.z0_epoch_resample),
                "pairs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.pairs},
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "clean_action_parity_max": parity_max}, args.out)
    print(json.dumps({"out": str(args.out), "train_states": train_states,
                      "val_states": val_states, "n_pairs": len(data),
                      "best_epoch": best_epoch, "best_val": best_val}))


if __name__ == "__main__":
    main()
