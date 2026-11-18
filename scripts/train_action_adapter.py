"""Train a zero-initialized action-consistent adapter on frozen CoRP pairs."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
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
    state = torch.tensor(rec["state_token"], dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    adapted = adapter(visual)
    tokens = torch.cat([adapted, lang, state], dim=1)
    return Representation(tokens=tokens, pooled_language=pooled, action_dtype=torch.float32)


def clean_rep(adapter, rec, device):
    visual = torch.tensor(rec["clean_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec["state_token"], dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    adapted = adapter(visual)
    return Representation(tokens=torch.cat([adapted, lang, state], dim=1),
                          pooled_language=pooled, action_dtype=torch.float32)


def clean_frozen_rep(rec, device):
    visual = torch.tensor(rec["clean_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec["state_token"], dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    return Representation(tokens=torch.cat([visual, lang, state], dim=1),
                          pooled_language=pooled, action_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--id-weight", type=float, default=1.0)
    ap.add_argument("--grip-weight", type=float, default=1.0)
    ap.add_argument("--z0-augment", type=int, default=1)
    ap.add_argument("--res-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z0-epoch-resample", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    data = json.loads(args.pairs.read_text())["records"]
    data = [r for r in data if r.get("pairing_pass")]
    states = sorted({int(r["state_id"]) for r in data})
    if len(states) < 2:
        raise ValueError("need at least two source states for grouped train/val")
    val_states = {states[-1]}
    train = [r for r in data if int(r["state_id"]) not in val_states]
    val = [r for r in data if int(r["state_id"]) in val_states]
    if not train or not val:
        raise ValueError("empty split")
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
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
        pair_total = 0.0; ident_total = 0.0; n = 0
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
            n += 1
        pair = pair_total / n; ident = ident_total / n
        loss = pair + args.id_weight * ident
        if args.res_weight > 0:
            visual = torch.tensor(rec["shift_visual_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
            adapted_visual = adapter(visual)
            res = (adapted_visual - visual).square().mean()
            loss = loss + args.res_weight * res
        return loss, pair, ident
    for epoch in range(args.epochs):
        adapter.train(); tr = 0.0
        for rec in train:
            opt.zero_grad(); loss, pair, ident = one(rec, epoch=epoch, resample=args.z0_epoch_resample); loss.backward(); opt.step()
            tr += float(loss.detach())
        adapter.eval(); va = 0.0
        with torch.no_grad():
            for rec in val:
                loss, _, _ = one(rec, epoch=0, resample=False); va += float(loss)
        print(json.dumps({"epoch": epoch, "train": tr / len(train), "val": va / len(val)}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "rank": args.rank, "dim": 768,
                "val_states": sorted(val_states), "train_states": sorted(set(states) - val_states),
                "id_weight": args.id_weight, "grip_weight": args.grip_weight,
                "z0_augment": args.z0_augment, "res_weight": args.res_weight,
                "training_seed": args.seed, "z0_epoch_resample": bool(args.z0_epoch_resample),
                "pairs_sha256": __import__("hashlib").sha256(args.pairs.read_bytes()).hexdigest(),
                "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest()}, args.out)
    print(json.dumps({"out": str(args.out), "train_states": sorted(set(states) - val_states),
                      "val_states": sorted(val_states), "n_pairs": len(data)}))


if __name__ == "__main__":
    main()
