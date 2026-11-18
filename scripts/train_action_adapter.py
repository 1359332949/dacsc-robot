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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--id-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
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
    lossf = nn.HuberLoss()
    def one(rec):
        rep = build_rep(adapter, rec, device)
        z0 = torch.tensor(rec["z0"], dtype=torch.float32, device=device).unsqueeze(0)
        target = torch.tensor(rec["clean_normalized_actions"], dtype=torch.float32, device=device)
        pred = sample_from_z0(policy, rep, z0)
        pair = lossf(pred, target)
        c_rep = clean_rep(adapter, rec, device)
        c_pred = sample_from_z0(policy, c_rep, z0)
        ident = lossf(c_pred, target)
        return pair + args.id_weight * ident, pair, ident
    for epoch in range(args.epochs):
        adapter.train(); tr = 0.0
        for rec in train:
            opt.zero_grad(); loss, pair, ident = one(rec); loss.backward(); opt.step()
            tr += float(loss.detach())
        adapter.eval(); va = 0.0
        with torch.no_grad():
            for rec in val:
                loss, _, _ = one(rec); va += float(loss)
        print(json.dumps({"epoch": epoch, "train": tr / len(train), "val": va / len(val)}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "rank": args.rank, "dim": 768,
                "val_states": sorted(val_states), "train_states": sorted(set(states) - val_states),
                "id_weight": args.id_weight}, args.out)
    print(json.dumps({"out": str(args.out), "train_states": sorted(set(states) - val_states),
                      "val_states": sorted(val_states), "n_pairs": len(data)}))


if __name__ == "__main__":
    main()
