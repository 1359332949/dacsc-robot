"""Offline action-consistency evaluation for a trained adapter."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402


def action(policy, adapter, rec, key, device):
    v = torch.tensor(rec[key], dtype=torch.float32, device=device).unsqueeze(0)
    lang = torch.tensor(rec["language_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.tensor(rec["state_token"], dtype=torch.float32, device=device).unsqueeze(0)
    pooled = torch.tensor(rec["pooled_language"], dtype=torch.float32, device=device).unsqueeze(0)
    rep = Representation(tokens=torch.cat([adapter(v), lang, state], dim=1),
                         pooled_language=pooled, action_dtype=torch.float32)
    z0 = torch.tensor(rec["z0"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return sample_from_z0(policy, rep, z0)[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ck = torch.load(args.adapter, map_location="cpu")
    trained = ResidualVisualAdapter(768, ck["rank"]).to(device)
    trained.load_state_dict(ck["adapter"]); trained.eval()
    zero = ResidualVisualAdapter(768, ck["rank"]).to(device); zero.eval()
    # zero adapter: output layer zero-init
    for p in zero.parameters(): p.requires_grad_(False)
    data = json.loads(args.pairs.read_text())["records"]
    rows = []
    for r in data:
        if not r.get("pairing_pass"): continue
        target = np.asarray(r["clean_normalized_actions"], dtype=float)
        a0 = action(policy, zero, r, "shift_visual_tokens", device)
        a1 = action(policy, trained, r, "shift_visual_tokens", device)
        def err(a): return {"rmse": float(np.sqrt(np.mean((a-target)**2))),
                            "xyz": float(np.sqrt(np.mean((a[:,:3]-target[:,:3])**2))),
                            "grip": float(np.sqrt(np.mean((a[:,6]-target[:,6])**2)))}
        rows.append({"state_id": r["state_id"], "chunk_index": r["chunk_index"],
                     "frozen": err(a0), "adapted": err(a1)})
    summary = {}
    for mode in ("frozen", "adapted"):
        summary[mode] = {k: float(np.mean([x[mode][k] for x in rows])) for k in ("rmse", "xyz", "grip")}
    report = {"summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
