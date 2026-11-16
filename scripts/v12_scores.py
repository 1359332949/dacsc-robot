"""V1.2 scores from captured paired DINO features.

s_pair : mean |F_clean - F_shift| per embedding dimension
s_stat : cross-state std of shift features per dimension (proxy, not nuisance)
s_aug  : optional augmentation sensitivity of target shift images
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
import torch  # noqa: E402


def policy_image(raw: np.ndarray, size: int) -> torch.Tensor:
    img = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
    if img.shape[:2] != (size, size):
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR))
    return torch.from_numpy(np.ascontiguousarray(img).copy()).unsqueeze(0)


def dino(policy, raw, view, instruction):
    size = int(policy.config["image_size"])
    device = next(policy.parameters()).device
    img = policy_image(raw, size).to(device)
    batch = prepare_observation({"images": {view: img},
                                 "state": torch.zeros(1, policy.config["state_dim"], device=device),
                                 "instruction": [instruction]},
                                config=policy.config, normalizer=policy.normalizer, device=device)
    with torch.no_grad():
        p = policy.representation_extractor.visual_encoder(batch["images"][view])
    return p[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    z = np.load(args.dir / "features.npz")
    manifest = json.loads((args.dir / "manifest.json").read_text())
    views = manifest["policy_views"]
    # normalize per-view absolute difference
    pair_scores = {}
    stat_scores = {}
    for v in views:
        c = z[f"clean_{v}"].astype(np.float64); s = z[f"shift_{v}"].astype(np.float64)
        pair_scores[v] = np.abs(c - s).mean(axis=(0, 1))
        stat_scores[v] = s.std(axis=0).mean(axis=(0, 1))
    pair = np.mean([pair_scores[v] for v in views], axis=0)
    stat = np.mean([stat_scores[v] for v in views], axis=0)

    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    dev = [i for i in manifest["dev_indices"] if i in valid]
    held = [i for i in manifest["heldout_indices"] if i in valid]
    pos = {state: j for j, state in enumerate(valid)}
    def sub_score(idx):
        vals = []
        for v in views:
            pidx = [pos[i] for i in idx]
            c = z[f"clean_{v}"][pidx].astype(np.float64); s = z[f"shift_{v}"][pidx].astype(np.float64)
            vals.append(np.abs(c - s).mean(axis=(0, 1)))
        return np.mean(vals, axis=0)
    dev_score = sub_score(dev); held_score = sub_score(held)
    mid = len(dev) // 2
    half_a, half_b = sub_score(dev[:mid]), sub_score(dev[mid:])
    rel = {
        "dev_vs_held_spearman": float(spearmanr(dev_score, held_score).statistic),
        "split_half_spearman": float(spearmanr(half_a, half_b).statistic),
        "view_correlation": float(spearmanr(pair_scores[views[0]], pair_scores[views[1]]).statistic) if len(views) > 1 else None,
    }
    topk = 20
    top = np.argsort(-pair)[:topk]
    out = {
        "num_states": len(valid), "dev": len(dev), "heldout": len(held), "skipped": len(manifest["records"]) - len(valid),
        "pair_mean": float(pair.mean()), "pair_std": float(pair.std()),
        "top_dims": top.tolist(), "top_values": pair[top].tolist(),
        "reliability": rel,
        "view_pair_scores": {v: pair_scores[v].tolist() for v in views},
        "stat": stat.tolist(),
    }
    if args.augment:
        policy = load_policy(CKPT, device=args.device); policy.eval()
        aug_scores = []
        for v in views:
            vals = []
            for i in range(len(dev) + len(held)):
                idx = (dev + held)[i]
                raw = np.asarray(Image.open(args.dir / "images" / f"state_{idx:03d}" / f"shift_{v}.png"))
                base = dino(policy, raw, v, manifest["clean_language"])
                for factor in (0.8, 1.2):
                    img = ImageEnhance.Brightness(Image.fromarray(raw)).enhance(factor)
                    aug_scores.append(np.abs(dino(policy, np.asarray(img), v, manifest["clean_language"]) - base).mean(axis=(0, 1)))
            pair_scores[v] = np.mean(vals, axis=0) if vals else pair_scores[v]
        aug = np.mean(aug_scores, axis=0)
        out["aug"] = aug.tolist()
        out["aug_vs_pair_spearman"] = float(spearmanr(aug, pair).statistic)
    (args.dir / "scores.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.dir / "scores.npz", pair=pair, stat=stat,
                        **{f"pair_{v}": pair_scores[v] for v in views})
    print(json.dumps({k: out[k] for k in ("num_states", "dev", "heldout", "reliability",
                                          "pair_mean", "top_dims")}, indent=2))
    if "aug_vs_pair_spearman" in out:
        print("aug_vs_pair_spearman", out["aug_vs_pair_spearman"])


if __name__ == "__main__":
    main()
