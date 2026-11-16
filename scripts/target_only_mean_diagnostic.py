"""Offline direction diagnostic for source-calibrated target-only Mean."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from dacsc.target_only_mean import ConditionalSourceBaseline, TargetOnlyMeanEstimator  # noqa: E402


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def pooled_visual(policy, patches, instruction):
    ext = policy.representation_extractor
    p = {v: torch.as_tensor(a, dtype=torch.float32, device=next(policy.parameters()).device).unsqueeze(0)
         for v, a in patches.items()}
    lang, pooled = ext.language_encoder(instruction)
    visual = ext.cross_attention_resampler(p, pooled)
    return visual.mean(dim=1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--momentum", type=float, default=0.5)
    args = ap.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    dev = [i for i in manifest["dev_indices"] if i in by_state]
    remaining = dev[10:]
    fit_ids = remaining[:len(remaining)//2]
    eval_ids = remaining[len(remaining)//2:]
    policy = load_policy(args.checkpoint, device="cuda"); policy.eval()
    views = manifest["policy_views"]
    def feat(state, kind):
        return {v: z[f"{kind}_{v}"][by_state[state]] for v in views}
    clean = {s: pooled_visual(policy, feat(s, "clean"), args.clean_language) for s in by_state}
    shift = {s: pooled_visual(policy, feat(s, "shift"), args.clean_language) for s in by_state}
    states = {s: z["state8"][by_state[s]] for s in by_state}
    source = ConditionalSourceBaseline(l2=args.l2)
    source.fit(np.stack([states[s] for s in fit_ids]), np.stack([clean[s] for s in fit_ids]))
    paired = np.mean([shift[s] - clean[s] for s in fit_ids], axis=0)
    per_state = []
    for s in eval_ids:
        # Each eval state is a separate target episode: no cross-state EMA.
        estimator = TargetOnlyMeanEstimator(source, momentum=args.momentum)
        delta = estimator.update(torch.from_numpy(shift[s][None, :]).float(), states[s][None, :])
        d = delta[0].numpy()
        cos = float(np.dot(d, paired) / (np.linalg.norm(d) * np.linalg.norm(paired) + 1e-12))
        per_state.append({"state": int(s), "cos_target_only_vs_paired": cos,
                          "target_only_norm": float(np.linalg.norm(d)),
                          "paired_norm": float(np.linalg.norm(paired))})
    out = {"capture": str(args.capture), "capture_manifest_sha256": sha(args.capture / "manifest.json"),
           "features_sha256": sha(args.capture / "features.npz"),
           "fit_ids": fit_ids, "eval_ids": eval_ids, "l2": args.l2,
           "per_state": per_state,
           "mean_cosine": float(np.mean([r["cos_target_only_vs_paired"] for r in per_state]))}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"fit": fit_ids, "eval": eval_ids, "mean_cosine": out["mean_cosine"]}, indent=2))


if __name__ == "__main__":
    main()
