"""V1.2 s_aug: target-image augmentation sensitivity, per view.

Definitions are frozen before running:
  s_aug_photo: brightness 0.8/1.2, contrast 0.8/1.2, saturation 0.8/1.2
  s_aug_geo:   translate +/-8 px, crop 95% + resize, scale 1.05/0.95

Only the target view is augmented; the other view stays original.  The full
two-view mapping enters CoRP's prepare_observation, matching deployment input.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402


def policy_image(raw: np.ndarray, size: int) -> torch.Tensor:
    img = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
    if img.shape[:2] != (size, size):
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR))
    return torch.from_numpy(np.ascontiguousarray(img).copy()).unsqueeze(0)


def features(policy, raw_by_view: dict[str, np.ndarray], views: list[str], instruction: str):
    size = int(policy.config["image_size"]); device = next(policy.parameters()).device
    imgs = {v: policy_image(raw_by_view[v], size).to(device) for v in views}
    batch = prepare_observation(
        {"images": imgs, "state": torch.zeros(1, policy.config["state_dim"], device=device),
         "instruction": [instruction]}, config=policy.config,
        normalizer=policy.normalizer, device=device)
    with torch.no_grad():
        p = policy.representation_extractor.visual_encoder(
            torch.cat([batch["images"][v] for v in views], dim=0))
    return {v: p[i].detach().cpu().numpy() for i, v in enumerate(views)}


def photo_aug(img: np.ndarray, kind: str, factor: float) -> np.ndarray:
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if kind == "brightness":
        im = ImageEnhance.Brightness(im).enhance(factor)
    elif kind == "contrast":
        im = ImageEnhance.Contrast(im).enhance(factor)
    elif kind == "saturation":
        im = ImageEnhance.Color(im).enhance(factor)
    return np.asarray(im)


def geo_aug(img: np.ndarray, kind: str, value: float) -> np.ndarray:
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)); w, h = im.size
    if kind == "translate":
        dx = int(value)
        im = im.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, value * 0.5), resample=Image.BILINEAR)
    elif kind == "crop":
        m = int(round(w * value)); box = ((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)
        im = im.crop(box).resize((w, h), Image.BILINEAR)
    elif kind == "scale":
        nw, nh = int(w * value), int(h * value)
        im = im.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    z = np.load(args.dir / "features.npz")
    manifest = json.loads((args.dir / "manifest.json").read_text())
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    pos = {s: j for j, s in enumerate(valid)}
    views = manifest["policy_views"]
    policy = load_policy(CKPT, device=args.device); policy.eval()
    photo_cfgs = [(k, f) for k in ("brightness", "contrast", "saturation") for f in (0.8, 1.2)]
    geo_cfgs = [("translate", -8), ("translate", 8), ("crop", 0.95), ("scale", 1.05), ("scale", 0.95)]
    out = {"photo": {}, "geo": {}, "num_states": len(valid),
           "views": views, "photo_cfgs": photo_cfgs, "geo_cfgs": geo_cfgs}
    for target in views:
        other = [v for v in views if v != target]
        photo_scores, geo_scores = [], []
        photo_state, geo_state = [], []
        for state in valid:
            d = args.dir / "images" / f"state_{state:03d}"
            raw = {v: np.asarray(Image.open(d / f"shift_{v}.png")) for v in views}
            base = z[f"shift_{target}"][pos[state]]
            ps, gs = [], []
            for kind, factor in photo_cfgs:
                imgs = dict(raw); imgs[target] = photo_aug(raw[target], kind, factor)
                f = features(policy, imgs, views, manifest["clean_language"])[target]
                score = np.abs(f - base).mean(axis=0); photo_scores.append(score); ps.append(score)
            for kind, value in geo_cfgs:
                imgs = dict(raw); imgs[target] = geo_aug(raw[target], kind, value)
                f = features(policy, imgs, views, manifest["clean_language"])[target]
                score = np.abs(f - base).mean(axis=0); geo_scores.append(score); gs.append(score)
            photo_state.append(np.mean(ps, axis=0)); geo_state.append(np.mean(gs, axis=0))
        photo = np.mean(photo_scores, axis=0); geo = np.mean(geo_scores, axis=0)
        out["photo"][target] = photo.tolist(); out["geo"][target] = geo.tolist()
        out.setdefault("photo_by_state", {})[target] = np.stack(photo_state).tolist()
        out.setdefault("geo_by_state", {})[target] = np.stack(geo_state).tolist()
        # dev/held indices are state ids; map through valid order
        for split_name, idxs in (("dev", manifest["dev_indices"]), ("heldout", manifest["heldout_indices"])):
            pidx = [pos[i] for i in idxs if i in pos]
            if pidx:
                out.setdefault(f"photo_{split_name}", {})[target] = np.mean(np.stack(photo_state)[pidx], axis=0).tolist()
                out.setdefault(f"geo_{split_name}", {})[target] = np.mean(np.stack(geo_state)[pidx], axis=0).tolist()
    (args.dir / "s_aug.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({v: {"photo_mean": float(np.mean(out["photo"][v])),
                          "geo_mean": float(np.mean(out["geo"][v]))} for v in views}, indent=2))


if __name__ == "__main__":
    main()
