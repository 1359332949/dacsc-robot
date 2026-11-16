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


def _translate_reflect(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    pad = max(abs(dx), abs(dy))
    if pad == 0:
        return arr.copy()
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    y0 = pad + dy; x0 = pad + dx
    return padded[y0:y0 + h, x0:x0 + w].copy()


def geo_aug(img: np.ndarray, kind: str, value: float) -> np.ndarray:
    """True 2D geometry proxy.

    translate: reflect padding (no black border);
    crop: center crop fraction then resize;
    zoom: center crop 1/value then resize (view really narrows).
    """
    arr = np.asarray(img, dtype=np.uint8); h, w = arr.shape[:2]
    if kind == "translate":
        return _translate_reflect(arr, int(value), int(value * 0.5))
    if kind == "crop":
        m = int(round(w * value)); box = ((w - m) // 2, (h - m) // 2,
                                         (w + m) // 2, (h + m) // 2)
        return np.asarray(Image.fromarray(arr).crop(box).resize((w, h), Image.BILINEAR))
    if kind == "zoom":
        if value < 1.0:
            raise ValueError("zoom proxy only supports zoom-in factors >= 1")
        m = int(round(w / value)); box = ((w - m) // 2, (h - m) // 2,
                                         (w + m) // 2, (h + m) // 2)
        return np.asarray(Image.fromarray(arr).crop(box).resize((w, h), Image.BILINEAR))
    raise ValueError(f"unknown geo kind {kind}")


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
    geo_groups = {
        "translate": [("translate", -8), ("translate", 8)],
        "crop": [("crop", 0.95)],
        "zoom": [("zoom", 1.05), ("zoom", 1.10)],
    }
    geo_cfgs = [x for xs in geo_groups.values() for x in xs]
    out = {"photo": {}, "geo": {}, "geo_groups": {}, "num_states": len(valid),
           "views": views, "photo_cfgs": photo_cfgs, "geo_cfgs": geo_cfgs}
    for target in views:
        other = [v for v in views if v != target]
        photo_scores, geo_scores = [], []
        photo_state, geo_state = [], []
        group_scores = {g: [] for g in geo_groups}
        group_state = {g: [] for g in geo_groups}
        for state in valid:
            d = args.dir / "images" / f"state_{state:03d}"
            raw = {v: np.asarray(Image.open(d / f"shift_{v}.png")) for v in views}
            base = z[f"shift_{target}"][pos[state]]
            ps, gs = [], []
            for kind, factor in photo_cfgs:
                imgs = dict(raw); imgs[target] = photo_aug(raw[target], kind, factor)
                f = features(policy, imgs, views, manifest["clean_language"])[target]
                score = np.abs(f - base).mean(axis=0); photo_scores.append(score); ps.append(score)
            grp = {g: [] for g in geo_groups}
            for kind, value in geo_cfgs:
                imgs = dict(raw); imgs[target] = geo_aug(raw[target], kind, value)
                f = features(policy, imgs, views, manifest["clean_language"])[target]
                score = np.abs(f - base).mean(axis=0); geo_scores.append(score); gs.append(score)
                for g, configs in geo_groups.items():
                    if (kind, value) in configs:
                        grp[g].append(score)
            for g in geo_groups:
                group_scores[g].append(np.mean(grp[g], axis=0))
                group_state[g].append(np.mean(grp[g], axis=0))
            photo_state.append(np.mean(ps, axis=0)); geo_state.append(np.mean(gs, axis=0))
        photo = np.mean(photo_scores, axis=0); geo = np.mean(geo_scores, axis=0)
        out["photo"][target] = photo.tolist(); out["geo"][target] = geo.tolist()
        out.setdefault("geo_groups", {})[target] = {
            g: np.mean(group_scores[g], axis=0).tolist() for g in geo_groups}
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
