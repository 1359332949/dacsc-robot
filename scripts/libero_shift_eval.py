"""Photometric-shift evaluation on standard LIBERO (thin driver).

The project's LIBERO-Plus adapter (scripts/eval_corp_libero_plus.py) selects
tasks through LIBERO-Plus's task_classification.json, which does not describe
the 40 standard LIBERO tasks. For a like-for-like clean-vs-shift comparison on
the released CoRP LIBERO checkpoint we therefore reuse, without reimplementing:

  * official suite/rollout/success logic: CoRP/scripts/evaluate_libero.py
    (configure_libero, evaluate_suite, MAX_STEPS, settling steps and the
    official per-task seed and initial-state index);
  * the project's observation-side photometric wrapper: PhotometricPolicy from
    scripts/eval_corp_libero_plus.py (apply_corruption on resized RGB);
  * the released checkpoint loader: corp.load_policy.

Task selection is the official default (all tasks of the suite). One condition
per call, so clean and shifted runs use identical task ids, seeds and initial
states and can be paired episode by episode.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import MAX_STEPS, configure_libero, evaluate_suite  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.corp_images import SHIFTS  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream, verify_release_checkpoint  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from scripts.eval_corp_libero_plus import PhotometricPolicy  # noqa: E402

# PDF-style test-time augmentation voting (arXiv:2604.18107): average action
# chunks over a few deterministic appearance variants of the SAME observation.
# 4-tuples (brightness, contrast, saturation, hue). The default set is the
# generic PDF-style augmentation vote; the "calib_*" sets add variants that
# partially invert the benchmark corruption (candidate variant, to be reviewed
# before it is claimed as a contribution).
VOTE_VARIANTS = (
    (1.0, 1.0, 1.0, 0.0),
    (0.9, 1.0, 1.0, 0.0),
    (1.1, 1.0, 1.0, 0.0),
    (1.0, 0.9, 1.0, 0.0),
    (1.0, 1.1, 1.0, 0.0),
)
# reference clean colour statistics, measured with libero_photometric_stats.py
# (2 tasks x 3 frames): channel order R>G>B, luminance std ~0.20, saturation ~0.13.
PHOTO_REF = {
    "channel_mean": (0.503, 0.471, 0.436),
    "luminance_mean": 0.477,
    "luminance_std": 0.197,
    "saturation_mean": 0.129,
}

CALIB_VARIANTS = {
    "color_jitter": (
        (1.0, 1.0, 1.0, 0.0),
        (0.7, 0.7, 0.5, 0.0),
        (0.85, 0.85, 1.5, 0.0),
        (0.6, 1.0, 0.6, 0.0),
        (1.0, 0.7, 1.5, 0.0),
    ),
    "contrast": (
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.3, 1.0, 0.0),
        (1.0, 1.6, 1.0, 0.0),
        (1.0, 0.8, 1.0, 0.0),
        (1.15, 1.0, 1.0, 0.0),
    ),
    "hue": (
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, -0.25),
        (1.0, 1.0, 1.0, -0.5),
        (1.0, 1.0, 1.0, 0.25),
        (1.0, 1.0, 1.0, 0.5),
    ),
    "hue_inverse": ((1.0, 1.0, 1.0, -0.5),),
    "hue_mild": (
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, -0.25),
        (1.0, 1.0, 1.0, 0.25),
    ),
}


def detect_family(images: dict[str, torch.Tensor]) -> str:
    """Classify the photometric corruption family from observation statistics."""
    image = next(iter(images.values()))
    x = image.to(torch.float32)
    if x.max() > 2:
        x = x / 255.0
    nchw = x.permute(0, 3, 1, 2).clamp(0, 1)
    r, g, b = nchw[0, 0], nchw[0, 1], nchw[0, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mx = nchw[0].max(dim=0).values
    mn = nchw[0].min(dim=0).values
    saturation = float(((mx - mn) / mx.clamp_min(1e-6)).mean())
    lum_std = float(luminance.std())
    lum_mean = float(luminance.mean())
    ref = PHOTO_REF
    if saturation < 0.02:
        return "grayscale"
    if lum_std < 0.6 * ref["luminance_std"]:
        return "low_contrast"
    if lum_mean > ref["luminance_mean"] + 0.12:
        return "color_jitter"
    if float(b.mean()) - float(r.mean()) > 0.02:
        return "hue_shift"
    return "none"


FAMILY_VARIANTS = {
    "hue_shift": CALIB_VARIANTS["hue"],
    "low_contrast": CALIB_VARIANTS["contrast"],
    "color_jitter": CALIB_VARIANTS["color_jitter"],
    "grayscale": None,
    "none": None,
}


class PSCTTAWrapper:
    """Full PSC test-time training wrapper (updates LayerNorm affines only)."""

    def __init__(self, policy, stats, device, lr, steps, lambda_res, interval=8, weighting="none", buffer_size=64):
        if weighting == "identifiability":
            from dacsc.psc_weighted import WeightedPSCTTA as _TTA
        else:
            from dacsc.psc_tta import PSCTTA as _TTA

        self.policy = policy
        self.device = device
        self.config, self.normalizer = policy.config, policy.normalizer
        self.tta = _TTA(
            policy, stats, lr=lr, steps=steps, lambda_res=lambda_res,
            update_interval=interval, buffer_size=buffer_size,
        )
        self.buffer: list[dict] = []
        self.buffer_size = self.tta.buffer_size

    def _batch(self, observations):
        batches = [
            prepare_observation(
                observation,
                config=self.config,
                normalizer=self.normalizer,
                device=self.device,
            )
            for observation in observations
        ]
        images = {
            name: torch.cat([batch["images"][name] for batch in batches], dim=0)
            for name in batches[0]["images"]
        }
        state = torch.cat([batch["state"] for batch in batches], dim=0)
        instruction = [item for batch in batches for item in batch["instruction"]]
        return {"images": images, "state": state, "instruction": instruction}

    def predict(self, observation):
        self.buffer.append(
            {
                "images": {k: v.clone() for k, v in observation["images"].items()},
                "state": torch.as_tensor(observation["state"]).clone(),
                "instruction": list(observation["instruction"]),
            }
        )
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        if len(self.buffer) >= 2:
            batch = self._batch(self.buffer)
            for _ in range(self.tta.steps):
                self.tta.update(batch)
        return self.policy.predict(observation)


class VotePolicy:
    """Average chunks over brightness/contrast variants (no labels, no grads)."""

    def __init__(self, policy, variants, device, auto_family=False):
        from torchvision.transforms import functional as TVF

        self.policy = policy
        self.variants = tuple(variants)
        self.device = device
        self._tvf = TVF
        self.config, self.normalizer = policy.config, policy.normalizer
        self.auto_family = auto_family
        self.last_family = None

    def _variant(self, image, factors):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = image.to(self.device).float()
        if x.max() > 2:  # uint8
            x = x / 255.0
        nchw = x.permute(0, 3, 1, 2).clamp(0, 1)
        b, c, ss, h = factors
        if b != 1.0:
            nchw = self._tvf.adjust_brightness(nchw, b)
        if c != 1.0:
            nchw = self._tvf.adjust_contrast(nchw, c)
        if ss != 1.0:
            nchw = self._tvf.adjust_saturation(nchw, ss)
        if h != 0.0:
            nchw = self._tvf.adjust_hue(nchw, h)
        # hand back the same format the environment produced: uint8 NHWC
        return (nchw.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)

    def predict(self, observation):
        variants = self.variants
        if self.auto_family:
            family = detect_family(observation["images"])
            variants = FAMILY_VARIANTS.get(family)
            if variants is None:
                return self.policy.predict(observation)
            self.last_family = family
        chunks = []
        for factors in variants:
            variant = dict(observation)
            variant["images"] = {
                name: self._variant(image, factors)
                for name, image in observation["images"].items()
            }
            chunks.append(self.policy.predict(variant))
        return torch.stack(chunks, dim=0).mean(dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    parser.add_argument("--suite", choices=list(MAX_STEPS), required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--shift", choices=SHIFTS, default="none")
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--libero-src", type=Path, default=_CORP_REPO / "third_party" / "LIBERO")
    parser.add_argument("--libero-config-path", type=Path, default=_CORP_REPO / ".cache" / "libero")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--psc-beta", type=float, default=0.0)
    parser.add_argument("--psc-tta-stats", type=Path, default=None)
    parser.add_argument("--psc-weighting", choices=["none", "identifiability"], default="none")
    parser.add_argument("--residual-shrink", type=Path, default=None)
    parser.add_argument("--residual-gamma-min", type=float, default=0.3)
    parser.add_argument("--residual-tau-s", type=float, default=None)
    parser.add_argument("--full-mean-align", type=Path, default=None)
    parser.add_argument("--full-mean-cmax", type=float, default=3.0)
    parser.add_argument("--cond-align", type=Path, default=None)
    parser.add_argument("--cond-cmax", type=float, default=3.0)
    parser.add_argument("--null-align", type=Path, default=None)
    parser.add_argument("--null-align-alpha", type=float, default=1.0)
    parser.add_argument("--null-align-cmax", type=float, default=3.0)
    parser.add_argument("--action-align", type=Path, default=None)
    parser.add_argument("--action-align-alpha", type=float, default=1.0)
    parser.add_argument("--action-align-cmax", type=float, default=3.0)
    parser.add_argument("--residual-affine", type=Path, default=None)
    parser.add_argument("--affine-start-step", type=int, default=None)
    parser.add_argument("--affine-alpha", type=float, default=1.0)
    parser.add_argument("--hook-block", type=int, default=0)
    parser.add_argument("--adaptive", type=Path, default=None)
    parser.add_argument("--residual-debias", type=Path, default=None)
    parser.add_argument("--debias-ratio-min", type=float, default=1.15)
    parser.add_argument("--topk-ratio-min", type=float, default=1.2)
    parser.add_argument("--topk-ratio-max", type=float, default=2.0)
    parser.add_argument("--residual-topk", type=Path, default=None)
    parser.add_argument("--residual-k", type=int, default=20)
    parser.add_argument("--interface-calib", type=Path, default=None)
    parser.add_argument("--write-calib-stats", type=Path, default=None)
    parser.add_argument("--write-calib-mask", type=Path, default=None)
    parser.add_argument("--write-calib-expo", type=float, default=1.0)
    parser.add_argument("--write-calib-mode", choices=["support", "diag"], default="support")
    parser.add_argument("--write-calib-lo", type=float, default=0.85)
    parser.add_argument("--write-calib-hi", type=float, default=1.15)
    parser.add_argument("--write-calib-all-steps", action="store_true")
    parser.add_argument("--psc-tta-lr", type=float, default=1e-3)
    parser.add_argument("--psc-tta-interval", type=int, default=8)
    parser.add_argument("--psc-tta-buffer", type=int, default=64)
    parser.add_argument("--psc-tta-steps", type=int, default=1)
    parser.add_argument("--psc-tta-lambda", type=float, default=1.0)
    parser.add_argument(
        "--classification",
        type=Path,
        default=None,
        help="LIBERO-Plus task_classification.json; enables per-category seeding",
    )
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--vote-k", type=int, default=0)
    parser.add_argument("--auto-family", action="store_true")
    parser.add_argument(
        "--vote-calib",
        choices=["none", "contrast", "hue", "hue_inverse", "hue_mild"],
        default="none",
    )
    parser.add_argument(
        "--psc-stats",
        type=Path,
        default=None,
        help="optional source-support stats (build_libero_psc_stats.py)",
    )
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    upstream = verify_upstream(["CoRP"])
    checkpoint_digest = verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if args.classification is not None and args.tasks is None:
        if args.per_category <= 0:
            raise ValueError("--classification requires positive --per-category")
        import random as _random

        suite_meta = json.loads(args.classification.read_text())[args.suite]
        groups: dict[str, list] = {}
        for item in suite_meta:
            if args.categories and item["category"] not in args.categories:
                continue
            groups.setdefault(item["category"], []).append(item)
        if not groups:
            raise ValueError("no tasks selected from classification")
        rng = _random.Random(args.seed)
        selected = []
        for category, items in sorted(groups.items()):
            if len(items) < args.per_category:
                raise ValueError(f"insufficient tasks in {category}")
            selected.extend(rng.sample(items, args.per_category))
        task_suite = benchmark.get_benchmark_dict()[args.suite]()
        names = {name: i for i, name in enumerate(task_suite.get_task_names())}
        missing = [item["name"] for item in selected if item["name"] not in names]
        if missing:
            raise ValueError(f"{len(missing)} classification tasks absent from suite, e.g. {missing[0]}")
        args.tasks = ",".join(str(names[item["name"]]) for item in selected)
        print(f"selected {len(selected)} tasks from {len(groups)} categories", flush=True)

    policy = load_policy(args.checkpoint, device=args.device)
    projector = None
    if args.psc_stats is not None:
        from dacsc.psc import install, load_stats

        projector = install(policy, load_stats(args.psc_stats), beta=args.psc_beta)
    residual_shrink = None
    if args.full_mean_align is not None:
        from dacsc.full_mean_alignment import FullMeanAlignment

        residual_shrink = FullMeanAlignment(
            policy.action_generator.blocks[0].attention,
            json.loads(args.full_mean_align.read_text()),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            c_max=args.full_mean_cmax,
        )
    elif args.cond_align is not None:
        from dacsc.conditional_alignment import ConditionalAlignment

        residual_shrink = ConditionalAlignment(
            policy.action_generator.blocks[0].attention,
            str(args.cond_align),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            c_max=args.cond_cmax,
        )
    elif args.null_align is not None:
        from dacsc.null_space_alignment import NullSpaceAlignment

        residual_shrink = NullSpaceAlignment(
            policy.action_generator.blocks[0].attention,
            json.loads(args.null_align.read_text()),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            alpha=args.null_align_alpha,
            c_max=args.null_align_cmax,
        )
    elif args.action_align is not None:
        from dacsc.action_subspace_alignment import ActionSubspaceAlignment

        residual_shrink = ActionSubspaceAlignment(
            policy.action_generator.blocks[0].attention,
            json.loads(args.action_align.read_text()),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            alpha=args.action_align_alpha,
            c_max=args.action_align_cmax,
        )
    elif args.residual_affine is not None:
        from dacsc.residual_affine import ResidualAffineCalibration

        spectrum = json.loads(args.residual_affine.read_text())
        residual_shrink = ResidualAffineCalibration(
            policy.action_generator.blocks[args.hook_block].attention,
            json.loads(args.residual_shrink.read_text()),
            spectrum,
            k=args.residual_k,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            start_step=args.affine_start_step,
            alpha=args.affine_alpha,
        )
    elif args.adaptive is not None:
        from dacsc.adaptive_residual import AdaptiveResidualMapping

        spectrum = json.loads(args.adaptive.read_text())
        residual_shrink = AdaptiveResidualMapping(
            policy.action_generator.blocks[0].attention,
            json.loads(args.residual_shrink.read_text()),
            spectrum,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            k=args.residual_k,
        )
    elif args.residual_debias is not None:
        from dacsc.residual_debias import ResidualDebias

        spectrum = json.loads(args.residual_debias.read_text())
        residual_shrink = ResidualDebias(
            policy.action_generator.blocks[0].attention,
            json.loads(args.residual_shrink.read_text()) if args.residual_shrink is not None else stats,
            spectrum,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            ratio_min=args.debias_ratio_min,
        )
    elif args.residual_topk is not None:
        from dacsc.residual_shrink_topk import TopKResidualShrink

        spectrum = json.loads(args.residual_topk.read_text())
        residual_shrink = TopKResidualShrink(
            policy.action_generator.blocks[0].attention,
            json.loads(args.residual_shrink.read_text()) if args.residual_shrink is not None else stats,
            spectrum,
            k=args.residual_k,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            gamma_min=args.residual_gamma_min,
            ratio_min=args.topk_ratio_min,
            ratio_max=args.topk_ratio_max,
        )
    elif args.residual_shrink is not None:
        from dacsc.residual_shrink import ResidualShrink

        stats = json.loads(args.residual_shrink.read_text())
        residual_shrink = ResidualShrink(
            policy.action_generator.blocks[0].attention,
            stats,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=int(policy.config["action_horizon"]),
            gamma_min=args.residual_gamma_min,
            tau_s=args.residual_tau_s,
        )
    interface_calibration = None
    if args.interface_calib is not None:
        from dacsc.interface_calibration import InterfaceCalibration

        calib = json.loads(args.interface_calib.read_text())
        interface_calibration = InterfaceCalibration(
            policy.action_generator.blocks[0].attention,
            calib,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
        )
    write_calibration = None
    if args.write_calib_stats is not None:
        from dacsc.write_calibration import WriteCalibration

        stats = json.loads(args.write_calib_stats.read_text())
        mask_indices = None
        if args.write_calib_mask is not None:
            mask_indices = json.loads(args.write_calib_mask.read_text())["mask_indices"]
        write_calibration = WriteCalibration(
            policy.action_generator.blocks[0].attention,
            stats,
            mode=args.write_calib_mode,
            mask_indices=mask_indices,
            expo=args.write_calib_expo,
            lo=args.write_calib_lo,
            hi=args.write_calib_hi,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            last_step_only=not args.write_calib_all_steps,
        )
    if args.psc_tta_stats is not None:
        stats = json.loads(args.psc_tta_stats.read_text())
        policy = PSCTTAWrapper(
            policy,
            stats,
            args.device,
            lr=args.psc_tta_lr,
            steps=args.psc_tta_steps,
            lambda_res=args.psc_tta_lambda,
            interval=args.psc_tta_interval,
            weighting=args.psc_weighting,
            buffer_size=args.psc_tta_buffer,
        )
    wrapped = PhotometricPolicy(policy, args.shift, args.blur_sigma, args.device)
    if args.auto_family:
        wrapped = VotePolicy(wrapped, (), args.device, auto_family=True)
    elif args.vote_k > 1 or args.vote_calib != "none":
        variants = (
            CALIB_VARIANTS[args.vote_calib]
            if args.vote_calib != "none"
            else VOTE_VARIANTS[: args.vote_k]
        )
        wrapped = VotePolicy(wrapped, variants, args.device)
    try:
        result = evaluate_suite(
            args,
            args.suite,
            wrapped,
            benchmark,
            get_libero_path,
            OffScreenRenderEnv,
        )
    finally:
        if projector is not None:
            projector.close()
        if write_calibration is not None:
            write_calibration.close()
        if interface_calibration is not None:
            interface_calibration.close()
        if residual_shrink is not None:
            counts = getattr(residual_shrink, "mode_counts", None)
            if counts:
                print(f"[adaptive] mode_counts={counts}", flush=True)
            residual_shrink.close()
    result.update(shift=args.shift, blur_sigma=args.blur_sigma)
    summary = {
        "suite": args.suite,
        "shift": args.shift,
        "blur_sigma": args.blur_sigma,
        "paired": True,
        "pairing": "official per-task seed and identical initial-state index",
        "episodes_per_task": args.episodes_per_task,
        "checkpoint_sha256": checkpoint_digest,
        "model_origin": "official CoRP LIBERO release",
        "psc_stats": str(args.psc_stats) if args.psc_stats is not None else None,
        "vote_k": args.vote_k,
        "psc_tta_stats": str(args.psc_tta_stats) if args.psc_tta_stats is not None else None,
        "psc_weighting": args.psc_weighting,
        "write_calib_stats": str(args.write_calib_stats) if args.write_calib_stats is not None else None,
        "interface_calib": str(args.interface_calib) if args.interface_calib is not None else None,
        "residual_shrink": str(args.residual_shrink) if args.residual_shrink is not None else None,
        "vote_calib": args.vote_calib,
        "upstream": upstream,
        "task_selection": (
            "LIBERO-Plus per-category seeded sample"
            if args.classification is not None
            else "official default (all tasks)"
        ),
        "classification": str(args.classification) if args.classification is not None else None,
        "per_category": args.per_category,
        "result": result,
        "checkpoint_unchanged": file_sha256(args.checkpoint) == checkpoint_digest,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"suite": args.suite, "shift": args.shift, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
