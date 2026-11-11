"""Two deployable test-time adaptation probes for the CoRP flow policy.

B1 (TENT-style): freeze the network, adapt ONLY the LayerNorm affine parameters
    of the action expert at test time, with a label-free objective. TENT
    (Sun et al., ICLR 2021) minimises prediction entropy for classification; a
    regression flow policy has no entropy, so the objective here is the
    Mean-Teacher-style velocity consistency between two appearance views of the
    same observation (MSE), plus an L2 anchor to the source parameters so that a
    single-sample adaptation cannot drift (anchoring follows SAR, Niu et al.,
    ICLR 2023). Only the LN affine parameters are updated, exactly the TENT
    parameter scope.

B2 (PSC-style): source-free subspace alignment at the cross-attention write.
    PSC (arXiv:2603.09338) aligns target features to the SOURCE predictive
    support instead of removing domain directions; here the support is the
    top-r PCA subspace of the source-style clean writes and the projection is
    applied at every Euler step. Leave-one-state-out: for each state the
    subspace is fitted on the OTHER states' clean writes only, so no clean
    information about the evaluated state is used. Control: random orthonormal
    subspace of the same rank.

No labels / no clean reference are used by either method at test time; the
clean actions are used only for scoring.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", nargs="+", type=int, default=[16, 29, 35, 41, 56, 57, 69, 76])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--tent-steps", type=int, default=3)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-anchor", type=float, default=1e-4)
    parser.add_argument("--psc-ranks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--sample-steps", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def load_episode(dataset: Path, episode: int):
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, states, model_xml


class WriteHook:
    """Capture or subspace-project the cross-attention write (attend output)."""

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.mean: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.target_calls: set[int] | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", mean=None, basis=None, target_calls=None):
        self.mode = mode
        self.mean = mean
        self.basis = basis
        self.target_calls = target_calls
        self.call = 0
        self.value = None

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if self.mode == "capture":
            self.value = output.detach().clone()
            return output
        if self.mode == "project" and self.basis is not None:
            if self.target_calls is not None and index not in self.target_calls:
                return output
            mean = self.mean.to(device=output.device, dtype=output.dtype)
            basis = self.basis.to(device=output.device, dtype=output.dtype)
            centred = output - mean
            return mean + (centred @ basis.T) @ basis
        return output

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def set_action_noise(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample(model, images, state, lang, steps, noise_seed):
    set_action_noise(noise_seed)
    with torch.no_grad():
        action = model.sample_action(images, state, lang, num_steps=steps)
    return action.detach().float().flatten()


def photometric_variant(images: dict[str, torch.Tensor], generator: torch.Generator) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in images.items():
        mean = IMAGENET_MEAN.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        rgb = (value * std + mean).clamp(0, 1)
        brightness = 0.85 + torch.rand(1, generator=generator).item() * 0.3
        contrast = 0.85 + torch.rand(1, generator=generator).item() * 0.3
        rgb = TVF.adjust_brightness(rgb, brightness)
        rgb = TVF.adjust_contrast(rgb, contrast)
        out[name] = ((rgb - mean) / std).to(dtype=value.dtype)
    return out


def ln_affine_params(model) -> list[torch.nn.Parameter]:
    params = []
    for module in model.action_generator.modules():
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def tent_adapt(model, images, state, lang, *, steps, lr, anchor, generator, action_horizon, action_dim):
    # TENT updates ONLY the BN/LN affine parameters; freeze everything else so
    # backward does not build graphs for the frozen network (TENT,
    # Sun et al. ICLR 2021, Algorithm 1).
    grad_state = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    params = ln_affine_params(model)
    originals = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=lr)
    variant = photometric_variant(images, generator)
    for _ in range(steps):
        eps = torch.randn(1, action_horizon, action_dim, device=images["robot0_agentview_left"].device)
        tau = torch.zeros(1, device=eps.device, dtype=eps.dtype)
        x_tau = eps
        with torch.no_grad():
            prefix_j = model.action_generator.prepare_prefix(
                model.extract(images, state, lang)
            )
            prefix_v = model.action_generator.prepare_prefix(
                model.extract(variant, state, lang)
            )
        velocity_j = model.action_generator(x_tau, tau, prefix_j)
        velocity_v = model.action_generator(x_tau, tau, prefix_v)
        loss = torch.nn.functional.mse_loss(velocity_j.float(), velocity_v.float().detach())
        for p, original in zip(params, originals):
            loss = loss + anchor * ((p - original) ** 2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for p in params:
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(grad_state.get(id(p), True))
    return originals


def tent_restore(model, originals):
    params = ln_affine_params(model)
    with torch.no_grad():
        for p, original in zip(params, originals):
            p.copy_(original)


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    episode_data = {}
    for episode in args.episodes:
        if episode not in metadata:
            raise ValueError(f"episode is absent: {episode}")
        meta, states, model_xml = load_episode(args.dataset, episode)
        if states.shape[0] <= max(args.frames):
            raise ValueError(f"episode {episode} lacks requested state frames")
        episode_data[episode] = {
            "style": metadata[episode]["style_id"],
            "meta": meta,
            "states": states,
            "model_xml": model_xml,
        }

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    hook = WriteHook(model.action_generator.blocks[0].attention)
    steps = args.sample_steps or model.action_generator.num_sample_steps
    styles = sorted({item["style"] for item in episode_data.values()})
    env = RoboCasaGymEnv(
        env_name="OpenDrawer",
        split=None,
        obj_instance_split="pretrain",
        style_ids=styles,
        layout_ids=LayoutType.TRAIN,
        camera_widths=256,
        camera_heights=256,
        seed=args.seed,
    )
    core = env.env
    records = []
    started = time.monotonic()
    try:
        for episode in args.episodes:
            info = episode_data[episode]
            for frame in args.frames:
                reset_to(
                    core,
                    {
                        "model": info["model_xml"],
                        "states": info["states"][frame],
                        "ep_meta": json.dumps(info["meta"]),
                    },
                )
                obs = env.get_observation(core._get_observations(force_update=True))
                images_c, state, lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "none"
                )
                images_j, state_j, lang_j = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, args.shift
                )
                if not torch.equal(state, state_j) or lang != lang_j:
                    raise RuntimeError("clean and shifted inputs are not paired")
                seed = args.seed * 1000 + len(records)
                hook.reset("capture")
                action_c = sample(model, images_c, state, lang, args.sample_steps, seed)
                write_c = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("capture")
                action_j = sample(model, images_j, state, lang, args.sample_steps, seed)
                write_j = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("idle")
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "write_shift": write_j,
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": seed,
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        action_horizon = int(records[0]["action_clean"].numel() // 12)
        action_dim = int(records[0]["action_clean"].numel() // action_horizon)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)

        # ---- B1: TENT-style LN adaptation on the shifted observation ----
        tent_rows = []
        for record in records:
            base_error = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            originals = tent_adapt(
                model,
                record["images_j"],
                record["state"],
                record["lang"],
                steps=args.tent_steps,
                lr=args.tent_lr,
                anchor=args.tent_anchor,
                generator=generator,
                action_horizon=action_horizon,
                action_dim=action_dim,
            )
            adapted = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            clean_cost = sample(model, record["images_c"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            tent_restore(model, originals)
            tent_rows.append(
                {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "tent_recovery": float((adapted - record["action_clean"]).norm() / base_error),
                    "tent_clean_cost": float((clean_cost - record["action_clean"]).norm() / base_error),
                }
            )
            print(f"tent ep={record['episode']} f={record['frame']} recovery={tent_rows[-1]['tent_recovery']:.3f}", flush=True)

        # ---- B2: PSC-style source-subspace alignment (leave-one-state-out) ----
        clean_writes = torch.stack([r["write_clean"] for r in records]).float()
        jitter_writes = torch.stack([r["write_shift"] for r in records]).float()
        psc_rows = {rank: [] for rank in args.psc_ranks}
        psc_random_rows = {rank: [] for rank in args.psc_ranks}
        for index, record in enumerate(records):
            base_error = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            train = torch.ones(len(records), dtype=torch.bool)
            train[index] = False
            source = clean_writes[train]
            source_mean = source.mean(dim=0)
            _, _, vh = torch.linalg.svd(source - source_mean, full_matrices=False)
            for rank in args.psc_ranks:
                r = min(rank, vh.shape[0])
                basis = vh[:r].float()
                hook.reset("project", mean=source_mean, basis=basis, target_calls=set(range(steps)))
                adapted = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                clean_cost = sample(model, record["images_c"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                random_basis = torch.linalg.qr(
                    torch.randn(basis.shape[1], r, generator=generator)
                )[0].T.float()
                hook.reset("project", mean=source_mean, basis=random_basis, target_calls=set(range(steps)))
                random_adapted = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                psc_rows[rank].append(
                    {
                        "episode": record["episode"],
                        "frame": record["frame"],
                        "recovery": float((adapted - record["action_clean"]).norm() / base_error),
                        "clean_cost": float((clean_cost - record["action_clean"]).norm() / base_error),
                    }
                )
                psc_random_rows[rank].append(
                    float((random_adapted - record["action_clean"]).norm() / base_error)
                )
            print(f"psc ep={record['episode']} f={record['frame']} elapsed={time.monotonic() - started:.1f}s", flush=True)

        def summarize(values):
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5)),
            }

        payload = {
            "schema": "dacsc_tta_adaptation_probe_v1",
            "method_references": {
                "tent": "Sun et al., ICLR 2021 (test-time entropy minimisation on BN/LN affine); regression objective replaced by velocity consistency (Mean Teacher 2017); anchor follows SAR (Niu et al., ICLR 2023)",
                "psc": "source-free subspace alignment for regression TTA, arXiv:2603.09338",
                "site": "action_generator.blocks[0].attention.attend",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "states": len(records),
                "tent": {
                    "steps": args.tent_steps,
                    "lr": args.tent_lr,
                    "anchor": args.tent_anchor,
                    "target": "velocity consistency at tau=0 between two appearance views",
                    "updated_params": "all LayerNorm weight/bias in the action expert",
                },
                "psc": {"ranks": list(args.psc_ranks), "leave_one_state_out": True},
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "tent_recovery": summarize([r["tent_recovery"] for r in tent_rows]),
                "tent_clean_cost": summarize([r["tent_clean_cost"] for r in tent_rows]),
                **{
                    f"psc_rank{rank}_recovery": summarize([r["recovery"] for r in psc_rows[rank]])
                    for rank in args.psc_ranks
                },
                **{
                    f"psc_rank{rank}_clean_cost": summarize([r["clean_cost"] for r in psc_rows[rank]])
                    for rank in args.psc_ranks
                },
                **{
                    f"psc_rank{rank}_random": summarize(psc_random_rows[rank])
                    for rank in args.psc_ranks
                },
            },
            "tent": tent_rows,
            "psc": {str(rank): psc_rows[rank] for rank in args.psc_ranks},
            "status": "completed",
            "elapsed_seconds": None,
        }
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        hook.close()


if __name__ == "__main__":
    raise SystemExit(main())
