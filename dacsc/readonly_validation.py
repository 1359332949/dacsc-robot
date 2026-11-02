"""Bounded real-input validation, without training, gates or automatic downloads.

Only glue lives here: official DP reading, CoRP loading/forward, Torchvision
operators and the existing read-only Registers observer do the actual work.
MuJoCo replay remains the official playback_dataset.py entry (see the runbook).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import hashlib
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback

import torch
from torchvision.transforms import ColorJitter, functional as TVF

from dacsc.corp_data import (
    CorpRoboCasaDataset,
    VIEW_TO_KEY,
    SHAPE_META,
    read_episode_meta,
)
from dacsc.corp_ffn_capture import CoRPFFNCapture, load_registers_reference
from dacsc.corp_images import (
    image_contract,
    prepare_image,
    resize_rgb,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from dacsc.eval_protocol import file_sha256
from dacsc.pixel_augmentation import TrainingPixelAugmentation
from dacsc.run_output import entry_output
from dacsc.upstream import ROOT, verify_upstream, verify_release_checkpoint

SCHEMA = "corp_readonly_validation_v1"
VIEWS = tuple(VIEW_TO_KEY)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(model):
    return hashlib.sha256(
        json.dumps(
            [(k, tensor_digest(v)) for k, v in model.state_dict().items()]
        ).encode()
    ).hexdigest()


def exact(actual, expected, label):
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not torch.isfinite(actual).all()
        or not torch.isfinite(expected).all()
        or not torch.equal(actual, expected)
    ):
        raise AssertionError(f"nonfinite or nonidentical {label}")


def rng_state():
    return [torch.get_rng_state().clone()] + (
        [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else []
    )


def assert_rng(before):
    after = rng_state()
    if len(before) != len(after) or any(
        not torch.equal(a, b) for a, b in zip(before, after)
    ):
        raise AssertionError("global CPU/CUDA RNG changed during read-only validation")


def environment():
    import torchvision

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def code_identity():
    """A server receipt must identify one clean committed tree, never local edits."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("commit local changes before real-input validation")
    analysis = json.loads((ROOT / "analysis_upstream.lock.json").read_text())
    repo = Path(os.environ.get("CORP_REPO", ROOT / "repos/CoRP")).resolve()
    for rel, expected in analysis["CoRPVisual"]["files"].items():
        if file_sha256(repo / rel) != expected:
            raise RuntimeError(f"CoRP visual execution source changed: {rel}")
    return {
        "commit": git("rev-parse", "HEAD"),
        "upstream": verify_upstream(),
        "analysis_lock_sha256": file_sha256(ROOT / "analysis_upstream.lock.json"),
    }


def source_module(module, expected):
    actual = Path(inspect.getfile(module)).resolve()
    if actual != expected.resolve():
        raise RuntimeError(f"import shadowing: {actual}; expected {expected}")
    return {"path": str(actual), "sha256": file_sha256(actual)}


def select_observations(root, seed=123):
    """Freeze at most 3 tasks x 3 independent source episodes x 3 distinct frames."""
    root = Path(root).resolve()
    atomic = root / "v1.0/pretrain/atomic"
    candidates = []
    for task in sorted(p for p in atomic.iterdir() if p.is_dir()):
        releases = sorted(task.glob("*/lerobot"))
        if len(releases) != 1:
            raise ValueError(f"expected one official release: {task}")
        path = releases[0]
        metadata = read_episode_meta(path)
        eligible = [
            ep
            for ep, m in sorted(metadata.items())
            if 11 <= m["style_id"] <= 30 and ep % 10 != 9
        ]
        if eligible:
            candidates.append((path, metadata, eligible))
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(3, len(candidates)))
    rows = []
    for path, metadata, episodes in chosen:
        for episode in sorted(rng.sample(episodes, min(3, len(episodes)))):
            m = metadata[episode]
            for frame in sorted({0, m["length"] // 2, m["length"] - 1}):
                rows.append(
                    {
                        "dataset_relpath": str(path.relative_to(root)),
                        "task": path.parent.parent.name,
                        "episode": episode,
                        "frame": frame,
                        "length": m["length"],
                        "style_id": m["style_id"],
                        "layout_id": m["layout_id"],
                        "instruction_sha256": hashlib.sha256(
                            m["lang"].encode()
                        ).hexdigest(),
                        "fold": "source_train",
                        "view_names": list(VIEWS),
                    }
                )
    if not rows:
        raise ValueError("no source-training episodes; never substitute val/test")
    return rows


class RecordingReader:
    """Observe the real upstream return once; no decoding/cache implementation."""

    def __init__(self, reader):
        self.reader, self.last = reader, None

    def __getitem__(self, index):
        self.last = self.reader[index]
        return self.last


def check_sample(sample, row, raw):
    if (
        sample["style_id"].item() != row["style_id"]
        or sample["episode"] != row["episode"]
        or sample["frame"] != row["frame"]
        or sample["task"] != row["task"]
        or sample["image_space"] != "imagenet_normalized"
        or hashlib.sha256(sample["instruction"].encode()).hexdigest()
        != row["instruction_sha256"]
    ):
        raise AssertionError(
            "sample identity/labels differ from frozen episode metadata"
        )
    for name, shape in (("state", (9,)), ("action", (10, 12))):
        if sample[name].shape != shape or not torch.isfinite(sample[name]).all():
            raise AssertionError(f"invalid {name}")
    mask = torch.arange(10) < row["length"] - row["frame"]
    if sample["action_mask"].dtype != torch.bool or not torch.equal(
        sample["action_mask"], mask
    ):
        raise AssertionError("invalid future-action mask")
    if not mask.all():
        expected = sample["action"][int(mask.sum()) - 1].expand_as(
            sample["action"][~mask]
        )
        exact(sample["action"][~mask], expected, "repeat-last padding")
    for view, key in VIEW_TO_KEY.items():
        image = raw["obs"][key][0]
        if image.shape != (3, 256, 256):
            raise AssertionError("official raw image must be CHW 256 RGB")
        exact(sample["images"][view], prepare_image(image), f"prepared {view}")


def video_stage(args, out):
    root = args.data_root.expanduser().resolve()
    if out.is_relative_to(root):
        raise ValueError("validation outputs must be outside the dataset")
    rows = select_observations(root)
    write_json(out / "selection.json", rows)  # before any video decode/model forward
    paths = list(dict.fromkeys(str(root / r["dataset_relpath"]) for r in rows))
    ds = CorpRoboCasaDataset(
        paths,
        style_min=11,
        style_max=30,
        horizon=10,
        episode_residues=set(range(9)),
        use_cache=False,
        shift="none",
    )
    dp = import_module("diffusion_policy.dataset.lerobot_dataset")
    groot = import_module("robocasa.utils.groot_utils.groot_dataset")

    modules = {
        "DP": source_module(
            dp,
            ROOT / "repos/diffusion_policy/diffusion_policy/dataset/lerobot_dataset.py",
        ),
        "LeRobot": source_module(
            groot, ROOT / "repos/robocasa/robocasa/utils/groot_utils/groot_dataset.py"
        ),
    }
    readers = list(ds.datasets)
    recordings = [RecordingReader(r) for r in readers]
    ds.datasets = recordings
    positions = {
        (str(Path(paths[r[0]]).relative_to(root)), r[5], r[6]): i
        for i, r in enumerate(ds.flat)
    }
    payload, manifests, digests = [], [], {}
    for index, row in enumerate(rows):
        pos = positions[(row["dataset_relpath"], row["episode"], row["frame"])]
        ds_index = ds.flat[pos][0]
        sample = ds[pos]  # real official __getitem__ once, existing adapter unchanged
        raw = recordings[ds_index].last
        check_sample(sample, row, raw)
        images, cameras = {}, {}
        for view, key in VIEW_TO_KEY.items():
            image = raw["obs"][key][0].clone()
            images[view] = image
            video = Path(
                readers[ds_index].get_video_path(
                    row["episode"],
                    SHAPE_META["obs"][key]["lerobot_keys"][0].removeprefix("video."),
                )
            )
            if not video.resolve().is_relative_to(root):
                raise ValueError("video path escapes dataset root")
            if video not in digests:
                digests[video] = file_sha256(video)
            cameras[view] = {
                "video": str(video.relative_to(root)),
                "video_sha256": digests[video],
                "decoded_sha256": tensor_digest(image),
                "backend": readers[ds_index].video_backend,
                "backend_kwargs": readers[ds_index].video_backend_kwargs,
            }
            if index == 0:
                TVF.to_pil_image(image).save(out / f"raw_{view}.png")
        path = root / row["dataset_relpath"]
        metadata = [
            path / "meta/episodes.jsonl",
            path / "meta/info.json",
            path / "meta/modality.json",
            path / f"extras/episode_{row['episode']:06d}/ep_meta.json",
        ]
        manifests.append(
            {
                **row,
                "cameras": cameras,
                "metadata_sha256": {
                    str(p.relative_to(root)): file_sha256(p) for p in metadata
                },
            }
        )
        payload.append({"identity": row, "raw_images": images, "sample": sample})
    torch.save(payload, out / "observations.pt")
    (out / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifests)
    )
    # Video imports may initialize libraries; this stage never loads a policy.
    return {
        "reader_imports": modules,
        "observations": len(rows),
        "view_frames": len(rows) * 3,
        "episodes": len({(r["dataset_relpath"], r["episode"]) for r in rows}),
        "selection_seed": 123,
        "requested_max": {"tasks": 3, "episodes_per_task": 3, "frames_per_episode": 3},
        "support": ds.support(
            [positions[(r["dataset_relpath"], r["episode"], r["frame"])] for r in rows]
        ),
        "training_augmentation": "not_instantiated",
        "conditions": ["none"],
        "model_forward": "not_run",
        "dataset_root": str(root),
    }


def replay_plan_stage(args, out):
    """Check exactly the episode the unchanged official --n 1 will use.

    The official implementation enumerates the first directory but reads states
    and XML using ordinal zero. Require episode_000000, not an arbitrary episode.
    This stage prepares commands; it never imports or runs the simulator.
    """
    dataset = args.dataset.expanduser().resolve()
    metadata = read_episode_meta(dataset)
    first = sorted((dataset / "extras").glob("episode_*"))[0].name
    if first != "episode_000000" or 0 not in metadata:
        raise ValueError("official --n 1 requires a real episode_000000")
    episode = metadata[0]
    if not 11 <= episode["style_id"] <= 30:
        raise ValueError(
            "official first episode is not source; choose another dataset, never rewrite metadata"
        )
    files = [
        dataset / "extras/episode_000000" / name
        for name in ("ep_meta.json", "states.npz", "model.xml.gz")
    ]
    from gzip import open as gzopen
    import numpy as np

    with np.load(files[1], allow_pickle=False) as states:
        value = states["states"]
        if (
            value.ndim != 2
            or value.shape[0] != episode["length"]
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid recorded states/episode length")
    with gzopen(files[2], "rb") as stream:
        xml = stream.read()
    from xml.etree import ElementTree

    ElementTree.fromstring(xml)  # no XML rewriting
    if args.replay_output.exists():
        raise FileExistsError("choose a new replay-output directory")
    playback = (
        ROOT / "repos/robocasa/robocasa/scripts/dataset_scripts/playback_dataset.py"
    )
    common = [
        str(args.sim_python.resolve()),
        str(playback),
        "--dataset",
        str(dataset),
        "--n",
        "1",
        "--camera_height",
        "224",
        "--camera_width",
        "224",
        "--verbose",
    ]
    return {
        "simulation_execution": "not_run",
        "dataset": str(dataset),
        "episode": {"index": 0, **episode},
        "source_files": {str(p): file_sha256(p) for p in files},
        "official_entry_sha256": file_sha256(playback),
        "commands": {
            "state": common
            + ["--video_path", str(args.replay_output / "state_replay.mp4")],
            "action": common
            + [
                "--use-actions",
                "--video_path",
                str(args.replay_output / "action_replay.mp4"),
            ],
        },
        "environment": {"MUJOCO_GL": "egl"},
        "acceptance": "commands only; inspect full logs, state divergence, finite states, decode full videos; success is separate",
    }


def passed_artifacts(path, stage, identity):
    path = Path(path).resolve()
    receipt = json.loads((path / "report.json").read_text())
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("stage") != stage
        or receipt.get("status") != "passed"
        or receipt.get("code") != identity
        or receipt.get("environment") != environment()
        or not receipt.get("artifacts")
    ):
        raise ValueError(f"incompatible or incomplete {stage} receipt")
    for name, digest in receipt["artifacts"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (path / relative).resolve().is_relative_to(path)
        ):
            raise ValueError("artifact path escapes its receipt directory")
        if file_sha256(path / relative) != digest:
            raise ValueError(f"artifact digest differs: {name}")
    required = (
        {"observations.pt", "sample_manifest.jsonl", "selection.json"}
        if stage == "video"
        else {"official_fp32.log", "official_bf16.log"}
    )
    if not required <= receipt["artifacts"].keys():
        raise ValueError("receipt lacks required stage artifacts")
    return receipt


def require_cuda(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing a CPU fallback")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 unsupported")
        torch.cuda.init()
        return {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }


def official_cuda_stage(args, out, identity):
    passed_artifacts(args.video_artifacts, "video", identity)
    gpu = require_cuda(args.device)
    repo = Path(os.environ.get("CORP_REPO", ROOT / "repos/CoRP")).resolve()
    checkpoint = repo / "checkpoints/corp_libero.pt"
    digest = verify_release_checkpoint(checkpoint)
    env = {
        **os.environ,
        "PYTHONPATH": str(repo),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    commands = []
    for precision in ("fp32", "bf16"):
        command = [
            sys.executable,
            str(repo / "scripts/smoke_test.py"),
            "--device",
            args.device,
            "--precision",
            precision,
        ]
        commands.append(command)
        with (out / f"official_{precision}.log").open("x") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        text = (out / f"official_{precision}.log").read_text()
        if (
            "smoke_ok" not in text
            or f"sha256={digest}" not in text
            or "parameters=48689159" not in text
        ):
            raise AssertionError("official demo did not confirm its release contract")
    return {
        "commands": commands,
        "gpu": gpu,
        "checkpoint_sha256": digest,
        "video_report_sha256": file_sha256(args.video_artifacts / "report.json"),
        "inference_only": True,
    }


def augmentation_checks(rgb):
    """Actual Torchvision and private RNG on real pixels, no parameter updates."""
    before = rng_state()
    augment = TrainingPixelAugmentation(123)
    saved = augment.state_dict()
    expected_transform = ColorJitter(0.4, 0.4, 0.4, 0.0)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(saved["rng"])
        expected = TVF.normalize(
            torch.stack([expected_transform(x) for x in rgb]),
            IMAGENET_MEAN.flatten().tolist(),
            IMAGENET_STD.flatten().tolist(),
        )
    actual = augment(rgb)
    exact(actual, expected, "upstream ColorJitter")
    augment.load_state_dict(saved)
    exact(augment(rgb), actual, "restored augmentation RNG")
    private = augment.generator.get_state().clone()
    # Evaluation uses a separate deterministic functional API, never this object.
    from dacsc.corp_images import apply_image_shift

    shifted = apply_image_shift(rgb, "color_jitter")
    exact(shifted, apply_image_shift(rgb, "color_jitter"), "deterministic eval shift")
    exact(
        augment.generator.get_state(), private, "untouched augmentation RNG during eval"
    )
    assert_rng(before)
    return {
        "torchvision_exact": True,
        "private_rng_restorable": True,
        "global_rng_unchanged": True,
        "used_in_model_forward": False,
    }


def observe_batch(model, reference, images):
    """One precision at a time; exact identity, raw-hook alignment and cleanup."""
    visual = model.representation_extractor.visual_encoder
    flat = torch.cat([images[v] for v in model.view_names])
    before_rng = rng_state()
    before_hooks = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )
    base = visual(flat)
    with CoRPFFNCapture(model, reference) as capture:
        observed = visual(flat)
        frame = capture.last
        exact(observed, base, "hooked visual output")
        # Independent raw hooks confirm every layer/sample/view/CLS/patch identity.
        raw = {}
        handles = []

        def record(kind, i):
            def hook(module, inputs, output):
                raw[kind, i] = output.detach().to(
                    device="cpu", dtype=torch.float32, copy=True
                )

            return hook

        try:
            for i, block in enumerate(visual.backbone.blocks):
                handles.extend(
                    [
                        block.register_forward_hook(record("blocks", i)),
                        block.mlp.act.register_forward_hook(record("ffn", i)),
                    ]
                )
            exact(visual(flat), base, "independent raw hook visual output")
        finally:
            for h in handles:
                h.remove()
        batch = flat.shape[0] // len(model.view_names)
        summaries = []
        for kind, layers in (("ffn", frame.ffn), ("blocks", frame.blocks)):
            for layer, value in layers.items():
                for vi, view in enumerate(model.view_names):
                    for sample in range(batch):
                        exact(
                            value[sample, vi],
                            raw[kind, layer][vi * batch + sample],
                            "view-major CLS+patch",
                        )
                        for group, sl in (
                            ("cls", slice(0, 1)),
                            ("patch", slice(1, None)),
                        ):
                            norm = torch.linalg.vector_norm(
                                value[sample, vi, sl], dim=-1
                            )
                            summaries.append(
                                {
                                    "kind": kind,
                                    "layer": layer,
                                    "sample": sample,
                                    "view": view,
                                    "tokens": group,
                                    "mean_l2": norm.mean().item(),
                                    "min_l2": norm.min().item(),
                                    "max_l2": norm.max().item(),
                                    "quantiles_l2": torch.quantile(
                                        norm, torch.tensor([0.25, 0.5, 0.75])
                                    ).tolist(),
                                }
                            )
    after_hooks = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )
    if before_hooks != after_hooks or any(m.training for m in model.modules()):
        raise AssertionError("observer changed hooks/model mode")
    assert_rng(before_rng)
    return (
        frame,
        summaries,
        {
            "exact_output": True,
            "all_layer_view_token_identity": True,
            "rng_unchanged": True,
            "hooks_restored": True,
            "shape": list(base.shape),
            "dtype": str(base.dtype),
        },
    )


def capture_stage(args, out, identity):
    video = passed_artifacts(args.video_artifacts, "video", identity)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (
        args.precision == "bf16" and device.type != "cuda"
    ):
        raise ValueError("use CPU FP32 or CUDA FP32/BF16; no fallback")
    if device.type == "cuda":
        require_cuda(args.device)
        if args.official_cuda_artifacts is None:
            raise ValueError("run the official CUDA stage first")
        official = passed_artifacts(
            args.official_cuda_artifacts, "official-cuda", identity
        )
        if official["details"]["video_report_sha256"] != file_sha256(
            args.video_artifacts / "report.json"
        ):
            raise ValueError(
                "official CUDA receipt belongs to different video artifacts"
            )
    digest = verify_release_checkpoint(args.checkpoint)
    reference = load_registers_reference(args.registers_source)
    from dacsc.corp_robocasa import load_corp_robocasa

    torch.manual_seed(123)
    model, transfer = load_corp_robocasa(args.checkpoint, device=device)
    repo = Path(os.environ.get("CORP_REPO", ROOT / "repos/CoRP")).resolve()
    actual_imports = {
        rel: source_module(import_module(name), repo / rel)
        for name, rel in (
            ("corp._vendor.dinov2", "corp/_vendor/dinov2.py"),
            (
                "corp.representation.visual_encoder",
                "corp/representation/visual_encoder.py",
            ),
            ("corp.representation.extractor", "corp/representation/extractor.py"),
        )
    }
    model.requires_grad_(False)
    before = state_digest(model)
    # Local artifact produced by video_stage, not an untrusted arbitrary checkpoint.
    observations = torch.load(
        args.video_artifacts / "observations.pt", map_location="cpu", weights_only=True
    )
    selected = json.loads((args.video_artifacts / "selection.json").read_text())
    if (
        not observations
        or len(observations) != video["details"]["observations"]
        or len(observations) != len(selected)
    ):
        raise ValueError("video observation support differs")
    # Preserve raw pixels and sample identity across conditions; no re-decoding.
    manifest = [
        json.loads(line)
        for line in (args.video_artifacts / "sample_manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    if len(manifest) != len(selected):
        raise ValueError("video sample manifest support differs")
    for item, row, record in zip(observations, selected, manifest, strict=True):
        if item["identity"] != row or any(record[k] != v for k, v in row.items()):
            raise ValueError("cached observation identity differs")
        raw = {
            "obs": {VIEW_TO_KEY[v]: item["raw_images"][v].unsqueeze(0) for v in VIEWS}
        }
        check_sample(item["sample"], row, raw)
        for v in VIEWS:
            if (
                tensor_digest(item["raw_images"][v])
                != record["cameras"][v]["decoded_sha256"]
            ):
                raise ValueError("decoded pixels differ from manifest")
    checks, summaries = [], []
    scope = {
        "full_policy": {
            "status": "not_run",
            "reason": "visual-only release validation; no adapted checkpoint/normalizer loaded",
        },
        "action_causality": {
            "status": "not_run",
            "reason": "read-only observations, no intervention",
        },
    }
    for start in range(0, len(observations), 2):
        batch = observations[start : start + 2]
        rgb = torch.stack(
            [
                torch.stack([resize_rgb(x["raw_images"][v]) for v in VIEWS])
                for x in batch
            ]
        ).to(device)
        augmented = augmentation_checks(rgb)
        for shift in ("none", "color_jitter"):
            images = {
                v: torch.stack(
                    [prepare_image(x["raw_images"][v], shift) for x in batch]
                ).to(device)
                for v in VIEWS
            }
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            ):
                frame, rows, check = observe_batch(model, reference, images)
            if (
                args.precision == "bf16"
                and "torch.bfloat16" not in frame.source_dtypes.values()
            ):
                raise AssertionError(
                    "requested BF16 but no BF16 activation was captured"
                )
            checks.append(
                {
                    "start": start,
                    "count": len(batch),
                    "shift": shift,
                    **check,
                    "source_dtypes": frame.source_dtypes,
                    "augmentation_isolation": augmented,
                }
            )
            for row in rows:
                row["identity"] = batch[row.pop("sample")]["identity"]
                row["shift"] = shift
            summaries.extend(rows)
            if start == 0:
                torch.save(
                    {
                        "identities": [x["identity"] for x in batch],
                        "ffn": frame.ffn,
                        "blocks": frame.blocks,
                        "source_dtypes": frame.source_dtypes,
                    },
                    out / f"first_batch_{shift}.pt",
                )
                for v in VIEWS:
                    from dacsc.corp_images import apply_image_shift

                    vi = VIEWS.index(v)
                    TVF.to_pil_image(apply_image_shift(rgb[0, vi].cpu(), shift)).save(
                        out / f"{shift}_{v}.png"
                    )
    if before != state_digest(model) or any(
        p.grad is not None for p in model.parameters()
    ):
        raise AssertionError("model state/gradient changed")
    write_json(
        out / "identity_checks.json",
        {"visual": checks, **scope, "state_dict_unchanged": True},
    )
    write_json(out / "activation_summary.json", summaries)
    write_json(
        out / "capture_manifest.json",
        {
            "device": str(device),
            "precision": args.precision,
            "view_names": list(VIEWS),
            "positions": ["mlp.act/GELU_output", "raw_block_output"],
            "conditions": ["none", "color_jitter"],
            "model_images": image_contract(),
            "official_checkpoint_sha256": digest,
            "actual_model_imports": actual_imports,
            "interface_initialization": transfer,
            "scope": "visual_only; randomly initialized RoboCasa interfaces never executed",
            "source_report_sha256": file_sha256(args.video_artifacts / "report.json"),
            "registers": reference.contract,
            "state_dict_sha256": before,
            **scope,
        },
    )
    return {
        "observations": len(observations),
        "precision": args.precision,
        "device": str(device),
        "conditions": ["none", "color_jitter"],
        **scope,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    video = sub.add_parser("video", help="real official reader only, no model")
    video.add_argument("--data-root", type=Path, required=True)
    cuda = sub.add_parser(
        "official-cuda", help="delegate both precisions to official smoke_test.py"
    )
    capture = sub.add_parser(
        "capture", help="read-only visual FFN, no actions or attribution"
    )
    replay = sub.add_parser(
        "replay-plan",
        help="validate source episode zero; print official commands without running them",
    )
    replay.add_argument("--dataset", type=Path, required=True)
    replay.add_argument("--sim-python", type=Path, required=True)
    replay.add_argument("--replay-output", type=Path, required=True)
    for entry in (cuda, capture):
        entry.add_argument("--video-artifacts", type=Path, required=True)
        entry.add_argument("--device", default="cuda:0" if entry is cuda else "cpu")
    capture.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="official LIBERO release with adjacent config.json",
    )
    capture.add_argument("--registers-source", type=Path, required=True)
    capture.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    capture.add_argument("--official-cuda-artifacts", type=Path)
    for entry in (video, cuda, capture, replay):
        entry.add_argument(
            "--out", type=Path, required=True, help="new empty directory"
        )
    args = parser.parse_args(argv)
    # No network fallback: callers prepare official sources/dependencies/weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with entry_output(args.out) as out:
        report = {
            "schema": SCHEMA,
            "stage": args.stage,
            "status": "running",
            "command": sys.argv if argv is None else argv,
            "environment": environment(),
        }
        try:
            report["code"] = code_identity()
            write_json(out / "report.json", report)
            with (out / "run.log").open("x") as log, redirect_stdout(
                log
            ), redirect_stderr(log):
                if args.stage == "video":
                    details = video_stage(args, out)
                elif args.stage == "official-cuda":
                    details = official_cuda_stage(args, out, report["code"])
                elif args.stage == "replay-plan":
                    details = replay_plan_stage(args, out)
                    write_json(out / "replay_plan.json", details)
                else:
                    details = capture_stage(args, out, report["code"])
            report.update(status="passed", details=details)
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (out / "error.txt").write_text(traceback.format_exc())
            raise
        finally:
            report["artifacts"] = {
                str(p.relative_to(out)): file_sha256(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p != out / "report.json"
            }
            write_json(out / "report.json", report)
    print(f"{args.stage}: passed; artifacts={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
