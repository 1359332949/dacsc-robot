"""CoRP 官方 checkpoint 在 LIBERO-Plus 上的零样本鲁棒性评测。

用法（smoke）：
  python3 scripts/eval_corp_libero_plus.py --suite libero_spatial \
    --per-category 2 --episodes-per-task 1 --categories Camera,Robot,Language,Light,Background,Noise,Layout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi

CORP = Path("/hy-tmp/models/corp/CoRP")
sys.path.insert(0, str(CORP))
sys.path.insert(0, str(CORP / "scripts"))

from evaluate_libero import (  # noqa: E402
    CAMERA_SIZE, DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS,
    configure_libero, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402

CATEGORIES = ["Camera Viewpoints", "Robot Initial States", "Language Instructions",
              "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]


def select_tasks(classification: dict, suite: str, categories: list[str], per_cat: int):
    entries = classification[suite]
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    out = []
    for cat in categories:
        items = by_cat.get(cat, [])
        if not items:
            print(f"[warn] category {cat!r} not found in {suite}", flush=True)
        out.extend(items[:per_cat])
    return out


def _apply_shift(arr: np.ndarray, shift: str) -> np.ndarray:
    if shift == "none":
        return arr
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    if shift == "grayscale":
        return np.asarray(img.convert("L").convert("RGB"))
    if shift == "low_contrast":
        return np.asarray(ImageEnhance.Contrast(img).enhance(0.3))
    if shift == "color_jitter":
        img = ImageEnhance.Brightness(img).enhance(1.6)
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = ImageEnhance.Color(img).enhance(2.2)
        return np.asarray(img)
    if shift == "hue_shift":
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 128) % 256
        return np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    return arr


def corrupt_observation(obs: dict, sigma: float, shift: str = "none") -> dict:
    if sigma <= 0 and shift == "none":
        return obs
    out = dict(obs)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in out:
            continue
        arr = _apply_shift(np.asarray(out[key], dtype=np.uint8), shift)
        if sigma > 0:
            arr = ndi.gaussian_filter(arr.astype(np.float32), sigma=(sigma, sigma, 0)).astype(np.uint8)
        out[key] = arr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CORP / "checkpoints" / "corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--blur-sigma", type=float, default=0.0)
    ap.add_argument("--shift", default="none",
                    choices=["none", "grayscale", "color_jitter", "low_contrast", "hue_shift"])
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/repos/LIBERO-plus"))
    ap.add_argument("--libero-config-path", type=Path,
                    default=Path("/hy-tmp/repos/LIBERO-plus/.cache/libero"))
    ap.add_argument("--output-dir", type=Path, default=Path("/hy-tmp/runs/corp_libero_plus"))
    args = ap.parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark(args.suite)()
    names = list(suite.get_task_names())
    name_to_id = {n: i for i, n in enumerate(names)}
    classification = json.loads(
        (args.libero_src / "libero" / "libero" / "benchmark" /
         "task_classification.json").read_text()
    )
    selected = select_tasks(classification, args.suite, categories, args.per_category)
    print(f"suite={args.suite} n_tasks={suite.n_tasks} selected={len(selected)}", flush=True)

    policy = load_policy(args.checkpoint, device=args.device)
    # LIBERO-Plus 用 torch.load 读本地 init states；PyTorch 2.6+ 默认 weights_only=True。
    # 仅对本地可信的 init states 放开（CoRP checkpoint 仍显式 weights_only=True）。
    _orig_torch_load = torch.load
    def _trusted_torch_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*a, **kw)
    torch.load = _trusted_torch_load
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    device = torch.device(args.device)
    started = time.monotonic()
    result_path = args.output_dir / f"{args.suite}.jsonl"
    with result_path.open("w") as output:
        for entry in selected:
            cat, name = entry["category"], entry["name"]
            if name not in name_to_id:
                print(f"[warn] task not in suite: {name}", flush=True)
                continue
            task_id = name_to_id[name]
            task = suite.get_task(task_id)
            bddl = (Path(get_libero_path("bddl_files")) /
                    task.problem_folder / task.bddl_file)
            initial_states = suite.get_task_init_states(task_id)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=CAMERA_SIZE,
                camera_widths=CAMERA_SIZE, render_gpu_device_id=args.render_gpu,
            )
            try:
                for episode in range(args.episodes_per_task):
                    seed = args.seed + task_id * 1000 + episode
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    env.seed(seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[episode])
                    success, done, steps = False, False, 0
                    for _ in range(SETTLING_STEPS):
                        observation, reward, done, info = env.step(DUMMY_ACTION)
                        success = is_success(env, info, reward, done)
                        if success or done:
                            break
                    while not success and not done and steps < MAX_STEPS[args.suite]:
                        batch = policy_observation(corrupt_observation(observation, args.blur_sigma, args.shift), task.language, policy)
                        with torch.autocast(device_type=device.type,
                                            dtype=torch.bfloat16,
                                            enabled=device.type == "cuda" and args.precision == "bf16"):
                            chunk = policy.predict(batch)[0].float().cpu().numpy()
                        for action in chunk:
                            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= MAX_STEPS[args.suite]:
                                break
                    rec = {"suite": args.suite, "category": cat, "task": name,
                           "task_id": task_id, "episode": episode, "seed": seed,
                           "success": bool(success), "policy_steps": steps,
                           "language": task.language}
                    output.write(json.dumps(rec) + "\n")
                    output.flush()
                    records.append(rec)
                    print(f"[{cat}] task={task_id} success={int(success)} steps={steps}", flush=True)
            finally:
                env.close()

    by_cat: dict[str, list[int]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(int(r["success"]))
    summary = {
        "suite": args.suite,
        "checkpoint": str(args.checkpoint),
        "episodes": len(records),
        "successes": sum(int(r["success"]) for r in records),
        "success_rate": (sum(int(r["success"]) for r in records) / len(records)) if records else 0.0,
        "by_category": {c: {"n": len(v), "successes": sum(v),
                            "success_rate": sum(v) / len(v)} for c, v in by_cat.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
