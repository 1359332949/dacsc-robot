"""Modified-protocol rollout with leftover-consistency interface adaptation (pilot).

Protocol change (must be disclosed): the official LIBERO evaluation executes the
full action chunk; leftover supervision needs the previous chunk tail, so this
driver executes `exec_len` of `horizon` actions and keeps the tail. A baseline
run under the SAME protocol is produced for a fair paired comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION,
    MAX_STEPS,
    SETTLING_STEPS,
    configure_libero,
    is_success,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import LowRankInterfaceTransform, leftover_loss, sample_with_grad  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--adapt-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["base", "tta"], required=True)
    ap.add_argument("--action-basis", type=Path, default=None,
                    help="stats json/flags with action_subspace for null-space protection")
    ap.add_argument("--adapt-seed", type=int, default=1234)
    ap.add_argument("--last-step-only", action="store_true")
    ap.add_argument("--verifier", choices=["leftover", "consensus", "smooth"], default=None,
                    help="if set, use candidate selection + disagreement gate before distillation")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--disagree-threshold", type=float, default=0.02)
    ap.add_argument("--restore", type=float, default=0.0, help="pull U,V toward zero by this factor each step")
    ap.add_argument("--adapt-once", action="store_true")
    ap.add_argument("--uv-clip", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    horizon = int(policy.config["action_horizon"])
    transform = None
    if args.mode == "tta":
        action_basis = None
        if args.action_basis is not None:
            blob = json.loads(args.action_basis.read_text())
            action = torch.tensor(blob["action_subspace"], dtype=torch.float32)
            u_a, s_a, _ = torch.linalg.svd(action, full_matrices=False)
            rank_a = int(torch.sum(s_a > s_a[0] * 1e-8))
            action_basis = u_a[:, :rank_a].contiguous()
        transform = LowRankInterfaceTransform(
            policy.action_generator.blocks[0].attention,
            int(policy.config["hidden_dim"]),
            rank=args.rank,
            reg=args.reg,
            action_basis=action_basis,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            last_step_only=args.last_step_only,
        )
        transform.optimizer = torch.optim.Adam([transform.u, transform.v], lr=args.lr)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(item) for item in args.tasks.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
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
                    previous_chunk = None
                    while not success and not done and steps < MAX_STEPS[args.suite]:
                        batch = policy_observation(observation, task.language, policy)
                        if transform is not None:
                            prepared = prepare_observation(
                                batch, config=policy.config, normalizer=policy.normalizer, device=args.device
                            )
                            if previous_chunk is not None and not (args.adapt_once and transform.adapted_once):
                                # adapt only the small low-rank interface transform;
                                # fork_rng keeps the execution stream identical to base
                                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                                target_chunk = None
                                if args.verifier is not None:
                                    # 1) sample K candidates WITHOUT the transform, fixed seeds
                                    candidates = []
                                    with torch.random.fork_rng(devices=devices):
                                        for k in range(args.candidates):
                                            torch.manual_seed(args.adapt_seed + k)
                                            transform.active = False
                                            transform.reset()
                                            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                                cand = policy.normalizer.unnormalize_action(
                                                    sample_with_grad(
                                                        policy.action_generator, representation,
                                                        policy.config["action_generator"]["num_sample_steps"],
                                                    )[0].float()
                                                )
                                            candidates.append(cand)
                                    cands = torch.stack(candidates)
                                    mean_c = cands.mean(dim=0)
                                    disagreement = float(torch.sqrt(((cands - mean_c) ** 2).mean()).item())
                                    if disagreement < args.disagree_threshold:
                                        transform.gated = True
                                    else:
                                        if args.verifier == "leftover":
                                            tgt = torch.tensor(previous_chunk[args.exec_len:], device=mean_c.device, dtype=mean_c.dtype)
                                            scores = [float(torch.nn.functional.mse_loss(c[: tgt.shape[0]], tgt)) for c in cands]
                                        elif args.verifier == "consensus":
                                            scores = [float(torch.sqrt(((c - mean_c) ** 2).mean())) for c in cands]
                                        else:  # smooth
                                            scores = [float((c[1:] - c[:-1]).abs().mean()) for c in cands]
                                        target_chunk = cands[int(min(range(len(scores)), key=lambda i: scores[i]))].detach()
                                with torch.random.fork_rng(devices=devices):
                                    for _ in range(args.adapt_steps):
                                        torch.manual_seed(args.adapt_seed)
                                        transform.active = True
                                        transform.reset()
                                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                            representation = policy.representation_extractor(
                                                prepared["images"], prepared["state"], prepared["instruction"]
                                            )
                                        # fixed sampling noise for a fair pre/post comparison
                                        with torch.random.fork_rng(devices=devices):
                                            torch.manual_seed(args.adapt_seed)
                                            chunk = policy.normalizer.unnormalize_action(
                                                sample_with_grad(
                                                    policy.action_generator, representation,
                                                    policy.config["action_generator"]["num_sample_steps"],
                                                )[0].float()
                                            )
                                        target = torch.tensor(
                                            previous_chunk[args.exec_len:], device=chunk.device, dtype=chunk.dtype
                                        )
                                        if target_chunk is not None:
                                            loss = torch.nn.functional.mse_loss(chunk, target_chunk)
                                        else:
                                            loss = leftover_loss(chunk, target, args.exec_len)
                                        pre = float(loss.detach())
                                        loss = loss + args.reg * (transform.u @ transform.v.T).pow(2).sum()
                                        transform.optimizer.zero_grad(set_to_none=True)
                                        loss.backward()
                                        transform.optimizer.step()
                                        if args.restore > 0:
                                            with torch.no_grad():
                                                transform.u.mul_(1.0 - args.restore)
                                                transform.v.mul_(1.0 - args.restore)
                                        if args.uv_clip > 0:
                                            with torch.no_grad():
                                                prod = transform.u @ transform.v.T
                                                n = float(prod.norm())
                                                if n > args.uv_clip:
                                                    transform.u.mul_(args.uv_clip / n)
                                                    transform.v.mul_(args.uv_clip / n)
                                        transform.adapted_once = True
                                        with torch.no_grad(), torch.random.fork_rng(devices=devices):
                                            torch.manual_seed(args.adapt_seed)
                                            post_chunk = policy.normalizer.unnormalize_action(
                                                sample_with_grad(
                                                    policy.action_generator, representation,
                                                    policy.config["action_generator"]["num_sample_steps"],
                                                )[0].float()
                                            )
                                            post = float(leftover_loss(post_chunk, target, args.exec_len))
                                        transform.loss_trace.append((pre, post, float((post_chunk - chunk).abs().mean())))
                            transform.active = True
                            transform.reset()
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                with torch.no_grad():
                                    chunk = policy.predict(batch)[0].float().cpu().numpy()
                            transform.active = False
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                with torch.no_grad():
                                    chunk = policy.predict(batch)[0].float().cpu().numpy()
                        previous_chunk = chunk
                        for action in chunk[: args.exec_len]:
                            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= MAX_STEPS[args.suite]:
                                break
                    record = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "episode_id": episode,
                        "seed": seed,
                        "success": success,
                        "policy_steps": steps,
                        "mode": args.mode,
                        "exec_len": args.exec_len,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(f"{args.suite} task={task_id} ep={episode} success={int(success)} steps={steps}", flush=True)
            finally:
                env.close()
    if transform is not None:
        print(f"[cpia] update_a_frac={transform.update_a_frac:.3f} gated_early={transform.gated}", flush=True)
        if transform.loss_trace:
            pre = [x[0] for x in transform.loss_trace]; post = [x[1] for x in transform.loss_trace]
            delta = [x[2] for x in transform.loss_trace]
            print(f"[cpia] loss_pre_mean={sum(pre)/len(pre):.4f} loss_post_mean={sum(post)/len(post):.4f} "
                  f"chunk_delta_mean={sum(delta)/len(delta):.4f} updates={len(pre)}", flush=True)
        transform.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
