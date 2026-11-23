"""Diagnostic A: t=0 z0 sensitivity of frozen CoRP per target-dev state.

No env rollout. For each state:
- settle once, fix observation
- sample K z0 from a fixed CUDA generator
- measure action dispersion across z0
- compare wipeout / immune / edge groups
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path('/hy-tmp/patch-diag')
sys.path[:0] = [str(ROOT), str(ROOT/'repos/CoRP'), str(ROOT/'repos/CoRP/scripts')]
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure, sample_z, visual_from_obs  # noqa

GROUPS = {
    'immune': [36, 39, 49],
    'wipeout': [34, 35, 43, 45],
    'edge': [37, 44, 48],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=7777)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--suite', default='libero_spatial')
    ap.add_argument('--lighting-id', type=int, default=2292)
    ap.add_argument('--libero-src', type=Path, default=Path('/hy-tmp/libero_plus'))
    ap.add_argument('--libero-config', type=Path, default=Path('/hy-tmp/libero_config'))
    ap.add_argument('--render-gpu', type=int, default=0)
    args = ap.parse_args()
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix['base_tasks'] if b['suite'] == args.suite)
    variant = next(v for v in task['variants']['lighting'] if int(v['classification_id']) == args.lighting_id)
    states = torch.load(task['state_split']['source_file'], map_location='cpu', weights_only=False)
    language = BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    policy = load_policy(args.checkpoint, device=torch.device('cuda:0'))
    policy.eval()
    rows = []
    for sid in [s for group in GROUPS.values() for s in group]:
        env = OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],
                                 camera_heights=256, camera_widths=256,
                                 render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            env.seed(args.seed); env.reset(); obs = env.set_init_state(states[sid])
            for _ in range(SETTLING_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            po = policy_observation(obs, language, policy)
            _, rep = visual_from_obs(policy, po)
            actions = []
            for j in range(args.k):
                g = torch.Generator(device='cuda:0'); g.manual_seed(args.seed + 100003 * sid + j)
                z0 = torch.randn((1, 12, 7), device='cuda:0', dtype=torch.float32, generator=g)
                a = policy.normalizer.unnormalize_action(sample_z(policy, rep, z0))[0].float().cpu().numpy()
                actions.append(np.clip(a, -1, 1))
            A = np.stack(actions)  # [K,12,7]
            xyz = A[..., :3]; rot = A[..., 3:6]; grip = A[..., 6]
            rows.append({
                'state_id': sid,
                'xyz_std_mean': float(np.mean(np.std(xyz, axis=0))),
                'rot_std_mean': float(np.mean(np.std(rot, axis=0))),
                'grip_std_mean': float(np.mean(np.std(grip, axis=0))),
                'xyz_pair_rms': float(np.mean([np.sqrt(np.mean((A[i, :, :3] - A[j, :, :3]) ** 2))
                                                for i in range(args.k) for j in range(i + 1, args.k)])),
                'grip_sign_flip_rate': float(np.mean([
                    np.mean(np.sign(A[i, :, 6]) != np.sign(A[j, :, 6]))
                    for i in range(args.k) for j in range(i + 1, args.k)])),
            })
            print(json.dumps({'state': sid, 'xyz_std': rows[-1]['xyz_std_mean'], 'grip_flip': rows[-1]['grip_sign_flip_rate']}), flush=True)
        finally:
            env.close()
    by_state = {int(r['state_id']): r for r in rows}
    summary = {}
    for name, ids in GROUPS.items():
        summary[name] = {k: float(np.mean([by_state[s][k] for s in ids]))
                         for k in ('xyz_std_mean', 'rot_std_mean', 'grip_std_mean', 'xyz_pair_rms', 'grip_sign_flip_rate')}
    args.out.write_text(json.dumps({'schema': 'diagnostic_a_z0_sensitivity_v1',
                                    'k': args.k, 'rows': rows, 'groups': summary}, indent=2) + '\n')


if __name__ == '__main__':
    main()
