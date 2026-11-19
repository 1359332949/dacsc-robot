"""Closed-loop privileged same-state Clean-patch oracle (B2) for target-dev pilot."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/hy-tmp/patch-diag')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'repos/CoRP')); sys.path.insert(0, str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, model_compatible, compare, camera_compare, digest, digest_model, digest_controller  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def make_rep(encoded, patches=None, context=None):
    """Use patches from encoded or a donor; context from encoded Shift."""
    ext = encoded['_extractor'] if '_extractor' in encoded else None
    raise RuntimeError('unused')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--split-json', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--suite', default='libero_spatial')
    ap.add_argument('--lighting-id', type=int, default=2292)
    ap.add_argument('--state-ids', default='36,37,43,45')
    ap.add_argument('--domain', choices=['shift'], default='shift')
    ap.add_argument('--env-seed', type=int, default=7777)
    ap.add_argument('--z0-seed', type=int, default=7777)
    ap.add_argument('--modes', default='frozen,oracle')
    ap.add_argument('--executed-k', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=220)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--render-gpu', type=int, default=0)
    ap.add_argument('--libero-src', type=Path, default=Path('/hy-tmp/libero_plus'))
    ap.add_argument('--libero-config', type=Path, default=Path('/hy-tmp/libero_config'))
    args = ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text()); task = next(b for b in matrix['base_tasks'] if b['suite'] == args.suite)
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get('base_task') or '', args.lighting_id)
    variant = next(v for v in task['variants']['lighting'] if int(v['classification_id']) == args.lighting_id)
    states = torch.load(task['state_split']['source_file'], map_location='cpu', weights_only=False)
    language = BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    ids = resolve_indices(split, 'target_dev', args.state_ids)
    device = torch.device(args.device); policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ext = policy.representation_extractor
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    records = []
    for state_id in ids:
        for mode in modes:
            shift = OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'], camera_heights=256, camera_widths=256, render_gpu_device_id=args.render_gpu)
            clean = OffScreenRenderEnv(bddl_file_name=variant['base_bddl'], camera_heights=256, camera_widths=256, render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.env_seed); torch.manual_seed(args.env_seed)
                shift.seed(args.env_seed); shift.reset(); obs = shift.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS): obs, _, _, _ = shift.step(DUMMY_ACTION)
                steps = 0; chunk = 0; success = False; done = False
                chunk_log = []
                while not success and not done and steps < args.max_steps:
                    po_shift = policy_observation(obs, language, policy)
                    torch.manual_seed(args.z0_seed + 100003 * int(state_id) + chunk)
                    z0 = torch.randn((1, policy.action_generator.action_horizon, policy.action_generator.action_dim), device=device, dtype=next(policy.parameters()).dtype)
                    shift_enc = encode_observation(policy, po_shift, z0)
                    chosen = shift_enc
                    shift_enc_pair = shift_enc
                    if mode in ('oracle', 'frozen_refreshed'):
                        # Refresh only derived data for the diagnostic branch.
                        shift.sim.forward(); shift._update_observables(force=True)
                        obs_shift_fresh = shift.env._get_observations()
                        po_shift_fresh = policy_observation(obs_shift_fresh, language, policy)
                        shift_enc_pair = encode_observation(policy, po_shift_fresh, z0)
                        shift_refresh_max = float((shift_enc_pair['state_input'] - shift_enc['state_input']).abs().max().item())
                        if mode == 'frozen_refreshed':
                            chosen = shift_enc_pair
                            shift_patch_hash = __import__('hashlib').sha256(
                                b''.join(shift_enc_pair['patches'][v].detach().float().cpu().numpy().tobytes()
                                         for v in sorted(shift_enc_pair['patches']))).hexdigest()
                            z0_hash = __import__('hashlib').sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest()
                            action_hash = __import__('hashlib').sha256(chosen['controller_actions'].detach().float().cpu().numpy().tobytes()).hexdigest()
                            state_hash = __import__('hashlib').sha256(chosen['state_token'].detach().float().cpu().numpy().tobytes()).hexdigest()
                            chunk_log.append({'chunk': chunk, 'mode': mode, 'patch_source': 'shift',
                                              'shift_patch_sha256': shift_patch_hash,
                                              'z0_sha256': z0_hash, 'action_sha256': action_hash,
                                              'state_token_sha256': state_hash,
                                              'shift_refreshed_state_max': shift_refresh_max})
                        else:
                            snap = snapshot(shift)
                            clean.seed(args.env_seed + 991 * chunk); clean.reset()
                            clean.env.sim.model.body_pos[:] = np.asarray(snap['model_signature']['body_pos'], float)
                            clean.env.sim.model.body_quat[:] = np.asarray(snap['model_signature']['body_quat'], float)
                            clean.env.sim.forward()
                            restore(clean, snap); clean._update_observables(force=True)
                            obs_clean = clean.env._get_observations()
                            snap_clean = snapshot(clean)
                            compat = model_compatible(snap_clean, snap, strict_bodies=True,
                                                      strict_geometry=True, strict_physics=True)
                            state_check = compare(snap, snap_clean, tol={"body_xpos": 1e-3, "eef_pos": 1e-3, "body_xquat_rad": 2e-3})
                            camera_check = camera_compare(snap, snap_clean, expect_equal=True, tol=1e-3)
                            po_clean = policy_observation(obs_clean, language, policy)
                            clean_enc = encode_observation(policy, po_clean, z0)
                            context_state_max = float((clean_enc['state_input'] - shift_enc_pair['state_input']).abs().max().item())
                            context_token_l2 = float((clean_enc['state_token'] - shift_enc_pair['state_token']).norm().item())
                            context_lang_max = float((clean_enc['language_tokens'] - shift_enc_pair['language_tokens']).abs().max().item())
                            context_pooled_max = float((clean_enc['pooled_language'] - shift_enc_pair['pooled_language']).abs().max().item())
                            pairing_pass = bool(compat["pass"] and state_check["all_pass"] and camera_check["pass"]
                                                and context_state_max <= 1e-6 and context_token_l2 <= 1e-5
                                                and context_lang_max <= 1e-6 and context_pooled_max <= 1e-6)
                            visual = ext.cross_attention_resampler(clean_enc['patches'], shift_enc_pair['pooled_language'])
                            rep = Representation(tokens=torch.cat([visual, shift_enc_pair['language_tokens'].to(visual.dtype), shift_enc_pair['state_token']], dim=1),
                                                 pooled_language=shift_enc_pair['pooled_language'], action_dtype=visual.dtype)
                            from dacsc.differentiable_policy import sample_from_z0
                            with torch.no_grad():
                                norm = sample_from_z0(policy, rep, z0)
                                ctrl = policy.normalizer.unnormalize_action(norm)
                            chosen = {'controller_actions': ctrl}
                            shift_patch_hash = __import__('hashlib').sha256(
                                b''.join(shift_enc_pair['patches'][v].detach().float().cpu().numpy().tobytes()
                                         for v in sorted(shift_enc_pair['patches']))).hexdigest()
                            clean_patch_hash = __import__('hashlib').sha256(
                                b''.join(clean_enc['patches'][v].detach().float().cpu().numpy().tobytes()
                                         for v in sorted(clean_enc['patches']))).hexdigest()
                            z0_hash = __import__('hashlib').sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest()
                            action_hash = __import__('hashlib').sha256(ctrl.detach().float().cpu().numpy().tobytes()).hexdigest()
                            state_hash = __import__('hashlib').sha256(shift_enc_pair['state_token'].detach().float().cpu().numpy().tobytes()).hexdigest()
                            chunk_log.append({'chunk': chunk, 'mode': mode, 'patch_source': 'same_state_clean',
                                              'pairing_pass': pairing_pass,
                                              'model_compat_pass': bool(compat['pass']),
                                              'state_compare_pass': bool(state_check['all_pass']),
                                              'camera_compare_pass': bool(camera_check['pass']),
                                              'context_state_max': context_state_max,
                                              'context_token_l2': context_token_l2,
                                              'context_language_max': context_lang_max,
                                              'context_pooled_max': context_pooled_max,
                                              'shift_snapshot_digest': digest(snap),
                                              'clean_snapshot_digest': digest(snap_clean),
                                              'shift_model_digest': digest_model(snap),
                                              'clean_model_digest': digest_model(snap_clean),
                                              'shift_controller_digest': digest_controller(snap),
                                              'clean_controller_digest': digest_controller(snap_clean),
                                              'shift_patch_sha256': shift_patch_hash,
                                              'clean_patch_sha256': clean_patch_hash,
                                              'z0_sha256': z0_hash,
                                              'action_sha256': action_hash,
                                              'state_token_sha256': state_hash,
                                              'clean_state_input_max': context_state_max,
                                              'state_token_l2': context_token_l2,
                                              'shift_refreshed_state_max': shift_refresh_max})
                            if not pairing_pass:
                                print(json.dumps({'oracle_audit_fail': True, 'state': int(state_id), 'chunk': chunk,
                                                  'model': compat['checks'], 'state_checks': {k: v for k, v in state_check['checks'].items() if not v.get('pass')},
                                                  'camera': camera_check['checks'], 'context': {'state': context_state_max, 'token': context_token_l2,
                                                                                                'language': context_lang_max, 'pooled': context_pooled_max}}, default=str), flush=True)
                                raise RuntimeError(f'oracle pairing audit failed state={state_id} chunk={chunk}')
                    else:
                        shift_patch_hash = __import__('hashlib').sha256(
                            b''.join(shift_enc['patches'][v].detach().float().cpu().numpy().tobytes()
                                     for v in sorted(shift_enc['patches']))).hexdigest()
                        z0_hash = __import__('hashlib').sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest()
                        action_hash = __import__('hashlib').sha256(shift_enc['controller_actions'].detach().float().cpu().numpy().tobytes()).hexdigest()
                        state_hash = __import__('hashlib').sha256(shift_enc['state_token'].detach().float().cpu().numpy().tobytes()).hexdigest()
                        chunk_log.append({'chunk': chunk, 'mode': mode, 'patch_source': 'shift',
                                          'shift_patch_sha256': shift_patch_hash,
                                          'z0_sha256': z0_hash, 'action_sha256': action_hash,
                                          'state_token_sha256': state_hash})
                    ctrl = chosen['controller_actions']
                    for k in range(min(args.executed_k, ctrl.shape[1])):
                        obs, reward, done, info = shift.step(np.clip(ctrl[0, k].detach().cpu().numpy(), -1, 1))
                        steps += 1
                        success = is_success(shift, info, reward, done)
                        if success or done or steps >= args.max_steps: break
                    chunk += 1
                records.append({'state_id': int(state_id), 'mode': mode, 'success': bool(success), 'steps': int(steps),
                                'hit_max_steps': bool(steps >= args.max_steps and not success), 'chunks': chunk_log})
                print(json.dumps(records[-1]), flush=True)
            finally:
                shift.close(); clean.close()
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps({'schema': 'patch_closed_loop_oracle_v1', 'records': records}, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'n': len(records)}))


if __name__ == '__main__':
    main()
