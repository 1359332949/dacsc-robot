"""Clean-five SOH closed-loop evaluation with resumable (seed, case) pairs.

Default run: seed 0 x (20 Sensor + 10 Background) = 30 pairs / 60 episodes.
Later seeds can be added without recomputing seed 0 because every pair is
stored by (seed, case_id).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

import phase3_smolvla_deepgate_closedloop as base
import phase3_smolvla_shift as shift
from phase3_smolvla_common import load_actor, reference_for, seed_torch
from dacsc.smolvla_soh import SOHModule, SOHModuleV2, SmolVLASOHAdapter

ROOT = Path('/hy-tmp/patch-diag')
DIAG = ROOT / 'runs/phase3_smolvla_unified_diag_v1'
CALIB = ROOT / 'runs/phase3_smolvla/spatial_init0_calib.npz'
OUT = Path(os.environ.get('SOH_EVAL_OUT', str(ROOT / 'runs/phase4_soh_clean5_eval')))
GROUPS = ('Sensor', 'Background')


def load_soh(path: Path):
    p = torch.load(path, map_location='cuda', weights_only=False)
    if p.get('schema') != 'phase4_soh_ckpt_v1':
        raise ValueError('bad schema')
    cls = SOHModuleV2 if p.get('arch') == 'v2' else SOHModule
    if 'v_basis' in p and 'eigenvalues' in p:
        m = cls(v_basis=p['v_basis'].cuda().float(), eigenvalues=p['eigenvalues'].cuda().float(),
                rank=int(p['rank']), num_bases=int(p['num_bases']),
                damping_beta=float(p.get('damping_beta', 1.0))).cuda()
    else:
        m = cls(p['u_geo'].cuda().float(), rank=int(p['rank']), num_bases=int(p['num_bases'])).cuda()
    m.load_state_dict(p['state_dict'])
    return m.eval()


def manifest(selected=GROUPS):
    m = json.loads((DIAG / 'manifest.json').read_text())
    cases = [dict(c, scored=True) for c in m['cases'] if c['group'] in selected]
    return m, cases


def preflight(soh_path: Path, groups=GROUPS):
    m, cases = manifest(groups)
    actor = load_actor(); soh = load_soh(soh_path)
    funcs = dict(shift.corruption_functions(), glass_blur=base.glass_blur_exact)
    checks = []
    for group in groups:
        c = next(x for x in cases if x['group'] == group)
        suite, _, raw = shift.setup(c)
        try:
            refs = reference_for(CALIB, c['task_id'])
            tau = m['protocol']['gate'][str(c['task_id'])]['threshold']
            obs = shift.perturb(raw, c, 0, funcs)
            actor.reset(0); seed_torch(0); frozen = actor.act(obs)
            actor.reset(0); seed_torch(0)
            off = SmolVLASOHAdapter(actor._model, refs, tau, soh, ema_beta=0.0, decision_override=False)
            off.begin(); a_off = actor.act(obs); d_off = off.end(); off.close()
            actor.reset(0); seed_torch(0)
            on = SmolVLASOHAdapter(actor._model, refs, tau, soh, ema_beta=0.0, decision_override=True)
            on.begin(); a_on = actor.act(obs); d_on = on.end(); on.close()
            assert np.array_equal(frozen, a_off), f'{group} reject differs'
            assert np.isfinite(a_on).all() and d_off['encoder_calls'] == 2 and d_on['encoder_calls'] == 4
            checks.append(dict(group=group, case=c['id'], reject_equals_frozen=True, accept_finite=True,
                               reject_encoder_calls=2, accept_encoder_calls=4))
        finally:
            suite.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preflight.json').write_text(json.dumps(dict(status='PASS', checks=checks,
        soh_sha256=hashlib.sha256(soh_path.read_bytes()).hexdigest(), ema_beta=0.0), indent=2) + '\n')
    print('PREFLIGHT_PASS', flush=True)


def pair_check(rows):
    f, g = rows['frozen'], rows['soh']
    assert f['start'] == g['start'] and f['observed_start'] == g['observed_start']
    for a, b in zip(f['trace'], g['trace']):
        assert a['step'] == b['step'] and a['rng_before'] == b['rng_before'] and a['rng_after'] == b['rng_after']
    return dict(initial_state_and_two_view_pixels_exact=True, rng_prefix_exact=True)


def worker(soh_path: Path, shard: int, seed: int, groups=GROUPS):
    m, cases = manifest(groups)
    actor = load_actor(); soh = load_soh(soh_path)
    funcs = dict(shift.corruption_functions(), glass_blur=base.glass_blur_exact)
    original_seed = seed_torch
    # base.rollout hard-codes seed_torch(0); for a requested policy seed, the
    # wrapper preserves the paired RNG prefix while changing the policy stream.
    base.seed_torch = lambda _unused: original_seed(seed)
    base.ARMS = ('frozen', 'soh')
    base.SmolVLADeepGate = lambda policy, references, threshold, **kw: SmolVLASOHAdapter(
        policy, references, threshold, soh, ema_beta=0.0, **kw)
    dest = OUT / 'pairs'; dest.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, c) for c in cases]
    try:
        for idx, (s, c) in enumerate(jobs):
            if idx % 4 != shard:
                continue
            path = dest / f's{s}_{c["id"]}.json'
            if path.exists():
                continue
            refs = reference_for(CALIB, c['task_id'])
            tau = m['protocol']['gate'][str(c['task_id'])]['threshold']
            rows = {}; pixel_cache = {}
            for arm in ('frozen', 'soh'):
                rows[arm] = base.rollout(actor, c, arm, refs, tau, funcs, pixel_cache)
                rows[arm]['policy_seed'] = s  # actual stream set by the paired seed wrapper above
                print('ARM', s, c['id'], arm, rows[arm]['success'], rows[arm]['steps'], flush=True)
            path.write_text(json.dumps(dict(seed=s, case=c, arms=rows, pairing=pair_check(rows)), indent=2) + '\n')
            print('PAIR_DONE', s, c['id'], flush=True)
    finally:
        base.seed_torch = original_seed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('preflight', 'worker'))
    p.add_argument('--soh', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'preflight':
        preflight(a.soh, a.groups)
    else:
        worker(a.soh, a.shard, a.seed, a.groups)
