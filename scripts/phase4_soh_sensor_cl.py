"""Sensor-only paired closed loop for the 20-epoch SOH checkpoint.

Uses the audited Phase-3 rollout/pairing/render-state/RNG harness.  The only
changed arm is the adapter factory; the 20 Sensor conditions are the edbac60
severity-3 bilateral set.  No Clean/Lighting/Background conditions are run.
"""
from __future__ import annotations

import argparse
import hashlib
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
OUT = ROOT / 'runs/phase4_soh_sensor_cl'


def load_soh(path: Path):
    p = torch.load(path, map_location='cuda', weights_only=False)
    if p.get('schema') != 'phase4_soh_ckpt_v1':
        raise ValueError('bad checkpoint schema')
    if 'v_basis' in p and 'eigenvalues' in p:
        cls = SOHModuleV2 if p.get('arch') == 'v2' else SOHModule
        m = cls(v_basis=p['v_basis'].cuda(), eigenvalues=p['eigenvalues'].cuda(),
                      rank=int(p['rank']), num_bases=int(p['num_bases']),
                      damping_beta=float(p.get('damping_beta', 1.0))).cuda()
    else:
        cls = SOHModuleV2 if p.get('arch') == 'v2' else SOHModule
        m = cls(p['u_geo'].cuda(), rank=int(p['rank']), num_bases=int(p['num_bases'])).cuda()
    m.load_state_dict(p['state_dict'])
    return m.eval()


def cases():
    m = json.loads((DIAG / 'manifest.json').read_text())
    cs = [dict(c, scored=True) for c in m['cases'] if c['group'] == 'Sensor']
    assert len(cs) == 20, len(cs)
    return m, cs


def preflight(soh_path: Path, ema_beta: float = 0.8):
    m, cs = cases()
    actor = load_actor()
    soh = load_soh(soh_path)
    funcs = dict(shift.corruption_functions(), glass_blur=base.glass_blur_exact)
    c = cs[0]
    suite, _, raw = shift.setup(c)
    try:
        refs = reference_for(CALIB, c['task_id'])
        tau = m['protocol']['gate'][str(c['task_id'])]['threshold']
        obs = shift.perturb(raw, c, 0, funcs)
        actor.reset(0); seed_torch(0)
        frozen = actor.act(obs)
        actor.reset(0); seed_torch(0)
        off = SmolVLASOHAdapter(actor._model, refs, tau, soh, ema_beta=ema_beta, decision_override=False)
        off.begin(); a_off = actor.act(obs); d_off = off.end(); off.close()
        actor.reset(0); seed_torch(0)
        on = SmolVLASOHAdapter(actor._model, refs, tau, soh, ema_beta=ema_beta, decision_override=True)
        on.begin(); a_on = actor.act(obs); d_on = on.end(); on.close()
        assert np.array_equal(frozen, a_off), 'reject branch differs from Frozen'
        assert np.isfinite(a_on).all(), 'accept branch non-finite'
        assert d_off['encoder_calls'] == 2 and d_on['encoder_calls'] == 4
        result = dict(status='PASS', case=c['id'], reject_equals_frozen=True,
                      accept_finite=True, reject_encoder_calls=2, accept_encoder_calls=4,
                      soh_sha256=hashlib.sha256(Path(soh_path).read_bytes()).hexdigest())
        (OUT).mkdir(parents=True, exist_ok=True)
        (OUT / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
        print('PREFLIGHT_PASS', c['id'], flush=True)
    finally:
        suite.close()


def pair_check(rows):
    f, g = rows['frozen'], rows['soh']
    assert f['start'] == g['start'] and f['observed_start'] == g['observed_start']
    for a, b in zip(f['trace'], g['trace']):
        assert a['step'] == b['step'] and a['rng_before'] == b['rng_before'] and a['rng_after'] == b['rng_after']
        if a['snapshot'] == b['snapshot']:
            assert a['action'] == b['action'] if not b['gate']['accepted'] else True
    return dict(initial_state_and_two_view_pixels_exact=True, rng_prefix_exact=True)


def worker(soh_path: Path, shard: int, ema_beta: float = 0.8):
    m, cs = cases()
    actor = load_actor()
    soh = load_soh(soh_path)
    funcs = dict(shift.corruption_functions(), glass_blur=base.glass_blur_exact)
    base.ARMS = ('frozen', 'soh')
    base.SmolVLADeepGate = lambda policy, references, threshold, **kw: SmolVLASOHAdapter(
        policy, references, threshold, soh, ema_beta=ema_beta, **kw)
    dest = OUT / 'pairs'; dest.mkdir(parents=True, exist_ok=True)
    for idx, c in enumerate(cs):
        if idx % 4 != shard:
            continue
        path = dest / (c['id'] + '.json')
        if path.exists():
            continue
        refs = reference_for(CALIB, c['task_id'])
        tau = m['protocol']['gate'][str(c['task_id'])]['threshold']
        rows = {}
        pixel_cache = {}
        for arm in ('frozen', 'soh'):
            rows[arm] = base.rollout(actor, c, arm, refs, tau, funcs, pixel_cache)
            print('ARM', c['id'], arm, rows[arm]['success'], rows[arm]['steps'], flush=True)
        handle = pair_check(rows)
        path.write_text(json.dumps(dict(case=c, arms=rows, pairing=handle), indent=2) + '\n')
        print('PAIR_DONE', c['id'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('preflight', 'worker', 'report'))
    p.add_argument('--soh', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--ema-beta', type=float, default=0.8)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'preflight':
        preflight(a.soh, a.ema_beta)
    elif a.mode == 'worker':
        worker(a.soh, a.shard, a.ema_beta)
    else:
        raise SystemExit('report not implemented')
