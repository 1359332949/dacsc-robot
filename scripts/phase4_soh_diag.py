"""Phase-4 SOH t=0 diagnostic on the audited edbac60 50-condition set.

Reuses phase3_smolvla_unified_diag.prepare/run unchanged for the Frozen and
DeepGate baselines, then adds a SOH arm on the same case/init/physics state,
same t=0 observation, same Flow RNG reset.  No environment step is taken.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import phase3_smolvla_unified_diag as diag
from dacsc.smolvla_soh import SOHModule, SmolVLASOHAdapter
from phase3_smolvla_common import reference_for, seed_torch


def _load_soh(path: Path):
    p = torch.load(path, map_location='cuda', weights_only=False)
    if p.get('schema') != 'phase4_soh_ckpt_v1':
        raise ValueError('bad SOH checkpoint')
    if 'v_basis' in p and 'eigenvalues' in p:
        m = SOHModule(v_basis=p['v_basis'].cuda(), eigenvalues=p['eigenvalues'].cuda(),
                      rank=int(p['rank']), num_bases=int(p['num_bases']),
                      damping_beta=float(p.get('damping_beta', 1.0))).cuda()
    else:
        m = SOHModule(p['u_geo'].cuda(), rank=int(p['rank']), num_bases=int(p['num_bases'])).cuda()
    m.load_state_dict(p['state_dict'])
    return m.eval()


def _reduce(err, base):
    out = {}
    for k in ('xyz', 'rotation', 'action7'):
        out[k] = 100.0 * (1.0 - err[k] / base[k]) if base[k] > 0 else None
    return out


def run(out: Path, soh_path: Path):
    # Prepare and run the frozen/deepgate audit first.
    if not (out / 'manifest.json').exists():
        diag.prepare(out)
    if not (out / 'cases').exists():
        diag.run(out)
    manifest = diag.read(out / 'manifest.json')
    actor = diag.base.load_actor()
    soh = _load_soh(soh_path)
    funcs = dict(diag.base.corruption_functions(), glass_blur=diag.glass_blur_exact)
    rows = []
    for i, case in enumerate(manifest['cases']):
        p = out / 'cases' / (case['id'] + '.json')
        row = diag.read(p)
        refs = reference_for(diag.CALIB, case['task_id'])
        tau = manifest['protocol']['gate'][str(case['task_id'])]['threshold']
        suite, _, raw = diag.base.setup(case)
        try:
            obs = diag.base.perturb(raw, case, 0, funcs)
            actor.reset(0)
            seed_torch(0)
            adapter = SmolVLASOHAdapter(actor._model, refs, tau, soh)
            try:
                adapter.begin()
                action = actor.act(obs)
                data = adapter.end()
            finally:
                adapter.close()
            assert action.shape == (7,) and np.isfinite(action).all()
            clean = np.asarray(row['clean_frozen_action'], dtype=np.float32)
            frozen = np.asarray(row['frozen_action'], dtype=np.float32)
            err = diag.errors(action, clean)
            base_err = diag.errors(frozen, clean)
            row['soh'] = dict(action=action.tolist(), errors=err, drift_from_frozen=diag.errors(action, frozen),
                              reduction_vs_frozen_pct=_reduce(err, base_err), gate=data,
                              accepted=bool(data['accepted']))
            diag.write(p, row)
            rows.append(row)
            print(json.dumps({'i': i + 1, 'case': case['id'], 'group': case['group'],
                              'accepted': row['soh']['accepted'],
                              'reduction': row['soh']['reduction_vs_frozen_pct']}), flush=True)
        finally:
            suite.close()

    groups = {}
    for g in ('Clean', 'Lighting', 'Background', 'Sensor'):
        grp = [r for r in rows if r['case']['group'] == g]
        groups[g] = dict(
            n=len(grp),
            soh_accepted=sum(r['soh']['accepted'] for r in grp),
            soh_mean_reduction_pct={k: float(np.mean([r['soh']['reduction_vs_frozen_pct'][k] for r in grp]))
                                    for k in ('xyz', 'rotation', 'action7')},
            deepgate_mean_reduction_pct={k: float(np.mean([
                100.0 * (1.0 - r['methods']['deep_gate_L3_a0.75']['errors'][k] / r['baseline_errors'][k])
                for r in grp if r['baseline_errors'][k] > 0])) for k in ('xyz', 'rotation', 'action7')},
            frozen_mean_error={k: float(np.mean([r['baseline_errors'][k] for r in grp])) for k in ('xyz', 'rotation', 'action7')},
        )
    summary = dict(schema='phase4_soh_diag_v1', soh_checkpoint_sha256=diag.base.sha(soh_path),
                   groups=groups,
                   comparison='DeepGate values are recomputed from the same diag rows; edbac60 report values may be rounded',
                   scope='t=0 only; previously accessed development conditions; not a success rate',
                   blind_access=False)
    diag.write(out / 'soh_summary.json', summary)
    lines = ['# SOH t=0 diagnostic', '',
             '| group | n | accepted | SOH xyz red% | SOH rot red% | DeepGate xyz red% | DeepGate rot red% |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g, x in groups.items():
        lines.append(f"| {g} | {x['n']} | {x['soh_accepted']} | {x['soh_mean_reduction_pct']['xyz']:+.2f} | "
                     f"{x['soh_mean_reduction_pct']['rotation']:+.2f} | {x['deepgate_mean_reduction_pct']['xyz']:+.2f} | "
                     f"{x['deepgate_mean_reduction_pct']['rotation']:+.2f} |")
    lines += ['', 'Negative reduction means the arm is worse than Frozen at the same t=0 state.', '']
    (out / 'PHASE4_SOH_DIAG.md').write_text('\n'.join(lines))
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--soh', type=Path, required=True)
    args = p.parse_args()
    run(args.out, args.soh)
