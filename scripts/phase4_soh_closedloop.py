"""Phase-4 closed-loop entry for SOH-VLA.

Reuses the audited Phase-3 environment/pairing/RNG harness but has its own
preflight, worker launch and report.  The Phase-3 preflight is intentionally not
reused: it asserts bit-exact equality with the old Alpha=.75 V1 branch, which is
false by construction once the SOH residual has a non-zero scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import phase3_smolvla_deepgate_closedloop as base
from dacsc.smolvla_soh import SOHModule, SOHModuleV2, SmolVLASOHAdapter


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_soh(path: Path):
    payload = torch.load(path, map_location='cuda', weights_only=False)
    if payload.get('schema') != 'phase4_soh_ckpt_v1':
        raise ValueError('unexpected SOH checkpoint schema')
    if 'v_basis' in payload and 'eigenvalues' in payload:
        v_basis = payload['v_basis'].cuda().float()
        eigenvalues = payload['eigenvalues'].cuda().float()
        cls = SOHModuleV2 if payload.get('arch') == 'v2' else SOHModule
        soh = cls(v_basis=v_basis, eigenvalues=eigenvalues,
                        rank=int(payload['rank']), num_bases=int(payload['num_bases'])).cuda()
    else:
        u_geo = payload['u_geo'].cuda().float()
        cls = SOHModuleV2 if payload.get('arch') == 'v2' else SOHModule
        soh = cls(u_geo, rank=int(payload['rank']), num_bases=int(payload['num_bases'])).cuda()
    soh.load_state_dict(payload['state_dict'])
    return soh.eval()


def _factory(soh):
    def make(policy, references, threshold, *, audit_inputs=False, decision_override=None):
        return SmolVLASOHAdapter(policy, references, threshold, soh,
                                 audit_inputs=audit_inputs, decision_override=decision_override)
    return make


def prepare(out: Path, soh_path: Path, subspace: Path):
    base.prepare(out)
    m = base.read(out / 'manifest.json')
    m['arms'] = ['frozen', 'soh_streaming_v1']
    m['soh_checkpoint_sha256'] = _sha(soh_path)
    m['subspace_sha256'] = _sha(subspace)
    m['method'] = ('deep Layer-12 gate; below threshold reuse pass-1 tokens exactly; '
                   'above threshold repair only Layer-3 in P_dom = I - U_geo U_geo^T; '
                   'streaming EMA fingerprint; native connector/VLM/Flow10 run once')
    m.pop('fingerprint', None)
    m['fingerprint'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    base.write(out / 'manifest.json', m)
    print('PREPARED', m['fingerprint'], flush=True)


def preflight(out: Path, soh):
    """Independent SOH preflight: off==Frozen, on calls 4 encoders, no weight drift."""
    m, gates = base.validate(out)
    actor = base.load_actor()
    funcs = dict(base.corruption_functions(), glass_blur=base.glass_blur_exact)
    case = next(c for c in m['cases'] if c['scored'] and c['group'] == 'Clean')
    suite, _, raw = base.setup(case)
    try:
        refs = base.reference_for(base.CALIB, case['task_id'])
        tau = gates[str(case['task_id'])]['threshold']
        obs = base.perturb(raw, case, 0, funcs)
        versions = [p._version for p in actor._model.parameters()]

        actor.reset(0); base.seed_torch(0)
        frozen = actor.act(obs)

        actor.reset(0); base.seed_torch(0)
        off = SmolVLASOHAdapter(actor._model, refs, tau, soh, decision_override=False)
        off.begin(); off_action = actor.act(obs); off_data = off.end(); off.close()
        if not np.array_equal(frozen, off_action):
            raise AssertionError('SOH reject branch drifted from Frozen')
        if off_data['encoder_calls'] != 2:
            raise AssertionError(f'reject encoder calls {off_data["encoder_calls"]} != 2')

        actor.reset(0); base.seed_torch(0)
        on = SmolVLASOHAdapter(actor._model, refs, tau, soh, decision_override=True)
        on.begin(); on_action = actor.act(obs); on_data = on.end(); on.close()
        if not np.isfinite(on_action).all():
            raise AssertionError('SOH accept branch produced non-finite action')
        if on_data['encoder_calls'] != 4:
            raise AssertionError(f'accept encoder calls {on_data["encoder_calls"]} != 4')
        if versions != [p._version for p in actor._model.parameters()]:
            raise AssertionError('policy parameters changed during preflight')
        base.write(out / 'preflight.json', dict(
            status='PASS', case_id=case['id'], reject_equals_frozen_bit_exact=True,
            accept_finite=True, reject_encoder_calls=2, accept_encoder_calls=4,
            soh_checkpoint_sha256=m['soh_checkpoint_sha256'],
            subspace_sha256=m['subspace_sha256'],
            weights_frozen_and_versions_unchanged=True,
            note='independent SOH preflight; does not assert equality with old Alpha=.75 V1'))
    finally:
        suite.close()


def report(out: Path, soh_path: Path):
    m, _ = base.validate(out)
    pairs = []
    for c in m['cases']:
        p = base.read(out / 'pairs' / (c['id'] + '.json'))
        if p['fingerprint'] != m['fingerprint'] or p['case'] != c:
            raise AssertionError('pair/manifest mismatch')
        base.paired_check(p['arms'])
        for v in p['arms'].values():
            if not (v['finite_actions'] and v['weights_frozen_and_versions_unchanged']
                    and len(v['trace']) == v['steps']):
                raise AssertionError('invalid arm record')
        pairs.append(p)
    if len(pairs) != 51:
        raise AssertionError(len(pairs))
    base.ARMS = ('frozen', 'soh_streaming_v1')
    groups = {g: base.stats([p for p in pairs if base.selected(p['case'], g)]) for g in base.GROUPS}
    extra = base.stats([p for p in pairs if not p['case']['scored']])
    compact = [dict(case=p['case'], frozen=p['arms']['frozen']['success'],
                    soh=p['arms'][base.ARMS[1]]['success'],
                    steps={a: p['arms'][a]['steps'] for a in base.ARMS},
                    first_gate_step=next((t['step'] for t in p['arms'][base.ARMS[1]]['trace']
                                          if t['gate']['accepted']), None)) for p in pairs]
    result = dict(schema='smolvla_soh_closedloop_v1', protocol=m['protocol'],
                  fingerprint=m['fingerprint'], soh_checkpoint_sha256=m['soh_checkpoint_sha256'],
                  subspace_sha256=m['subspace_sha256'], groups=groups,
                  supplemental_task0_clean_init1=extra, cases=compact,
                  preflight=base.read(out / 'preflight.json'), latency=base.read(out / 'latency.json'),
                  raw_sha256={c['id']: base.sha(out / 'pairs' / (c['id'] + '.json')) for c in m['cases']},
                  limitations=['SOH-VLA exploratory paired pilot; one seed and previously accessed development conditions',
                               'not an independent confirmation, not a SALT reproduction',
                               'task-cluster bootstrap has only ten clusters',
                               'SOH result does not prove universal geometry preservation'])
    base.write(out / 'summary.json', result)
    lines = ['# SOH-VLA: deep-gated orthogonal-subspace paired pilot', '',
             '50 conditions x Frozen/SOH = 100 episodes; supplemental task0 init1 pair excluded from the main table.', '',
             '| group | Frozen | SOH | rescued/harmed | net | exact two-sided p | task-cluster 95% CI pp |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group, x in groups.items():
        lo, hi = x['task_cluster_bootstrap95_pp']
        lines.append(f"| {group} | {x['frozen']}/{x['n']} | {x['soh']}/{x['n']} | {x['rescued']}/{x['harmed']} | {x['net']:+d} | {x['mcnemar_exact_two_sided']:.6g} | [{lo:+.1f}, {hi:+.1f}] |")
    lines += ['', '## Gate behaviour', '',
              '| group | accepted replans/total | step rate % | episode macro % | episodes ever | first frame |',
              '|---|---:|---:|---:|---:|---:|']
    for group, x in groups.items():
        g = x['gate']
        lines.append(f"| {group} | {g['accepted_replans']}/{g['total_replans']} | {100*g['micro_rate']:.2f} | {100*g['episode_macro_rate']:.2f} | {g['episodes_ever_accepted']}/{x['n']} | {g['first_observation_accepted']}/{x['n']} |")
    lines += ['', f"SOH checkpoint SHA256: {m['soh_checkpoint_sha256']}. Subspace SHA256: {m['subspace_sha256']}.", '']
    (out / 'PHASE4_SOH_CLOSEDLOOP.md').write_text('\n'.join(lines))


def run(out: Path, soh_path: Path):
    m, _ = base.validate(out)
    if base.read(out / 'preflight.json')['status'] != 'PASS' or not (out / 'latency.json').exists():
        raise AssertionError('preflight/latency incomplete')
    start = time.monotonic()
    processes, logs = [], []
    python = '/hy-tmp/smolvla_venv/bin/python'
    try:
        for k in range(4):
            log = (out / f'worker{k}.log').open('a')
            logs.append(log)
            processes.append(subprocess.Popen(
                [python, '-u', str(Path(__file__).resolve()), 'worker', '--out', str(out),
                 '--shard', str(k), '--soh', str(soh_path)],
                stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failed:
                raise RuntimeError(f'worker failure {failed}')
            complete = list((out / 'pairs').glob('*.json'))
            base.write(out / 'status.json', dict(phase='running', pairs_complete=len(complete),
                                                pairs_total=51, episodes_complete=2 * len(complete),
                                                seconds=time.monotonic() - start,
                                                pids=[p.pid for p in processes]))
            time.sleep(5)
        if not all(p.returncode == 0 for p in processes):
            raise RuntimeError('worker failed')
        report(out, soh_path)
        base.write(out / 'status.json', dict(phase='complete', main_episodes=100,
                                            supplemental_episodes=2, pairs=51,
                                            seconds=time.monotonic() - start,
                                            fingerprint=m['fingerprint']))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'preflight', 'latency', 'worker', 'run', 'report'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--soh', type=Path, default=Path('runs/phase4_soh/soh_ckpt.pt'))
    parser.add_argument('--subspace', type=Path, default=Path('runs/phase4_soh/subspace_init0.npz'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base.ARMS = ('frozen', 'soh_streaming_v1')
    base.__file__ = str(Path(__file__).resolve())

    if args.mode == 'prepare':
        prepare(args.out, args.soh, args.subspace)
        return
    if args.mode == 'report':
        report(args.out, args.soh)
        return
    soh = load_soh(args.soh)
    base.SmolVLADeepGate = _factory(soh)
    try:
        if args.mode == 'preflight':
            preflight(args.out, soh)
        elif args.mode == 'latency':
            base.latency(args.out)
        elif args.mode == 'worker':
            base.worker(args.out, args.shard)
        else:
            run(args.out, args.soh)
    except Exception as e:
        base.write(args.out / (args.mode + '_failure.json'), dict(error=str(e)))
        if args.mode == 'run':
            base.write(args.out / 'status.json', dict(phase='failed', error=str(e)))
        raise


if __name__ == '__main__':
    main()
