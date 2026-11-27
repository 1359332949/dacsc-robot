"""Phase-4 closed-loop entry for SOH-VLA.

This file deliberately reuses the audited Phase-3 harness (same paired cases,
render-state checks, RNG protocol, 220-step budget and reporting).  It changes
only the adapter factory and the arm name; the old files stay untouched as
comparison baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import phase3_smolvla_deepgate_closedloop as base
from dacsc.smolvla_soh import SOHModule, SmolVLASOHAdapter


def load_soh(path: Path):
    payload = torch.load(path, map_location='cuda', weights_only=False)
    if payload.get('schema') != 'phase4_soh_ckpt_v1':
        raise ValueError('unexpected SOH checkpoint schema')
    u_geo = payload['u_geo'].cuda().float()
    soh = SOHModule(u_geo, rank=int(payload['rank']), num_bases=int(payload['num_bases'])).cuda()
    soh.load_state_dict(payload['state_dict'])
    soh.eval()
    return soh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'preflight', 'latency', 'worker', 'run', 'report'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--soh', type=Path, default=Path('runs/phase4_soh/soh_ckpt.pt'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    soh = load_soh(args.soh)

    def factory(policy, references, threshold, *, audit_inputs=False, decision_override=None):
        return SmolVLASOHAdapter(policy, references, threshold, soh,
                                 audit_inputs=audit_inputs, decision_override=decision_override)

    base.SmolVLADeepGate = factory
    base.ARMS = ('frozen', 'soh_streaming_v1')
    base.__file__ = str(Path(__file__).resolve())
    try:
        if args.mode == 'prepare':
            base.prepare(args.out)
        elif args.mode == 'preflight':
            base.preflight(args.out)
        elif args.mode == 'latency':
            base.latency(args.out)
        elif args.mode == 'worker':
            base.worker(args.out, args.shard)
        elif args.mode == 'report':
            base.report(args.out)
        else:
            base.run(args.out)
    except Exception as e:
        base.write(args.out / (args.mode + '_failure.json'), dict(error=str(e)))
        if args.mode == 'run':
            base.write(args.out / 'status.json', dict(phase='failed', error=str(e)))
        raise


if __name__ == '__main__':
    main()
