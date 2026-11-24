"""Offline audit of patch-pair NPZ/manifest: pass rates and state-input forensics."""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/hy-tmp/patch-diag')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'repos/CoRP')); sys.path.insert(0, str(ROOT / 'repos/CoRP/scripts'))
from corp import load_policy  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    return {
        'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
        'min': float(x.min()), 'p50': float(np.median(x)),
        'p95': float(np.quantile(x, .95)), 'max': float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=Path, default=Path('/hy-tmp/patch-diag/runs/patch_diag/source_patch_pairs.npz'))
    ap.add_argument('--manifest', type=Path, default=None)
    ap.add_argument('--phase0', type=Path, default=Path('/hy-tmp/patch-diag/runs/patch_diag/phase0_interface.json'))
    ap.add_argument('--phase1', type=Path, default=Path('/hy-tmp/patch-diag/runs/patch_diag/phase1_report.json'))
    ap.add_argument('--b0b3', type=Path, default=Path('/hy-tmp/patch-diag/runs/patch_diag/b0_b3_report.json'))
    ap.add_argument('--checkpoint', type=Path, default=Path('/hy-tmp/patch-diag/repos/CoRP/checkpoints/corp_libero.pt'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest_path = args.manifest or args.pairs.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    npz = np.load(args.pairs, allow_pickle=False)
    records = manifest['records']
    policy = load_policy(args.checkpoint, device=torch.device('cpu'))
    norm = policy.normalizer
    state_std = norm.state_std.detach().cpu().numpy().reshape(-1)
    state_mean = norm.state_mean.detach().cpu().numpy().reshape(-1)
    names = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'grip_l', 'grip_r']
    rows = []
    for r in records:
        i = int(r['feature_row'])
        cs = np.asarray(npz['clean_state'][i], dtype=float).reshape(-1)
        ss = np.asarray(npz['shift_state'][i], dtype=float).reshape(-1)
        d = ss - cs
        raw = d * state_std
        absd = np.abs(d)
        j = int(np.argmax(absd))
        cst = np.asarray(npz.get('clean_state_token', np.zeros((len(records), 1)))[i], dtype=float).reshape(-1)
        sst = np.asarray(npz.get('shift_state_token', np.zeros((len(records), 1)))[i], dtype=float).reshape(-1)
        rows.append({
            'feature_row': i, 'state_id': int(r['state_id']), 'role': r['role'],
            'window': r['window'], 'chunk_index': int(r['chunk_index']), 'step': int(r['step']),
            'pairing_pass': bool(r['pairing_pass']),
            'physical_state_compare_pass': bool(r['physical_state_compare_pass']),
            'model_compatibility_pass': bool(r['model_compatibility_pass']),
            'camera_compare_pass': bool(r['camera_compare_pass']),
            'controller_compare_pass': bool(r['controller_compare_pass']),
            'patch_shape_pass': bool(r['patch_shape_pass']),
            'state_input_max_abs': float(r['state_input_max_abs']),
            'delta': d.tolist(), 'raw_delta': raw.tolist(),
            'abs_delta': absd.tolist(),
            'max_dim': names[j], 'max_dim_index': j,
            'clean_state': cs.tolist(), 'shift_state': ss.tolist(),
            'state_token_l2': float(np.linalg.norm(sst - cst)),
        })
    # pass table by role/window/chunk
    table = []
    for role in ('source_fit', 'source_validation'):
        for window in ('early', 'late'):
            for chunk in sorted({r['chunk_index'] for r in rows if r['role'] == role and r['window'] == window}):
                sub = [r for r in rows if r['role'] == role and r['window'] == window and r['chunk_index'] == chunk]
                table.append({'role': role, 'window': window, 'chunk': chunk,
                              'total': len(sub), 'pass': sum(x['pairing_pass'] for x in sub),
                              'pass_rate': (sum(x['pairing_pass'] for x in sub) / len(sub)) if sub else 0.0,
                              'states_total': len({x['state_id'] for x in sub}),
                              'states_pass': len({x['state_id'] for x in sub if x['pairing_pass']})})
    # per-state chunk matrix
    state_chunks = defaultdict(list)
    for r in rows:
        state_chunks[(r['role'], r['state_id'])].append((r['window'], r['chunk_index'], r['pairing_pass'], r['state_input_max_abs']))
    # per-dim stats pass/fail
    dim_stats = {}
    for name, j in zip(names, range(8)):
        dim_stats[name] = {
            'all': stats([r['abs_delta'][j] for r in rows]),
            'pass': stats([r['abs_delta'][j] for r in rows if r['pairing_pass']]),
            'fail': stats([r['abs_delta'][j] for r in rows if not r['pairing_pass']]),
        }
    fail_physical = [r for r in rows if not r['physical_state_compare_pass']]
    fail_state = [r for r in rows if not r['pairing_pass'] and r['physical_state_compare_pass']]
    out = {
        'schema': 'corp_patch_pairs_audit_v1',
        'inputs': {str(p): sha(p) for p in [args.pairs, manifest_path, args.phase0, args.phase1, args.b0b3, args.checkpoint] if p and Path(p).exists()},
        'records_total': len(rows), 'records_pass': sum(r['pairing_pass'] for r in rows),
        'state_field_names': names,
        'state_mean': state_mean.tolist(), 'state_std': state_std.tolist(),
        'pass_table_by_chunk': table,
        'per_state_chunk': {f'{k[0]}/{k[1]}': v for k, v in sorted(state_chunks.items())},
        'dim_stats_normalized': dim_stats,
        'fail_reasons': {
            'physical_state_compare_fail': len(fail_physical),
            'only_state_input_fail': len(fail_state),
            'model_compat_fail': sum(not r['model_compatibility_pass'] for r in rows),
            'camera_fail': sum(not r['camera_compare_pass'] for r in rows),
            'controller_fail': sum(not r['controller_compare_pass'] for r in rows),
            'patch_shape_fail': sum(not r['patch_shape_pass'] for r in rows),
        },
        'state_input_stats': {
            'pass': stats([r['state_input_max_abs'] for r in rows if r['pairing_pass']]),
            'fail': stats([r['state_input_max_abs'] for r in rows if not r['pairing_pass']]),
        },
        'top_fail_dims': Counter(r['max_dim'] for r in rows if not r['pairing_pass']).most_common(),
        'top_all_dims': Counter(r['max_dim'] for r in rows).most_common(),
        'records': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'records': len(rows), 'pass': out['records_pass'],
                      'fail_reasons': out['fail_reasons'], 'pass_table': table}, indent=2))


if __name__ == '__main__':
    main()
