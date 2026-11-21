"""Reanalyse privileged target-dev residuals with outer-state leakage excluded.

Ridge alpha and features are frozen. Scalar calibration uses fitted predictions
on OUTER TRAIN states only (in-sample calibration, not an independent estimate).
No target-test data, new data collection or policy weight updates are involved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

KINDS = ('M0', 'M1a', 'M1w', 'M1both')
METRICS = ('rel', 'agg', 'cos', 'mag', 'G_row', 'G_agg')


def features(row, kind):
    base = np.r_[row['h'], row['state8'], row['tau'], row['k'] / 12]
    if kind == 'M0': return base
    if kind == 'M1a': return np.r_[base, row['z_agent']]
    if kind == 'M1w': return np.r_[base, row['z_eye']]
    if kind == 'M1both': return np.r_[base, row['z_agent'], row['z_eye']]
    raise ValueError(kind)


def ridge(x, y, alpha=100.):
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = (x - mu) / sd
    # Centered fit equals an unpenalized intercept in the legacy design matrix.
    ym = y.mean(0)
    if z.shape[1] <= z.shape[0]:
        b = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - ym))
    else:
        b = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(z.shape[0]), y - ym)
    return mu, sd, b, ym


def predict(x, model):
    mu, sd, b, ym = model
    return ((x - mu) / sd) @ b + ym


def fold_predictions(x, y, state_ids, held_out, alpha=100.):
    train, test = state_ids != held_out, state_ids == held_out
    if not train.any() or not test.any(): raise ValueError('empty train/test fold')
    model = ridge(x[train], y[train], alpha)
    train_prediction = predict(x[train], model)
    beta = float(np.clip(np.sum(train_prediction * y[train]) /
                         (np.sum(train_prediction ** 2) + 1e-12), 0, 1))
    constant = y[train].mean(0)
    p = predict(x[test], model)
    return p, np.broadcast_to(constant, p.shape).copy(), beta * p, {
        'train_states': sorted(set(state_ids[train].tolist())),
        'held_out_state': int(held_out), 'beta_train_in_sample': beta,
        'constant_train_mean': constant.tolist(),
        'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
    }


def metrics(p, y):
    rel = float(np.mean(np.linalg.norm(p-y, axis=1) / (np.linalg.norm(y, axis=1)+1e-9)))
    agg = float(np.sqrt(np.sum((p-y)**2) / (np.sum(y**2)+1e-12)))
    cos = float(np.mean(np.sum(p*y, axis=1) / (np.linalg.norm(p, axis=1)*np.linalg.norm(y, axis=1)+1e-9)))
    mag = float(np.mean(np.linalg.norm(p, axis=1)) / (np.mean(np.linalg.norm(y, axis=1))+1e-12))
    return dict(rel=rel, agg=agg, cos=cos, mag=mag, G_row=1-rel, G_agg=1-agg)


def group(data, ids):
    return {k: float(np.mean([data[str(s)][k] for s in ids])) for k in METRICS} if ids else None


def analyse(rows, pilot=(34, 43, 45)):
    y = np.asarray([r['r'] for r in rows], dtype=np.float64)
    s = np.asarray([r['state_id'] for r in rows])
    states = sorted(set(s.tolist())); pilot = sorted(set(pilot) & set(states))
    unseen = sorted(set(states) - set(pilot))
    if y.ndim != 2 or y.shape[1] != 7 or not np.isfinite(y).all():
        raise ValueError('residual must be finite Nx7')
    # Do not silently change the weighting compared with the archived experiment.
    counts = [int((s == i).sum()) for i in states]
    if len(set(counts)) != 1: raise ValueError('unequal rows/state; choose weighting explicitly')
    out = {'schema': 'b3_oracle10_leakage_fixed_v2', 'states': states,
           'unseen7': unseen, 'pilot3': pilot,
           'claim_scope': 'privileged target-dev oracle diagnostic; state is the independent unit',
           'calibration': 'outer-train in-sample scalar; not nested validation or deployable TTA',
           'aggregation': 'equal state mean; G_agg is mean of per-state G_agg',
           'zero': metrics(np.zeros_like(y), y)}
    for kind in KINDS:
        x = np.stack([features(r, kind) for r in rows])
        if not np.isfinite(x).all(): raise ValueError('nonfinite features')
        raw, const, calibrated, folds = {}, {}, {}, {}
        for state in states:
            p, c, cp, audit = fold_predictions(x, y, s, state)
            key = str(state); truth = y[s == state]
            raw[key], const[key], calibrated[key] = metrics(p, truth), metrics(c, truth), metrics(cp, truth)
            folds[key] = audit
        out[kind] = {'states': raw, 'constant': const, 'calibrated': calibrated,
                     'fold_audit': folds, 'all': group(raw, states),
                     'unseen7': group(raw, unseen), 'pilot3': group(raw, pilot),
                     'cal_all': group(calibrated, states), 'const_all': group(const, states),
                     'cal_unseen7': group(calibrated, unseen), 'const_unseen7': group(const, unseen)}
        print(json.dumps({'model': kind, 'all': out[kind]['all'],
                          'constant': out[kind]['const_all'], 'calibrated': out[kind]['cal_all']}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--legacy', type=Path)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    result = analyse(json.loads(args.input.read_text())['rows'])
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.legacy:
        legacy = json.loads(args.legacy.read_text())
        result['legacy_raw_metric_max_abs_diff'] = max(abs(result[k]['states'][s][m] - legacy[k]['states'][s][m])
            for k in KINDS for s in result[k]['states'] for m in METRICS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__': main()
