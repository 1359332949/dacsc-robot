"""Actual CoRP lifecycle check only; no rollout or success-rate claim."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.baselines_tta import ActMADInspired


def main(a):
    torch.set_num_threads(1)
    rows = [r for p in sorted(a.input.glob('index_*.json')) for r in json.loads(p.read_text())]
    row = next(r for r in rows if r['factor'] == 'clean' and not r['geometry'])
    path = a.input / (row['id'] + '.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['file_sha']
    z = np.load(path, allow_pickle=False)
    raw = {'images': {v: torch.from_numpy(z[v].copy())[None] for v in ('agentview', 'eye_in_hand')},
           'state': torch.from_numpy(z['state'].copy()), 'instruction': [row['lang']]}
    references = []
    tasks = []
    for p in sorted(a.calibration.glob('task_*.npz')):
        source = np.load(p, allow_pickle=False)
        task = str(source['task'].item())
        if task == row['task']:
            continue
        tasks.append(task)
        references.append((torch.from_numpy(source['mu_l']).cuda(), torch.from_numpy(source['ss_l']).cuda()))
    assert len(tasks) == 9 and row['task'] not in tasks
    mu = torch.stack([m for m, s in references]).mean(0)
    second = torch.stack([s.square() + m.square() for m, s in references]).mean(0)
    sd = (second - mu.square()).clamp_min(1e-6).sqrt()
    calibration = SimpleNamespace(ref=lambda i: (mu[i], sd[i], torch.zeros_like(mu[i])))
    policy = load_policy(a.checkpoint, device='cuda:0').eval().requires_grad_(False)
    batch = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer, device='cuda:0')
    def predict():
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(37); torch.cuda.manual_seed_all(37)
            return policy.predict(raw, num_steps=12).detach().clone()
    baseline = predict()
    adapter = ActMADInspired(policy, calibration)
    inactive = predict()
    assert torch.equal(baseline, inactive), 'installing inactive adapter changed prediction'
    updates = []
    try:
        for n in range(1, 5):
            loss = adapter.update(batch)
            grad_norm_change = adapter.last_param_l2
            assert np.isfinite(loss) and np.isfinite(grad_norm_change) and grad_norm_change > 0
            assert all(torch.isfinite(p).all() for p in adapter.backbone.parameters())
            updates.append({'frames': n, 'loss': loss, 'parameter_step_l2': grad_norm_change})
        adapted = predict()
        assert torch.isfinite(adapted).all()
        assert not torch.equal(adapted, baseline), 'adapter updated parameters but did not affect action'
        adapter.reset()
        reset = predict()
        assert torch.equal(reset, baseline), 'reset did not restore Frozen prediction'
        assert not adapter.frames and not adapter.opt.state
    finally:
        adapter.close()
    closed = predict()
    assert torch.equal(closed, baseline), 'close did not restore Frozen prediction'
    assert all(not p.requires_grad for p in policy.parameters())
    result = {'status': 'PASS', 'scope': 'real-model lifecycle check, not a closed-loop evaluation',
              'implementation': adapter.implementation_id, 'case': row['id'],
              'target_task': row['task'], 'source_tasks': tasks, 'strict_task_loto': True,
              'input_sha256': row['file_sha'], 'updates': updates,
              'inactive_exact_identity': True, 'reset_exact_identity': True,
              'close_exact_identity': True, 'finite_adapted_action': True,
              'action_change_l2': float((adapted - baseline).norm()),
              'action_shape': list(baseline.shape), 'H': 12, 'Euler_steps': 12}
    assert not a.out.exists()
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
