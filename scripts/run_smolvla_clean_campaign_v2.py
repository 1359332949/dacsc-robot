"""Historical handwritten campaign, superseded by the unmodified upstream evaluator.

Originally planned two replay anchors then 30 initial states; stopped per user
request after confirmed input/loading differences. Do not use as canonical repro.

Runs on the GPU host; no TTA, no hyperparameter sweep, no Held-out/Blind access.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(a):
    root = a.run.resolve()
    source = root / 'source'
    root.mkdir(parents=True, exist_ok=True)
    assert (source / 'smolvla_clean_repro.py').exists()
    assert 1 <= a.workers <= 4
    manifest = {
        'version': 'smolvla_clean_reproduction_v2',
        'method': 'frozen_only', 'assets': 'vanilla_LIBERO', 'suite': 'libero_spatial',
        'pipeline': 'legacy', 'lerobot_commit': '483be9aac217c2d8ef16982490f22b2ad091ab46',
        'policy': 'HuggingFaceVLA/smolvla_libero',
        'checkpoint_sha256': '71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f',
        'policy_n_action_steps': 1, 'euler_steps': 10, 'max_steps': 280,
        'image_size': 256, 'settling_steps': 10, 'noise': 0, 'renderer': 'EGL',
        'seed_rule': '10 * task_id; same policy/environment seed across initial states',
        'workers': a.workers, 'adaptation': False,
        'protected_pools_accessed': False,
        'interpretation': 'new initial states of previously examined tasks; not independent task generalization',
        'anchors': [{'task': 0, 'init': 0}, {'task': 4, 'init': 0}],
        'cases': [{'task': t, 'init': i} for i in (1, 2, 3) for t in range(10)],
        'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source.rglob('*.py')},
    }
    assert not (root / 'manifest.json').exists(), 'fresh run directory required'
    write_json(root / 'manifest.json', manifest)
    env = dict(os.environ)
    env.update(LIBERO_CONFIG_PATH='/hy-tmp/vanilla_libero_config', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl',
               PYGLFW_LIBRARY='/hy-tmp/corp_venv/lib/python3.10/site-packages/glfw/x11/libglfw.so',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH='/hy-tmp/lerobot_community_revision:/hy-tmp/LIBERO-vanilla:'
               '/hy-tmp/smolvla_venv/lib/python3.10/site-packages:'
               '/hy-tmp/corp_venv/lib/python3.10/site-packages')
    python = '/hy-tmp/smolvla_venv/bin/python'
    test_env = dict(env, PYTHONPATH=str(source))
    with (root / 'unit_tests.log').open('w') as log:
        subprocess.run([python, '-m', 'unittest', 'discover', '-s', str(source / 'tests'),
                        '-p', 'test_actmad_view_layout.py', '-v'], cwd=source,
                       env=test_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(root / 'status.json', {'phase': 'anchors', 'completed': 0, 'total': 32})

    def run_case(c, kind):
        dest = root / kind / f"t{c['task']}_i{c['init']}"
        dest.mkdir(parents=True, exist_ok=False)
        cmd = [python, '-u', str(source / 'smolvla_clean_repro.py'), '--pipeline', 'legacy',
               '--assets', 'vanilla', '--task', str(c['task']), '--init-index', str(c['init']),
               '--max-steps', '280', '--out', str(dest)]
        with (dest / 'rollout.log').open('w') as log:
            p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f"case {c} exit {p.returncode}: {dest / 'rollout.log'}")
        files = list(dest.glob('*.json'))
        assert len(files) == 1, (dest, files)
        row = json.loads(files[0].read_text())
        assert row['weights_keys_matched'] == 788
        assert row['init_index'] == c['init'] and row['task_id'] == c['task']
        return row

    anchors = []
    old = Path('/hy-tmp/patch-diag/runs/smolvla_reproduction_v1')
    for c, ref in zip(manifest['anchors'], [old / 'legacy_fixed/legacy_vanilla_t0.json',
                                          old / 'legacy_even/legacy_vanilla_t4.json']):
        row = run_case(c, 'anchors')
        prior = json.loads(ref.read_text())
        fields = {'success': row['success'] == prior['success'],
                  'steps': row['steps'] == prior['steps'],
                  'settled_state': row['settled_state'] == prior['settled_state'],
                  'raw_agent': row['first']['raw_agent'] == prior['first']['raw_agent'],
                  'raw_wrist': row['first']['raw_wrist'] == prior['first']['raw_wrist'],
                  'actions_sha256': row['actions_sha256'] == prior['actions_sha256']}
        anchors.append({'task': c['task'], 'reference': str(ref), 'matches': fields})
        write_json(root / 'anchor_checks.json', anchors)
        if not all(fields.values()):
            write_json(root / 'status.json', {'phase': 'anchor_mismatch', 'checks': anchors})
            raise RuntimeError('anchor changed: investigate before expanding evaluation')
        print('anchor', c, fields, flush=True)
    rows = []
    errors = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = {executor.submit(run_case, c, 'clean_new'): c for c in manifest['cases']}
        for fut in as_completed(futures):
            try:
                row = fut.result(); rows.append(row)
                print('result', row['task_id'], row['init_index'], row['success'], row['steps'], flush=True)
            except Exception as exc:
                errors.append({'case': futures[fut], 'error': str(exc)})
            write_json(root / 'progress.json', {'completed': len(rows), 'errors': errors,
                        'successes': sum(r['success'] for r in rows), 'total': 30,
                        'elapsed_seconds': time.monotonic() - started})
    rows.sort(key=lambda r: (r['task_id'], r['init_index']))
    summary = {'n': len(rows), 'successes': sum(r['success'] for r in rows),
               'errors': errors, 'anchor_checks': anchors,
               'by_task': {str(t): {'n': sum(r['task_id'] == t for r in rows),
                                    'successes': sum(r['success'] for r in rows if r['task_id'] == t)}
                           for t in range(10)},
               'rows': rows, 'manifest': manifest}
    write_json(root / 'summary.json', summary)
    write_json(root / 'status.json', {'phase': 'complete' if not errors else 'failed_cases',
                                     'completed': len(rows), 'errors': errors})
    if errors:
        raise RuntimeError(f'{len(errors)} rollouts failed; do not count as valid outcomes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        write_json(args.run / 'error.json', {'error': str(exc)})
        raise
