"""Fixed edbac60 deep gate: native paired closed-loop pilot and real latency.

50 main conditions x2 arms, plus one separate authorized init1 Clean recheck.
No threshold fitting, outcome selection, protected pools, or decoder reruns.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import binomtest

import phase3_smolvla_shift as base
from phase3_smolvla_common import reference_for, rng_hash, seed_torch
from dacsc.plus_glass_exact import glass_blur_exact
from dacsc.smolvla_deepgate import SmolVLADeepGate
from dacsc.smolvla_v1_hook import SmolVLAV1Hook

ROOT = Path('/hy-tmp/patch-diag')
DIAG = ROOT / 'runs/phase3_smolvla_unified_diag_v1'
CALIB = ROOT / 'runs/phase3_smolvla/spatial_init0_calib.npz'
ARMS = ('frozen', 'deep_gate_L3_a0.75')
GROUPS = ('Clean', 'Lighting', 'Background', 'Sensor', 'Shift40', 'All50',
          'table_texture', 'scene_texture', *base.FAMILIES)


def render_state(suite):
    """SimState alone omits the actual camera/geometry/light render transforms."""
    d, m = suite._env.sim.data, suite._env.sim.model
    h = hashlib.sha256()
    for owner, names in ((d, ('geom_xpos', 'geom_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir', 'mocap_pos', 'mocap_quat')),
                         (m, ('geom_rgba', 'light_pos', 'light_dir', 'light_diffuse', 'light_ambient', 'light_specular'))):
        for name in names:
            if hasattr(owner, name):
                h.update(name.encode())
                h.update(base.array_hash(np.asarray(getattr(owner, name))).encode())
    return h.hexdigest()


def canonical_rgb(raw, key, cache, audit):
    """Pair-local raw RGB reuse only at identical full physical/render state.

    Same pre-existing CoRP rounding bound: <=64 pixels/view, <=1 uint8 LSB.
    Outside that numerical allowance stop and retain a diagnostic, never hide it.
    """
    if key not in cache:
        cache[key] = tuple(np.array(raw[k], copy=True) for k in base.KEYS)
        return raw
    result = dict(raw)
    for name, reference in zip(base.KEYS, cache[key]):
        current = np.asarray(raw[name])
        assert current.shape == reference.shape and current.dtype == reference.dtype == np.uint8
        if not np.array_equal(current, reference):
            delta = np.abs(current.astype(np.int16) - reference.astype(np.int16))
            pixels, maximum = int((delta.max(-1) > 0).sum()), int(delta.max())
            event = dict(step=key[0], view=name, pixels=pixels, max_lsb=maximum,
                         physical_hash=key[1], render_hash=key[-1], native_sha256=base.array_hash(current),
                         canonical_sha256=base.array_hash(reference))
            audit.append(event)
            if pixels > 64 or maximum > 1:
                raise AssertionError(f'paired RGB outside pre-existing numerical bound: {event}')
            result[name] = reference
    return result


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def prepare(out):
    assert not (out / 'manifest.json').exists()
    old = read(DIAG / 'manifest.json')
    assert read(DIAG / 'final_audit.json')['status'] == 'PASS'
    assert old['calib_sha256'] == base.sha(CALIB)
    assert old['source_gate_sha256'] == base.sha(DIAG / 'source_gate.json')
    gates = read(DIAG / 'source_gate.json')
    cases = [dict(c, scored=True) for c in old['cases']]
    extra = dict(next(c for c in cases if c['task_id'] == 0 and c['group'] == 'Clean'),
                 id='t00_clean_i1_supplement', init_index=1, scored=False)
    assert len(cases) == 50
    # New gate is copied exactly, never refit on any current observation.
    write(out / 'source_gate.json', gates)
    assert (out / 'source_gate.json').read_bytes() == (DIAG / 'source_gate.json').read_bytes()
    weights = read(DIAG / 'weights_receipt.json')
    for value in weights.values():
        assert base.sha(value['path']) == value['sha256']
    write(out / 'weights_receipt.json', weights)
    source = out / 'source_code'
    m = dict(schema='smolvla_deepgate_closedloop_v2_renderpair', cases=cases + [extra], arms=list(ARMS),
        calib_sha256=base.sha(CALIB), source_gate_sha256=base.sha(out / 'source_gate.json'),
        diagnostic_manifest_sha256=base.sha(DIAG / 'manifest.json'),
        corruption_source_sha256=base.sha(base.PLUS / 'libero/libero/envs/env_wrapper.py'),
        source_sha256={str(p.relative_to(source)): base.sha(p) for p in source.rglob('*.py')},
        protocol=dict(main_conditions=50, main_episodes=100, supplemental_conditions=1, supplemental_episodes=2,
            policy_seed=0, workers=4, H=50, E=1, Euler_steps=10, max_steps=220,
            vision='unchanged native BF16, no CLS, two serial cameras, original CPU pixel division',
            model='load_vlm_weights=True; all parameters frozen, unchanged versions',
            method='unchanged edbac60 source gate G at raw layer12; same scalar decision for both views; raw reuse or layer3 alpha=.75 second vision pass',
            execution='native actor.act once; connector once per camera; native VLM prefix once and Flow10 once; no detector RNG draw or queue advance',
            calibration='unchanged init0 target-task LOTO; gate copied from edbac60 nested9/8/7 source-only fit',
            environment='native Vanilla base with existing audited official Plus scene selectors; EGL env.noise=0',
            sensor='same 20 bilateral severity3 cases, independent view seeds, t0 through entire episode',
            pairing='same initial physical/render state and RGB; identical CPU/CUDA RNG prefixes; pair-local raw RGB canonicalization only when full physical AND render state match, <=64 pixels/view and <=1 uint8 LSB; otherwise abort',
            scope='one-policy-seed exploratory development pilot, not independent confirmation or a SALT reproduction',
            blind_access=False, main_expansion_autostart=False))
    m['fingerprint'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    write(out / 'manifest.json', m)
    print('PREPARED', m['fingerprint'], flush=True)


def validate(out):
    m = read(out / 'manifest.json')
    fp = m.pop('fingerprint')
    assert hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest() == fp
    m['fingerprint'] = fp
    assert base.sha(CALIB) == m['calib_sha256']
    assert base.sha(out / 'source_gate.json') == m['source_gate_sha256']
    assert base.sha(base.PLUS / 'libero/libero/envs/env_wrapper.py') == m['corruption_source_sha256']
    for p, digest in m['source_sha256'].items():
        assert base.sha(out / 'source_code' / p) == digest
    for c in m['cases']:
        assert c['init_index'] in (1, 2) and c['policy_seed'] == 0
        assert not any(ch.isdigit() for ch in c['language'])
        if 'bddl' in c:
            assert base.sha(base.PLUS / c['bddl']) == c['bddl_sha256']
            assert base.sha(base.PLUS / 'libero/libero/assets' / c['scene_settings']['scene_xml']) == c['scene_sha256']
    return m, read(out / 'source_gate.json')['tasks']


class CallAudit:
    """Count actual encoder computation and direct native forward calls."""
    def __init__(self, policy):
        self.policy = policy
        self.counts = dict(encoder=0, connector=0, flow_noise=0, denoise=0, vlm_prefill=0, expert_suffix=0)

    def __enter__(self):
        self.stack = ExitStack()
        core = self.policy.model
        vision = core.vlm_with_expert.vlm.model.vision_model
        connector = core.vlm_with_expert.vlm.model.connector
        for module, key in ((vision.encoder.layers[0], 'encoder'), (connector, 'connector')):
            def hook(mod, args, output, key=key):
                self.counts[key] += 1
            handle = module.register_forward_hook(hook)
            self.stack.callback(handle.remove)
        for module, name, key in ((core, 'sample_noise', 'flow_noise'), (core, 'denoise_step', 'denoise'),
                                  (core.vlm_with_expert, 'forward', 'expert')):
            original = getattr(module, name)
            def wrapped(*args, original=original, key=key, **kwargs):
                k = ('vlm_prefill' if kwargs.get('fill_kv_cache') else 'expert_suffix') if key == 'expert' else key
                self.counts[k] += 1
                return original(*args, **kwargs)
            self.stack.enter_context(patch.object(module, name, new=wrapped))
        return self

    def __exit__(self, *args):
        self.stack.close()


def predict(actor, obs, refs=None, threshold=None, override=None, count=False):
    actor.reset(0)
    seed_torch(0)
    before = rng_hash()
    adapter = SmolVLADeepGate(actor._model, refs, threshold, audit_inputs=True, decision_override=override) if refs is not None else None
    try:
        with ExitStack() as stack:
            audit = stack.enter_context(CallAudit(actor._model)) if count else None
            if adapter:
                adapter.begin()
            action = actor.act(obs)
            data = adapter.end() if adapter else None
        assert np.isfinite(action).all()
        return action, data, (before, rng_hash()), audit.counts if audit else None
    finally:
        if adapter:
            adapter.close()


def preflight(out):
    m, gates = validate(out)
    actor = base.load_actor()
    versions = [p._version for p in actor._model.parameters()]
    funcs = dict(base.corruption_functions(), glass_blur=glass_blur_exact)
    previous = {r['case']['id']: r for r in read(DIAG / 'summary.json')['cases']}
    history = read(DIAG / 'historical_first_frames.json')
    checks, branches = [], []
    seen = set()
    dynamic_obs = None
    for c in m['cases']:
        suite, _, raw = base.setup(c)
        try:
            obs = base.perturb(raw, c, 0, funcs)
            refs = reference_for(CALIB, c['task_id'])
            tau = gates[str(c['task_id'])]['threshold']
            action, data, rng, counts = predict(actor, obs, refs, tau, count=True)
            expected_counts = dict(encoder=4 if data['accepted'] else 2, connector=2, flow_noise=1,
                                   denoise=10, vlm_prefill=1, expert_suffix=10)
            assert counts == expected_counts, (c['id'], counts)
            if c['scored']:
                old = previous[c['id']]
                assert base.snapshot(suite, raw) == history[c['id']]['frozen']['start']
                assert base.snapshot(suite, obs) == history[c['id']]['frozen']['observed_start']
                assert np.array_equal(action, np.asarray(old['methods']['deep_gate_L3_a0.75']['action'], dtype=action.dtype))
                assert data['accepted'] == old['gate']['accepted']
                assert abs(data['score'] - old['gate']['score']) <= 1e-7
                assert rng == (old['rng_before'], old['rng_after'])
            checks.append(dict(case_id=c['id'], scored=c['scored'], optimized_action_exact_to_t0_diagnostic=True if c['scored'] else None,
                               accepted=data['accepted'], call_counts=counts, same_flow_rng=True))
            if c['group'] not in seen:
                seen.add(c['group'])
                frozen, _, f_rng, native_counts = predict(actor, obs, count=True)
                off, _, off_rng, off_counts = predict(actor, obs, refs, tau, override=False, count=True)
                assert np.array_equal(frozen, off) and f_rng == off_rng
                assert off_counts == native_counts
                actor.reset(0)
                seed_torch(0)
                h = SmolVLAV1Hook(actor._model, refs, layers=(3,), active_layer=3, alpha=.75)
                try:
                    h.begin()
                    v1 = actor.act(obs)
                    h.end()
                    v1_rng = rng_hash()
                finally:
                    h.close()
                on, _, on_rng, on_counts = predict(actor, obs, refs, tau, override=True, count=True)
                assert np.array_equal(v1, on) and v1_rng == on_rng[1]
                assert on_counts == dict(native_counts, encoder=4)
                branches.append(dict(group=c['group'], forced_reject_equals_native=True, forced_accept_equals_v1=True,
                                     native_counts=native_counts, reject_counts=off_counts, accept_counts=on_counts))
                if c['group'] == 'Clean':
                    dynamic_obs = (obs, refs, tau)
        finally:
            suite.close()
        print('PREFLIGHT', len(checks), c['id'], data['accepted'], flush=True)
    # A changing decision must not advance queues, reset noise, or retain a hook.
    obs, refs, tau = dynamic_obs
    decisions = (False, True, False)
    actor.reset(0)
    seed_torch(0)
    oracle = []
    for active in decisions:
        before = rng_hash()
        h = SmolVLAV1Hook(actor._model, refs, layers=(3,), active_layer=3 if active else None, alpha=.75)
        try:
            h.begin()
            a = actor.act(obs)
            h.end()
            oracle.append((a.copy(), before, rng_hash()))
        finally:
            h.close()
    actor.reset(0)
    seed_torch(0)
    adapter = SmolVLADeepGate(actor._model, refs, tau, audit_inputs=True)
    try:
        for active, (a, before, after) in zip(decisions, oracle):
            adapter.decision_override = active
            assert rng_hash() == before
            adapter.begin()
            actual = actor.act(obs)
            adapter.end()
            assert np.array_equal(actual, a) and rng_hash() == after
            assert len(actor._model._queues['action']) == 0
    finally:
        adapter.close()
    assert versions == [p._version for p in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
    # The image guard must preserve identity, repair a single rounding LSB,
    # reject a substantive mismatch, and never reuse another state's image.
    raw = {k: np.full((256, 256, 3), 100, dtype=np.uint8) for k in base.KEYS}
    cache, events = {}, []
    key = (0, 'state', 'eef', 'quat', 'gripper', 'render')
    assert canonical_rgb(raw, key, cache, events) is raw
    small = {k: v.copy() for k, v in raw.items()}
    small[base.KEYS[0]][0, 0, 0] += 1
    repaired = canonical_rgb(small, key, cache, events)
    assert np.array_equal(repaired[base.KEYS[0]], raw[base.KEYS[0]]) and len(events) == 1
    different = (*key[:-1], 'different_render')
    assert canonical_rgb(small, different, cache, events) is small
    large = {k: v.copy() for k, v in raw.items()}
    large[base.KEYS[0]][0, 0, 0] += 10
    try:
        canonical_rgb(large, key, cache, events)
    except AssertionError:
        pass
    else:
        raise AssertionError('RGB mismatch guard failed to reject substantive change')
    native_files = [inspect.getfile(type(actor)), inspect.getfile(type(actor._model.model)),
                    inspect.getfile(type(actor._model.model.vlm_with_expert)), inspect.getfile(type(adapter.vision))]
    write(out / 'preflight.json', dict(status='PASS', conditions=checks, branch_checks=branches,
        source_fingerprint=m['fingerprint'], no_decoder_rerun=True, frozen_and_v1_forced_branch_actions_bit_exact=True,
        dynamic_off_on_off_three_step_queue_and_rng_exact=True, weights_unchanged=True, all_frozen=True,
        actual_encoder_calls_reject=2, actual_encoder_calls_accept=4, connector_calls=2, vlm_prefill=1, flow_steps=10,
        native_sources={p: base.sha(p) for p in native_files},
        bounded_rgb_guard_identity_one_lsb_repair_large_change_reject_and_state_isolation=True))


def latency(out):
    m, gates = validate(out)
    assert read(out / 'preflight.json')['status'] == 'PASS'
    actor = base.load_actor()
    funcs = dict(base.corruption_functions(), glass_blur=glass_blur_exact)
    observations = {}
    for group in ('Clean', 'Lighting', 'Background', 'Sensor'):
        c = next(c for c in m['cases'] if c['scored'] and c['group'] == group)
        suite, _, raw = base.setup(c)
        try:
            observations[group] = (c, base.perturb(raw, c, 0, funcs), reference_for(CALIB, c['task_id']))
        finally:
            suite.close()
    modes = [(g, a) for g in observations for a in ARMS]
    rows = []
    for round_index in range(-1, 5):
        order = modes if round_index < 0 else modes[round_index:] + modes[:round_index]
        for group, arm in order:
            c, obs, refs = observations[group]
            actor.reset(0)
            seed_torch(0)
            adapter = SmolVLADeepGate(actor._model, refs, gates[str(c['task_id'])]['threshold']) if arm != 'frozen' else None
            try:
                if round_index < 0:
                    for _ in range(2):
                        actor.reset(0)
                        seed_torch(0)
                        if adapter:
                            adapter.begin()
                        actor.act(obs)
                        if adapter:
                            adapter.end()
                    continue
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                if adapter:
                    adapter.begin()
                action = actor.act(obs)
                data = adapter.end() if adapter else None
                torch.cuda.synchronize()
                ms = 1000 * (time.perf_counter() - start)
                assert np.isfinite(action).all()
                rows.append(dict(group=group, case_id=c['id'], arm=arm, round=round_index, ms=ms,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                    gate_accepted=data['accepted'] if data else None))
            finally:
                if adapter:
                    adapter.close()
    summary = {}
    for group in observations:
        part = {a: [r for r in rows if r['group'] == group and r['arm'] == a] for a in ARMS}
        med = {a: float(np.median([r['ms'] for r in part[a]])) for a in ARMS}
        summary[group] = dict(median_ms=med, relative=med[ARMS[1]] / med['frozen'],
            peak_allocated_mib={a: float(np.median([r['peak_allocated_mib'] for r in part[a]])) for a in ARMS},
            gate_decisions=[r['gate_accepted'] for r in part[ARMS[1]]])
    write(out / 'latency.json', dict(rounds=5, rows=rows, groups=summary,
        scope='end-to-end native actor.act including CPU preprocessing, device transfer, actual gate/stats, connectors, one VLM prefix and Flow10; synchronized CUDA; episode reset/seeding/adapter construction excluded',
        representative_selection='first manifest case per category, before current outcomes; actual gate decision, never forced',
        instrumentation='no call-count mocks or full cache-pixel checks during timing; same actual evaluation adapter'))


def rollout(actor, case, arm, refs, threshold, funcs, pixel_cache, max_steps=220):
    suite, spec, raw = base.setup(case)
    start = None
    observed_start = None
    actor.reset(0)
    seed_torch(0)
    versions = [p._version for p in actor._model.parameters()]
    adapter = SmolVLADeepGate(actor._model, refs, threshold) if arm != 'frozen' else None
    trace = []
    roundings = []
    ah = hashlib.sha256()
    began = time.monotonic()
    try:
        success = False
        for step in range(max_steps):
            raw_state = base.snapshot(suite, raw)
            render = render_state(suite)
            key = (step, *(raw_state[k] for k in ('sim', 'eef', 'quat', 'gripper')), render)
            raw = canonical_rgb(raw, key, pixel_cache, roundings)
            obs = base.perturb(raw, case, step, funcs)
            if step == 0:
                start = base.snapshot(suite, raw)
                observed_start = base.snapshot(suite, obs)
            frame = dict(step=step, snapshot=base.snapshot(suite, obs), render_state=render, rng_before=rng_hash())
            if adapter:
                adapter.begin()
            action = actor.act(obs)
            gate = adapter.end() if adapter else None
            frame.update(rng_after=rng_hash(), action=action.tolist(), gate=gate)
            assert action.shape == (7,) and np.isfinite(action).all()
            assert len(actor._model._queues['action']) == 0
            ah.update(action.tobytes())
            trace.append(frame)
            result = suite.step(action)
            raw = result.obs
            success = bool(result.success)
            if result.done or success:
                break
        assert versions == [p._version for p in actor._model.parameters()]
        assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
        return dict(case_id=case['id'], arm=arm, task_id=case['task_id'], group=case['group'], subgroup=case['subgroup'],
            scored=case['scored'], init_index=case['init_index'], policy_seed=0, environment_seed=spec.seed,
            success=success, steps=len(trace), seconds=time.monotonic() - began, start=start,
            observed_start=observed_start, trace=trace, action_sha256=ah.hexdigest(), rounding_events=roundings,
            weights_frozen_and_versions_unchanged=True, finite_actions=True)
    finally:
        if adapter:
            adapter.close()
        suite.close()


def paired_check(rows):
    f, g = rows['frozen'], rows[ARMS[1]]
    assert f['start'] == g['start'] and f['observed_start'] == g['observed_start']
    identical_state_frames = 0
    physical_equal_render_different = 0
    for a, b in zip(f['trace'], g['trace']):
        assert a['step'] == b['step'] and a['rng_before'] == b['rng_before'] and a['rng_after'] == b['rng_after']
        assert b['gate']['decision_rule'] == 'source'
        same_physical = all(a['snapshot'][k] == b['snapshot'][k] for k in ('sim', 'eef', 'quat', 'gripper'))
        same_state = same_physical and a['render_state'] == b['render_state']
        physical_equal_render_different += int(same_physical and not same_state)
        if same_state:
            assert a['snapshot'] == b['snapshot'], 'same physical state rendered different paired RGB'
            identical_state_frames += 1
            if not b['gate']['accepted']:
                assert a['action'] == b['action'], 'reject drifted from Frozen at identical input'
    return dict(initial_state_and_two_view_pixels_exact=True, common_prefix_cpu_cuda_rng_exact=True,
                identical_physical_render_state_rgb_checks=identical_state_frames,
                physical_equal_render_different_frames=physical_equal_render_different,
                reject_at_equal_input_action_exact=True)


def worker(out, shard):
    m, gates = validate(out)
    assert read(out / 'preflight.json')['status'] == 'PASS'
    actor = base.load_actor()
    funcs = dict(base.corruption_functions(), glass_blur=glass_blur_exact)
    dest = out / 'pairs'
    dest.mkdir(exist_ok=True)
    for index, case in enumerate(m['cases']):
        if index % 4 != shard:
            continue
        p = dest / (case['id'] + '.json')
        if p.exists():
            v = read(p)
            assert v['fingerprint'] == m['fingerprint'] and set(v['arms']) == set(ARMS)
            paired_check(v['arms'])
            continue
        refs = reference_for(CALIB, case['task_id'])
        tau = gates[str(case['task_id'])]['threshold']
        rows, pixel_cache = {}, {}
        for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
            rows[arm] = rollout(actor, case, arm, refs, tau, funcs, pixel_cache)
            print('ARM_DONE', case['id'], arm, rows[arm]['success'], rows[arm]['steps'], flush=True)
        check = paired_check(rows)
        write(p, dict(fingerprint=m['fingerprint'], case=case, arms=rows, pairing=check))
        print('PAIR_DONE', case['id'], flush=True)


def selected(c, group):
    if group == 'All50':
        return c['scored']
    if group == 'Shift40':
        return c['scored'] and c['group'] != 'Clean'
    return c['scored'] and (c['group'] == group or c['subgroup'] == group)


def stats(pairs):
    assert pairs
    n = len(pairs)
    f = np.array([p['arms']['frozen']['success'] for p in pairs], dtype=bool)
    g = np.array([p['arms'][ARMS[1]]['success'] for p in pairs], dtype=bool)
    rescued, harmed = int((~f & g).sum()), int((f & ~g).sum())
    tasks = sorted({p['case']['task_id'] for p in pairs})
    nums = np.array([sum(int(p['arms'][ARMS[1]]['success']) - int(p['arms']['frozen']['success']) for p in pairs if p['case']['task_id'] == t) for t in tasks])
    dens = np.array([sum(p['case']['task_id'] == t for p in pairs) for t in tasks])
    rng = np.random.default_rng(2026112607)
    draws = rng.integers(0, len(tasks), size=(10000, len(tasks)))
    effects = 100 * nums[draws].sum(1) / dens[draws].sum(1)
    traces = [p['arms'][ARMS[1]]['trace'] for p in pairs]
    accepts = [sum(fr['gate']['accepted'] for fr in trace) for trace in traces]
    return dict(n=n, frozen=int(f.sum()), deepgate=int(g.sum()), rescued=rescued, harmed=harmed,
        net=rescued - harmed, delta_pp=100 * (rescued - harmed) / n,
        mcnemar_exact_two_sided=float(binomtest(rescued, rescued + harmed, .5).pvalue) if rescued + harmed else 1.,
        task_cluster_bootstrap95_pp=np.percentile(effects, [2.5, 97.5]).tolist(), task_clusters=len(tasks),
        gate=dict(accepted_replans=sum(accepts), total_replans=sum(map(len, traces)),
            micro_rate=sum(accepts) / sum(map(len, traces)), episode_macro_rate=float(np.mean([a / len(t) for a, t in zip(accepts, traces)])),
            episodes_ever_accepted=sum(a > 0 for a in accepts), first_observation_accepted=sum(t[0]['gate']['accepted'] for t in traces)))


def report(out):
    m, _ = validate(out)
    pairs = []
    for c in m['cases']:
        p = read(out / 'pairs' / (c['id'] + '.json'))
        assert p['fingerprint'] == m['fingerprint'] and p['case'] == c
        paired_check(p['arms'])
        for v in p['arms'].values():
            assert v['finite_actions'] and v['weights_frozen_and_versions_unchanged'] and len(v['trace']) == v['steps']
        pairs.append(p)
    assert len(pairs) == 51
    groups = {g: stats([p for p in pairs if selected(p['case'], g)]) for g in GROUPS}
    extra = stats([p for p in pairs if not p['case']['scored']])
    compact = [dict(case=p['case'], frozen=p['arms']['frozen']['success'], deepgate=p['arms'][ARMS[1]]['success'],
        steps={a: p['arms'][a]['steps'] for a in ARMS}, pairing=p['pairing'],
        first_gate_step=next((t['step'] for t in p['arms'][ARMS[1]]['trace'] if t['gate']['accepted']), None),
        gate_score_minmax=[min(t['gate']['score'] for t in p['arms'][ARMS[1]]['trace']), max(t['gate']['score'] for t in p['arms'][ARMS[1]]['trace'])]) for p in pairs]
    result = dict(schema='smolvla_deepgate_closedloop_v1', protocol=m['protocol'], fingerprint=m['fingerprint'],
        calib_sha256=m['calib_sha256'], source_gate_sha256=m['source_gate_sha256'], groups=groups,
        supplemental_task0_clean_init1=extra, cases=compact, preflight=read(out / 'preflight.json'), latency=read(out / 'latency.json'),
        raw_sha256={c['id']: base.sha(out / 'pairs' / (c['id'] + '.json')) for c in m['cases']},
        limitations=['one policy seed, fifty previously accessed development conditions, not untouched confirmation',
            'nominal McNemar is not task-dependence or multiplicity adjusted; task bootstrap has only ten clusters',
            'source gate is pointwise95, not trajectory-wise95; t0 zero activation does not guarantee zero later Clean activation',
            'no concurrent unconditional V1 arm; old V1 results are descriptive historical context, not a new paired three-arm comparison',
            'no result establishes pretraining causality, universal geometry preservation, or CoRP deep-gate performance',
            'extra task0/init1 pair excluded from fifty-condition main table; all harms and paired discordances retained'])
    write(out / 'summary.json', result)
    lines = ['# SmolVLA：冻结深层总闸的同期配对闭环', '',
        '50条件×Frozen/DeepGate=100局；另加task0 Clean init1的独立补充1对（2局），不混入主表。策略seed0，H50/E1/Euler10，220步，EGL，原版BF16/CPU像素协议。', '',
        '| 条件 | Frozen | DeepGate | 救回/误伤 | 净增 | 精确双侧p | 任务聚类95%区间pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group, x in groups.items():
        lo, hi = x['task_cluster_bootstrap95_pp']
        lines.append(f"| {group} | {x['frozen']}/{x['n']} | {x['deepgate']}/{x['n']} | {x['rescued']}/{x['harmed']} | {x['net']:+d} | {x['mcnemar_exact_two_sided']:.6g} | [{lo:+.1f}, {hi:+.1f}] |")
    lines += ['', '## 整局总闸行为', '',
        '| 条件 | 放行replan/总replan | 逐步放行率% | 逐局平均放行率% | 至少触发一次的局数 | 首帧触发局数 |',
        '|---|---:|---:|---:|---:|---:|']
    for group, x in groups.items():
        g = x['gate']
        lines.append(f"| {group} | {g['accepted_replans']}/{g['total_replans']} | {100*g['micro_rate']:.2f} | {100*g['episode_macro_rate']:.2f} | {g['episodes_ever_accepted']}/{x['n']} | {g['first_observation_accepted']}/{x['n']} |")
    lines += ['', '## task0 init1补充检查（不计入主表）', '',
        f"Frozen {extra['frozen']}/{extra['n']}，DeepGate {extra['deepgate']}/{extra['n']}；救{extra['rescued']}、伤{extra['harmed']}，放行{extra['gate']['accepted_replans']}/{extra['gate']['total_replans']}次。", '',
        '## 端到端延迟（5轮交错、CUDA同步）', '', result['latency']['scope'], '',
        '| 固定代表输入 | Frozen中位ms | DeepGate中位ms | 倍数 | 实际触发 | Frozen/DeepGate峰值allocated MiB中位 |',
        '|---|---:|---:|---:|---:|---:|']
    for group, x in result['latency']['groups'].items():
        med, mem = x['median_ms'], x['peak_allocated_mib']
        lines.append(f"| {group} | {med['frozen']:.2f} | {med[ARMS[1]]:.2f} | {x['relative']:.3f}× | {x['gate_decisions']} | {mem['frozen']:.2f}/{mem[ARMS[1]]:.2f} |")
    lines += ['', '## 审计与判读', '',
        '- 完全复制edbac60总闸与原始校准包，未重新选阈值/步长。检测对两眼对称，G取原始Layer12双眼均值；修正只在Layer3、alpha=.75。',
        '- 第一遍仅计算未干预视觉；拒绝时缓存复用，接受时仅多跑两路视觉编码。connector每眼一次，原生VLM prefix一次、Flow10一次；模型参数与版本保持冻结。',
        '- 开跑前全部50首帧优化实现与edbac60动作逐位相同；强制开/关两分支对照原V1/Frozen；off→on→off连续重规划的动作、队列和RNG逐位一致。实际调用计数排除了重复LLM/Flow。',
        '- 五十组同期初始物理状态/双目像素严格配对；共同步CPU/CUDA RNG前后均匹配。只有完整物理和相机/几何/灯光渲染状态均匹配才允许共享原始RGB；沿用既有CoRP数值界限（每眼≤64像素、≤1灰度LSB），每次替换保留原/标准哈希与差异量。界限外继续中止，不同状态不复用。',
        '- 原逐字节渲染检查试跑在一个Clean配对被拦停，单进程动作回放与模型复现均一致；原试跑归档排除，正式批次在本修订配对协议下全部重跑。未改门控阈值、alpha、模型或条件。',
        '- 放行率分别给出按步与按局平均；早成功的局数步数更少，因此不混淆两个统计口径。源95%按帧拟合，不是整局95%不误触发。',
        '- 这是单种子开发诊断，保留全部失败与误伤。与旧V1只作历史参照，不冒称本轮三臂同期对比；不能把域分组AUROC当作适配收益预测准确率。',
        '- 未访问Blind/Held-out，未扩跑其他套件。SmolVLA闭环结果不证明CoRP加同一总闸也有效。', '',
        f"校准SHA256：{m['calib_sha256']}。总闸SHA256：{m['source_gate_sha256']}。", '']
    (out / 'PHASE3_SMOLVLA_DEEPGATE_CL.md').write_text('\n'.join(lines))
    print(json.dumps(groups, indent=2), flush=True)


def run(out):
    m, _ = validate(out)
    assert read(out / 'preflight.json')['status'] == 'PASS' and (out / 'latency.json').exists()
    start = time.monotonic()
    processes, logs = [], []
    python = '/hy-tmp/smolvla_venv/bin/python'
    try:
        for k in range(4):
            log = (out / f'worker{k}.log').open('a')
            logs.append(log)
            processes.append(subprocess.Popen([python, '-u', __file__, 'worker', '--out', str(out), '--shard', str(k)], stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failed:
                raise RuntimeError(f'worker failure {failed}; preserve complete pairs and inspect logs')
            complete = list((out / 'pairs').glob('*.json'))
            write(out / 'status.json', dict(phase='running', pairs_complete=len(complete), pairs_total=51,
                episodes_complete=2 * len(complete), seconds=time.monotonic() - start, pids=[p.pid for p in processes]))
            time.sleep(5)
        assert all(p.returncode == 0 for p in processes)
        report(out)
        write(out / 'status.json', dict(phase='complete', main_episodes=100, supplemental_episodes=2,
            pairs=51, seconds=time.monotonic() - start, fingerprint=m['fingerprint']))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'preflight', 'latency', 'worker', 'run', 'report'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == 'prepare':
            prepare(args.out)
        elif args.mode == 'preflight':
            preflight(args.out)
        elif args.mode == 'latency':
            latency(args.out)
        elif args.mode == 'worker':
            worker(args.out, args.shard)
        elif args.mode == 'report':
            report(args.out)
        else:
            run(args.out)
    except Exception as e:
        write(args.out / (args.mode + '_failure.json'), dict(error=str(e)))
        if args.mode == 'run':
            write(args.out / 'status.json', dict(phase='failed', error=str(e)))
        raise
