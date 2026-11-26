"""Source-calibrated, two-view SmolVLA t=0 diagnosis; never a rollout verdict.

Deep gate is a new hypothesis, not a reproduction of SALT's temporal gate.
Its scalar threshold is fitted only to nested held-out source Clean frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import phase3_smolvla_shift as base
from phase3_smolvla_common import reference_for, rng_hash, seed_torch
from dacsc.plus_glass_exact import glass_blur_exact
from dacsc.smolvla_v1_hook import LAYERS, SmolVLAV1Hook

ROOT = Path('/hy-tmp/patch-diag')
SOURCE = ROOT / 'runs/phase3_smolvla'
OLD = ROOT / 'runs/phase3_smolvla_shift'
SEV3 = ROOT / 'runs/phase3_smolvla_sev3'
CALIB = SOURCE / 'spatial_init0_calib.npz'
GROUPS = ('Clean', 'Lighting', 'Background', 'Sensor')
CONFIGS = tuple((layer, alpha) for layer in (3, 6) for alpha in (.25, .50, .75))
ARMS = tuple(f'L{l}_a{a:.2f}' for l, a in CONFIGS) + ('deep_gate_L3_a0.75',)
FLOOR = 1e-6


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized(d2, tau):
    return d2 / tau.clamp_min(FLOOR)


def source_gate(out):
    """Outer9 deployment; held-out source scored against8, their tau against7.

    Neither the target task nor evaluation Clean/Shift observations contribute.
    Scores/thresholds are empirical diagnostics, not conformal guarantees.
    """
    torch.set_num_threads(1)
    raw, hashes = {}, {}
    for task in range(10):
        p = SOURCE / 'source' / f'task{task}.npz'
        meta = read(p.with_suffix('.json'))
        assert meta['init_index'] == 0 and base.sha(p) == meta['moments_sha256']
        with np.load(p, allow_pickle=False) as z:
            raw[task] = tuple(torch.from_numpy(z[k][:, 3].copy()).float() for k in ('mean', 'second'))
        assert raw[task][0].shape[1:] == (2, 768)
        hashes[str(task)] = base.sha(p)
    pools, refs = {}, {}

    def pooled(tasks):
        tasks = tuple(tasks)
        if tasks not in pools:
            m, q = [torch.cat([raw[t][k] for t in tasks]) for k in (0, 1)]
            mu = m.mean(0)
            pools[tasks] = mu, (q.mean(0) - mu.square()).clamp_min(0).sqrt()
        return pools[tasks]

    def score(task, tasks):
        mu, ss = pooled(tasks)
        m, q = raw[task]
        sd = (q - m.square()).clamp_min(0).sqrt()
        return ((m - mu).square() + (sd - ss).square()) / (ss.square() + FLOOR)

    def fitted(tasks):
        tasks = tuple(tasks)
        if tasks not in refs:
            tau = torch.cat([score(h, [t for t in tasks if t != h]) for h in tasks]).quantile(.95, dim=0)
            refs[tasks] = (*pooled(tasks), tau)
        return refs[tasks]

    gates, arrays = {}, {}
    for target in range(10):
        outer = [t for t in range(10) if t != target]
        deployment = fitted(outer)
        frozen = reference_for(CALIB, target, device='cpu')[12]
        assert all(torch.equal(a, b) for a, b in zip(deployment, frozen)), ('source reference drift', target)
        values, folds = [], {}
        for hold in outer:
            inner = [t for t in outer if t != hold]
            assert len(inner) == 8 and target not in inner and hold not in inner
            tau = fitted(inner)[2]
            g = normalized(score(hold, inner), tau).mean((1, 2))
            assert torch.isfinite(g).all()
            values.append(g)
            arrays[f'target{target}_hold{hold}'] = g.numpy()
            folds[str(hold)] = dict(fit_tasks=inner, channel_threshold_inner7={str(h): [t for t in inner if t != h] for h in inner}, frames=len(g))
        g = torch.cat(values)
        gates[str(target)] = dict(threshold=float(g.quantile(.95)), source_frames=len(g), outer_tasks=outer,
                                  source_crossfit_exceedance=float((g > g.quantile(.95)).float().mean()), folds=folds)
    np.savez_compressed(out / 'source_gate_scores.npz', **arrays)
    write(out / 'source_gate.json', dict(tasks=gates, source_sha256=hashes,
        formula='G = mean over both views and all 768 channels of d2 / clamp_min(tau95, 1e-6); accept iff G > source_crossfit_quantile95',
        calibration='target excluded; fit8/channel_tau_inner7+1, score held source task; pool9 held-out source trajectories; deploy original fit9 reference',
        limitation='crossfit reference fit-size8 vs deployed9; empirical quantile, no 95% target false-positive guarantee'))
    return gates


def prepare(out):
    assert not (out / 'manifest.json').exists()
    old, sev = read(OLD / 'manifest.json'), read(SEV3 / 'manifest.json')
    assert old['calib_sha256'] == sev['calib_sha256'] == base.sha(CALIB)
    sensor = {c['id']: c for c in sev['cases']}
    cases = [sensor[c['id']] if c['group'] == 'Sensor' else c for c in old['cases']]
    assert len(cases) == 50 and len({c['id'] for c in cases}) == 50
    assert {g: sum(c['group'] == g for c in cases) for g in GROUPS} == dict(Clean=10, Lighting=10, Background=10, Sensor=20)
    assert all(c['init_index'] >= 1 and c['policy_seed'] == 0 for c in cases)
    assert all(c['severity'] == 3 for c in cases if c['group'] == 'Sensor')
    assert base.sha(base.PLUS / 'libero/libero/envs/env_wrapper.py') == old['corruption_source_sha256']
    for c in cases:
        assert not any(ch.isdigit() for ch in c['language'])
        if 'bddl' in c:
            assert base.sha(base.PLUS / c['bddl']) == c['bddl_sha256']
    gates = source_gate(out)
    write(out / 'manifest.json', dict(schema='smolvla_unified_diag_v1', cases=cases, calib_sha256=base.sha(CALIB),
        source_gate_sha256=base.sha(out / 'source_gate.json'), source_gate_scores_sha256=base.sha(out / 'source_gate_scores.npz'),
        previous_manifest_sha256=base.sha(OLD / 'manifest.json'), severity3_manifest_sha256=base.sha(SEV3 / 'manifest.json'),
        source_sha256={str(p.relative_to(out / 'source_code')): base.sha(p) for p in (out / 'source_code').rglob('*.py')},
        protocol=dict(scope='50 previously accessed development observations; t=0 only; zero environment steps after reset; not a success-rate experiment',
            candidate_configs=[dict(layer=l, alpha=a) for l, a in CONFIGS], gate=gates,
            clean_reference='Frozen, same task/init/physics/state/language/flow RNG; used only in action-distance evaluation',
            model='unchanged native BF16, load_vlm_weights=True, CPU pixel normalization, H50 E1 Euler10',
            sensor='bilateral severity3 from first policy observation; unchanged case IDs and independent view seeds',
            calibration='unchanged init0 outer9/inner8+1; scalar deep gate adds nested8/7 source-only calibration',
            gate_compute='unadapted Layer12 first; separate Layer3-adapted forward if accepted; no same-pass retroactive hook',
            blind_access=False, closed_loop_autostart=False)))
    print('PREPARED', len(cases), base.sha(CALIB), flush=True)


class ProbeHook(SmolVLAV1Hook):
    def _hook(self, layer):
        parent = super()._hook(layer)

        def hook(module, args, output):
            view = self.counts[layer]
            x = output[0] if isinstance(output, tuple) else output
            result = parent(module, args, output)
            mu, ss, tau = (v[view].float()[None] for v in self.references[layer])
            m, sd = x.float().mean(1), x.float().std(1, unbiased=False)
            d2 = ((m - mu).square() + (sd - ss).square()) / (ss.square() + FLOOR)
            assert torch.isfinite(d2).all()
            self.stats[layer, view].update(d2_mean=d2.mean().detach(), normalized_mean=normalized(d2, tau).mean().detach(),
                                           tau_floor_fraction=(tau < FLOOR).float().mean().detach())
            return result
        return hook


def prediction(actor, obs, refs, layer=None, alpha=.75):
    actor.reset(0)
    seed_torch(0)
    before = rng_hash()
    h = ProbeHook(actor._model, refs, layers=LAYERS, active_layer=layer, alpha=alpha)
    try:
        h.begin()
        action = actor.act(obs)
        entries = h.end()
        assert action.shape == (7,) and np.isfinite(action).all()
        stats = {str(l): {k: [float(entries[l, v][k]) for v in range(2)] for k in
                 ('active_fraction', 'd2_mean', 'normalized_mean', 'tau_floor_fraction')} for l in LAYERS}
        return action.copy(), stats, (before, rng_hash())
    finally:
        h.close()


def physical_snapshot(suite, obs):
    x = base.snapshot(suite, obs)
    return {k: x[k] for k in ('sim', 'eef', 'quat', 'gripper')}


def errors(action, clean):
    return {name: float(np.linalg.norm(action[sl] - clean[sl])) for name, sl in
            (('xyz', slice(0, 3)), ('rotation', slice(3, 6)), ('action7', slice(0, 7)))}


def run(out):
    manifest = read(out / 'manifest.json')
    assert base.sha(CALIB) == manifest['calib_sha256']
    for p, digest in manifest['source_sha256'].items():
        assert base.sha(out / 'source_code' / p) == digest
    assert base.sha(out / 'source_gate.json') == manifest['source_gate_sha256']
    assert base.sha(out / 'source_gate_scores.npz') == manifest['source_gate_scores_sha256']
    assert not (out / 'cases').exists(), 'fresh diagnostic run only; retain any failed evidence'
    (out / 'cases').mkdir()
    actor = base.load_actor()
    versions = [p._version for p in actor._model.parameters()]
    funcs = dict(base.corruption_functions(), glass_blur=glass_blur_exact)
    old_summary = read(OLD / 'summary.json')
    old_sensor = read(SEV3 / 'summary.json')
    history = {c['case_id']: c for c in old_summary['pairs'] if c['group'] != 'Sensor'}
    history.update({c['case_id']: c for c in old_sensor['cases']})
    clean_cache = {}
    start = time.monotonic()
    identity_checks = []
    for index, case in enumerate(manifest['cases']):
        refs = reference_for(CALIB, case['task_id'])
        key = (case['task_id'], case['init_index'])
        if key not in clean_cache:
            clean_case = dict(case, group='Clean')
            suite, _, obs = base.setup(clean_case)
            try:
                action, _, noise = prediction(actor, obs, refs)
                clean_cache[key] = dict(action=action, physical=physical_snapshot(suite, obs),
                                       physics={k: base.array_hash(v) for k, v in base.physics(suite).items()}, noise=noise)
            finally:
                suite.close()
        clean = clean_cache[key]
        suite, _, raw = base.setup(case)
        try:
            assert physical_snapshot(suite, raw) == clean['physical'], (case['id'], 'clean counterfactual physical state')
            assert {k: base.array_hash(v) for k, v in base.physics(suite).items()} == clean['physics']
            obs = base.perturb(raw, case, 0, funcs)
            if case['group'] == 'Sensor':
                assert all(not np.array_equal(obs[k], raw[k]) for k in base.KEYS)
            rowdir = SEV3 if case['group'] == 'Sensor' else OLD
            old = {a: read(rowdir / 'pairs' / f"{case['id']}_{a}.json") for a in ('frozen', 'v1')}
            assert base.snapshot(suite, raw) == old['frozen']['start'], (case['id'], 'historical raw RGB/state')
            assert base.snapshot(suite, obs) == old['frozen']['observed_start']
            frozen, stats, noise = prediction(actor, obs, refs)
            assert noise == clean['noise'] and noise[0] == old['frozen']['rng_before_replans'][0]
            assert np.array_equal(frozen, np.asarray(old['frozen']['frames'][0]['action'], dtype=frozen.dtype))
            if case['group'] == 'Clean':
                assert np.array_equal(frozen, clean['action'])
            methods = {}
            for layer, alpha in CONFIGS:
                action, _, rng = prediction(actor, obs, refs, layer, alpha)
                assert rng == noise
                methods[f'L{layer}_a{alpha:.2f}'] = dict(action=action.tolist(), errors=errors(action, clean['action']),
                    drift_from_frozen=errors(action, frozen))
            assert np.array_equal(np.asarray(methods['L3_a0.75']['action'], dtype=frozen.dtype),
                                  np.asarray(old['v1']['frames'][0]['action'], dtype=frozen.dtype))
            # Identity once per category, at both intervened depths.
            if case['group'] not in {x['group'] for x in identity_checks}:
                for layer in (3, 6):
                    zero, _, rng = prediction(actor, obs, refs, layer, 0)
                    assert np.array_equal(zero, frozen) and rng == noise
                    identity_checks.append(dict(case_id=case['id'], group=case['group'], layer=layer, alpha0_action_bit_exact=True))
            g = float(np.mean(stats['12']['normalized_mean']))
            threshold = manifest['protocol']['gate'][str(case['task_id'])]['threshold']
            accepted = g > threshold
            if accepted:
                gated, _, rng = prediction(actor, obs, refs, 3, .75)
                assert rng == noise and np.array_equal(gated, np.asarray(methods['L3_a0.75']['action'], dtype=frozen.dtype))
            else:
                gated = frozen.copy()
            methods['deep_gate_L3_a0.75'] = dict(action=gated.tolist(), errors=errors(gated, clean['action']),
                                                drift_from_frozen=errors(gated, frozen))
            previous = history[case['id']]
            row = dict(case=case, layers=stats, gate=dict(score=g, threshold=threshold, accepted=accepted),
                frozen_action=frozen.tolist(), clean_frozen_action=clean['action'].tolist(),
                baseline_errors=errors(frozen, clean['action']), methods=methods,
                previous_closed_loop=dict(frozen=bool(previous['frozen']), v1=bool(previous['v1']),
                    harmed=bool(previous['frozen'] and not previous['v1']), rescued=bool(not previous['frozen'] and previous['v1'])),
                audit=dict(same_clean_physics_state=True, historical_t0_pixels_exact=True, native_frozen_action_exact=True,
                    native_v1_action_exact=True, flow_rng_exact=True, no_closed_loop_steps=True),
                rng_before=noise[0], rng_after=noise[1])
            write(out / 'cases' / (case['id'] + '.json'), row)
            write(out / 'status.json', dict(phase='running', conditions_complete=index + 1, conditions_total=50,
                                           seconds=time.monotonic() - start))
            print('DIAG', index + 1, case['id'], 'deep_gate', accepted, 'G', round(g, 3), 'tau', round(threshold, 3), flush=True)
        finally:
            suite.close()
    assert versions == [p._version for p in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
    write(out / 'preflight.json', dict(status='PASS', alpha0_checks=identity_checks, model_weights_unchanged=True,
        historical_50_frozen_and_v1_t0_actions_bit_exact=True, paired_counterfactual_physics_and_rng=True,
        vision_dtype='torch.bfloat16', no_cls_1024_patches=True, two_serial_cameras=True))
    report(out)
    write(out / 'status.json', dict(phase='complete', conditions=50, closed_loop_episodes=0,
                                    seconds=time.monotonic() - start))


def report(out):
    m = read(out / 'manifest.json')
    rows = [read(out / 'cases' / (c['id'] + '.json')) for c in m['cases']]
    assert len(rows) == 50 and read(out / 'preflight.json')['status'] == 'PASS'
    layers, actions, gates = {}, {}, {}
    for group in GROUPS:
        part = [r for r in rows if r['case']['group'] == group]
        layers[group] = {str(l): {metric: np.mean([r['layers'][str(l)][metric] for r in part], axis=0).tolist()
                        for metric in ('active_fraction', 'normalized_mean', 'd2_mean', 'tau_floor_fraction')} for l in LAYERS}
        gates[group] = dict(n=len(part), accepted=sum(r['gate']['accepted'] for r in part),
            previously_harmed=sum(r['previous_closed_loop']['harmed'] for r in part),
            previous_harms_gate_would_accept=sum(r['previous_closed_loop']['harmed'] and r['gate']['accepted'] for r in part),
            previously_rescued=sum(r['previous_closed_loop']['rescued'] for r in part),
            previous_rescues_gate_would_accept=sum(r['previous_closed_loop']['rescued'] and r['gate']['accepted'] for r in part))
        actions[group] = {}
        for arm in ARMS:
            entry = dict(n=len(part))
            for metric in ('xyz', 'rotation', 'action7'):
                b = np.array([r['baseline_errors'][metric] for r in part])
                a = np.array([r['methods'][arm]['errors'][metric] for r in part])
                drift = np.array([r['methods'][arm]['drift_from_frozen'][metric] for r in part])
                entry[metric] = dict(frozen_mean_error=float(b.mean()), adapted_mean_error=float(a.mean()),
                    pooled_distance_reduction_pct=float(100 * (b.sum() - a.sum()) / b.sum()) if b.sum() > 1e-8 else None,
                    mean_drift_from_frozen=float(drift.mean()), median_drift_from_frozen=float(np.median(drift)),
                    closer=int((a < b - 1e-8).sum()), farther=int((a > b + 1e-8).sum()), near_zero_baseline=int((b <= 1e-8).sum()))
            actions[group][arm] = entry
    lighting = [dict(case_id=r['case']['id'], previously_harmed=r['previous_closed_loop']['harmed'],
                    layer3=r['layers']['3'], layer12=r['layers']['12'], gate=r['gate'], baseline_errors=r['baseline_errors'])
                for r in rows if r['case']['group'] == 'Lighting']
    # Group labels and old outcome labels are diagnostic targets only, never inputs to the gate.
    pos = [np.mean(r['layers']['12']['normalized_mean']) for r in rows if r['case']['group'] in ('Background', 'Sensor')]
    neg = [np.mean(r['layers']['12']['normalized_mean']) for r in rows if r['case']['group'] in ('Clean', 'Lighting')]
    auc = float(np.mean([float(p > n) + .5 * float(p == n) for p in pos for n in neg]))
    result = dict(schema='smolvla_unified_diag_v1', protocol=m['protocol'], calib_sha256=m['calib_sha256'],
        manifest_sha256=base.sha(out / 'manifest.json'), layer_stats=layers, action_stats=actions, gate_stats=gates,
        lighting_cases=lighting, deep_score_posthoc_domain_group_auc=auc, cases=rows,
        preflight=read(out / 'preflight.json'), raw_sha256={c['id']: base.sha(out / 'cases' / (c['id'] + '.json')) for c in m['cases']},
        limitations=['t=0, one policy noise seed, 50 previously accessed development conditions; not a closed-loop improvement',
            'Clean init2; every Shift uses its own same-task/init1 Clean counterfactual; target Clean never enters fitting/gating',
            'no inference that equal aggregate Clean/Lighting success means lighting invariance',
            'gate is source-only but its mean token/channel statistic can miss position-specific or action-sensitive errors',
            'hypothesis selection and alpha scan are exploratory; no target success labels used by inference',
            'gate needs an unadapted deep pass before shallow intervention; deployment efficiency not benchmarked here',
            'source percentile calibration is empirical, not a guaranteed 5% target false-positive rate',
            'cross-backbone action-drift ratios do not isolate pretraining causes or comparable action scales'])
    write(out / 'summary.json', result)
    lines = ['# SmolVLA：深层检测与浅层步长纯前向诊断', '',
        '50个既有开发条件，固定策略噪声seed0；仅t=0前向，闭环执行局数为0。保留原版BF16、CPU像素归一化、H50/E1/Euler10。', '',
        '## 表1：原生Frozen各层偏移', '',
        '每格为主眼/腕眼。激活率按d2>tau95；R为mean(d2/clamp_min(tau95,1e-6))。', '',
        '| 条件 | L3激活% | L3 R | L6激活% | L6 R | L9激活% | L9 R | L12激活% | L12 R | 深层总闸接受 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        cells = []
        for l in LAYERS:
            x = layers[group][str(l)]
            cells += [' / '.join(f'{100*v:.2f}' for v in x['active_fraction']), ' / '.join(f'{v:.3f}' for v in x['normalized_mean'])]
        lines.append('| ' + group + ' | ' + ' | '.join(cells) + f" | {gates[group]['accepted']}/{gates[group]['n']} |")
    lines += ['', '## 表2：动作误差缩小与Clean漂移', '',
        'Shift每格为XYZ/Rotation距离缩小百分比：100×(ΣFrozen误差−Σ适配误差)/ΣFrozen误差。各例参照同任务、同init、同物理状态、同噪声的Clean Frozen动作。正数表示离线更接近，负数表示更远。',
        'Clean列为XYZ/Rotation绝对漂移，不能计算相对自身零误差的百分比。不混合平移与旋转单位。', '',
        '| 方法 | Clean绝对漂移 XYZ/Rot | Lighting缩小% XYZ/Rot | Background缩小% XYZ/Rot | Sensor sev3缩小% XYZ/Rot |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        clean = actions['Clean'][arm]
        cells = [' / '.join(f"{clean[k]['mean_drift_from_frozen']:.6f}" for k in ('xyz', 'rotation'))]
        for g in ('Lighting', 'Background', 'Sensor'):
            cells.append(' / '.join('N/A' if actions[g][arm][k]['pooled_distance_reduction_pct'] is None else
                                   f"{actions[g][arm][k]['pooled_distance_reduction_pct']:+.2f}" for k in ('xyz', 'rotation')))
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 原Lighting误伤3例与其余7例（逐例）', '',
        '| case | 此前误伤 | L3 R主/腕 | L12 R主/腕 | 深层G | 源阈值 | 接受 |', '|---|---:|---:|---:|---:|---:|---:|']
    for x in lighting:
        cells = [' / '.join(f'{v:.3f}' for v in x[k]['normalized_mean']) for k in ('layer3', 'layer12')]
        g = x['gate']
        lines.append(f"| {x['case_id']} | {x['previously_harmed']} | {' | '.join(cells)} | {g['score']:.3f} | {g['threshold']:.3f} | {g['accepted']} |")
    lines += ['', '非误伤7例包含共同失败，不等于7例均安全成功。历史结局仅作事后分组，不参与阈值、门控或动作。', '',
        '## 总闸定义与防循环论证', '',
        'G为未干预Layer12的双视角、全通道平均R。各目标任务T的阈值只用其余9任务init0完整源轨迹：依次留1源任务，均值/方差拟合8任务，通道tau用这8任务内部7+1留一拟合；计算被留出源任务G，合并后取固定95%分位。部署评分仍使用原9任务参考。目标任务、init1/2及成功标签均不拟合阈值。',
        '先进行未干预深层前向，接受后再运行浅层适配前向；不能在同一次前向读完Layer12再倒回去改Layer3。双相机共用一个对称总闸与同一alpha。此诊断未实测部署延迟。',
        f'事后按Clean+Lighting vs Background+Sensor分组，Layer12 G的AUROC={auc:.4f}；此分组本身不证明哪些例需要适配。', '',
        '| 分组 | 此前误伤 | 其中总闸仍接受 | 此前救回 | 其中总闸仍接受 |', '|---|---:|---:|---:|---:|']
    for group, x in gates.items():
        lines.append(f"| {group} | {x['previously_harmed']} | {x['previous_harms_gate_would_accept']} | {x['previously_rescued']} | {x['previous_rescues_gate_would_accept']} |")
    lines += ['', '以上只是首帧是否放行，不推断新门控整局能避免多少误伤或保留多少救回。', '',
        '## 文献核对与解释边界', '',
        '- SALT arXiv:2610.07946v1 Appendix B.4：检测器与更新分离，阈值按Clean任务交叉拟合；其检测量是相邻帧接口token余弦变化，使用稳健尺度，并非本诊断的深层通道矩偏移。本文不冒称复现SALT或把新gate说成论文已证实机制。',
        '- On the Limits of Representation-Based Test-Time Adaptation for Multimodal LLMs, arXiv:2610.19892v1：讨论position-shared/sample-blind纠偏的局限；其结果不直接证明SmolVLA深层检测有效或无效。',
        '- arXiv:2607.24611的TADD是视频动作识别的双蒸馏适配，不是VLA闭环论文，不能作为SmolVLA因果解释。',
        '- 原论文入口：https://arxiv.org/html/2610.07946v1 ，https://arxiv.org/abs/2610.19892 ，https://arxiv.org/abs/2607.24611 。', '',
        '离线距离只是算子诊断；不把域标签分组当作推理分诊，不把同一统计距离下降作为动作安全证据，不把10例同分当光照免疫，也不把CoRP/SmolVLA漂移倍数当预训练因果证据。保持现有主表与SmolVLA旧负结果，未启动新闭环或访问Blind/Held-out。', '',
        f"校准SHA256：{m['calib_sha256']}。全部50例Frozen与原V1首帧动作逐位复现，双相机顺序、BF16、alpha0和无权重更新均通过实际模型检查。", '']
    (out / 'PHASE3_SMOLVLA_DIAG.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gate_stats=gates, action_stats=actions), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'run', 'report'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'prepare':
        prepare(a.out)
    elif a.mode == 'report':
        report(a.out)
    else:
        try:
            run(a.out)
        except Exception as e:
            write(a.out / 'status.json', dict(phase='failed', error=str(e)))
            raise
