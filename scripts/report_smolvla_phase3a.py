"""Summarize completed Phase3A; no result-based tuning or phase3B launch."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np


def main(a):
    root=a.out
    status=json.loads((root/'status.json').read_text());assert status['phase']=='complete'
    protocol=json.loads((root/'protocol.json').read_text())
    folds=json.loads((root/'folds.json').read_text())
    h=hashlib.sha256((root/'spatial_init0_calib.npz').read_bytes()).hexdigest();assert h==folds['calib_sha256']
    all_ids=set(range(10))
    for t,fold in folds['folds'].items():
        assert set(fold['outer'])==all_ids-{int(t)}
        for held,names in fold['inner'].items():assert set(names)==all_ids-{int(t),int(held)}
    cases=[];scans=[];source=[]
    for t in range(10):
        src=json.loads((root/'source'/f'task{t}.json').read_text());assert src['init_index']==0;source.append(src)
        d=json.loads((root/'guard'/f'task{t}.json').read_text());f,v=d['rows']
        assert f['task_id']==v['task_id']==t and f['init_index']==v['init_index']==1
        assert f['start']==v['start'] and d['start_exact']
        common=min(f['steps'],v['steps']);assert f['rng_before_replans'][:common]==v['rng_before_replans'][:common]
        assert src['start']['sim']!=f['start']['sim'],'calibration and guard state duplicated'
        rows=[]
        for r in (f,v):
            assert len(r['frames'])==r['steps']==len(r['rng_before_replans'])
            assert all(np.isfinite(frame['action']).all() for frame in r['frames'])
            rows.append({k:r[k] for k in ('mode','success','steps','seconds')})
            rows[-1]['mean_activation_by_view']=np.mean([frame['active'] for frame in r['frames']],axis=0).tolist()
            rows[-1]['mean_gain_abs_by_view']=np.mean([frame['gain_abs'] for frame in r['frames']],axis=0).tolist()
        cases.append({'task_id':t,'task':f['task'],'rows':rows,'rescued':not f['success'] and v['success'],
                      'harmed':f['success'] and not v['success'],'initial_state_and_rgb_exact':True,'rng_prefix_exact':True})
        assert [x['layer'] for x in d['scan']]==[3,6,9,12]
        assert all(x['alpha0_action_exact'] for x in d['scan'])
        scans.extend([dict(x,task_id=t) for x in d['scan']])
    rescued=sum(c['rescued'] for c in cases);harmed=sum(c['harmed'] for c in cases)
    frozen=sum(c['rows'][0]['success'] for c in cases);v1=sum(c['rows'][1]['success'] for c in cases)
    disc=rescued+harmed
    p=1. if not disc else min(1.,2*sum(math.comb(disc,k) for k in range(min(rescued,harmed)+1))/2**disc)
    latency=json.loads((root/'latency.json').read_text());pre=json.loads((root/'preflight.json').read_text())
    layer_summary={str(l):{'xyz_drift_mean':statistics.mean(r['xyz_drift_l2'] for r in scans if r['layer']==l),
                         'rotation_drift_mean':statistics.mean(r['rotation_drift_l2'] for r in scans if r['layer']==l),
                         'action_drift_mean':statistics.mean(r['action_drift_l2'] for r in scans if r['layer']==l),
                         'activation_macro_by_view':np.mean([r['active'] for r in scans if r['layer']==l],axis=0).tolist()}
                   for l in (3,6,9,12)}
    gate=(harmed==0 and v1>=frozen)
    postchecks={}
    if (root/'oracle_tests.log').exists():
        log=(root/'oracle_tests.log').read_text()
        postchecks['original_v1_oracle_tests_passed']=('Ran 2 tests' in log and '\nOK' in log)
    if (root/'replay_task0.json').exists():
        replay=json.loads((root/'replay_task0.json').read_text())
        postchecks['harmed_case_replay']={k:v for k,v in replay.items() if k!='rows'}
        postchecks['harmed_case_replay']['outcomes']=[{k:r[k] for k in ('mode','success','steps')} for r in replay['rows']]
    summary={'schema':'smolvla_phase3a_v1','protocol':protocol,'status':'completed',
       'clean_guard':{'n':10,'frozen_successes':frozen,'v1_successes':v1,'rescued':rescued,'harmed':harmed,
                      'net':rescued-harmed,'mcnemar_exact_p':p,'observed_zero_harm_gate':gate,
                      'role':'previously examined DEVELOPMENT Clean, not independent unseen confirmation'},
       'activation_macro_by_view':{arm:np.mean([c['rows'][i]['mean_activation_by_view'] for c in cases],axis=0).tolist()
                                  for i,arm in enumerate(('frozen','v1'))},
       'cases':cases,'clean_t0_layer_scan':layer_summary,'alpha0_exact_action_checks':40+len(pre['alpha0_full_action_identity_layers']),
       'preflight':pre,'latency':latency,'calibration':{'sha256':h,'total_frames':sum(s['steps'] for s in source),
                  'source_outcomes_not_filtered':True,'source_successes':sum(s['success'] for s in source),'source_task_frames':{str(s['task_id']):s['steps'] for s in source}},
       'durations':status['durations'],'phase3b_autostarted':False,'postchecks':postchecks,
       'decision':'Clean guard passed for this pilot; requires user review before main table' if gate else 'NO-GO: Clean guard harmed a previously successful case; do not launch main table',
       'limitations':['one policy seed, ten previously examined init1 states; no population zero-risk claim',
             'scan measures Clean t0 drift, not physical geometry sensitivity or shifted-domain repair',
             '95% nested-source threshold does not guarantee 5% activation on held-task Clean',
             'SALT Table13 uses LIBERO10 H50/E8; this Spatial E1 pilot is not a direct SALT comparison']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['# Phase 3A：SmolVLA V1 校准、Clean 护栏与延迟','',
          f"**本轮判决：{'样本内 Clean 护栏通过，等待主表协议确认' if gate else 'NO-GO：Clean 出现误伤，暂不进入主表'}。** 同期 Frozen {frozen}/10，V1 {v1}/10；救回 {rescued}、误伤 {harmed}、净 {rescued-harmed:+d}。不是三域效果或独立盲测。",'',
          '## 1. 实现与隔离','',
          '- 使用已复现的兼容环境，原版 ro boeval 的 SmolVLA.load / act / 输入归一化 / 动作后处理，以及原版 suite.reset / step。外层仅选择指定init、控制配对RNG和注册hook。未修改上游源码或权重。',
          '- 原始Hub配置load_vlm_weights=True，视觉BF16；hook内mean/std/d2/gain/out均FP32，返回BF16。无CLS，完整1024patch；四层均校验每次replan恰好两次串行调用，真实像素输入顺序已核对。',
          '- 模型保持原生n_action_steps=1、Euler10、chunk50；EGL、env.noise=0、settling10、max220。没有套用CoRP H=E12。',
          '- 环境Python/NumPy/libc srand及上游env.seed固定为task ID；每个回合在环境reset/settling后显式重置Torch CPU和CUDA seed0。两臂首帧物理state/RGB逐位一致，所有共同replan的RNG状态hash前缀相同。',
          '- source仅十任务init0全部实际Frozen轨迹，包含失败轨迹、不按成功筛选；全轨迹帧等权，与冻结V1定义一致。目标任务T的mu/ss来自其他9任务，tau由这9任务内部8+1留任务评分后取95%分位。',
          '- **init1此前已被诊断访问**，本轮称“校准状态不重叠的开发Clean”，不能称未见/独立确认。每任务source/guard的实际初始模拟状态hash亦不同；不打开Blind/Held-out清单。',
          f"- 校准共 {sum(s['steps'] for s in source)} 个实际replan；文件 spatial_init0_calib.npz，SHA {h}。",'',
          '## 2. 配对Clean结果','',
          '| Task | Frozen | V1 | 救/伤 | V1主眼/腕眼平均激活率 |','|---:|---:|---:|---|---|']
    for c in cases:
        f,v=c['rows'];rates=v['mean_activation_by_view']
        text.append(f"| {c['task_id']} | {int(f['success'])} | {int(v['success'])} | {int(c['rescued'])}/{int(c['harmed'])} | {100*rates[0]:.1f}% / {100*rates[1]:.1f}% |")
    text += ['',f"精确McNemar p={p:.6g}。本轮只有一个策略种子、10个任务/状态；即使零误伤，也只能说本样本未观察到误伤。 Frozen宏平均激活率：主眼{100*summary['activation_macro_by_view']['frozen'][0]:.2f}%、腕眼{100*summary['activation_macro_by_view']['frozen'][1]:.2f}%；V1：主眼{100*summary['activation_macro_by_view']['v1'][0]:.2f}%、腕眼{100*summary['activation_macro_by_view']['v1'][1]:.2f}%。",'',
      '## 3. 浅/深层单层扫查','',
      '固定alpha=.75，四层分别独立干预，同一缓存Clean t0和同一随机数；不是多层叠加，不据此挑选新主方法。这里只测动作漂移，不能称为真实移物体的几何保真或三域修复。', '',
      '| 层 | XYZ平均漂移L2 | Rotation平均漂移L2 | 动作平均漂移L2 |','|---:|---:|---:|---:|']
    for l,r in layer_summary.items():text.append(f"| {l} | {r['xyz_drift_mean']:.6f} | {r['rotation_drift_mean']:.6f} | {r['action_drift_mean']:.6f} |")
    text += ['',f"全模型alpha0精确恒等共 {summary['alpha0_exact_action_checks']} 次：真实接口预检4层，10个开发起点各4层。所有动作finite，六项CPU测试通过；4路源/评测worker均校验参数版本未改变、梯度关闭。",'',
       '## 4. SmolVLA延迟/显存表','',
       '单GPU空闲后测；预热2对，5轮交错、CUDA前后同步。包含原版CPU预处理、完整replan及动作CPU回传；V1包含激活统计开销。不包含环境step和加载模型。','',
       '| 方法 | replan中位ms | 均值ms | Frozen倍数 | 峰值allocated MB |','|---|---:|---:|---:|---:|']
    for arm in ('frozen','v1'):
        rr=[r for r in latency['rows'] if r['arm']==arm]
        text.append(f"| {arm} | {latency['median_ms'][arm]:.2f} | {statistics.mean(r['ms'] for r in rr):.2f} | {latency['median_ms'][arm]/latency['median_ms']['frozen']:.3f} | {max(r['peak_allocated_mb'] for r in rr):.1f} |")
    text += ['', '5个样本用于工程量级估计，不包装为精确性能分布；原始逐轮数据见JSON。','',
       '## 5. 推进边界','',summary['decision'],
       'SALT原文Appendix B.6 Table13确实从首个replan施扰，但其LIBERO10使用H50/E8（Table17）。本轮Spatial/E1和Clean护栏不构成同协议对比；后续主表需先锁任务套件、执行频率、双视角扰动定义、严重度、同时Frozen、多种子及source-data预算。',
       'SALT来源：https://arxiv.org/html/2610.07946v1 。不把离线Clean漂移或启动成功率当成TTA闭环收益。','']
    if postchecks:
        text += ['## 补充复核（不并入主批次计数）','']
        if postchecks.get('original_v1_oracle_tests_passed'):
            text += ['另两项独立oracle测试通过：直接执行冻结CoRP V1源码中的align/fit/score；串行BF16视角输出与原FP32公式再转BF16逐位相同，全轨迹LOTO的mu/ss/tau也逐位相同。加上原六项共八项测试。没有修改本轮算法或调参。','']
        if 'harmed_case_replay' in postchecks:
            rr=postchecks['harmed_case_replay']
            text += [f"仅对首个误伤task0额外回放一次，动作/初始状态/RNG/步数/成败全部精确复现：{rr['all_fields_exact']}。",
                     '该回放属于诊断复核，不是新的独立样本，不替换原结果。','']
    (root/'PHASE3_SMOLVLA_V1_PILOT.md').write_text('\n'.join(text).replace('ro boeval','roboeval'))
    print(json.dumps(summary['clean_guard']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
