"""Complete-case paired reporting. No partial results or silent pair dropping."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np


def exact_mcnemar(rescued,harmed):
    n=rescued+harmed
    return min(1.,2*sum(math.comb(n,k) for k in range(min(rescued,harmed)+1))/2**n) if n else 1.


def summarize(pairs):
    assert pairs
    f=sum(p[0]['success'] for p in pairs);v=sum(p[1]['success'] for p in pairs)
    r=sum(not p[0]['success'] and p[1]['success'] for p in pairs)
    h=sum(p[0]['success'] and not p[1]['success'] for p in pairs)
    tasks=sorted({p[0]['task_id'] for p in pairs});rng=np.random.default_rng(2026112603)
    # Resample whole task clusters, keeping all conditions of each selected task.
    cells={t:[p for p in pairs if p[0]['task_id']==t] for t in tasks};boot=[]
    for _ in range(10000):
        sample=[p for t in rng.choice(tasks,len(tasks),replace=True) for p in cells[t]]
        boot.append(100*sum(int(p[1]['success'])-int(p[0]['success']) for p in sample)/len(sample))
    active={}
    for i,arm in enumerate(('frozen','v1')):
        active[arm]=np.mean([np.mean([f['active'] for f in p[i]['frames']],axis=0) for p in pairs],axis=0).tolist()
    return dict(n=len(pairs),frozen=f,v1=v,rescued=r,harmed=h,net=v-f,delta_pp=100*(v-f)/len(pairs),
                mcnemar_exact_two_sided=exact_mcnemar(r,h),task_cluster_bootstrap95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                active_fraction_episode_macro=active)


def main(a):
    manifest=json.loads((a.run/'manifest.json').read_text());pre=json.loads((a.run/'preflight.json').read_text())
    assert pre['status']=='PASS';pairs=[]
    for c in manifest['cases']:
        p=[json.loads((a.run/'pairs'/f"{c['id']}_{arm}.json").read_text()) for arm in ('frozen','v1')]
        check=json.loads((a.run/'pairs'/f"{c['id']}_pair.json").read_text())
        assert check['start_exact'] and check['paired_rng_prefix_exact']
        assert p[0]['start']==p[1]['start'] and p[0]['observed_start']==p[1]['observed_start']
        n=min(p[0]['steps'],p[1]['steps']);assert p[0]['rng_before_replans'][:n]==p[1]['rng_before_replans'][:n]
        pairs.append(p)
    assert len(pairs)==50
    groups={g:summarize([p for p in pairs if p[0]['group']==g]) for g in ('Clean','Sensor','Lighting','Background')}
    groups.update({g:summarize([p for p in pairs if p[0]['subgroup']==g]) for g in
                  ('motion_blur','gaussian_blur','zoom_blur','glass_blur','fog','table_texture','scene_texture')})
    groups['Shift40']=summarize([p for p in pairs if p[0]['group']!='Clean'])
    groups['All50']=summarize(pairs)
    # Pool init1 only when it is exactly the native Vanilla Clean protocol.
    old=json.loads((a.phase3a/'summary.json').read_text())
    old_pairs=[]
    for task in range(10):
        rows=json.loads((a.phase3a/'guard'/f'task{task}.json').read_text())['rows']
        p=sorted(rows,key=lambda r:r['mode']);assert [r['mode'] for r in p]==['frozen','v1']
        for row in p:
            assert row['policy_seed']==0 and row['init_index']==1 and row['vision_dtype']=='torch.bfloat16'
            row['group']='Clean';row['subgroup']='Clean'
        old_pairs.append(p)
    assert old['protocol']['model']['load_vlm_weights'] is True
    folds=json.loads((a.phase3a/'folds.json').read_text())
    assert folds['calib_sha256']==manifest['calib_sha256']
    assert old['protocol']['model']['n_action_steps']==manifest['protocol']['E']==1
    assert old['protocol']['environment']['max_steps']==manifest['protocol']['max_steps']==220
    groups['Clean20_development_init1_init2']=summarize(old_pairs+[p for p in pairs if p[0]['group']=='Clean'])
    compact=[{**{k:p[0][k] for k in ('case_id','task_id','group','subgroup','init_index')},
              'frozen':p[0]['success'],'v1':p[1]['success'],'steps':[r['steps'] for r in p],
              'active':[np.mean([f['active'] for f in r['frames']],axis=0).tolist() for r in p],
              'start_exact':True,'rng_prefix_exact':True} for p in pairs]
    shift=groups['Shift40'];clean=groups['Clean20_development_init1_init2']
    result=dict(schema='smolvla_phase3b_pilot_v1',protocol=manifest['protocol'],groups=groups,pairs=compact,
                calib_sha256=manifest['calib_sha256'],manifest_sha256=hashlib.sha256((a.run/'manifest.json').read_bytes()).hexdigest(),
                preflight=pre,decision='exploratory result for review; no automatic formal-table launch',
                limitations=['one policy seed; tiny subgroup counts; task bootstrap has only ten clusters',
                    'Clean20 cannot establish a 1–2pp noninferiority margin; Phase3A reproduced harm retained',
                    'official Plus scene selectors on reproduced Vanilla base; not unmodified full Plus evaluator',
                    'Plus severity5 is on its ten-level corruption scale; not a matched SALT leaderboard',
                    'all conditions development-only, not untouched confirmation'])
    a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# SmolVLA Phase 3B：冻结 V1 的开发集 Shift 摸底','',
        '50 条件 × Frozen/V1，共 100 局。模型、校准包、Layer 3 与 α=0.75 不变；单策略种子 0。',
        '本轮是开发探索，不能当作独立确认或 Clean 非劣证明。',
        '', '| 分组 | 条件数 | Frozen | V1 | 救回 / 误伤 | 净增 | 精确 McNemar p | 任务聚类95%区间(pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in groups.items():
        lo,hi=g['task_cluster_bootstrap95_pp']
        lines.append(f"| {name} | {g['n']} | {g['frozen']}/{g['n']} | {g['v1']}/{g['n']} | {g['rescued']} / {g['harmed']} | {g['net']:+d} | {g['mcnemar_exact_two_sided']:.4f} | [{lo:+.1f}, {hi:+.1f}] |")
    lines+=['','## 协议核验','',
        '- 原生 SmolVLA load_vlm_weights=True，BF16 视觉、CPU 像素归一化、E=1/H=50/Euler10，220 步。',
        '- env.noise=0；双目 Sensor 自首个策略观察全程生效，两眼使用独立、逐步确定的扰动随机数。',
        '- 灯光/背景使用官方 BDDL 和对应场景 XML/样式，映射到未改动的 Vanilla 基类；保留原版相机/控制流程。20 个物理变体全部预检物理/相机参数与初始状态一致、画面确实变化。',
        '- 所有条件使用原始任务的干净语言，不把变体编号写进提示。50 对初始状态、两眼 RGB、共同步 Torch CPU/CUDA RNG 哈希全部一致。',
        '- 10任务 init0 全轨迹校准不变，任务外9/内8+1 LOTO；测试 init1/2，未访问封存池。',
        '- 选择资产只依赖已有历史开发池、任务/子类与固定排序，不看成败。官方池历史已跑 Frozen，本轮不称盲测。',
        '', '## 激活率（先每局取均值，再对局平均）','',
        '| 分组 | Frozen 主/腕 | V1 主/腕 |','|---|---:|---:|']
    for name in ('Clean','Sensor','Lighting','Background','Shift40'):
        g=groups[name];f=g['active_fraction_episode_macro']['frozen'];v=g['active_fraction_episode_macro']['v1']
        lines.append(f'| {name} | {100*f[0]:.2f}% / {100*f[1]:.2f}% | {100*v[0]:.2f}% / {100*v[1]:.2f}% |')
    lines+=['','## 判读边界','',
        f"Shift40 净增 {shift['net']:+d}（救 {shift['rescued']}、伤 {shift['harmed']}）。Clean20 合计净增 {clean['net']:+d}（救 {clean['rescued']}、伤 {clean['harmed']}），保留此前 init1 的所有结果。",
        'p 不显著不等于无伤害，5% 源阈值不保证目标激活率为5%。子类仅4或5例，不能据此宣称稳定收益。',
        '本轮不满足与 SALT 直接排名的同协议条件，也未增加 ActMAD 比较。完整大表是否推进依据以上完整救伤数据另行评审。',
        '', f"校准 SHA256：`{manifest['calib_sha256']}`。",'']
    a.md.parent.mkdir(parents=True,exist_ok=True);a.md.write_text('\n'.join(lines))
    print(json.dumps({k:groups[k] for k in ('Clean','Sensor','Lighting','Background','Shift40','Clean20_development_init1_init2')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--phase3a',required=True,type=Path)
    p.add_argument('--json',required=True,type=Path);p.add_argument('--md',required=True,type=Path);main(p.parse_args())
