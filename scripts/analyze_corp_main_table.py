"""Independent complete-case analysis of the paired five-arm raw JSON files."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ARMS=('frozen','pixel_align','deep_block12','actmad_inspired_viewwise_v2','ours_v1')
GROUPS=('clean','sensor_bilateral','lighting','background','table_texture','scene_texture','total')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def selected(c,group):return group=='total' or c['factor']==group or (group in ('table_texture','scene_texture') and c['sub_group']==group)
def stats(rows,arm,comparator):
    if not rows:return dict(n=0,success=0,comparator_success=0,rescued=0,harmed=0,net=0,delta_pp=None,p_exact=None,ci95_task_pp=None,task_clusters=0)
    x=np.array([[r['success'][arm],r['success'][comparator]] for r in rows],dtype=bool)
    rescued=int(np.sum(x[:,0]&~x[:,1]));harmed=int(np.sum(~x[:,0]&x[:,1]));discordant=rescued+harmed
    p=float(binomtest(rescued,discordant,.5,alternative='two-sided').pvalue) if discordant else 1.
    tasks=sorted({r['task'] for r in rows});num=np.array([sum(int(r['success'][arm])-int(r['success'][comparator]) for r in rows if r['task']==t) for t in tasks])
    den=np.array([sum(r['task']==t for r in rows) for t in tasks]);rng=np.random.default_rng(2026112604)
    draws=rng.integers(0,len(tasks),size=(10000,len(tasks)));effects=100*num[draws].sum(1)/den[draws].sum(1)
    return dict(n=len(rows),success=int(x[:,0].sum()),comparator_success=int(x[:,1].sum()),rescued=rescued,harmed=harmed,
        net=rescued-harmed,delta_pp=100*(rescued-harmed)/len(rows),p_exact=p,
        ci95_task_pp=np.percentile(effects,[2.5,97.5]).tolist(),task_clusters=len(tasks))
def analyze(root,suite):
    m=json.loads((root/(suite+'_manifest.json')).read_text());protocol=json.loads((root/'run_protocol.json').read_text())
    rows=[];rawsha={}
    for c in m['cases']:
        for seed in m['seeds']:
            p=root/suite/'jobs'/f"{c['case_id']}__s{seed}.json";v=json.loads(p.read_text())
            assert v['run_fingerprint']==protocol['fingerprints'][suite] and v['seed']==seed and v['case_id']==c['case_id']
            assert v['factor']==c['factor'] and v['sub_group']==c['sub_group'] and v['task']==c['task']
            assert v['paired_initial_state_pixels_and_flow'] and v['all_weights_restored_and_frozen']
            assert [x['arm'] for x in v['arms']]==list(ARMS)
            frozen=v['arms'][0]['trace']
            for arm in v['arms']:
                assert arm['finite'] and 0<arm['steps']<=220 and arm['chunks']==len(arm['trace'])
                assert all(arm['trace'][0][k]==frozen[0][k] for k in ('state','robot','images','z'))
                assert all(a['z']==b['z'] for a,b in zip(arm['trace'],frozen))
            rows.append(dict(case_id=c['case_id'],seed=seed,task=c['task'],factor=c['factor'],sub_group=c['sub_group'],
                bilateral=c['bilateral'],success={x['arm']:x['success'] for x in v['arms']}));rawsha[p.name]=sha(p)
    assert len(rows)==300 and len({(r['case_id'],r['seed']) for r in rows})==300
    assert all(r['bilateral'] for r in rows if r['factor']=='sensor_bilateral')
    assert {g:sum(r['factor']==g for r in rows) for g in ('clean','sensor_bilateral','lighting','background')}==dict(clean=30,sensor_bilateral=90,lighting=90,background=90)
    groups={}
    for group in GROUPS:
        part=[r for r in rows if selected(r,group)]
        groups[group]={cmp:{arm:stats(part,arm,cmp) for arm in ARMS} for cmp in ('frozen','actmad_inspired_viewwise_v2')}
    byseed={str(seed):{g:{arm:stats([r for r in rows if r['seed']==seed and selected(r,g)],arm,'frozen') for arm in ARMS} for g in GROUPS} for seed in m['seeds']}
    return dict(schema='corp_main_table_v2',suite=suite,conditions=100,seeds=m['seeds'],episodes=1500,groups=groups,per_seed=byseed,
        cases=rows,raw_sha256=rawsha,protocol=protocol,calibration_receipt=json.loads((root/'calibration_receipt.json').read_text()),
        limitations=['historically accessed benchmark development pools, not untouched blind confirmation',
            'ActMAD-inspired viewwise VLA port, not faithful original CIFAR spatial L1 ActMAD',
            'p-values are nominal paired binomial tests, not multiplicity or task-dependence adjusted',
            'bootstrap resamples all conditions/seeds of each task jointly, only ten task clusters; degenerate intervals do not establish equivalence',
            'H=E=12 is chunk/execution length; native checkpoint uses 10 Euler iterations; both suites capped at220 by requested frozen protocol'])
def report(root,suites,export):
    export.mkdir(parents=True,exist_ok=True);allresults={}
    for suite in suites:
        result=analyze(root,suite);write(export/(suite+'_5arm.json'),result);allresults[suite]=result
    lines=['# CoRP Route A：同期双目公平物理五臂主表','',
        '主问题仅含Clean、双目Sensor、物理Lighting与Background；没有单眼Sensor。每套100条件×3种子×5臂=1500局，EGL/FP32，H=E=12，原生Euler10，220步上限。',
        'Spatial严格外9/内8+1任务LOTO；Ours沿用指定冻结full.npz。Object四个适配臂仅使用Spatial10全局参考，0条Object轨迹拟合。',
        'ActMAD-v2是明确标注的ActMAD-inspired VLA移植。旧ActMAD/污染语言/错误shared阈值的历史数字不补填。','']
    for suite,result in allresults.items():
        lines+=['## '+suite,'','| 组 | Frozen | Pixel-Align | Deep B12 | ActMAD-inspired v2 | Ours V1 |','|---|---:|---:|---:|---:|---:|']
        for group in GROUPS:
            g=result['groups'][group]['frozen'];cells=[f"{g[a]['success']}/{g[a]['n']}" for a in ARMS]
            lines.append('| '+group+' | '+' | '.join(cells)+' |')
        for cmp in ('frozen','actmad_inspired_viewwise_v2'):
            lines+=['','### 配对比较：相对 '+cmp,'','| 组 | 臂 | 救回/误伤 | 净增 | Δpp | 精确双侧p | 任务聚类95%区间pp |','|---|---|---:|---:|---:|---:|---:|']
            for group in GROUPS:
                for arm in ARMS:
                    if arm==cmp:continue
                    x=result['groups'][group][cmp][arm];ci=x['ci95_task_pp'];ci='—' if ci is None else f'[{ci[0]:+.1f}, {ci[1]:+.1f}]'
                    p='—' if x['p_exact'] is None else f"{x['p_exact']:.6g}"
                    delta='—' if x['delta_pp'] is None else f"{x['delta_pp']:+.1f}"
                    lines.append(f"| {group} | {arm} | {x['rescued']}/{x['harmed']} | {x['net']:+d} | {delta} | {p} | {ci} |")
        lines+=['','### 各种子Ours对Frozen净局数','', '| 组 | seed0 | seed1 | seed2 |','|---|---:|---:|---:|']
        for group in GROUPS:lines.append('| '+group+' | '+' | '.join(f"{result['per_seed'][str(s)][group]['ours_v1']['net']:+d}" for s in (0,1,2))+' |')
    if (root/'latency.json').exists():
        l=json.loads((root/'latency.json').read_text());lines+=['','## 更新后的效率实测','',l['scope'],'','| 臂 | 中位ms | 相对Frozen | 峰值allocated MiB中位 |','|---|---:|---:|---:|']
        for arm in ARMS:
            mem=float(np.median([r['peak_allocated_mib'] for r in l['rows'] if r['arm']==arm]));lines.append(f"| {arm} | {l['median_ms'][arm]:.3f} | {l['relative'][arm]:.3f}× | {mem:.2f} |")
    if len(suites)<2:lines+=['','另一套件尚未完成，未补入任何历史数据。']
    lines+=['','## 审计与解释边界','',
        '- 所有统计直接读取完整原始JSON，binomtest().pvalue不再乘2；bootstrap10,000次按任务整组重采样，保留各任务全部条件与种子。',
        '- 每个case/seed五臂一起原子提交；断点仅跳过通过指纹/完整性核验的完整记录，未完成job按原种子重跑，不按成败筛选。',
        '- 初始物理状态、机器人读数、双目原始输入像素及Flow噪声前缀逐对检查。旧协议限定同物理状态下≤64像素、≤1灰度的EGL舍入统一，逐项记录；不同状态绝不复用图片。',
        '- 参考来自同十条Frozen源轨迹101推理块；第3层统计逐位复现，并与冻结参考逐位复算一致。拟合全部在原生Clean前向，未叠适配。',
        '- 只有十个任务簇，结果不是未见任务总体保证。Clean误伤、负结果、近地板及无显著性均如实保留；不得只凭total最高宣称安全或普适。',
        '- Object零样本指零Object校准/选参；官方池历史已访问，不能冒称未触碰盲测。保护池未读取。','']
    (export.parent.parent/'CORP_MAIN_TABLE_V2.md').write_text('\n'.join(lines))
    return allresults
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--suites',default='spatial,object');p.add_argument('--export',type=Path,required=True);a=p.parse_args()
    results=report(a.out,a.suites.split(','),a.export)
    for suite,x in results.items():print(suite,{arm:x['groups']['total']['frozen'][arm] for arm in ARMS},flush=True)
