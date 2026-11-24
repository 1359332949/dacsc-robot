"""Build a report from the verified paired standard and wrist-stress tables."""
import json, sys
from collections import Counter
from pathlib import Path


def render(summary):
    a=summary['aggregate'];seeds=summary['seeds'];rows=summary['rows'];names={'continuous':'单眼连续','fixed':'同门控固定 0.10','symmetric':'对称双眼'}
    lines=['# EGL 门控消融与对称双视角压力测试','',
    '2026-11-24。标准 Dev：同一100个实例，3个策略种子，每例各执行 Frozen、单眼连续、同门控固定0.10、对称双眼，共1200条。反向压力测试：30个 Sensor 配置改为只扰动腕部，Seed 0，重跑自己的 Frozen 和对称方法，共60条。包含 fog 和四类 blur；后者是自定义压力测试而非官方新 Blind。', '',
    '|种子|方法|Clean /10|Sensor /30|Background /30|Lighting /30|总成功 /100|救回 / 损失|总净增|配对 p|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i in range(3):
        x=seeds['standard_'+str(i)]['methods'];fs=['clean','sensor_noise','background','lighting'];d=x['continuous'];lines.append(f'|{i}|同期 Frozen|'+ '|'.join(str(d[f]['base']) for f in fs)+f"|{d['all']['base']}|—|—|—|")
        for m,d in x.items():
            all_=d['all'];lines.append(f'|{i}|{names[m]}|'+'|'.join(str(d[f]['success']) for f in fs)+f"|{all_['success']}|{all_['rescued']}/{all_['harmed']}|{all_['net']:+d}|{all_['exact_p']:.4f}|")
    lines+=['','## 相同实例跨三个策略种子的重复结果','','|方法|Sensor 成功 / 90|Sensor 增益 pp [按任务聚类区间]|所有成功 / 300|总恢复 / 总损失|三个种子总非Sensor触发|','|---|---:|---:|---:|---:|---:|']
    for m in names:
        total=a[m+'/all'];ss=a[m+'/sensor_noise'];lo,hi=ss['task_bootstrap_95'];lines.append(f"|{names[m]}|{ss['base']} → {ss['success']}|{ss['gain_pp']:+.2f} [{lo:.2f}, {hi:.2f}]|{total['base']} → {total['success']}|{total['rescued']}/{total['harmed']}|{total['accepted']-ss['accepted']}|")
    lines+=['','## 同一触发集合里，连续权重有无增量？','',
           '|种子|共同接受|连续成功|固定成功|固定额外救回 / 额外损失（相对连续）|','|---|---:|---:|---:|---:|']
    for i in range(3):
        d=seeds['standard_'+str(i)]['fixed_vs_continuous_accepted'];lines.append(f"|{i}|{d['n']}|{d['base']}|{d['success']}|{d['rescued']}/{d['harmed']}|")
    direct=summary['direct_fixed_vs_continuous_accepted'];lines+=['',f"共同触发合计 {direct['n']} 次，来自 {direct['unique_cases']} 个不同实例。直接配对：仅固定权重成功 {direct['fixed_only']} 次，仅连续权重成功 {direct['continuous_only']} 次。该小样本比较不是等效性检验。"]
    lines+=['','## 对称方法选择了哪只眼','', '|条件|种子|因素|接受数|退主眼|退腕眼|同期 Frozen → 对称|救回 / 损失|','|---|---:|---|---:|---:|---:|---:|---:|']
    for label,s in seeds.items():
        regime,i=label.split('_');d=s['methods']['symmetric'];
        for f,v in d.items():
            if f=='all':continue
            lines.append(f"|{regime}|{i}|{f}|{v['accepted']} / {v['n']}|{v['agent']}|{v['wrist']}|{v['base']} → {v['success']}|{v['rescued']}/{v['harmed']}|")
    lines+=['','## 腕部压力：分子类型','', '|类型|实例|Frozen → 对称|救回 / 损失|退主眼 / 退腕眼|','|---|---:|---:|---:|---:|']
    w=[r for r in rows if r['regime']=='wrist' and r['method']=='symmetric']
    for t in sorted({r['subtype'] for r in w}):
        rr=[r for r in w if r['subtype']==t];lines.append(f"|{t}|{len(rr)}|{sum(r['base'] for r in rr)} → {sum(r['success'] for r in rr)}|{sum(r['delta']>0 for r in rr)}/{sum(r['delta']<0 for r in rr)}|{sum(r['eye']=='agent' for r in rr)}/{sum(r['eye']=='wrist' for r in rr)}|")
    ws=seeds['wrist_0']['methods']['symmetric']['all'];lines+=['',f"腕部压力下，投影优化后、验收前的候选方向计数：{ws['candidate_eyes']}。最终接受主眼 {ws['agent']}，腕眼 {ws['wrist']}。优化曾选中某只相机，并不保证独立噪声验收也通过。"]
    lines+=['','## 判决与限制','',f"预设标准 Dev 关口：{summary['standard_screen']}；反向腕部关口：{summary['wrist_screen']}；同时通过：{summary['both_pass']}。历史 Frozen/连续/接受状态与本次不完全一致的 seed-case 数：{len(summary['prior_differences'])}，已逐例保留；所有方法成绩均以本次同期 Frozen 配对。",'',
        '门控优化和验收不读哪只眼受扰、当前域标签、成败或 Clean 统计；扰动投放位置只由评测器控制。对称的是两个系数的约束及其优化，并非假定冻结策略本身以相同方式利用两路画面。未触发不等于识别对了受损相机。一次种子内 p 为配对不一致计数的二项精确检验，未校正多重比较；重复种子不是更多独立实例。未测试全局几何是否被保留，也未打开任何保护数据。', '',
        '注意：固定 0.10 组仍然需要拟合连续候选来生成同一份门控判定，所以这是对执行权重的消融，并没有消除拟合的计算成本。该实验没有“所有样本都固定衰减”的新对照，不直接识别门控本身相对无条件衰减的因果贡献。腕部30个配置只有一个策略种子；这限制了外推，但不足以把本轮失败写成已经通过。', '',
        '证据：`data/egl_trust_symmetry/summary.json`、`evidence.tar.xz`、`protocol.md` 和 `verification.json`。归档含逐例两路权重、验收噪声/损失、状态/输入/动作哈希、终点、步数和舍入校正记录。','']
    return '\n'.join(lines)
if __name__=='__main__':Path(sys.argv[2]).write_text(render(json.loads(Path(sys.argv[1]).read_text())))
