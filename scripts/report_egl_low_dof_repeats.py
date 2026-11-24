"""Append the predeclared seed qualification and fixed-method follow-up to a fresh report."""
import json,sys
from pathlib import Path
from report_egl_low_dof import render

def main(root,out):
    s=[json.loads((root/f'seed{i}_summary.json').read_text()) for i in range(3)];a=json.loads((root/'trust_three_seeds.json').read_text());g=a['groups'];lines=[render(s[0]),'## 只复测入选方法：三个策略噪声种子','',
    '|策略 seed|同期 Frozen → 仅视角 /100|Sensor /30|Clean /10|恢复/损害|净增|Sensor 接受门控|配对 p（描述性）|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for i,su in enumerate(s):
        m=su['methods']['trust'];all_=m['all'];se=m['sensor_noise'];cl=m['clean'];lines.append(f"|{i}|{all_['frozen']} → {all_['success']}|{se['frozen']} → {se['success']}|{cl['frozen']} → {cl['success']}|{all_['rescued']}/{all_['harmed']}|{all_['net']:+d}|{se['accepted']}/30|{all_['exact_p']:.4f}|")
    lines+=['',f"合计 {g['all']['baseline_successes']}/300 → {g['all']['method_successes']}/300，恢复 {g['all']['recovered']}、损害 {g['all']['harmed']}。Sensor {g['sensor_noise']['baseline_successes']}/90 → {g['sensor_noise']['method_successes']}/90。不同种子改变策略采样，不扩大扰动资产数量。门控拟合和验收随机流没有跟着 policy seed 重抽；三个种子间接受状态改变的实例数为 {a['gate_changed_across_seeds']}。",'',
    '跨种子区间对基础任务一起重采样，保留同一实例的全部种子：5000次、固定随机种子。仅10个任务，且 Seed 0 同时用于方法筛选；这不是独立盲测。', '',
    '|因素|固定实例数|成对运行数|平均百分点差|按任务聚类的描述性95%区间|','|---|---:|---:|---:|---:|']
    for k,v in g.items():
        lo,hi=v['task_bootstrap_gain_pp_95'];lines.append(f"|{k}|{v['cases']}|{v['n']}|{v['gain_pp']:+.2f}|[{lo:.2f}, {hi:.2f}]|")
    lines+=['','各子类型和逐任务净增见 `trust_three_seeds.json`。这是一项端到端闭环比较，不应把净收益全部归因于自由度；未包含“相同门控、触发后固定退权”对照，无法分离触发选择与连续权重优化的各自贡献。不同种子复用同一批开发实例；尚未建立跨任务或独立测试池上的确认性结果。',''];recovered=[r for r in a['rows'] if r['delta']>0]
    lines+=['',f"23次恢复的去重口径：{len({r['case_id'] for r in recovered})} 个不同实例，{len({r['task'] for r in recovered})} 个基础任务。Seed 2 的同期 Frozen 比历史多赢1个 Background 实例，没有计入方法收益。",'',
            '下一步最小消融：在相同接受集合上，直接比较优化得到的连续权重与固定保留 0.10 的 Value；两者拒绝时都严格回到 Frozen。锁定规则后再寻找方法开发外的数据。']
    out.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
