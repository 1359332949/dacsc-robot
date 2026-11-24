"""Condense the full candidate grid without choosing new strengths or layers."""
import json,sys
from pathlib import Path

def report(s):
    cand=s['candidates'];keys=s['candidate_order'][1:];fmt=lambda x:f'{x:.2f}'
    lines=['# 双视角外观不变性与物理位置敏感性：单层算子 V1','',
           f"已用 EGL 采集 {s['n_original']} 个 Dev t=0 实例、{s['n_geometry_frames']} 帧额外的几何反事实（含0位移）。单层顺序为 DINOv2 的第3/6/9/12个 residual block 输出，均在最终 LayerNorm 前；保持 CLS，两个相机使用同一公式和强度，不依赖另一只清晰相机。模型冻结、无梯度、未执行候选动作、未访问 Blind。",'',
           f"四个层的恒等旁路校验 {s['identity_hooks_verified']} 次；原生单噪声官方输出与手动批量采样的最大逐坐标差 {s['max_official_action_error']:.3g}。LOTO 严格排除本任务 Clean；逐通道阈值由内层8任务拟合、在第9个源任务上打分再取分位数。不是按目标动作误差择优。",'',
           '## 预设判决','',f"24 个非恒等配置中，通过名义开发关口：{s['nominal_pass'] or '无'}；对 24 配置 × 3 域 × 2 个分量做多重比较敏感性检查后：{s['simultaneous_pass'] or '无'}。",'',
           f"B 有 {s['pair_counts']['spatially_separated']} 个同任务位置配对，其中 {s['pair_counts']['eligible']} 个通过 Frozen XYZ 响应下限，来自 {s['n_geometry_tasks']} 个任务。它们共享位置与噪声，不能按独立轨迹计数。Frozen 位移命令差与目标物理平移的平均方向余弦（逐对中位数）为 {s['frozen_delta_world_offset_cos_median']:.3f}；几何指标评价的是保留 Frozen 响应，不是专家合理性证明。",'',
           '建议固定第 3 层、95% 死区、α=0.50 做下一轮 Dev 闭环对照。两种第 3 层 α=0.50 配置在多重比较检查中通过 A1 和 A3；A2 的旋转下界未超过零。死区版与均匀版的外观收益接近，但 Clean 平均 XYZ 漂移从 0.01596 降到 0.00847。死区版双眼 Sensor、背景、灯光分别仍有 10/30、8/30、4/30 的 XYZ 参考距离变远。只有 10 个任务，校正后下尾只涉及约 5 个 bootstrap 抽样，不能作确认性总体保证。','',
           f"首段平均 XYZ 命令差与物体位移同向的只有 {s['geometry_checks']['same_direction_world_offset_n']}/{s['pair_counts']['eligible']} 对；运动坐标和动作时序也会影响此检查。没有专家动作或实际执行，不能据此判定抓取正确性。",'',
           'A：下表为四噪声 clipped 12步动作到同任务 Clean Frozen 输出的组内总距离缩小百分比，正为拉近。XYZ/旋转控制坐标分别计算，旋转不是 SO(3) 测地距离。B 为四噪声平移动作差向量的中位余弦与中位幅度比。所有条件同一集合、同一四噪声；没有基于结果补跑参数。','',
           '|层/门控/α|双眼 Sensor XYZ/rot %|Background XYZ/rot %|Lighting XYZ/rot %|Clean XYZ 均/最大漂移|B余弦/幅比|B逐对同时合格|过关的A组|','|---|---:|---:|---:|---:|---:|---:|---|']
    for k in keys:
        v=cand[k];g=v['geometry']['xyz'];c=v['clean']['xyz'];lines.append('|'+k+'|'+'|'.join('/'.join(fmt(v[f][ch]['pooled_gain_pct']) for ch in ('xyz','rot')) for f in ('A1','A2','A3'))+f"|{c['mean']:.3f}/{c['max']:.3f}|{g['cos_med']:.3f}/{g['ratio_med']:.3f}|{g['joint_pass']}/{g['n']}|{','.join(v['check']['families']) or '—'}|")
    lines+=['','## 背景分项与 Flow 事后检查','', '完整 JSON 同时列出 `table`、`scene`、`A1_blur` 的组内动作距离及置信区间、Clean 的绝对 XYZ/rot 漂移、B 两个动作分量的尾部分布和逐例相对曲率。以下摘要每个范围是在冻结的24个配置之间，不是独立确认区间。','']
    for f in ('A1_blur','table','scene','A3'):
        ranges=[(min(cand[k][f][ch]['pooled_gain_pct'] for k in keys),max(cand[k][f][ch]['pooled_gain_pct'] for k in keys)) for ch in ('xyz','rot')];lines.append(f"- {f}：XYZ {ranges[0][0]:.2f}% 至 {ranges[0][1]:.2f}%；rot {ranges[1][0]:.2f}% 至 {ranges[1][1]:.2f}%。")
    for f,ls in s['flow_diagnostic'].items():
        acc=[r['xyz_near_min_agreement']/r['n'] for r in ls.values()];lines.append(f"- {f}：对每个层/门控固定后，在含恒等的四档强度中，最小曲率与最小 XYZ 参考误差的逐例吻合率范围 {min(acc):.1%}–{max(acc):.1%}（相同噪声、事后、含平局，不能作在线门控保证）。")
    for f,r in s['flow_diagnostic'].items():
        d=r['3_dead'];lines.append(f"- 第 3 层死区 {f}：曲率最小选择 α=0.50 的次数 {d['min_energy_choice_counts'].get('3_dead_0.5',0)}/{d['n']}；与逐例最小 XYZ 参考误差吻合 {d['xyz_near_min_agreement']}/{d['n']}。前者是固定候选选择频数，后者是事后逐例比较，均未用于门控。")
    lines+=['','## 如何解释','', 'A 的扰动相机是评测器设置，不提供给算子。A1 含29条双眼模糊和1条双眼雾；A2/A3 是物理一致的官方场景渲染，不是往两眼粘同一块纹理。B 在有相同 tabletop 语义的任务内实际移动目标自由关节，不用不同语言/任务构造假几何配对。运动后执行 forward 刷新，不执行物理控制动作。排除新增/改变目标接触伙伴、或额外穿透>1mm的变体；不保证所有姿态都能闭环成功。', '', '判断只适用于当前参考统计、阈值、层位、强度和首帧；若无合格候选则停止这组算子，不叠多层/WCT/EMA，也不推出所有浅层修正都不可能。若通过，仅获得下一轮同期 EGL 多种子闭环资格；没有任何闭环收益、安全性或 Clean 保持性主张。', '']
    return '\n'.join(lines)
if __name__=='__main__':Path(sys.argv[2]).write_text(report(json.loads(Path(sys.argv[1]).read_text())))
