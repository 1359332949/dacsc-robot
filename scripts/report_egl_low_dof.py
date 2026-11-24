"""Render a human-readable report from the verified seed 0 paired summary."""
import argparse,json
from pathlib import Path

NAMES={'rgb':'仅 RGB','trust':'仅视角门控','joint':'RGB＋视角门控'}

def render(s):
    hist=s['frozen_vs_archive'];out=[f'# EGL Seed {s["seed"]}：低自由度干预的闭环对照', '',
    f'实际执行同一份 Dev 100 个实例 × 4 个方法（含同期 Frozen）。历史 Frozen {hist["frozen"]}/100，同期 Frozen {hist["success"]}/100；历史→同期恢复 {hist["rescued"]}，损失 {hist["harmed"]}。表中成败转移均与同期 Frozen 配对。', '',
    '|方法|Clean /10|Sensor /30|Background /30|Lighting /30|总成功 /100|恢复 / 损害|净增|精确 McNemar p|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    g=['clean','sensor_noise','background','lighting'];out.append('|同期 Frozen|'+'|'.join(str(s['methods']['rgb'][f]['frozen']) for f in g)+f'|{hist["success"]}|—|—|—|')
    for m,d in s['methods'].items():
        a=d['all'];out.append('|'+NAMES[m]+'|'+'|'.join(str(d[f]['success']) for f in g)+f'|{a["success"]}|{a["rescued"]} / {a["harmed"]}|{a["net"]:+d}|{a["exact_p"]:.4f}|')
    out+=['','## 逐因素配对与门控','','每格依次为「恢复 / 损害；接受退权数」。仅 RGB 不拟合门控。','', '|方法|Clean|Sensor|Background|Lighting|','|---|---|---|---|---|']
    for m,d in s['methods'].items():out.append('|'+NAMES[m]+'|'+'|'.join(f"{d[f]['rescued']} / {d[f]['harmed']}；{d[f].get('accepted','—')}" for f in g)+'|')
    j=s['joint_vs_rgb'];out+=['',f'联用直接对比仅 RGB：恢复 {j["rescued"]}、损害 {j["harmed"]}、净增 {j["net"]:+d}。拒绝退权的 {s["joint_rollback_hashes_verified"]} 个实例与仅 RGB 的实际执行动作摘要、步数、成败一致。', '',
    '## 复现、解释与限制','',
    '原 EGL 实现把统计 seed 放在环境创建之后。预检发现一个 Lighting 案例在相同物理状态重置10次时，主视角有两种不同哈希；实测差别仅3个像素、最大1个灰度级。未据此推断所有差异的驱动根因。前3次预检分别保留，未混入正式成功率。', '',
    '最终版本在构造前固定 Python、NumPy、Torch 和 libc 随机状态。所有臂各自 hard reset、各自真实步进；同一实例内如果同一步、同一模拟状态及机械臂读数完全相同，则检查两路实际图像，只允许最多64个像素、最大1级的舍入差异，并统一为此状态的第一张图。不同状态不共用画面。这是配对评估工具，不是需随方法部署的输入。对照方法不读取目标域标签或成败，也不以历史数据替代执行。', '',
    '门控与校色细节及文献核对见 `data/egl_low_dof/protocol.md` 和 `literature.md`。Clean 仅10个起点；同任务实例、跨方法和多重比较存在相关性，精确 McNemar p 仅作描述。本轮不能建立几何不变性、确认集泛化或安全性定理。', '',
    f'按预设重复种子筛选（同期净增≥6、总成功≥79、Clean 10/10），入选：{s["qualifiers"] or "无"}。', '']
    return '\n'.join(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(render(json.loads(a.summary.read_text())))
