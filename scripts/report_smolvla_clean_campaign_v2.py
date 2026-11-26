"""Summarize a complete locked Clean campaign; never count process errors as failures."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(a):
    full = json.loads((a.input / 'summary.json').read_text())
    assert not full['errors'] and full['n'] == 30
    assert all(all(r['matches'].values()) for r in full['anchor_checks'])
    rows = full['rows']
    assert {(r['task_id'], r['init_index']) for r in rows} == {(t, i) for t in range(10) for i in (1, 2, 3)}
    assert len({(r['task_id'], r['raw_init_sha256']) for r in rows}) == 30
    assert all(r['weights_keys_matched'] == 788 for r in rows)
    assert all(r['first']['vision_dtype'] == 'torch.float32' for r in rows)
    assert all(r['seed'] == 10 * r['task_id'] for r in rows)
    by_task = full['by_task']
    task_rates = np.array([by_task[str(t)]['successes'] / 3 for t in range(10)])
    rng = np.random.default_rng(20261126)
    samples = task_rates[rng.integers(0, 10, size=(50000, 10))].mean(1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    model = json.loads((a.input / 'actmad_real_model.json').read_text())
    assert model['status'] == 'PASS'
    compact = {'version': 'repro_baseline_v2', 'n': 30, 'successes': full['successes'],
               'rate': full['successes'] / 30, 'task_cluster_bootstrap_ci95': interval,
               'bootstrap_seed': 20261126, 'bootstrap_resamples': 50000,
               'by_task': by_task, 'anchor_checks': full['anchor_checks'],
               'unique_initial_states': 30, 'weights_checked_per_case': 788,
               'protocol': {k: v for k, v in full['manifest'].items() if k != 'source_sha256'},
               'actmad_port_model_check': model,
               'limitations': ['same ten previously examined tasks; not blind or cross-task confirmation',
                   'one policy seed per task; three initial states, not three policy seeds',
                   'not the full community 100-episode reproduction',
                   'all states in this smoke are development-accessed and excluded from future untouched Clean claims',
                   'no TTA closed-loop success comparison in this campaign',
                   'ActMAD-inspired port remains different from original spatial L1 ActMAD']}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(compact, indent=2) + '\n')
    text = [
        '# Frozen 基线复现与对照实现修复 V2', '',
        f"真 vanilla LIBERO + 固定旧版 LeRobot，30 个额外初始状态的完整 Clean 闭环：{full['successes']}/30（{100*full['successes']/30:.1f}%）。",
        f"按10个任务聚类bootstrap的95%区间：[{100*interval[0]:.1f}%, {100*interval[1]:.1f}%]。这不是与社区成绩的显著性比较。", '',
        '## 协议与重放', '',
        '- 固定 HuggingFaceVLA/smolvla_libero 权重，旧版LeRobot 483be9aa，真vanilla资产；EGL、视觉FP32且模型含BF16、env.noise=0；此构建路径尚不等同原版。',
        '- n_action_steps=1、Euler10、256像素、settling10、最多280步；没有TTA、参数更新或选参。CoRP的H=E=12没有强加给SmolVLA。',
        '- 十个任务各init1/2/3，环境/策略种子均为10*task_id；不是三个独立策略种子。',
        '- 先重放task0/init0成功和task4/init0失败，成功标记、步数、初始物理状态、两路RGB及完整动作hash均与旧记录一致，才展开30局。锚点不计入30局。',
        '- 每任务3个初始状态互异，每局788个权重张量核对匹配。没有打开封存Blind/Held-out清单；本轮所有状态记为已接触开发体检，不能再充当独立未见Clean。', '',
        '## 逐任务结果', '', '| Task ID | init1/2/3 成功数 |', '|---:|---:|']
    text += [f"| {t} | {by_task[str(t)]['successes']}/3 |" for t in range(10)]
    text += ['', '## ActMAD-inspired 对照修复', '',
        '- 修正view-major多帧数据被当作frame-major的错误，聚合时不再混合双相机。去掉256patch/384channel的硬编码，加入shape/view-order检查。',
        '- 三项CPU测试通过：相机所有权与梯度、更新/重置/移除生命周期、非法数量拒绝。',
        '- 真实CoRP检查：使用已校验Dev缓存；参考统计排除当前任务，仅其余九任务。没有使用目标成败标签。',
        '- 安装未更新时动作逐位恒等；4步更新均finite且参数变化；动作变化；reset与close后逐位恢复Frozen。H=E=12。',
        f"- 四步后动作L2变化{model['action_change_l2']:.8f}。这是执行链验证，不是适配收益。",
        '- 明确标为ActMAD-inspired：仍采用空间池化、均值/标准差标准化平方损失，非作者原始逐空间位置的均值/方差L1。兼容旧类名不使旧actmad_true成绩自动有效。', '',
        '## 能说与不能说', '',
        '- 本轮把底座多起点稳定性和适配对照执行链分开验证；不把程序finite当成任务成功。',
        '- 不把本轮与此前8/10直接比较为方法涨跌：初始状态不同。',
        '- 仍不是社区100局完全复现，不是独立新任务或盲测，也没有V1第二底座闭环增益。',
        '- 旧错误ActMAD对照和错误共享校准文件的成绩仍需隔离，不能用新代码反向追认。',
        '- 下一阶段先据逐任务失败分布判断Frozen是否足够稳定，再预注册同底座、同参考预算与同测试清单的正式TTA比较。', '']
    a.report.write_text('\n'.join(text))
    print(json.dumps({'successes': full['successes'], 'n': 30, 'ci95': interval}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    main(p.parse_args())
