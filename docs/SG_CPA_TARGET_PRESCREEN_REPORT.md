# SG-CPA 目标预筛报告（随机数修复后，纯离线）

> 触发：审查发现视图噪声与动作 latent 在 episode 内复用；修复后重新比较三种目标：
> **Medoid incl identity / Medoid excl identity / Anchored Mean α=0.1**。
> 数据：4 域 × 2 套件 × 1 任务 × 4 chunks = **32 个观测**，每个观测 2 个 replan seeds（共 64 样本/目标）。
> 所有目标共享相同 4 视图、参数化、A 投影；仅监督目标不同。
> 原始 JSON：`/hy-tmp/validation/sg_cpa_prescreen/prescreen.json`；脚本 `scripts/sg_cpa_target_prescreen.py`。
> **更新（η 标定后）**：预筛脚本原先手写更新且少负号，现已改为走真实 `SGCpa.step()`；
> controller 单位变化改为 `unnormalize(a1)−unnormalize(a0)`；数字与首版基本一致（da≈1e-6），
> 因为 η=0.05 一步本来就极小。真正的尺度问题见 `SG_CPA_ETA_CALIBRATION_REPORT.md`。

## 1. 结果

| 目标 | 域 | identity 频率 | target−identity | 初始蒸馏 loss | 梯度范数 | 投影后范数 | retain | **一步后动作变化** | β 范数 |
|---|---|---|---|---|---|---|---|---|---|
| medoid incl | Camera | 0.812 | 0.00090 | 4.6e-4 | 5e-4 | 3.1e-4 | 0.573 | **~0** | 2e-5 |
| medoid incl | Background | 0.750 | 0.00247 | 7.3e-4 | 6e-4 | 3.8e-4 | 0.612 | **~0** | 2e-5 |
| medoid incl | Noise | 0.625 | 0.00419 | 4.6e-3 | 2.5e-3 | 1.35e-3 | 0.613 | **~1e-6** | 7e-5 |
| medoid incl | Light | **0.938** | 0.00011 | 6.1e-4 | 6e-4 | 3.1e-4 | 0.553 | **~0** | 2e-5 |
| medoid excl | Camera | 0 | 0.00683 | 5.0e-4 | 6e-4 | 4.0e-4 | 0.609 | ~0 | 2e-5 |
| medoid excl | Background | 0 | 0.01232 | 8.7e-4 | 9e-4 | 4.5e-4 | 0.543 | ~0 | 2e-5 |
| medoid excl | Noise | 0 | 0.02097 | 3.7e-3 | 2.0e-3 | 1.16e-3 | 0.631 | ~1e-6 | 6e-5 |
| medoid excl | Light | 0 | 0.00795 | 6.5e-4 | 6e-4 | 3.3e-4 | 0.557 | ~0 | 2e-5 |
| anchored mean | Camera | 0 | 0.00098 | 4.4e-4 | 5e-4 | 3.0e-4 | 0.588 | ~0 | 1e-5 |
| anchored mean | Background | 0 | 0.00106 | 7.4e-4 | 6e-4 | 3.6e-4 | 0.596 | ~0 | 2e-5 |
| anchored mean | Noise | 0 | 0.00295 | 4.4e-3 | 2.4e-3 | 1.30e-3 | 0.580 | ~1e-6 | 6e-5 |
| anchored mean | Light | 0 | 0.00119 | 5.8e-4 | 5e-4 | 2.8e-4 | 0.561 | ~0 | 1e-5 |

（identity 频率：medoid 选中 identity 的比例；excl/anchored 按构造为 0。）

## 2. 关键发现

1. **修复随机数后，medoid 仍然强烈偏向 identity**（Camera 0.81、Background 0.75、Light 0.94、Noise 0.63）；
2. Medoid excl-identity 把目标–identity 距离提高约 **5–7 倍**（0.001 → 0.007–0.021），
   但**初始蒸馏 loss、梯度范数、投影后范数几乎不变**（同一量级）；
3. 三种目标的**单步动作变化都 ≈ 1e-6（归一化动作空间）**，即实际动作几乎没变；
4. β 范数只有 2e-5–7e-5，远低于 0.15 边界——**不是投影把更新吃掉了，而是更新本身太小**。

## 3. 结论

- 之前 600 episodes 的负结果**不只是 RNG 缺陷或 identity 主导**：即便排除 identity，
  当前"η=0.05 + K-projection β + 每 replan 一次投影 SGD"的更新尺度也只能带来 ~1e-6 的动作变化；
- 这说明 **SG-CPA 的失败首先是更新幅度/参数化尺度问题**，其次才是目标信号问题；
- 在改变更新尺度（例如 η、每 replan 更新次数、β 参数化）之前，再换 medoid/mean 或再跑 rollout 都不会有信息量；
- 因此**不建议**直接进入新的 120-episode 实验；若要继续，应先做一个"更新尺度 vs 动作变化"的离线标定
  （η 扫描，纯离线，看一步/十步后动作变化是否能达到可感知幅度），再讨论新协议。

## 4. 与主线的关系

- 本次仍保留 4.2 的 600 episodes 为"当前实现"的负结果；
- 预筛说明：**负结果不能归因于"反事实共识一定无效"**，更准确的说法是
  "在当前更新尺度下，三种目标都没有产生可测的动作改变"；
- 任何后续方法改动都属于新协议，需要单独批准；本报告不构成 rollout 许可。
