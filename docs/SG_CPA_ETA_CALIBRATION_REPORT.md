# SG-CPA 离线 η 标定报告

> 触发：4.2 负结果 + 目标预筛显示"一步动作变化 ~1e-6"；审查指出预筛脚本符号写反且未走真实更新路径。
> 本工单：**修好测量后，只改变更新尺度（η、每 replan 更新次数），冻结其余所有配置**，纯离线，不跑机器人。
> 原始 JSON：`/hy-tmp/validation/sg_cpa_eta_cal/{aggregate.json,raw_rows.json}`；脚本 `scripts/sg_cpa_eta_calibration.py`。

## 1. 测量修正

| 问题 | 修复 |
|---|---|
| 预筛脚本 `d_beta` 少负号（实际测了梯度上升方向） | 删除手写更新；统一走真实 `SGCpa.step()`（含投影、β 边界、A guard、backtrack/rollback） |
| 预筛未应用 β 全局步长缩放 / A guard / backtrack / rollback | 同上：测量直接来自 `step()` 返回的 accepted 更新与诊断 |
| controller 单位动作变化算错（对差值做 unnormalize，带 offset） | 改为 `unnormalize(a1) − unnormalize(a0)` |
| action latent 覆盖不足 | 新增 identity-only 测试：固定 view seed、只改 action seed，验证 target 改变 |

`SGCpa.step()` 新增可选 `target_info`（固定教师目标）与 `raw_d_beta_norm`（区分提议步长 vs 最终步长）。
测试：`tests/test_sg_cpa.py` **22 passed**（全量 54 passed）。

## 2. 标定协议（冻结）

- 观测：4 域 × 2 套件 × 1 任务 × 2 chunks = **16 个固定观测**；
- 每个观测：教师目标、4 个视图、action latent、view latent **全部固定**，从 β=0 连续更新 10 步；
- η ∈ {0.05, 0.5, 5, 20, 50, 100}，N_updates = 10（记录 step 1/3/5/10）；
- 其余冻结：Medoid 含 identity、Identity+Noise+Blur+Photo、直接蒸馏、K-projection SVD β、投影开、A guard 开（tol=5e-3）、β 边界 0.15、SGD。

## 3. 结果（step 10，16 观测平均）

| η | L10 | 累计 loss 降幅 (L0−L10) | raw ‖Δβ‖ | accepted ‖Δβ‖ | β 最大 | β 饱和比例 | **identity 动作变化（normalized）** | controller 单位 | 4 视图平均变化 | backtrack 率 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 9.975e-4 | 1e-7 | 2.1e-5 | 2.1e-5 | 1.0e-4 | 0 | **4.1e-6** | 3.4e-6 | 4.2e-6 | 0 |
| 0.5 | 9.962e-4 | 1.4e-6 | 2.1e-4 | 2.1e-4 | 1.0e-3 | 0 | 4.1e-5 | 3.4e-5 | 4.2e-5 | 0 |
| 5 | 9.837e-4 | 1.4e-5 | 2.0e-3 | 2.0e-3 | 9.7e-3 | 0 | 3.9e-4 | 3.2e-4 | 4.0e-4 | 0 |
| 20 | 9.508e-4 | 4.7e-5 | 6.7e-3 | 6.7e-3 | 3.5e-2 | 0 | 1.3e-3 | 1.1e-3 | 1.4e-3 | 0 |
| 50 | 9.241e-4 | 7.4e-5 | 1.25e-2 | 1.01e-2 | 6.3e-2 | 1.6e-4 | 2.3e-3 | 1.9e-3 | 2.3e-3 | 19% |
| 100 | 9.069e-4 | **9.1e-5（9.1%）** | 2.12e-2 | 1.36e-2 | **9.1e-2** | 3.3e-4 | **3.0e-3** | **2.4e-3** | 3.1e-3 | **25%** |

（A guard 接受率在所有 η 下均为 100%；η≥50 出现少量 backtrack，没有 rollback。）

## 4. 如何解读

1. **动作变化随 η 近似线性放大**：4e-6 → 3.0e-3（约 720×）；controller 单位 3.4e-6 → 2.4e-3；
2. **loss 确实下降**：η=100 时 10 步累计下降 9.1%（η=20 为 4.7%，η=5 为 1.4%）；
3. **β 没有饱和**：η=100 时 βmax=0.091（边界 0.15 的 61%），饱和比例 3e-4；
4. **A guard 没有大规模拒绝**：接受率 100%，η=100 时 25% 的步发生一次 backtrack，仍被接受；
5. **但动作变化绝对量仍小**：3e-3 normalized ≈ 归一化动作范围的 0.3%；controller 2.4e-3 vs 动作范围 0.9375 ≈ 0.26%。

## 5. 按批准标准的判定

| 标准 | 结果 |
|---|---|
| 存在稳定区间：动作变化明显 >1e-6、loss 下降、guard 接受 | **✅ 存在**（η≈50–100，10 步：动作变化 2–3e-3，loss −7–9%，接受率 100%，β 未饱和） |
| 只有极大 η 才有效，且 β 饱和 / 大量 backtrack | 不完全符合：η=20 已有 1.3e-3；η=100 时 β 仅 61% 边界、backtrack 25% |
| 所有 η 都无有意义变化 | 不符合（最大 3e-3） |

**因此：当前 K-projection 参数空间并非"完全无法改变动作"；它需要 η 提高约 3 个数量级才能产生可测变化。**

## 6. 关键限制：成本

- 标定实测单步（真实 step，含 J_A/写回/guard）约 **0.4–0.5 s**；
- 达到 3e-3 动作变化需要 **η=100 且 10 次更新/replan** → 单次 replan 适配约 **4–5 s**（不计观测/执行）；
- SALT 基线约 **0.5 s/replan** → SG-CPA 在此设置下约 **10×** 慢；
- 对真实闭环控制，这是不可接受的时延（LIBERO 每步约 0.1–0.2 s 量级）。
- 即使只看机制：9% 的 loss 下降换来 0.3% 动作变化，**优化效率很低**。

## 7. 结论与建议

1. **不要再跑 rollout**：在可用时延内（η≤5，≤1 次更新）动作变化仍只有 1e-4–1e-5，几乎不可能改变成功率；
2. **不允许把 η=100/10 次更新直接带进正式实验**：它超出实时预算，且未证明动作变化"正确"（只证明"变了"）；
3. 当前参数化/更新机制的组合，在"可部署时延"约束下没有找到可用更新区间——**这比"medoid 选 identity"更根本**；
4. 若仍要继续 SG-CPA，需要重新设计**参数化或更新方式**（不是继续调 η/medoid/视图），并先给出可接受的时延预算与离线证据；这属于新方向，需单独审批；
5. 建议把当前结论写成："在 K-projection + 投影 SGD + 实时预算下，反事实共识目标没有形成可用的控制更新"，而不是"反事实共识一定无效"。

## 8. 交付

- 代码修复：`dacsc/sg_cpa.py`（`target_info`、`raw_d_beta_norm`）、`scripts/sg_cpa_target_prescreen.py`（负号/真实路径/controller 度量）、`scripts/sg_cpa_eta_calibration.py`；
- 测试：22 passed（SG-CPA），全量 54 passed；
- 数据：`/hy-tmp/validation/sg_cpa_eta_cal/`（aggregate.json, raw_rows.json）；
- 本报告与 commit。
