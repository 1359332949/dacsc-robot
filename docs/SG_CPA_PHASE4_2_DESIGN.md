# 工单 4.2 实验设计（批准设计，暂不批准 rollout）

> 前置：4.1 三个 P0 修复已完成（β 边界、累计 A 参考、RNG 隔离、SVD 零空间），回归测试 14/14，
> 全量测试 46 passed。**本文件只做设计；rollout 需单独批准。**

## 1. 要回答的问题

1. SG-CPA 是否优于 Frozen 和 SALT（视觉扰动下成功率）？
2. 提升（若有）来自**反事实共识自监督**还是 **A 零空间保护**？（关键消融）
3. SG-CPA 是否按设计工作（保留率、累计 A 漂移、回退率、β 饱和）？
4. 在 Clean/Light 等正常视觉条件下是否伤害原有行为？

## 2. 方法与消融

| 组 | 说明 |
|---|---|
| Frozen CoRP | 原始策略，基线 |
| SALT | 已有 TTA 基线（always-on、5 episodes 同协议） |
| **SG-CPA** | 完整方法（4.1 冻结配置） |
| **SG-CPA w/o A-projection** | 同自监督目标/增强/参数化，仅取消零空间投影（消融，子集） |

## 3. 实验域与任务

| 类型 | 域 | 说明 |
|---|---|---|
| 主验证 | **Camera** | 跨域视觉扰动（静态） |
| 主验证 | **Background** | 外观扰动 |
| 参考 | **Noise** | 独立视觉扰动参考 |
| 保真对照 | **Light / Clean** | 正常视觉条件，检查不伤害 |

- 任务集：沿用 `configs/phase3_final_tasks.json`（pilot 独立选择，已被冻结）；每 cell 3 任务 × 2 套件；
- **Noise|object** 的 pilot 全是 4/4（无提升空间）→ 不进入主结论，只作"不伤害"参考；
- 所有方法使用**相同任务、相同初始状态、相同 seed** 严格配对；
- **旧 Phase 3 数据只作历史参考**：Frozen/SALT 全部用新 seeds 重新跑，不与旧数据混合。

## 4. Seeds 与规模

- 正式 seeds：**30–34**（5 episodes/任务；Phase 3 用 7–11，pilot 202–205，smoke 303，均不重叠）；
- 初始状态索引 = episode 0..4（三方法一致）；
- 主实验：3 方法 × 8 cells × 3 tasks × 5 eps = **360 episodes**；
- 消融：4 cells（Camera/Background × spatial/object）× 3 tasks × 5 eps × 1 组 = **60 episodes**；
- 总计 **420 episodes**（不含 4.1 离线）。

## 5. 冻结参数（不调）

K=4（identity+3 noise σ=0.05）；anchored mean α=0.1；直接动作蒸馏；K-projection SVD β；
projected SGD η=0.05；A=big2（768×5）；always-on；η_param=0.15 边界；有限步 tol=5e-3；
SALT 基线参数与 Phase 3 完全一致（η=0.15, K=5, lr=0.06, draws=8, exec_len=6）。

## 6. 指标

**主指标**
- 每域成功率；配对成功率差 Δ = SR(SG-CPA) − SR(SALT)、SR(SG-CPA) − SR(Frozen)、SR(SALT)−SR(Frozen)；
- McNemar（精确）；任务级 cluster bootstrap 90% CI。

**机制指标（每次 update / episode）**
- `R_retain = ‖P_null g‖ / ‖g‖`（投影保留率）；
- `D_A_local = ‖Aᵀ(z_{t+1}−z_t)‖`，`D_A_cum = ‖Aᵀ(z_t−z_ref)‖`（z_ref = episode 初始 β=0 写回）；
- β 饱和比例（|β|≥0.95η、0.99η）；有限步回退率/平均 backtracks；
- 实际动作变化（normalized 与 controller 单位）；
- 单次 replan 耗时、每 episode 更新次数。

**保真对照**
- Light / Clean 上 SG-CPA 相对 Frozen 的非劣效（配对，操作阈值 ±5pp，仅描述）。

## 7. 统计与判定（预注册）

| 结果 | 结论 |
|---|---|
| ≥2 个主域 Δ(SG-CPA−SALT) > 0 且方向一致 | 值得扩大样本 |
| 仅 Camera 有效 | 限定 Camera-specific |
| 全部域无提升 | 停止 SG-CPA，不再调参 |
| 消融：完整 vs w/o A-projection 差异 <2pp | 不能把收益归因于 A 零空间（诚实报告） |
| Light/Clean 退化 >5pp | 记录并优先修复行为保真 |

## 7.5 Smoke 结果（seed 404，task 705，1 episode，非正式）

| 方法 | 结果 | 步数 | updates | 耗时 | 备注 |
|---|---|---|---|---|---|
| Frozen | 成功 | 122 | 0 | **18 s** | 基线 |
| SALT | 成功 | 120 | 95 | **24 s** | β 全饱和（已知 SALT 现象） |
| SG-CPA | 失败 | 220（max） | 36 | **40 s** | accepted 100%、backtrack 0%、R_retain 0.66、β_max 0.014、D_A_cum 7.6e-4 |

- 链路与记录字段验证通过（R_retain / D_A_local / D_A_cum / backtracks / β_max / replan 耗时均写入）；
- SG-CPA 单 episode 失败**不作结论**，但它跑满 220 步、β 更新量很小，正式实验需重点观察；
- 修复了 smoke 中发现的 **episode 级 z_ref 混入观测变化**问题：改为"当前观测在 β=0 的写回"作为参考。

## 8. 成本估算

- Smoke 实测单 episode：Frozen 18 s / SALT 24 s / SG-CPA 40 s；
- 主实验 360 eps（每方法 120 eps，8 并行按 cell 分批）估算：串行约 165 min，8 并行 + GPU 竞争约 **1.5–2.5 h**；
- 消融 60 eps：约 **20–40 min**；
- **总计约 2–3 h**（保守预算 3–4 h）；全程记录单步/单 episode 耗时，若单 episode >10 min 则暂停检查。

## 9. 停止条件

1. 任一主域 SG-CPA 比 Frozen 低 >10pp 且连续 ≥10 episodes → 暂停该域并检查诊断；
2. 回退率 >50% 或 β 饱和 >90% → 暂停，说明机制未按设计工作；
3. 实际单步耗时 >预算 2× → 暂停优化成本；
4. 中途禁止改 λ/η/K/A/seeds；如需改，必须新开协议并重新审批。

## 10. 交付

- 冻结配置 JSON（任务、seeds、方法、参数）；
- 每 cell JSONL（success/updates/R_retain/D_A/backtracks/β 饱和/耗时）；
- 统计脚本与结果表；
- 报告（含消融结论与失败案例）；
- commit SHA + 部署文件 sha256。

## 11. rollout 前置条件（全部满足才可申请）

1. 4.1 三个 P0 修复完成 + 回归测试通过（**已完成**）；
2. 本设计获批准（当前状态：设计已提交，待确认）；
3. 三方法 smoke 各 1 episode（非正式 seed）验证链路与记录字段；
4. 单独获得正式 rollout 批准。

_设计稿；未启动任何 rollout。_
