# B2 离线收尾 + B1/B2 联合分析

> 无新增仿真 rollout；只使用已有 B2 正式 JSONL 与 B1 capture 数据。
> 目的：修正记录口径、做 raw 候选 paired contrasts、检查 censor 一致性、验证 B1/B2 分支观测对齐，
> 并探索“更新前信号是否预测更新效用”。结论用于决定是否启动 B3。

## 1. 记录口径修正

B2 正式 JSONL 共 **165 条**，不是 165 条候选记录：

| 类型 | 数量 |
|---|---|
| 候选记录（4 candidates × 39 eligible branches） | **156** |
| branch-level stage skip | **9** |
| JSONL 总计 | **165** |

| candidate | complete | censored | total |
|---|---|---|---|
| prod_step_005 | 33 | 6 | 39 |
| raw_proj_100 | 33 | 6 | 39 |
| raw_full_100 | 32 | 7 | 39 |
| raw_sens_100 | 32 | 7 | 39 |
| 合计 | 130 | 26 | 156 |

Success 数字 `update-only 3 / control-only 0 / both 20 / neither 130 / not_observed 3`
是 **156 条候选级 paired observations 的汇总**，不是 24 个 episode 数；同一 branch 被 4 个候选分别干预，
不能视为 4 个独立实验单位。报告中不再称“全局成功率”。

## 2. Raw / production paired contrasts

在同一 `(suite, task_id, episode, branch)` 上计算候选间 τ 差（只保留两组都有完整 τ 的配对），
task-cluster bootstrap 90% CI：

| contrast | n pairs | mean Δ | median Δ | 90% CI |
|---|---|---|---|---|
| raw_proj − raw_full | 32 | -2.33e-3 | -1.78e-4 | [-1.04e-2, +3.90e-3] |
| raw_proj − raw_sens | 32 | -1.18e-3 | -1.30e-5 | [-6.87e-3, +4.88e-3] |
| raw_full − raw_sens | 32 | +1.15e-3 | 0.0 | [-2.43e-3, +5.68e-3] |
| raw_sens − prod | 32 | -2.71e-4 | 0.0 | [-1.20e-2, +8.40e-3] |
| raw_proj − prod | 33 | -9.86e-4 | -2.06e-4 | [-6.50e-3, +2.80e-3] |
| raw_full − prod | 32 | +8.76e-4 | -1.17e-4 | [-8.93e-3, +1.11e-2] |

**所有 paired contrast 的 CI 都跨 0**：没有证据表明 A-nullspace、full、A-sensitive 或 production
之间在局部 proximity 上有系统差异。marginal CI 重叠不能替代此 paired 检验，但 paired 检验同样为零。

## 3. censor 一致性

对 26 条 censor 记录检查 `done_*` 与 `success_*`、两侧执行步数：

- **0 条冲突**：不存在“done=true 但 success=false 被算成单边 success”或步数不等却被分类为 success 差异；
- censor 原因全部为 `termination_or_success`；无 action-cap censor。

## 4. B1/B2 分支观测对齐

按 `(suite, task_id, episode, replan=8/12)` 对齐；B1 和 B2 使用相同 seed 公式、相同 6 步执行节奏：

| 检查 | 结果 |
|---|---|
| 对齐 branch 数 | **33** |
| B1 vs B2 EEF-object 距离最大差 | **0.0** |
| B1 vs B2 object-target 分支距离最大差 | **0.0** |

→ 对齐可信：B2 的 branch 观测与 B1 capture 是同一状态。

## 5. 更新前信号 → 更新效用（探索性）

B1 信号：多视图一致性 `mv`（=−平均 view pairwise distance）、历史连续性 `hist`（=−leftover distance）。
B2 结果：`tau_sens`、`tau_proj`、`tau_full`、`tau_sens−tau_proj`。leave-one-task-out 线性预测；
baseline 为 leave-one-task-out 任务均值。

| outcome | n | baseline LOO ρ | mv overall ρ | mv LOO ρ | mv perm p | hist LOO ρ | hist perm p |
|---|---|---|---|---|---|---|---|
| tau_sens | 32 | -0.58 | -0.20 | -0.32 | 0.48 | -0.48 | 0.53 |
| tau_proj | 33 | -0.37 | -0.08 | -0.31 | 0.08 | -0.12 | 0.71 |
| tau_full | 32 | -0.54 | -0.20 | -0.52 | 0.19 | -0.29 | 0.80 |
| tau_sens−tau_proj | 32 | -0.59 | +0.08 | -0.63 | 0.06 | -0.36 | 0.72 |

**结论（限定）：这两个自洽类信号没有表现出正向、可泛化的预测力；不能推断所有任务正确性信号都无效。**
- 所有 LOO 相关系数为负或接近 0；
- 所有 permutation p > 0.05（最小 0.06，方向为负）；
- baseline 任务均值本身也是负 LOO ρ，说明当前 33 个 branch / 12 个任务簇的样本量下，
  跨任务预测极不稳定，signal 没有超过 baseline 的证据。

## 5.1 必须保留的分析限制

1. **33 个对齐 branch 是 complete-case 样本**：只包含至少一个候选有完整 τ 的 branch，不是全部 39 个 eligible branch；
   四种候选都被 censor 的 6 个 branch 不在联合分析中。
2. **对齐证据不是完整状态逐位相同**：验证的是确定性重放 + EEF-object / object-target 两个标量距离（最大差 0.0），
   没有比较完整 body pose、proprioception、图像或 observation 张量。
3. **LOO 模型未控制阶段/域**：单信号 `y~x`，没有 branch 8/12、suite/domain 协变量；
   也没有对 `signal − baseline` 的增量做正式 bootstrap/置换检验；baseline 与 signal 的 ρ 只是描述性比较。
4. **连续 τ 预测不含 success/censor 效用**：只回答“信号能否预测未提前成功/终止的完整窗口连续 proximity 效应”，
   没有检验信号能否预测 success、termination 或更早成功。
5. **只测试了两个自洽类信号**（多视图一致性、历史连续性）；未测试未来视觉预测或 Oracle 成功动作；
   “这两个信号无正向泛化证据”**不等于**“所有任务正确性信号无效”。

## 6. 解释边界

1. τ 只对“双方完整 24 步且未 success/termination”的 branch 计算，是 **complete-case / survivor-conditioned** estimand；
   不能外推到全部 eligible branch，尤其不能忽略 update-only success。
2. raw 三组虽然都用 η=100，但实际 applied ‖β‖ 不同（proj 0.115、full 0.143、sens 0.133），
   因此方向与幅度的效应仍混在一起；本轮不做等范数补实验。
3. `action_change` 是 normalized action chunk 的平均差，不等于 controller-space 动作或物理安全性。
4. 上游 CoRP 是 12 步 full-chunk 评估、支持 BF16；本实验是 FP32、6 步重规划的局部反事实协议。
5. 只改 action generator 的 K-β 子空间，冻结 representation extractor；不能称之为完整 representation 保护。

## 7. 结论与下一步

- **B2 可以冻结**：当前 SG-CPA production 更新在该协议下没有可测的局部 proximity 收益；
  raw 方向也没有稳定方向差异；没有证据支持继续调大 η 或移除 A-projection。
- **B3 不启动**：没有“更新前信号能在未见任务上区分正/负效用”的证据；gate 会变成未经验证的包装。
- **不再扩展 B2/仿真**：下一步若继续，应做机制解释或重新定义更接近任务完成的终点，
  但需要新的、明确的研究问题与授权；不是继续增加实验矩阵。

## 8. 交付

- B2 正式数据：`/hy-tmp/validation/b2v3_formal/`
- 离线汇总：`/hy-tmp/validation/b2v3_formal/offline.json`
- B1/B2 对齐与信号：`/hy-tmp/validation/b1_b2_alignment.json`
- 代码：`scripts/analyse_b2_offline.py`、`scripts/align_b1_b2.py`
