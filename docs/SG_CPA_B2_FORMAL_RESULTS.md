# B2 正式实验报告（24 episodes）

> 授权范围：12 tasks × 2 episodes、branch 8/12、raw proj/full/sens η=100、production `SGCpa.step()` 单独报告。
> 主要终点：完整 24 步窗口的局部 `τ_prox`；success/censor 独立报告。
> 数据：`/hy-tmp/validation/b2v3_formal/b2_v3.jsonl`（**165 条 JSONL：156 条候选记录 + 9 条 stage skip**）。
> 运行：固定 CoRP checkpoint、FP32、每 replan 执行 6 步、双环境重放、共同 action latent。

## 1. 数据完整性

| 项 | 值 |
|---|---|
| episodes | 24（12 tasks × 2） |
| JSONL 总记录 | 165（候选 156 + stage skip 9） |
| stage skip（contact>0.10 或分支前已 success） | 9 |
| 每候选完整窗口 | prod 33 / raw_full 32 / raw_proj 33 / raw_sens 32 |
| 执行期 censor | 6–7 / candidate，全部为 `termination_or_success`；无 action-cap censor |
| success 四类（156 条候选级 paired observations） | update-only **3**、control-only **0**、both 20、neither 130、not_observed 3 |

## 2. 主要结果（局部 proximity，24 步窗口）

| candidate | n complete | mean τ | median τ | task-cluster 90% CI | 分类（有益/不可辨识/有害） | mean action change | mean applied ‖β‖ |
|---|---|---|---|---|---|---|---|
| prod_step_005 | 33 | -3.5e-4 | 0.0 | [-8.0e-4, +5.7e-5] | 0 / 33 / 0 | 3.0e-5 | 1.26e-3 |
| raw_proj_100 | 33 | -1.34e-3 | -1.33e-4 | [-6.87e-3, +2.31e-3] | 5 / 24 / 4 | 2.09e-3 | 0.1148 |
| raw_full_100 | 32 | +7.12e-4 | 0.0 | [-9.20e-3, +1.08e-2] | 3 / 25 / 4 | 3.51e-3 | 0.1432 |
| raw_sens_100 | 32 | -4.35e-4 | 0.0 | [-1.23e-2, +8.10e-3] | 5 / 25 / 2 | 4.15e-3 | 0.1328 |

- 所有候选的 τ 均值都在 1e-3 量级或更小，**task-cluster 90% CI 全部跨 0**；
- 分类以 `unidentifiable` 为主（24–33 / 32–33），只有少量受益/有害计数，且**任务级样本 n=2–4**；
- raw 候选的动作变化（2–4e-3）比 production（3e-5）大约两个量级，但连续 τ 仍无稳定方向；
- production 的 τ CI 完全贴近 0（[-8e-4, +6e-5]），在本窗口内没有可测的局部效应。

## 3. Success（候选级，独立报告）

| 结果 | 全局 | prod | raw_proj | raw_full | raw_sens |
|---|---|---|---|---|---|
| update-only | 3 | 0 | 0 | 1 | 2 |
| control-only | 0 | 0 | 0 | 0 | 0 |
| both success | 20 | 6 | 6 | 5 | 3 |
| neither（完整窗口） | 130 | 33 | 33 | 32 | 32 |
| not_observed | 3 | 0 | 0 | 1 | 2 |

`update-only 3 vs control-only 0` 是 156 条候选级观察的描述性计数，不是 3 个独立 episode 获益；，样本极小、无统计检验支持；不能据此声称更新提升成功率。

## 4. 解释与边界

1. **B2 没有发现可重复的局部 proximity 改善**：raw 方向干预更大，但 τ 仍以不可辨识为主；production 几乎为 0。
2. **raw_proj 的 A-nullspace 更新**没有表现出与 full/sens 不同的稳定方向效应；三者的 CI 大面积重叠。
3. **任务异质性明显**：个别 task 的 τ 均值达 ±0.08（n=2），但这是小样本噪声，不能作为效应证据。
4. **不能外推**：τ 是 24 步 object→target 距离变化，不是任务完成、不是成功率、不是上游 12 步 full-chunk 评估结果；
   本实验固定 checkpoint、FP32、6 步重规划，且只更新 action generator 的 β 子空间。
5. **成功/censor**：完整窗口 130 条 neither；3 条 update-only 与 0 条 control-only 不足以支持任何成功率结论。
6. **未采集**：多视图/历史信号、controller-space 动作变化、实际 On/In 几何关系；B2 不回答信号预测问题。

## 4.1 离线收尾补充

- paired contrasts（raw 之间、raw vs production）全部跨 0：无系统方向差异；
- censor 一致性检查：0 条 `done/success` 冲突；
- B1/B2 branch 对齐：33 个 branch，contact/d_branch 最大差均为 0.0；
- 更新前信号对 τ 的 LOO 预测无正向证据（详见 `SG_CPA_B2_OFFLINE_JOINT_REPORT.md`）。

## 5. 结论

> 在固定 CoRP、FP32、6 步重规划、24 步局部窗口下，**没有观测到任何候选更新方向带来可重复的 proximity 改善**；
> production guard 的更新幅度很小且 τ 贴近 0。raw 方向（包括 A-nullspace、full、A-sensitive）虽然动作变化更大，
> 但局部任务进展效应不可辨识、且任务间方向不一致。

**这不是“更新无用”的证明，也不是“A 无意义”的证明**；它说明当前局部测量窗口/尺度下，
参数更新的行为后果没有被稳定地转化为 proximity 进展。下一步不应调大 η 或继续加实验框架，
而应决定：是换更接近任务完成的终点/更长窗口重新测量，还是接受“局部 proximity 对这类更新不敏感”并转向机制解释。

## 6. 交付

- 数据：`/hy-tmp/validation/b2v3_formal/`；归档 `docs/data/b2_formal.tar.gz`
- 分析：`/hy-tmp/validation/b2v3_formal/analysis.json`
- 代码：`scripts/sg_b2_v3.py`、`scripts/analyse_b2_v3.py`
- 本报告；commit 见本次提交。
