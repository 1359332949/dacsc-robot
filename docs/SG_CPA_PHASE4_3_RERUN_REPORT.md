# 工单 4.3 报告：修复随机数后的 SG-CPA 主比较复跑（360 episodes）

> 背景：4.2 的 600 episodes 是**旧实现**（视图噪声 seed 固定、动作 latent 每个 episode 复用）。
> 修复后（`b65c55b` + 本轮 diag 增强）必须重跑 Frozen / SALT / SG-CPA 主比较，才能评价"修复版 SG-CPA"。
> 本轮只修复随机数管理，**不改变 η（保持 0.05）、每 replan 一次更新、目标/视图/投影方式**；
> 不跑 η=20/100，不重跑消融组。seeds 30–34，always-on，diag-out 全开。
> 原始数据：`/hy-tmp/validation/p43/`；分析：`/hy-tmp/validation/p43_analysis/`；CSV：`docs/data/phase4_3_summary.csv`；
> 图：`docs/figures/p43_sr_by_domain.png`、`docs/figures/p43_delta_vs_salt.png`。

## 1. 执行

| 批次 | 时间 | 用时 |
|---|---|---|
| Frozen 120 | 10:19–10:41 | 22.5 min |
| SALT 120 | 10:41–11:07 | 25.5 min |
| SG-CPA(fixed) 120 | 11:07–11:41 | 34 min |

共 **360 episodes**，8 路并行；SG-CPA 每个 replan 一次 `SGCpa.step()`（η=0.05），记录每次 replan 的
action seed / view seed / medoid index / loss / grad / β / A guard / backtrack / 耗时。

## 2. 结果

| 域 | Frozen | SALT | **SG-CPA(fixed)** | Δ(SG−SALT) | McNemar p | 90% CI |
|---|---|---|---|---|---|---|
| Camera | 60.0% | 60.0% | **63.3%** | +3.3 pp | 1.000 | [−10.0, +16.7] |
| Background | 53.3% | **66.7%** | 50.0% | **−16.7 pp** | 0.125 | **[−30.0, −3.3]** |
| Noise | 90.0% | 86.7% | **93.3%** | +6.7 pp | 0.625 | [−6.7, +20.0] |
| Light | 93.3% | 86.7% | 90.0% | +3.3 pp | 1.000 | [−13.3, +23.3] |
| **Overall** | 74.2% | **75.0%** | **74.2%** | **−0.8 pp** | 1.000 | **[−8.3, +5.8]** |

- SG-CPA 与 Frozen 总体完全相同（74.2%），与 SALT 差 −0.8 pp（不显著）；
- Background 上 SALT 比 Frozen 高 13.3 pp（CI [3.3,23.3]），SG-CPA 没有复制这个收益，反而比 SALT 低 16.7 pp（负向趋势，p=0.125）；
- Camera/Noise/Light 上 SG-CPA 方向性略高，但全部不显著。

## 3. 修复版随机数是否真的生效？

对 120 个 SG-CPA episode 的 diag 检查：

| 检查 | 结果 |
|---|---|
| action seed 每个 replan 增加 7919 | **120/120 episodes 成立** |
| 同一 replan 内 teacher/student 共享 latent | 由实现保证（同一 seed 传入 build_target/loss） |
| 视图噪声每 replan 重画 | 由 `view_seed=action_seed+1` 与单测覆盖 |

→ 随机数修复在真实 rollout 中确实生效，不再是旧版"整 episode 固定噪声"。

## 4. 实际机制诊断（修复版 rollout，3197 次更新）

| 指标 | 值 |
|---|---|
| updates / episode | 26.6 |
| **medoid = identity 的比例** | **81.4%**（Camera 77.0%、Background 82.7%、Noise 83.5%、Light 83.3%） |
| medoid index 分布 | identity 2601、photo 413、blur 176、noise 7（共 3197） |
| 投影保留率 R_retain | 0.628 |
| 更新接受率 | 95.8% |
| 触发 backtrack 的更新比例 | **4.2%**（最终仍接受；rollback 0） |
| β 最大绝对值（episode 均值 / 全局） | 0.0055 / 0.0375（边界 0.15） |
| 单次 replan 适配耗时 | 2.14 s |

→ 修复后 medoid 仍然主要在选 identity；β 仍只用到边界的 ~3.7%；动作改变量极小。
这与 η 标定一致：η=0.05、每 replan 一次更新时，单步动作变化约 **4e-6**，几乎不可能影响成功率。

## 5. 旧版 vs 修复版对比

| | 旧 4.2（600 eps，含额外组） | 修复版 4.3（360 eps） |
|---|---|---|
| Frozen | 74.2% | 74.2% |
| SALT | 75.8% | 75.0% |
| SG-CPA | 74.2% | 74.2% |
| SG−SALT | −1.7 pp | −0.8 pp |

- SALT 自身在相同协议/seed 下两次运行差 **0.8 pp**（75.8→75.0）→ 本轮 SG−SALT 的 −0.8 pp 处于运行噪声量级；
- SG-CPA 在修复版下总体仍为 74.2%，没有因随机数修复而变好；
- 分域上修复改变了结果（Camera 56.7→63.3、Background 56.7→50、Noise 90→93.3、Light 93.3→90），但整体结论不变。

## 6. 结论

1. **修复随机数后，SG-CPA 仍未超过 Frozen/SALT**（overall 74.2% vs 75.0%/74.2%）；
2. 因此"4.2 的负结果是否只是旧代码 bug"这一问题已得到回答：**不是**；
3. 负结果的直接机制是：medoid 81.4% 选 identity + η=0.05 单步更新 → 动作变化 ~1e-6 量级；
4. 按预注册规则（无主域显著改善）应**暂停当前 SG-CPA**；不调 η、不换 medoid、不扩样本；
5. 消融组（no-projection / medoid-excl-identity）**没有**在修复版下重跑，旧消融数字只代表旧代码，不能作为修复版证据。

## 7. 局限

- 每域 3 任务 × 5 episodes = 30，任务级 cluster bootstrap 90% CI，统计功效有限；
- Background 的 McNemar p=0.125 与 CI 不含 0 不一致，只能称"负向趋势"；
- 未做多重比较校正；
- 评测扰动从 episode 开始即静态存在（always-on 协议）；
- 未测 η>0.05 的修复版任务成功率（η 标定只做离线）。

## 8. 交付

- 原始数据归档：`docs/data/p43_results.tar.gz`（360 episodes JSONL）
- 汇总 CSV：`docs/data/phase4_3_summary.csv`
- 图：`docs/figures/p43_sr_by_domain.png`、`docs/figures/p43_delta_vs_salt.png`
- 代码：本轮 diag 增强（action/view seed、medoid index、guard 统计）+ 修复版实现
- commit 见本次提交。
