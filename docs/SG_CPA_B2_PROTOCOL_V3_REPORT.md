# B2 协议 v3：预检审计与正式实验设计（正式实验待授权）

> 针对审查意见，先完成**环境恢复、执行协议、目标映射、抓取阶段、统计口径**的修正与预检；
> **未运行正式 B2 分支实验**，等待授权。

## 1. 关键预检结论

### 1.1 `env.sim` 状态恢复不可靠 → 放弃单环境 save/restore

| 检查 | 结果 |
|---|---|
| 恢复后 body 位置最大误差 | **1.2e-5–1.5e-5 m**（非逐位） |
| control vs zero-update（save/restore，4×6 步） | **轨迹最大差 0.043 m、dp 差 0.112** |

→ 仅保存 `env.sim.get_state()` 不足以恢复 wrapper/控制器/计数器状态；第一轮 v2 的配对方式被否决。

### 1.2 双环境重放是逐位确定的

| 检查 | 结果 |
|---|---|
| 两个独立 env、同 init/seed、相同动作序列（8 步） | **body 最大差 0.0** |
| 两个独立 env、4×6 步 zero-update 全流程（2 任务） | **轨迹最大差 0.0**（各 24 步） |

→ 正式 B2 使用**双环境重放**：相同 init/seed，冻结策略重放相同前缀；分支后两环境独立观测、独立重规划，
每个 replan 使用匹配的 action latent seed。

### 1.3 目标映射逐任务验证

从 12 个任务的 BDDL goal 解析 On/In literal，与硬编码映射比对：**0 mismatch**。
正式记录将写入 `target_pairs`、body 名称、literal，不匹配则中止任务。

### 1.4 抓取阶段审计（replan 8/12，24 episodes）

| 机械判据 | 满足数 / 48 |
|---|---|
| 夹爪-物体距离 ≤ 0.10 | **39/48（81%）** |
| 物体抬升 ≥ 0.02 m | **11/48（23%）** |

→ 预注册分支条件采用 **`eef_obj_dist ≤ 0.10`（"接近/接触"阶段）**；
不满足或已 success 的分支记录 `skip_reason`；预期覆盖率约 81%。
（更严格的"已抓取"条件覆盖率仅 23%，留作敏感性分析，不用于主分析。）

## 2. 冻结的正式 B2 协议（v3）

| 项 | 值 |
|---|---|
| 分支点 | replan 8 / 12；机械判据 `eef_obj_dist ≤ 0.10` 且 not success；否则 skip 并记录 |
| 执行协议 | 每 replan 执行 **6 步**；窗口 **4 段 × 6 = 24 环境步** |
| control | β=0 整个窗口 |
| update | 分支点写入候选 β，**4 段内保持**；每段重新观测、重新采样 |
| latent | 每个 replan 两分支使用同一 action seed；按各自新观测独立采样 |
| 分支实现 | 两个独立 env 从同一 init 重放相同前缀（双环境确定性已预检） |
| 主要终点 | `τ = Δp_update − Δp_control`（proximity，H=24）；δ=**0.01** 预注册 |
| 分类 | `τ>δ` 有益 / `|τ|≤δ` 不可辨识 / `τ<−δ` 有害 |
| 次要终点 | success、物体位移、抓取/释放、动作变化（normalized + controller）、`‖Δz_A‖` |
| 统计 | task-cluster bootstrap 90% CI、任务级分布、分域、scipy Spearman（处理 ties）；每候选额外报告分类比例 |
| 更新族 A（机制） | raw 方向 `proj/full/sens` × η∈{1,10,100}；记录 raw/effective Δβ、坐标截断比例、global shrink |
| 更新族 B（生产） | 通过 `SGCpa.step()` 的候选（β 边界/A guard/backtrack），单独报告，不与 A 混合 |
| 停止条件 | 若仅 η=100 产生可测 τ，或有效更新严重偏离 raw/动作变化超安全阈值，则只记为"大干预有作用"，不外推 TTA |

δ 依据：zero-update 预检轨迹精确一致（噪声 0），δ=0.01 为预注册的最小任务相关进度变化（初始 object-target 距离的 1%）。

## 3. 统计与代码修正（将在正式运行前落地）

- 默认路径改为 `/hy-tmp/validation/b2v3/`；
- Spearman 使用 `scipy.stats.spearmanr`，常数输入返回 NaN 并显式处理；
- 引入 δ 三分类，不再用 `τ>0`；
- 每候选报告 task-cluster bootstrap 90% CI、任务级效应与分域结果；
- 信号（多视图/历史连续性）对 `τ>δ` 的预测仅作探索性，不称验证器性能。

## 3.5 全流程 zero-update 配对审计（24 episodes，branch 8 / 12）

| 分支点 | 可用分支（满足 eef≤0.10 且未 success） | skip | obs 最大差 | action chunk 最大差 | 物体轨迹最大差 | EEF 轨迹最大差 | 步数一致 |
|---|---|---|---|---|---|---|---|
| 8 | **22** | 7 | 0.0 | 0.0 | 0.0 | 0.0 | ✅ |
| 12 | **18** | 6 | 0.0 | 0.0 | 0.0 | 0.0 | ✅ |

结论：在正式 24 episodes 上，双环境重放 + 4×6 步协议在 branch 8/12 的 zero-update 配对达到**逐位一致**
（观测、动作块、物体/EEF 轨迹、步数），配对协议通过预检。
（skip 均记录原因：接触距离不足或已成功；覆盖率约 76%/75%。）

## 3.6 正式代码与分析落地

- `scripts/sg_b2_v3.py`：正式分支脚本已实现（双环境、4×6、raw/prod 候选、终止/截断/动作上限、
  raw/effective β、clip/shrink、δ 分类），带 `--authorize-formal` 门禁；**尚未运行**。
- `scripts/analyse_b2.py`：默认路径改为 b2v3；δ=0.01 三分类；scipy Spearman（ties/常数处理）；
  task-cluster bootstrap 90% CI；分域/任务级结果。
- 已知：正式脚本尚未做一次 dry-run 联调；正式运行前应先用 1 个 episode/1 候选做端到端 dry-run。

## 4. 当前状态与授权请求

| 工作 | 状态 |
|---|---|
| 预检：双环境确定性、zero 4×6、目标映射、抓取阶段 | ✅ 完成（本报告 §1） |
| 正式 B2 协议 v3 | 已冻结（§2），代码待最后联调 |
| 正式 24 episodes 分支实验 | ⏳ **等待授权** |
| η 网格 | 条件支持（机制剂量扫描；不与生产约束结论混合） |
| rollout / 调参 / 移除 A-projection | ❌ 未授权 |

**当前请求**：先批准对 `scripts/sg_b2_v3.py` 做 1-episode dry-run 与代码审查（不产生正式结论）；
dry-run 通过并冻结脚本后，再单独请求正式 24-episode B2 授权。
（本轮只完成代码实现与全覆盖预检，未运行正式分支实验。）
