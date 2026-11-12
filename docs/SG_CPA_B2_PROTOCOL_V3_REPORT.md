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
| 主要终点 | **仅局部 proximity**：`τ_prox = Δp_prox,update − Δp_prox,control`（H=24，固定 episode `d_ref`）；δ=0.01 操作性阈值 |
| 分类 | `τ>δ` / `|τ|≤δ` / `τ<−δ`（仅辅助解释，主要结果看连续分布） |
| 次要终点（本轮范围） | 每段 normalized action change、success 四类结果、production guard 诊断；**不采集** controller-space、实际 On/In 几何、多视图/历史信号 |
| 统计 | task-cluster bootstrap 90% CI（**estimand：按 eligible branch 记录加权**；若需 task 等权，应先算 task 内均值再 bootstrap）；任务级分布、分域、scipy Spearman |
| 更新族 A（机制） | **仅 η=100 单剂量**：raw `proj/full/sens`；**整向量统一缩放**到 β 边界（α=min(1,0.15/‖Δβ‖∞)），不做逐坐标截断；记录 proposal/applied 范数、α、`‖J_A·Δβ_applied‖` |
| 更新族 B（生产） | 通过 `SGCpa.step()` 的候选（β 边界/A guard/backtrack），单独报告；记录 `raw_d_beta_norm`/`d_beta_norm`、step_scale、accepted、backtracks |
| 停止条件 | 若仅 η=100 产生可测 τ，或有效更新严重偏离 raw/动作变化超安全阈值，则只记为"大干预有作用"，不外推 TTA |

δ 依据：zero-update 预检轨迹精确一致（噪声 0），δ=0.01 为预注册的最小任务相关进度变化（初始 object-target 距离的 1%）。

## 3. 统计与代码修正（将在正式运行前落地）

- 默认路径改为 `/hy-tmp/validation/b2v3/`；
- Spearman 使用 `scipy.stats.spearmanr`，常数输入返回 NaN 并显式处理；
- 引入 δ 三分类，不再用 `τ>0`；
- 每候选报告 task-cluster bootstrap 90% CI、任务级效应与分域结果；
- 本轮**不采集也不分析**多视图/历史连续性信号；不声称 B2 验证了信号预测能力。

## 3.5 全流程 zero-update 配对审计（修正分母后）

此前把 branch 8 的 skip 与 branch 12 的 eligible 混在一起（22+7=29），分母不是 24；已用**逐分支独立审计**修正：

| 分支点 | 分母 | 可用（contact≤0.10 且未 success） | skip | 物体轨迹差 | EEF 轨迹差 | 步数一致 |
|---|---|---|---|---|---|---|
| 8 | 24 | **19**（ep0 10 / ep1 9） | 5 | 0.0 | 0.0 | ✅ |
| 12 | 24 | **20**（ep0 10 / ep1 10） | 4 | 0.0 | 0.0 | ✅ |

（修正：此前 audit 的 `mkenv` 对所有 episode 都用 `states[0]`；现已改为 `states[ep % len(states)]`，
branch 8 的 ep1 有 1 个 skip；branch 12 的 18→20。episode 1 的双环境 zero 配对同样为 0.0 轨迹差。）

skip 原因（接触距离 >0.10 / 分支前已 success 或 done）逐条记录在
`/hy-tmp/validation/b2_zero_branch_audit.json`；每个 episode 的 branch 8/12 独立判定，互不混淆。
zero 配对在有效分支上轨迹/步数逐位一致；部分 zero 分支在窗口内成功，正式 τ 将按规则 censor。

## 3.6 正式代码修正（针对 P0 审查）

已修复：

| 问题 | 修正 |
|---|---|
| `chunk_of()` 隐式清零 β | 改为 `sample_chunk(..., beta)` **必须显式传 β**；control 传零向量，update 传候选 |
| 每个候选后环境重置但未重放 | 每个 candidate **重建 env 对并重放冻结前缀到同一 branch**；branch 8/12 分别重放 |
| 终止/censor 不严格 | 记录 stepsA/B、doneA/B、succA/B；仅当 **stepsA==stepsB==24 且无 success/termination** 才计算 τ |
| τ 参考距离取错 | 记录 episode 初始 `d_ref`、分支时 `d_branch_control/update`、终点 `d_end_*`；`Δp=(d_branch−d_end)/d_ref`，各用自身环境 |
| 输出与分析 schema 不匹配 | 统一为**每 candidate×episode×branch 一条记录**，字段含 `tau_prox`/censor；`scripts/analyse_b2_v3.py` 直接读取 |
| 记录不足 | 增加 target_pair/body_ids、每段 action_changes、raw/effective β、clip_ratio、beta_max、step_scale、accepted/backtracks、local/cumulative A change |
| 分析 `τ>0` | 改为 **δ=0.01 三分类**；scipy Spearman（ties/常数）；task-cluster bootstrap 90% CI；分域/分套件 |

- dry-run 门禁：`--dry-run --task-id --branch --candidate`（强制单任务/单分支/单候选），
  不需要 `--authorize-formal`；正式 24-episode 仍必须 `--authorize-formal`。
- **尚未运行 dry-run 或正式实验**；正式脚本仍待一次受控 dry-run 联调。

## 3.7 raw 三候选受限 dry-run（task 705 / ep0 / branch 8，输出隔离）

| candidate | proposal/applied norm | α | beta_max | predicted ‖J_A·Δβ‖ | 动作变化（4 段） | 24 步 | τ_prox | censor |
|---|---|---|---|---|---|---|---|---|
| raw_proj_100 | 0.02049 / 0.02049 | 1.0 | 8.10e-3 | **1.02e-17** | 2.5e-4–4.0e-4 | ✅ | -0.01127 | 无 |
| raw_full_100 | 0.04264 / 0.04264 | 1.0 | 2.17e-2 | 2.24e-2 | 5.4e-4–1.6e-3 | ✅ | -0.01222 | 无 |
| raw_sens_100 | 0.03739 / 0.03739 | 1.0 | 1.73e-2 | 2.24e-2 | 5.9e-4–2.3e-3 | ✅ | -0.00227 | 无 |

- 三个 raw 路径均经模拟器执行验证；统一缩放未触发（α=1），无 action-cap censor；
- `raw_proj` 的预测 A 变化 ~1e-17，验证投影方向保持；full/sens 的 A 变化同量级（2.2e-2）；
- 动作变化在 2.5e-4–2.3e-3，明显大于 production η=0.05（~2e-7），但仍远低于 0.05 cap；
- 单 episode dry-run **不构成效应结论**；仅验证路径与记录。

## 4. 当前状态与授权请求

| 工作 | 状态 |
|---|---|
| 预检：双环境确定性、zero 4×6、目标映射、抓取阶段 | ✅ 完成（本报告 §1） |
| 正式 B2 协议 v3 | 已冻结（§2），代码待最后联调 |
| 正式 24 episodes 分支实验 | ⏳ **等待授权** |
| η 网格 | **本轮不做剂量扫描**（仅 η=100 单剂量） |
| rollout / 调参 / 移除 A-projection | ❌ 未授权 |

**范围冻结**：本轮 B2 只回答"在固定操作阶段、相同初始状态和 4×6 执行协议下，
η=100 的 raw 方向更新与 production guard 更新是否造成可测的**局部 proximity 变化**"；
不外推任务成功率、TTA 总体效益、剂量响应或信号预测能力。
**当前请求**：审查 P0 修正（raw 统一缩放、success 分解、episode 状态索引、范围收窄），
通过后再申请正式 24-episode B2 授权。
