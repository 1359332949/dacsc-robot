# IR-ASTTA / DRAIL 数据资产审计（Gate 0）

> 目的：在写任何 IR-ASTTA 代码之前，确认现有专家轨迹、CoRP rollout、域配对样本是否足以训练并验证一个 IRL 奖励模型。
> **本报告不训练模型、不跑 rollout；只盘点数据与样本构造条件。**

## 1. 方法需要什么数据

若采用 DRAIL 式奖励：

- 专家正样本：`(h_t, h_{t+1}, l)`，来自 LIBERO 专家轨迹；
- 策略负样本：`(h_t, h_{t+1}, l)`，来自 CoRP 在目标域的实际执行轨迹；
- 域平衡：专家样本与策略样本不能只靠“Clean vs Camera/Background”区分，否则判别器会学域分类而不是行为质量；
- 任务/阶段平衡：同一任务、相近执行阶段要有正负样本，才能把“行为质量”与“任务阶段/物体位置”分开；
- 规模：扩散判别器/奖励模型需要足够多的 transition；24 个 episode 的 replan 不能当独立样本。

## 2. 现有数据盘点

### 2.1 本地仓库

- **没有 LIBERO 专家 demonstration HDF5**（`find` 只找到 Python 包自带测试文件，与本项目无关）；
- 没有 CoRP 的完整轨迹数据（图像+状态+动作+下一状态）。

### 2.2 B1 capture v2

| 项 | 数量/内容 |
|---|---|
| episodes | **24**（12 tasks × 2） |
| replans | **667** |
| 域 | Camera、Background（目标域，静态扰动） |
| 字段 | 双相机图像路径、state、instruction、动作块、view actions、body/geom pose、success 标签 |
| 是否含专家正样本 | **否**（只有冻结 CoRP 的策略轨迹） |
| 是否含 Clean 配对 | **否**（没有同状态 clean/shift 对） |

用途：可以做离线诊断/小规模可行性，但不足以训练高容量 IRL 奖励模型，也不能声称跨任务泛化。

### 2.3 B2 formal

| 项 | 数量/内容 |
|---|---|
| JSONL | 165（156 candidate + 9 stage skip） |
| 字段 | 候选 τ、动作变化标量、β 范数、A 诊断、success/censor |
| 是否含 transition/动作块/图像 | **否** |

用途：只能做候选更新的聚合分析；不能构造奖励模型训练样本。

### 2.4 历史 rollout 日志（p42/p43）

- 每个 episode 一行 JSON：`success, steps, updates, beta 统计, salt_loss_last, ...`；
- **没有**图像、状态、动作、下一状态；
- 不能作为 IRL transition 数据。

### 2.5 服务器资产

- LIBERO 专家 demonstration、LIBERO-Plus 数据集、CoRP checkpoint 预期在 `/hy-tmp`；
- 当前 GPU 主机 `i-2.gpushare.com:20119` 连接被拒，**无法核对实际数量与格式**；
- 即使专家 demo 存在，它们通常是 **Clean** 专家轨迹；Camera/Background 扰动下的专家轨迹并不存在，需要重新渲染/仿真或做域增广。

## 3. 关键缺口

| 缺口 | 影响 | 是否可用现有数据补 |
|---|---|---|
| 无专家正样本（本地） | 无法训练 expert-vs-policy 判别器 | 否，需要 LIBERO demo / 服务器 |
| 无目标域策略完整 transition | 负样本不足 | B1 只有 24 eps，远不够 |
| 无域平衡的正/负样本 | 奖励可能只学 Camera/Background 风格 | 否，需要构造匹配增广/配对 |
| 无任务/阶段平衡 | 奖励可能学“离目标近”而非行为质量 | 否，需要分层采样 |
| 样本量小 | 扩散奖励过拟合，无法验证泛化 | 否，需要更多 episode/任务 |

## 4. 结论

**Gate 0 当前不通过（且部分被算力阻塞）。**

- 用 B1/B2 的 24 episodes 直接训练 DRAIL 奖励模型，统计上不可信；
- 必须补充：
  1. LIBERO 专家轨迹（Clean）；
  2. CoRP 在 Clean/Camera/Background 下的 policy rollouts（包含 transition）；
  3. 域匹配的专家/策略样本构造方案；
  4. 任务/阶段分层采样；
  5. 留出未见扰动参数做验证。

在这些完成之前，实现 IR-ASTTA 的 DRAIL 部分没有可靠训练基础。

## 5. Gate 1 修订：Terminal Outcome Predictability Audit（episode-level）

原“Reward Gate”设计有 credit assignment 和样本独立性问题，已按审查意见改为：

- **每个 episode 一个特征向量**（固定 replan 8 或 12），标签为最终 success；
- 不做 transition 级标签；
- Leave-one-task-out 逻辑回归；
- 任务簇 bootstrap 90% CI（探索性）；
- 比较三组：
  - A：域/任务套件/初始距离等元数据；
  - B：A + 当前状态、当前物体-目标距离、EEF-物体距离、抬升；
  - C：B + 执行动作统计、状态变化、距离变化。

### 结果（B1，24 episodes，13 success，12 task clusters）

| replan | A AUC | B AUC | C AUC | ΔAUC(B−A) | ΔAUC(C−B) |
|---|---|---|---|---|---|
| 8 | 0.455 | 0.469 | 0.406 | +0.014 | **-0.063** |
| 12 | 0.643 | 0.895 | 0.937 | +0.252 | **+0.042** |

（AUC 为 LOTO 预测的 pooled AUC；90% CI 很宽，例如 replan 12 的 C 为 [0.837, 1.0]。）

### 解释

1. **replan 8**：三组都接近随机，transition 特征没有增量；
2. **replan 12**：当前状态（B）已经能强烈预测最终 success（AUC 0.895），
   这主要是**任务阶段/当前几何进展**的信息，不是动作质量；
   transition 特征只增加约 0.04 AUC，且 CI 宽、12 个 task cluster，不能作为稳定证据；
3. 因此 Gate 1 的结论是：**没有可靠证据表明 transition 特征提供了超出当前状态的行为质量信号**；
   不能用它支持 DRAIL 训练。

### 仍未完成的 Gate 2

现有数据无法做“同一状态下不同候选动作 → 完整 episode success”的反事实比较：

- B2 只有 39 个 eligible branch、156 条候选记录，终点是局部中心距离；
- 成功/终止分支被 censor；
- 没有完整 episode 的配对反事实。

所以即使 Gate 1 有信号，也不能直接进入 Reward 指导的 A-active 更新。

## 6. 建议的降复杂度路径

在补大数据之前，先做一个**低成本 Reward Gate**：

1. 用 B1 的 24 episodes（667 transitions）训练一个**小型** transition 分类器（逻辑回归/小 MLP），标签用最终 success；
2. 按任务留出（leave-one-task-out），控制任务/阶段，检查能否区分成功与失败 transition；
3. 如果连这个都做不到，DRAIL 高容量模型也不会凭空产生任务信号；
4. 如果能做到，再考虑用 DRAIL 替代小模型，并补域平衡数据。

这样避免一上来就训练扩散奖励模型，也符合“不要复杂”的原则。

## 7. 与 IR-ASTTA 的关系

- IR-ASTTA 的潜在贡献不在“用 DRAIL”（已有工作），而在：
  > 把行为质量奖励通过 advantage-weighted flow matching 路由到 A-active 参数子空间。
- 但这个贡献只有在 Reward Gate 通过后才有意义；
- 当前数据审计的结论是：**先补数据/做小模型 Gate，不实现 DRAIL，不跑 rollout。**

## 8. 参考

- DiffAIL: arXiv:2312.06348
- DRAIL: arXiv:2412.06805
- Advantage Weighted Matching (AWM), ICML 2026（按用户提供信息）
