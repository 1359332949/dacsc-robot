# 研究主线重校准 + B1 收尾报告

> 结论先行：**研究主线回归 "Action-Sensitivity-Aware TTA"**；B1 只作为辅助实验收尾，不再扩展；
> B2（候选更新效用）需单独授权；不做 rollout、不调 η、不移除 A-projection。

## 1. 重新冻结的研究主目标

**Source-informed Action-Sensitivity-Aware Test-Time Adaptation**

> 利用源域 VLA 的动作敏感子空间 A，刻画目标域在线适配中的动作变化风险；
> 利用独立的目标域反馈判断候选更新是否具有任务效用；
> 在保护已有行为补偿能力的同时，允许真正有益的动作相关更新。

核心科学问题：
> 当目标域适配需要改变动作敏感方向时，什么证据能判断这次更新是有益的，而不是破坏已有行为？

工单准入检查：

| 检查 | 判定 |
|---|---|
| 加深对 A 动作敏感性的理解 | 主线 |
| 理解 A 约束对更新/动作的影响 | 主线 |
| 判断候选更新是否有独立反馈支持 | 主线 |
| 仅让某个 loss 下降更多 | 辅助 |
| 仅提高 episode 进展预测指标 | 辅助 |
| 只修补数据/评估设施而不回到更新效用 | **偏航风险，拒绝** |

## 2. B1 收尾（辅助实验，已完成）

### 2.1 数据

24 episodes / 667 replans / 2620 images / 64 MB（`/hy-tmp/validation/b1_capture_v2/`），
12 个 task 固定列表，图像 manifest 含 sha256；字段含完整 view action blocks、body pose/quat、geom 几何。

### 2.2 进度指标验收

| 指标 | 结果 |
|---|---|
| `proximity_progress` | 可用作基线（成功 episode p_last 常 0.5–0.9） |
| `geometric_progress`（On/In 真实几何） | **未通过验收**：object In basket 的成功 episode 多为 0；物理上 basket 是浅容器，物体中心在 rim 上方，而 success 判定与此不一致；**不依据 success 去拟合公式**（会变成用结果选指标） |

几何模块已独立实现并用**合成几何**测试（5 passed）：On 的 contact×inside、In 的容器内壁盒、旋转容器、姿态变换；但真实容器几何与 simulator success 语义不一致，故 geometric 仅作诊断，不进入 B1 主结论。

### 2.3 B1 信号筛选（探索性）

- 样本：619 个 replan（有 H=12 的 `dp12_proximity`；history 595）；
- 结果变量：`dp12_proximity`（H=12；geometric 未通过验收）；
- 评价：LOO task、不含 task-id 的基线模型（suite/domain、时间、当前进度、动作幅度）、
  任务簇 bootstrap、任务内 permutation；
- 四信号中：
  - **多视图一致性**：LOO Spearman 0.295（基线）→ **0.338**（Δ+0.043，perm p=0.004）；
  - **历史动作连续性**：0.297 → **0.354**（Δ+0.056，perm p=0.002）；
  - **未来视觉预测**：延迟信号，需按 LOO 折训练预测器；本轮**未评估**（明确标记）；
  - **已验证成功动作**：LOO 下不可估计，**Oracle/排除**。

**B1 结论（限定）**：
> 多视图一致性与历史动作连续性在 24 episodes 上对短期 proximity 进展有**探索性增量预测力**；
> 但这不是任务成功率结论，也不能说明某次 A 敏感更新有因果收益。

## 3. 研究主线回归（B2，未授权）

后续核心实验对象改为**候选参数更新**，而非信号：

1. 这次更新改变了什么：`AᵀΔz`、`A⊥Δz`、动作块变化；
2. 是否有独立反馈证据（未来视觉/进展证据）；
3. 同初始状态下的更新 vs 不更新分支：`τ_t = Δp_{t,H}(θ+Δθ) − Δp_{t,H}(θ)`；
4. 哪种**更新前可获得**的信号能预测 `τ_t` 的符号。

**B2 需要模拟器成对反事实分支（匹配 RNG/latent/步数），必须单独授权。**
在获得授权前：不实现、不执行、不声称任何因果结论。

## 4. 明确不再做

- 不扩展 B1 的信号种类、指标种类或采集规模；
- 不调 η、不移除 A-projection、不重跑 360 episodes；
- 不把 B1 的预测力包装成"适配有效"；
- 不做物理机器人 rollout。

## 5. 交付

- 主线目标与准入表：本文件 §1
- B1 数据：`/hy-tmp/validation/b1_capture_v2/`（图像）；run 归档 `docs/data/b1_capture_v2_meta.tar.gz`
- B1 筛选结果：`/hy-tmp/validation/b1_screen/screen.json`
- 几何模块/测试：`dacsc/b1_progress_geometry.py`、`tests/test_b1_progress_geometry.py`
- commit 见本次提交。
