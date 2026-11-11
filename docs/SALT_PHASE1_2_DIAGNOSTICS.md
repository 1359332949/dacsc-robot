# 工单 1.2 诊断报告：SALT 更新的可解释性（只诊断，不改算法）

> 基于 `7c5452a`。只增加日志/诊断与离线分析；**未改 lr/η/gate threshold/token 数/loss/regularizer，
> 未跑大规模成功率实验，未实现 SG-SALT。**

## 0. 方法与原始数据

- 新增诊断：replan 级 gate score（含未触发样本）、每 optimizer step 的 loss/grad/Δβ/β 分位数与边界比例、
  固定噪声+固定 τ 的 probe loss（before/after，不参与更新）。
- 原始逐 episode JSON：
  - clean + always：`/hy-tmp/validation/diag_clean_always/*.json`（3 episodes）
  - clean + feature gate：`/hy-tmp/validation/diag_clean_feature/*.json`（3）
  - camera 648/650/641 + feature gate：`/hy-tmp/validation/diag_cam_feature/*.json`（3）
- 分析脚本：`scripts/analyse_salt_diag.py`；校准文件 `p11_calib/gate.json`（threshold=4.428）。

## A. Gate score 分布（feature-gate 运行）

| 组 | replan 分数数 | above threshold | min | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| clean | 46 | **0** | 0.000 | 0.306 | 2.228 | 2.488 | 3.613 | **4.389** |
| camera | 111 | **0** | 0.000 | **0.000** | 0.815 | 1.095 | 1.944 | 2.206 |

（threshold = 4.428）

**发现**：
1. camera 任务的 gate 分数**低于 clean**（median 0 vs 0.306）。LIBERO-Plus camera 扰动从 episode 开始就是静止的，
   相邻 replan 之间视觉 token 变化很小，所以**基于"相邻 replan 变化"的 SALT gate 在原理上检测不到这种静态扰动**；
2. clean 的分数最高 4.389，距离阈值 4.428 仅 ~0.9%，说明 cross-fit 阈值几乎等于 clean 的最大分数，**余量很小**；
3. 两组都没有超过阈值 → "gate 没触发"不是数值 bug，而是该 gate 与静态 camera 扰动不匹配。

**证据等级**：静态扰动不可被 change-gate 检测 —— **已确认（直接测量）**；
需要换成 episode 级 shift 检测或 mid-episode shift 协议 —— **较可能（推断）**。

## B. β 饱和来源（clean + always）

每个 episode 第一次适配发生在 replan 1（需要 leftover）。第一轮 5 个 optimizer step（lr=2η/K=0.06）实测：

| task | step | Δβ_l2 | grad_l2 | β_l2 | frac≥0.99η |
|---|---|---|---|---|---|
| 0 | 0 | **1.654** | 0.0228 | 1.654 | 0.000 |
| 0 | 2 | 0.865 | 0.0028 | 3.497 | 0.534 |
| 0 | 4 | 0.285 | 0.0017 | 3.626 | 0.706 |
| 1 | 0 | **1.594** | 0.0005 | 1.594 | 0.000 |
| 1 | 2 | 0.994 | 0.0112 | 2.709 | 0.102 |
| 1 | 4 | 0.667 | 0.0228 | 2.827 | 0.253 |
| 2 | 0 | **1.657** | 0.0128 | 1.657 | 0.000 |
| 2 | 2 | 0.857 | 0.0392 | 3.798 | 0.688 |
| 2 | 4 | 0.319 | 0.0057 | 3.795 | 0.772 |

episode 结束时 `β_l2 = 4.141–4.157`（理论上限 `0.15√768 ≈ 4.157`），`frac≥0.99η ≈ 0.99–1.00`。

**关键机制（已确认）**：
- 单个 Adam step 的 `Δβ_l2 ≈ 1.65 ≈ lr·√768 = 0.06×27.7`，**与 grad_l2（0.0005–0.28）几乎无关** → Adam 的逐坐标步长≈lr；
- 5 步总可移动 `≈2η=0.30`，而 β 上限仅 0.15 → **第 3 步左右即触边**；
- 因此饱和来自 **optimizer 步长（lr=2η/K）+ Adam 的尺度无关性**，**不是梯度爆炸**；且 β 在 episode 内保留 → 后续 replan 基本被钉在边界。

**证据等级**：饱和由步长/优化器造成 —— **已确认（直接数据）**；
"这是否与 SALT 原文实现一致"（其 Adam/lr 调度、是否每 replan 重置）—— **尚无法判断（需原文/代码核对）**。

## C. Loss 曲线与目标稳定性

### C1. 训练 loss（每 step 平均，fresh draws）

| task | step0 | step1 | step2 | step3 | step4 |
|---|---|---|---|---|---|
| 0 | 0.0812 | 0.1118 | 0.1257 | 0.0898 | 0.0850 |
| 1 | 0.0734 | 0.0852 | 0.0713 | 0.0933 | 0.0784 |
| 2 | 0.0781 | 0.0803 | 0.0960 | 0.0856 | 0.0872 |

训练 loss 在 step 间波动（fresh draws），**没有单调下降**；单看训练 loss 不能判定优化发散。

### C2. 固定噪声/固定 τ 的 probe loss（不参与更新）

每个 replan 的 `probe_before → probe_after`：
- 组内平均相对变化（after vs before）：task0 **+3.7%**、task1 **+0.9%**、task2 **+2.7%**（单次最大 +31.6%）；
- 跨 replan：probe_before 从 ~0.0017–0.0057 升到 ~0.011–0.017（状态/目标随轨迹变化）。

**发现**：**更新后 probe loss 平均略微上升，而不是下降** → 在当前设置下（β 已饱和、Adam 步长固定），
更新并没有改善固定目标；跨 replan 的上升主要来自状态/目标漂移，不是适配本身。

**证据等级**：更新未降低固定 probe —— **已确认**；
原因是步长过大、β 饱和、还是 flow 目标噪声 —— **较可能为步长/饱和**，需下一轮控制实验区分。

## D. 证据分级汇总

| 结论 | 等级 |
|---|---|
| camera 静态扰动时 gate 分数低于 clean，change-gate 不触发 | 已确认 |
| β 饱和由 Adam 步长 lr=2η/K 造成（非梯度爆炸） | 已确认 |
| 更新后固定 probe loss 平均上升 | 已确认 |
| 饱和/上升的具体因果（步长 vs 目标噪声 vs 原文实现差异） | 尚无法判断 |
| camera 子集不能用于效果比较（frozen 0/3） | 已确认 |

## E. 建议（不自行执行）

1. **Gate 适用性**：静态 camera 扰动需要 episode 级检测或 **mid-episode shift（A′）** 协议；
   **不要**把阈值从 4.428 调低（clean 最大 4.389，几乎贴线，会误触发）。
2. **β 饱和**：根因是优化器步长。建议下一轮"控制诊断"（需批准）：
   - 比较"β 每 replan 重置"与"episode 内保留"；
   - 记录饱和 β 相对 β=0 的动作块变化（固定观测/seed），确认饱和更新是否真的改变行为；
   - 若要改 lr/步数，必须先批准，并作为单独消融报告。
3. **Loss**：增加"梯度与更新方向一致性"诊断（`<g, Δβ>` 是否为正/负），区分优化方向问题与目标噪声。
4. **公式同步**：任何后续 SG-SALT 任务书必须使用 CoRP-native 方向
   `X_τ=(1−τ)ε+τY, v*=Y−ε`，禁止再写反方向。

## F. 结论与阶段建议

- 本轮**没有**任何性能结论；
- 当前 SALT 基础实现在 CoRP 上：gate 不适用于静态 shift；β 会迅速饱和；固定 probe 不下降；
- **建议暂不进入 Phase 3**，先做上述 2–3 个控制诊断（Phase 1.3 候选），确认 SALT 更新是否在任何设置下能改善目标，
  再决定是否值得进入小规模成功率筛选或 SG-SALT。
