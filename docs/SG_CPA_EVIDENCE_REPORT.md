# SG-CPA 证据报告：任务级负结果、机制诊断与 η 响应

> 本报告合并 4.2（旧实现，600 episodes）、4.3（修复随机数后的 360 episodes 主比较）与离线 η 标定，
> 用于论文/汇报材料。所有数字可追溯到 `/hy-tmp/validation/` 下的原始 JSONL/JSON 与本仓库 commit。

## 1. 研究问题与方法

**目标**：在冻结 CoRP 的前提下，用"反事实视觉共识"构造无标签自监督目标，并用动作敏感子空间 A 的零空间投影约束测试时更新。

**方法（SG-CPA）**
- 视图：Identity + Gaussian Noise（σ=0.05）+ Gaussian Blur（k=3, σ=0.5）+ Photometric（1.05），像素空间施加后再归一化；
- 目标：4 视图共享动作噪声采样，取 **Medoid** 作为教师目标；
- 损失：动作块直接蒸馏（CoRP 归一化动作空间）；
- 参数化：K-projection SVD 缩放 β（冻结 CoRP 其余参数）；
- 更新：`Δβ = −η P_null g`，`P_null = I − V_r V_rᵀ`（对真实 Jacobian `J_A = Aᵀ ∂z/∂β` 做 SVD）；
- 保护：每步对同一观测的 β=0 写回做有限步 A guard，超限则步长减半/回退。

## 2. 任务级结果

### 2.1 旧实现（4.2，600 episodes；只代表旧代码）

| 组 | Episodes | 成功率 |
|---|---|---|
| Frozen | 120 | 89/120 = 74.2% |
| SALT | 120 | 91/120 = 75.8% |
| SG-CPA（旧） | 120 | 89/120 = 74.2% |
| SG-CPA w/o projection（旧，扩展组） | 120 | 90/120 = 75.0% |
| SG-CPA medoid-excl-id（旧，诊断组） | 120 | 93/120 = 77.5% |

SG−SALT = −1.7 pp（McNemar p=0.839；任务级 90% CI [−7.5,+5.0]）。
**注意**：后两组是超出原计划的扩展诊断；旧实现的视图噪声/动作 latent 存在复用缺陷，消融解释也受 A guard 混淆。

### 2.2 修复随机数后的主比较（4.3，360 episodes）

| 域 | Frozen | SALT | SG-CPA(fixed) | Δ(SG−SALT) | p | 90% CI |
|---|---|---|---|---|---|---|
| Camera | 60.0% | 60.0% | 63.3% | +3.3 pp | 1.000 | [−10.0,+16.7] |
| Background | 53.3% | 66.7% | 50.0% | −16.7 pp | 0.125 | [−30.0,−3.3] |
| Noise | 90.0% | 86.7% | 93.3% | +6.7 pp | 0.625 | [−6.7,+20.0] |
| Light | 93.3% | 86.7% | 90.0% | +3.3 pp | 1.000 | [−13.3,+23.3] |
| **Overall** | **74.2%** | **75.0%** | **74.2%** | **−0.8 pp** | 1.000 | **[−8.3,+5.8]** |

- 修复版仍未超过 Frozen/SALT；
- SALT 在相同协议/seed 下两次运行差 0.8 pp（75.8→75.0），说明 run-to-run 噪声约 1 pp；
- Background 上 SALT 相对 Frozen +13.3 pp（CI [3.3,23.3]），SG-CPA 未复制该收益（负向趋势，p=0.125）。

## 3. 机制诊断

### 3.1 目标退化：Medoid 多数就是 Identity

| 来源 | 域 | Medoid=Identity 比例 |
|---|---|---|
| 离线（每 replan 新 seed） | Camera | 0.77–0.81 |
| 离线 | Background | 0.75 |
| 离线 | Noise | 0.63 |
| 离线 | Light | 0.94 |
| **修复版真实 rollout（3197 次更新）** | 全部 | **81.4%**（Camera 77.0%、Background 82.7%、Noise 83.5%、Light 83.3%） |

medoid index 分布：identity 2601 / photo 413 / blur 176 / noise 7。
→ 大多数时刻教师目标 = 冻结策略在 identity 输入上的动作，蒸馏信号本身很小
（β=0 时的蒸馏 loss：离线 4e-4–2e-3；真实 rollout 平均 7.5e-3）。

### 3.2 更新幅度与保护行为（修复版 rollout）

| 指标 | 值 |
|---|---|
| updates / episode | 26.6 |
| R_retain = ‖P_null g‖/‖g‖ | 0.628 |
| 更新接受率（含 backtrack 后接受） | 95.8% |
| 触发 backtrack 的更新比例 | **4.2%**（rollback 0） |
| β 最大绝对值（episode 均值 / 全局） | 0.0055 / 0.0375（边界 0.15） |
| 单次 replan 适配耗时（8 路并行实测） | 2.14 s |

### 3.3 随机数修复验证

120/120 个 SG-CPA episode 的 action seed 在每个 replan 递增 7919；
同一 replan 内 teacher/student 共享同一 latent；视图噪声每 replan 用新 view seed（单测覆盖）。

## 4. η 响应（离线，固定观测）

**实验口径**：16 个固定观测；教师目标、四视图、动作 latent 全部固定；从 β=0 连续 10 次更新
（这是固定观测上的局部优化，**不是**机器人连续 10 次 replan）。

| η | 累计 loss 降幅 | β 最大 | identity 动作变化（normalized，mean abs） | backtrack 更新比例 |
|---|---|---|---|---|
| 0.05 | ~0 | 1.0e-4 | 4.1e-6 | 0 |
| 0.5 | 1.4e-6 | 1.0e-3 | 4.1e-5 | 0 |
| 5 | 1.4e-5 | 9.7e-3 | 3.9e-4 | 0 |
| 20 | 4.7e-5 | 3.5e-2 | 1.3e-3 | 0 |
| 50 | 7.4e-5 | 6.3e-2 | 2.3e-3 | 19% |
| 100 | 9.1e-5（9.1%） | 9.1e-2 | 3.0e-3 | 25% |

补充（η=100，10 步后的 controller-space 动作差 L2）：
- xyz **3.30e-2**、rotation **4.21e-3**、gripper **1.15e-2**、合计 **3.55e-2**；
- 对应 normalized 平均绝对变化：xyz 4.53e-3、rot 1.52e-3、grip 2.64e-3。
（η=20 时分别为 xyz 1.53e-2、rot 1.79e-3、grip 5.04e-3、合计 1.63e-2。）

**读法**：K-projection 参数空间并非不可动；但可测动作变化需要 η 提高约 3 个数量级
（0.05→50–100）并做 10 次更新，而 loss 只下降 7–9%，优化效率很低。
"接受率 100%"指**回溯后仍被接受**；η=100 时有 25% 的更新触发过 backtrack（rollback 0）。

## 5. 效率分析

| 项 | 数值 | 口径 |
|---|---|---|
| 单次 SG-CPA step（离线标定，单进程） | **0.4–0.5 s** | 固定教师目标，不含每 replan 重建目标 |
| 4.3 rollout 单次 replan 适配 | **2.14 s** | 8 路并行实测，含教师目标重建/写回/J_A/guard |
| 10 次更新/replan（η=100 方案） | **约 4–5 s（估算）** | 0.4–0.5 s ×10，单进程；若按 rollout 口径约 21 s |
| SALT replan（估算） | **约 0.5 s** | 0.1 s/update ×5，非同一计时边界；同 GPU 型号、不同并行负载 |

→ 在可接受的 replan 预算内（≤1 次更新、η≤5），动作变化只有 1e-6–1e-4 量级；
要达到 1e-3–3e-3 的动作变化，需要 10 次更新 + η≈50–100，成本超出实时预算。

## 6. 结论与局限

**结论（建议采用）**

> 在冻结 CoRP 的实现中，SG-CPA 的反事实共识目标能够驱动 K-projection 参数产生可测更新；但旧实现（4.2）
> 存在随机数复用缺陷，修复后的主比较（4.3，360 episodes）仍未显示出相对 Frozen/SALT 的任务收益
> （overall 74.2% vs 75.0%/74.2%；SG−SALT = −0.8 pp，p=1.0，CI [−8.3,+5.8]）。
> 机制上，修复版 rollout 中 medoid 仍有 81.4% 选中 identity，且 η=0.05 时单步动作变化约 1e-6；
> 离线 η 标定显示要获得 1e-3–3e-3 的动作变化需要 η 提高约 3 个数量级并迭代 10 次，
> 计算成本（每 replan 约 4–5 s 估算）超出既有 replan 预算。因此，当前 K-projection + 投影 SGD
> 参数化尚未形成兼顾任务收益与实时效率的可用控制更新。该结果不意味着反事实共识目标本身无效，
> 而是说明当前实现路径缺乏足够的效率证据。

**局限（必须保留）**
1. η 标定是固定观测下的局部优化（16 观测、固定目标/视图/latent、连续 10 次更新），
   不代表真实连续 10 次 replan，也不证明对新观测/新任务有相同响应；
2. 每域仅 3 任务 × 5 episodes = 30，任务级 cluster bootstrap 90% CI；未做多重比较校正；
3. Background 的 McNemar p=0.125 与 CI 不含 0 不一致，只能表述为负向趋势；
4. "动作变化"不等于"动作正确/安全"，loss 下降也不等于任务成功率提升；
5. 4.2 的消融组（no-projection / medoid-excl-id）只代表旧代码，修复版下未重跑；
6. SALT 为 CoRP 上的 always-on 端口，不是原论文完整的 Transition Anchoring 协议；
7. 时延为离线 GPU 实测/估算，未做真实部署测量；SALT 的 0.5 s/replan 为估算值。

_原始数据：`/hy-tmp/validation/p42/`（旧）、`/hy-tmp/validation/p43/`（修复版）、`/hy-tmp/validation/sg_cpa_eta_cal/`（η）。_
