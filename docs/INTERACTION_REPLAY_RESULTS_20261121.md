# 最新实验结论与修正记录（2026-11-21）

本轮完成：旧 B3-2 统计泄漏修正与原数据重分析；Lighting 日志核验；受控回放采集器、固定动作状态探针及 RTX 3090 / MuJoCo 实测。未训练风险观测器、Adapter 或更新 CoRP 权重。本轮未访问 target-test。

## 1. 旧残差结论必须纠正

`6fba252` 的常数预测读取了外层测试状态残差均值，beta 也读取了测试标签。原常数 / 校准结果已在原 JSON 标记部分撤回，`PATCH_DIAGNOSTICS.md` 已改写。

服务器 509 MB 原始十状态文件仍在。本次按固定特征与 alpha=100 重算，无须重新采集；每折训练均值、尺度、常数和 beta 均只使用外层训练状态。测试标签污染回归测试通过。

G 为相对零修正的改善量，正值优于零修正。下表是各状态等权平均；G_agg 是每状态整体误差比对应的改善量，再跨状态平均。

| 方法 | G_row：10 状态 | G_agg：10 状态 | G_row：未参与三状态 pilot 的 7 状态 |
|---|---:|---:|---:|
| 零修正 | 0 | 0 | 0 |
| 正确训练折常数 | -0.090 | 0.058 | -0.208 |
| M0：hidden+state+流时间/位置 | -0.047 | 0.127 | -0.162 |
| M1a：M0+主视角 | 0.016 | 0.158 | -0.065 |
| M1w：M0+腕部视角 | -0.330 | -0.002 | -0.474 |
| M1both：M0+双视角 | 0.001 | 0.150 | -0.090 |

- 原“常数基线 0.233 / 0.309 全面胜出”结论撤回；正确常数并未全面优于学习模型。
- 原始学习模型指标与旧结果最大差 `2.892e-12`，两个具体泄漏未改变这些原始指标。
- beta 改为外层训练内拟合后各折均裁剪为 1，结果与未校准模型相同。该校准仍是训练内、特权配对诊断，不是独立验证或部署可用估计。
- 七个非 pilot 状态上原始学习模型 G_row 均为负，部分 G_agg 为正；尚无稳定跨状态收益、Clean No-Op 或闭环收益证据。维持冻结直接残差支线，但理由不能再使用错误的常数比较。
- 4800 行不是 4800 个独立样本；这些仍是 **privileged target-dev oracle diagnostic**。

证据：`docs/data/b3_oracle10_leakage_fixed.json`（含各折训练状态、常数、beta、数据 SHA256）；`docs/data/b3_oracle10.json` 的 `_validity`；`scripts/b3_oracle10.py`。

## 2. Lighting 成败已按实际日志核实

只读取 frozen 方法，按 (state_id, seed) 配对，核对相同 z0 schedule，不把 first_observation 方法混入。

| seed | Clean 成功 | Lighting 2292 成功 | Lighting 成功状态 |
|---|---:|---:|---|
| 7777 | 7/10 | 5/10 | 36、37、39、44、49 |
| 8888 | 6/10 | 3/10 | 36、39、49 |
| 9999 | 8/10 | 3/10 | 36、39、49 |
| 合计 | 21/30 | 11/30 | 10 个初始状态的三次重复 |

“只有 36、37 成功”来自旧协议探索记录，不能替代当前三种子日志；36、39、49 在当前日志中确实成功。以上是本轮重新核验的历史记录，未重新运行这 60 条独立策略轨迹。

证据：`docs/data/lighting_episode_audit_20261121.json`；`scripts/audit_lighting_episodes.py`。

## 3. 新受控回放：小规模验收通过

从 seed=7777 的真实日志自动选择一个失败状态和一个成功状态，按编号确定为 34、36。该结果分层采样只服务于 instrumentation，不作风险分布或预测精度估计。

| 运行 | 配对状态 | 每域 chunk | 每域执行步数 | 末尾情况 |
|---|---|---:|---:|---|
| pilot | 34 | 3 | 36 | 采集上限：truncated=true，done=false |
| pilot | 36 | 3 | 36 | 采集上限：truncated=true，done=false |
| terminal | 36（重复） | 9 | 106 | 成功结束：8 个完整 chunk + 1 个 10 步 chunk |

只有 **2 个独立初始状态、1 个 seed**；terminal 的 state36 是重复和边界条件验证，不是新增独立样本。共 3 组配对 / 6 条受控轨迹 / 30 个 transition，含重复前缀。

- 完全相同的 Clean 策略裁剪动作驱动两域；每步测得 qpos、qvel、时间、body pose、EEF/site pose、ctrl 的最大差均为 **0**。
- 边界 state8 差为 **0**；已列举静态结构、几何和物理参数检查通过。未据此宣称所有 MuJoCo 内部状态完全等价。
- `[观测_i, 实际动作_i, 观测_(i+1)]` 对齐；模拟时间增量等于执行步数 / 控制频率；动作长度、mask、特征尺寸、数值有限性通过。
- 成功提前终止的 10 步 chunk 保留了实际长度和 mask，`fixed_horizon_eligible=false`，不能混入 12 步预测样本。
- 原始 NPZ 下载到本地后重新读取，hash 与契约全部通过。
- state36 两轮重叠的 36 步在 RGB、state8、两路 tokens、探针、动作、z0 上最大差均为 **0**。

终止 API 保留真实语义：success 来自 task checker；legacy done 原样保存；terminated 去除显式 TimeLimit；外部采集截止单独记 truncated，不推断“任务失败”。

## 4. 固定动作状态探针与推理路径通过

- 主探针固定 `x_probe[1,12,7]`、`tau=0.5`、动作位置 `k=0`。
- 每个配对 episode 内，本体状态锚点固定为初始 state8，语言固定，只有图像变化。不同初始状态间的锚点不同，不能跨状态直接归因纯视觉。
- action hidden 来自官方 generator 的 `output_norm` hook，velocity 来自正式 forward，未手工重写 DiT block。
- 同一条件重复测量差为 **0**。
- 初始真实 Clean/Lighting 输入上，`policy.predict()` 注入固定 z0，与显式 Euler 的每一步 x、velocity、最终 controller action、clipped command 的最大差均为 **0**。结论只覆盖这些测试输入。
- 单视角替换，另一视角 token 最大差 **0**，被替换视角有非零变化。视角归属成立于当前 checkpoint 与测试输入。

同物理状态、同动作仍有视觉响应，短回放的 RMS 如下（未白化、未经风险校准）：

| 状态 | 主视角 token 差 | 腕部 token 差 | 主视角 transition 差 | 腕部 transition 差 |
|---|---:|---:|---:|---:|
| 34 | 0.569 | 0.272 | 0.524 | 0.313 |
| 36 | 0.422 | 0.220 | 0.374 | 0.265 |

这里 transition 差是 `delta_z_clean - delta_z_shift` 的 RMS，**不是训练过的下一帧预测残差**。它说明原始视觉变化量仍对光照敏感；不能据此声称风险可预测，也不能证明一个未来的条件观测器一定失败。

## 5. 工程验证与证据边界

- 本地与服务器均通过 **8 个针对性单元测试**：测试标签污染、与旧 ridge 原始预测一致、部分 chunk、mask/时间/特征/探针破坏、两域动作错配、终止后继续执行、超时不等于成功、success 与 done 分离。
- 已实际运行冻结 CoRP / RTX 3090 / OSMesa，不是 mock 测试。
- 旧 baseline state36/7777 记录成功步数 107，本轮官方 predict 采集为 106；本轮证明的是新采集路径的重复一致和配对一致，**未证明历史 baseline 整条轨迹逐步复现**。不将本次受控回放混入 21/30、11/30 的基线统计。
- 本次不拟合正常观测器，不产生 chunk 风险标签、AUROC/AUPRC、5% 误报保证或提前量。
- 历史 target-test 已存在 admission 访问，不能改称从未访问；本次未访问，并限制为 target-dev Lighting 2292。

服务器原始数据：

- `/hy-tmp/patch-diag/runs/interaction_replay/20261121_pilot/`
- `/hy-tmp/patch-diag/runs/interaction_replay/20261121_terminal/`
- `/hy-tmp/patch-diag/runs/patch_diag/b3_flow_pairs_10.json`（旧实验原数据）

本地原始副本：

`/Users/wentiliangkaihua/Documents/ChatGPT/专利论文/remote_artifacts/interaction_replay_20261121/`

仓库内报告：`docs/data/interaction_replay_20261121/`，包含两次 summary、重新打开数据的验证和重复前缀对比。

## 6. 下一阶段可做什么

最小数据与探针验收已通过，可以开始单一“正常环境观测器”的 Source Fit / Source Validation 实验。先拟合动作条件下的下一视觉变化，并和 state+动作、视觉不带动作两种对照比较。

仍需证明三件事：正常交互预测能跨状态成立；无害光照不会制造大量假报警；Lighting 内部的分数能预测任务相关风险而非仅识别域。分数只能在下一观测到达后使用，提前量必须从该可用时刻计算。具体信息预算、标签、阈值与状态级统计规则见 `INTERACTION_RISK_PROTOCOL.md`。

当前结论是 **数据和探针已具备下一阶段实验条件，风险信号的有效性尚未建立**。
