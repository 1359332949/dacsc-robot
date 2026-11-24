# CoRP / LIBERO-Plus Lighting 2292 测试时适配研究：完整交接总结

> 目的：把一个长周期的实验、代码、数据协议、负结论和下一步方向完整交给另一个 AI/工程师继续执行。
> 本文所有数字均来自仓库 `fix/patch-freshness-v1` 分支与服务器实验产物；未重新跑过的实验明确标注。
> 最重要的纪律：**代码跑通、数学正确、协议有效、性能提高，是四种不同结论。** 不要把 B-Flow 的 Oracle 结果当成可部署 TTA。

---

## 0. 一句话结论

在冻结的 CoRP（DINOv2 + Resampler + DiT Flow Matching）上，LIBERO-Plus 的 Lighting 2292 视觉偏移是真实存在的；在特权 Clean/Shift 配对 Oracle 下，`velocity_projection` 之前的有界 hidden correction 有非常大的动作命令修复空间；但是：

1. 单帧/池化视觉特征 + hidden 的 readout **不能稳定预测 7 维 flow velocity residual**；
2. 简单 B0 correction 完整 Euler 后只有很小改善，且 Clean drift 约 3 mm，不能部署；
3. 第一 chunk 后的失败是混合型：核心 13 条失败里 7 条切回 Clean 视觉可继续成功，3 条 Shift 反而成功，3 条两边都失败；
4. 因此当前不应训练 Adapter、FAVR gate 或 VANE rollback；下一步应转向 **target-only 时序诊断 + selective abstention（何时不修正）**，而不是继续堆残差回归器。

---

## 1. 研究对象与基础协议

### 1.1 模型

- 策略：CoRP（Compact Robot Policy）
- checkpoint：`repos/CoRP/checkpoints/corp_libero.pt`
- 视觉编码器：DINOv2 ViT-S/14（frozen）
- 语言：T5（frozen）
- 视觉压缩：instruction-conditioned CrossAttentionResampler，每个 view 48 tokens
- 动作生成：DiT Flow Matching
- action horizon `H = 12`
- action dim `7`
- 推理 Euler steps `N = 10`
- 官方 evaluator 每次执行完整 12 个动作后才 replan，即 `E = 12`，`L = H - E = 0`
- 因此**没有 action-chunk leftover**；SALT 式监督不能直接接入。不要为了 SALT 改 `E`，历史上 `r=6` 已让 Shift 更差。

### 1.2 动作空间

环境 controller 为 OSC_POSE：

```text
输入 [-1, 1]
XYZ   -> [-0.05, 0.05] m
Rot   -> [-0.5, 0.5] rad axis-angle
gripper 单独通道
```

`dacsc/action_metrics.py` 已实现：

- clip 前/后；
- normalized command error；
- physical XYZ（米）；
- SO(3) geodesic（弧度）；
- gripper 误差；
- 与真实 `controller.scale_action()` 的 parity。

实测 parity：

```text
max abs error = 2.77e-17
5 tests passed
```

**报告时必须写 “XYZ controller-command error”，不是机器人实际 EEF 位移。**

### 1.3 数据与任务

- benchmark：LIBERO-Plus
- suite：`libero_spatial`
- 目标扰动：Lighting classification id **2292**
- 基础 BDDL vs lighting BDDL
- 每个 episode 使用 `state_split.source_file` 中的 10 维 state 索引
- 每个 state 在 baseline 中跑 3 个 seed：`7777, 8888, 9999`
- 30 个 episode 不是 30 个独立物理场景：只有 10 个 initial states；统计必须按 state cluster。

### 1.4 数据角色（非常重要）

来自 `docs/data/tta_split_libero_spatial_perturbation.json`：

```text
source_fit:
[0, 1, 2, 4, 6, 8, 10, 13, 14, 17]

source_validation:
[18, 20, 21, 22, 25, 27, 29, 31, 32, 33]

target_dev:
[34, 35, 36, 37, 39, 43, 44, 45, 48, 49]
```

- target-test 已处于 `admission_accessed` 状态（历史 2312 admission seed 2014）；**不能用于方法选择或调参**。
- 最终确认需要新的、未参与方法选择的扰动配置/状态划分。
- B3/B3-2 使用的是 target-dev 的 34–49 状态，所以文档必须标成 **privileged target-dev oracle diagnostic**，不能写成 source-only。

---

## 2. 实验时间线（由浅入深）

### 2.1 P0：正式基线

用途：确认 Frozen CoRP 在 Clean vs Lighting 2292 下是否真的有闭环差距。

协议：

- 10 states × 3 seeds = 30 paired episodes
- 同一 initial state、同一 seed、同一 `z0_schedule`
- 第一次 chunk 执行 12 个动作
- 官方 `is_success`

结果：

```text
Clean: 21/30
Shift 2292: 11/30
```

分 seed：

| domain | 7777 | 8888 | 9999 | total |
|---|---:|---:|---:|---:|
| Clean | 7/10 | 6/10 | 8/10 | 21/30 |
| Shift | 5/10 | 3/10 | 3/10 | 11/30 |

配对：

```text
both success:             8
Clean success/Shift fail: 13
Clean fail/Shift success: 3
both fail:                6
```

### 逐 state 的双峰结构（baseline_ledger）

| 分组 | states | Clean 成功 | Shift 2292 成功 |
|---|---|---:|---:|
| 完全免疫组 | 36, 39, 49 | 6/9 | **9/9 (100%)** |
| 完全团灭组 | 34, 35, 43, 45 | **10/12** | **0/12 (0%)** |
| 边缘组 | 37, 44, 48 | 5/9 | 2/9 |

> Lighting 2292 不是均匀拉低所有状态，而是双峰分化；全局修正会扰动免疫组和 Clean。

其他负结论：

- `p3_pilot.json`：固定光度仿射闭环后 Clean 4/4→3/4、Shift 2/4→1/4，破坏 state 37；
- `local_energy_*.json`：Clean 8/8，Shift 4/10，未救回核心失败且丢 state 44；
- Source Observer V1：agentview 768D 动力学可预测增益约 1.66%。

重要旧结果：

- 旧探索 `8/10 Clean / 2/10 Shift`、`r=6 8/10 / 1/10` 不能混入正式账本。
- 30 对 `z0_schedule` 全部匹配。
- `target-test` admission 状态是 `admission_accessed`，不是 `locked_unused`。

产物：

```text
docs/data/perturb_dev_clean_v2.json
docs/data/perturb_dev_v2.json
docs/data/baseline_ledger.json
```

---

### 2.2 P1.1：动作度量物理语义

问题：动作是 controller command，不是 world-space EEF 轨迹。

已完成：

- 真实 controller `output_min/max`；
- clip → physical mapping；
- normalized/physical XYZ、rotation geodesic、gripper；
- state-balanced 聚合；
- 5 个 unit tests；
- parity `2.77e-17`。

产物：

```text
dacsc/action_metrics.py
tests/test_action_metrics.py
```

---

### 2.3 B0：velocity head 数值秩

对象：

```text
action_generator.velocity_projection.weight  W: [7, 768]
```

实测：

```text
rank = 7
singular values = [0.514871, 0.428695, 0.419352, 0.407011, 0.367111, 0.354071, 0.343088]
condition number = 1.5007
```

结论：

> 最后线性头不是 7 维 velocity residual 的表达瓶颈；任意 7 维 residual 都有 hidden 解。因此**无约束 velocity Oracle 通过没有诊断意义**，必须加 hidden 范数约束和完整 Euler 积分。

产物：

```text
docs/data/velocity_head_audit.json
```

---

### 2.4 P2：真实 RGB DINO LayerNorm Oracle

问题：DINO 的小参数组是否能在 Clean/Shift 配对下显著降低动作误差？

协议：

- 20 个 source 配对（10 fit + 10 validation）
- 同一物理状态，Clean/Shift 双视角 224×224 RGB
- 只用 `backbone.blocks[0].norm1`（768 params）
- 其余 DINO/Resampler/DiT frozen
- 20 steps, lr 1e-4
- 固定 θ0 Clean reference
- 在 fresh RGB 上重跑

结果：

| 指标 | init | final |
|---|---:|---:|
| fit loss | 0.004593 | 0.004536 |
| held-out loss | 0.005326 | 0.005313 |
| clean drift | — | 1.69e-7 |

物理 Shift gap：

```text
XYZ controller-command: 8.998 mm
SO(3): 0.01178 rad
```

结论：

> block0.norm1 的 held-out 改善仅约 0.245%，弱，不足以支撑这条 DINO 参数组路线。但这不等于所有 DINO 微调都无效。

重要经验：**RGB 采集必须统一 refresh 路径**。

旧 collector Clean 侧用了 `step()` 返回的旧 obs；freshness smoke 发现 Clean 图像/state8 与统一 refresh 不一致。修复后的路径：

```text
sim.forward()
_update_observables(force=True)
_get_observations()
```

产物：

```text
docs/data/rgb_oracle_pairs_fresh.npz
docs/data/rgb_oracle_pairs_fresh.npz.manifest.json
docs/data/oracle_dino_p2_fresh.json
scripts/collect_rgb_oracle_pairs_v2.py
scripts/oracle_dino_p2.py
```

---

### 2.5 B-Flow-2：真实 Euler 轨迹的 bounded Oracle

目的：不是 z0 上的局部 velocity，而是在真实 10-step Euler 轨迹上，同一 x 计算 Clean/Shift velocity 差，再做有界 hidden correction。

协议：

```text
H=12, N=10, dt=1/10
x0 = z0
每步同一 x 计算 v_C 和 v_S
r = v_C - v_S
δh = W† r
δh_rho = δh * min(1, rho||h|| / (||δh||+eps))
```

Source 20 配对，`rho=0.10`：

| 条件 | XYZ command | SO(3) |
|---|---:|---:|
| 未修正 Shift | 8.642 mm | 0.011883 rad |
| rho=0.10 | **0.151 mm** | **0.000237 rad** |
| rho=0.15 | 4.66e-6 m | 8.58e-6 rad |

逐 token：

```text
mean ratio = 0.0397
P95 = 0.0864
max = 0.1631
超 0.1 的 token = 2.25%
集中在 flow step 8/9
```

结论：

> 有界 correction 在完整 Euler 下可以把 source 动作命令误差从 8.64 mm 降到 0.151 mm；这证明接口和积分管线自洽，但仍是特权 Oracle。

产物：

```text
scripts/b_flow2_velocity_oracle.py
docs/data/b_flow2.json
```

---

### 2.6 B-Flow-3：target-dev 第一决策 chunk 的 headroom

协议：

- 30 个 target-dev 配对（10 states × 3 seeds）
- 严格匹配 P0 z0 schedule hash
- Clean/Shift state8 max diff = 0
- 重放同一第一决策 chunk，rho=0.10

结果：

| 分组 | n | Shift XYZ | rho=0.10 XYZ |
|---|---:|---:|---:|
| Clean成功 / Shift失败 | **13** | **5.94 mm** | **0.020 mm** |
| both success | 8 | 4.71 mm | 0.122 mm |
| both fail | 6 | 5.25 mm | 0.025 mm |
| Clean失败 / Shift成功 | 3 | 6.57 mm | 0.123 mm |

结论：

> target-dev 核心失败样本也存在充足 bounded hidden correction headroom；接口不是瓶颈。

产物：

```text
scripts/collect_bf3_target_pairs.py
scripts/b_flow2_velocity_oracle.py
docs/data/bf3_target_pairs.npz
docs/data/b_flow3.json
```

---

### 2.7 Gate 0：chunk overlap 审计

官方 evaluator 控制流：

```text
for action in chunk:
    env.step(...)
```

所以：

```text
H=12, E=12, L=0
```

- 没有 leftover；
- SALT 类 leftover supervision 当前不可用；
- 改 E=6 会改变 baseline，所有方法必须统一重跑；
- 过去 `r=6` 已让 Shift 从 2/10 变 1/10，不能当默认协议。

审计脚本还验证了 sampler 是正向 Euler；训练 target velocity sign 在 released inference repo 中未定义，不能声称已核对训练目标符号。

产物：

```text
scripts/gate0_chunk_audit.py
docs/data/gate0_chunk_audit.json
```

---

### 2.8 Gate 1：第一 chunk 可恢复性

协议：

- 30 个 P0 Shift episode
- 执行第一 chunk 的 12 个动作
- 保存第 12 步物理 snapshot
- 分叉：
  - Frozen-Shift：继续 Shift 视觉
  - Frozen-Clean：从同一 snapshot 切 Clean 视觉（仅离线诊断）
- 严格匹配 P0 z0 schedule

结果：

| 分组 | n | Shift 后续成功 | Clean 后续成功 |
|---|---:|---:|---:|
| Clean成功 / Shift失败（核心） | 13 | 3 | **7** |
| both success | 8 | 6 | 6 |
| both fail | 6 | 1 | 4 |
| Clean失败 / Shift成功 | 3 | 3 | 2 |
| 全部 | 30 | 13 | 19 |

**Gate 1 复现审计（2026-11-21）：**

- P0 ledger `max_steps=220`，Gate1 原脚本硬编码 `max_steps=300`；
- 即使 z0 schedule hash 相同，Gate1 `shift_recovery` 与 P0 `shift_success` 在 30 条里翻转 6 条（20%）：
  - P0 fail → Gate1 success: 35(9999), 43(7777), 44(8888), 45(9999)；
  - P0 success → Gate1 fail: 36(7777), 39(9999)。
- 说明 Gate1 的 renderer/snapshot 链路没有严格复现 P0 原生 Shift rollout；
- 在消除这 6/30 漂移前，7/3/3 只能当粗略定性信号，不能当精确可恢复率上限。
- 修复方向：Gate1 使用 max_steps=220、固定 renderer、逐步记录 action/state hash 并先做同轨迹复现。

核心 13 的具体 state：

- Clean-only recovery: 34(2), 35(1), 37(1), 43(1), 45(2) = 7
- Shift-only: 35(9999), 43(7777), 45(9999) = 3
- both fail: 43(8888), 44(9999), 48(8888) = 3
- 没有两者都恢复的 core 样本

解释：

> 第一 chunk 后约一半 core 失败仍有 Clean 视觉恢复空间，约一半已经难以恢复；失败是混合型，不是单纯 first-frame cold start，也不是单纯后续 TTA 可解决。

边界：

- 第一 chunk 强制跑满 12 步，无 early exit；
- Clean branch 使用 `reset_episode_flags(env,0)`；
- 没有完整 model compatibility/compare 审计；
- 因此 7/13 不能等于“可由无标签 TTA 救回”，3/13 双失败也不能等于“物理不可恢复”。

产物：

```text
scripts/gate1_first_chunk_recovery.py
docs/data/gate1_first_chunk.json
```

---

### 2.9 B2：相同动作受控 replay

目的：把视觉偏移和物理轨迹变化分离。

协议：

- 同一 initial state
- 同一组 12 个 clipped actions
- Clean vs Shift
- 比较：

```text
policy_state8
qpos
qvel
act
body_xpos
body_xquat
eef_pos
```

10 个 target-dev state 结果：

```text
policy_state8 maxdiff = 0.0
measured physics maxdiff = 0.0
state mapping pass
body/site structure pass
```

视觉：

| 视角 | cosine |
|---|---:|
| agentview | 0.824–0.947 |
| eye-in-hand | 0.961–0.989 |

结论：

> 在受控相同动作下，已测物理状态一致；Lighting 2292 主要影响视觉观测，尤其 agentview。
>
> 不能写成完整 MuJoCo physics equivalence；`act` 为空是正常，`scene_model_compatible` 只覆盖 body/site 结构。

产物：

```text
scripts/b2_replay_control.py
docs/data/b2_replay2.json
docs/data/b2_replay3.json
docs/data/b3_admission.json
```

---

### 2.10 Experiment B：chunk 边界一致性 pilot

18 条边界（3 states × Clean/Shift × 3 chunks）：

| 指标 | Clean | Shift |
|---|---:|---:|
| mean ||Δz|| | 13.945 | 12.049 |
| mean ||Δstate8|| | 0.1500 | 0.1395 |
| mean ||ΣA|| | 15.898 | 15.486 |

相关：

```text
corr(||dz||, ||dstate||) = -0.027
corr(||dz||, ||action_sum||) = -0.222
```

Clean-fit action→state 预测：

```text
Clean relative error 0.064
Shift relative error 1.224
```

限制：

- `action_sum` 用的是 clip 前动作，不是实际执行；
- 只有 9+9 边界；
- 不能把 1.224 当 TTA 信号，只能当待复核线索。

产物：

```text
scripts/experiment_b_boundary.py
docs/data/experiment_b_boundary.json
```

---

### 2.11 R0：单帧 readout 诊断

数据：

- 20 个 source states
- 每条 4 个 z0
- 10 Euler steps
- 12 action positions
- 共 19,200 行（9,600 Shift + 9,600 Clean No-Op）
- 特征：visual mean、action hidden、state8、tau、k
- teacher：在未修正 Shift trajectory 同一 x 上 `r* = v_C - v_S`
- Clean 样本标签零

结果：

| 特征 | CV rel | val rel | val agg | val cos | Clean residual |
|---|---:|---:|---:|---:|---:|
| B0 h+state+τ/k | 0.621 | **0.675** | 0.684 | **0.764** | 0.124 |
| B1 visual+state+τ/k | 0.884 | 0.758 | 0.730 | 0.687 | 0.159 |
| B2 visual+h+state+τ/k | 0.758 | 0.692 | **0.657** | 0.696 | 0.200 |

kNN：rel 0.861，agg 0.847，cos 0.620。

方差分解（修正后）：

```text
between-state 35.9%
within-state 64.1%
```

注意：

- 64.1% 不是 temporal variance，混合了 z0、flow step、action position；
- 旧 26.4/73.6 是错误算法（把 total variance 当 within）；
- 单帧 h 有弱信号，但跨 state 不够，Clean No-Op 也不满足。

产物：

```text
scripts/build_r0_dataset.py
scripts/train_r0_ridge2.py
scripts/r0_analysis.py
docs/data/r0_analysis2.json
```

---

### 2.12 B3/B4：token retrieval 与 learned readout

- B3a per-view pooled：没有超过 B0。
- B3b fixed random token retrieval：cos 稍好、Clean residual 低，但 rel/agg 更差。
- B4fa/B4fb 是**固定随机投影近均匀 pooling**，不是 learned retrieval；命名必须写 random-projection controls。
- B4 learned joint training 发散；旧结果无效，不能作为“learned retrieval 无效”的证明。
- 后续发现的错误：旧 B4 relative error 把 Clean 零标签混入分母，产生巨大 pseudo-error；已修正为 Shift-only。
- 视角 token 归属单测通过：

```text
agentview change only -> tokens[0:48] 变化 43.35，tokens[48:96] 变化 0
eye change only      -> tokens[0:48] 变化 0，tokens[48:96] 变化 26.59
```

产物：

```text
scripts/build_r0_tokens.py
scripts/r0_b3_analysis.py
scripts/r0_b4.py
scripts/b3_view_ownership.py
docs/data/b3_view_ownership.json
```

---

### 2.13 B3-1：多帧状态转移 pilot

3 states × Clean/Shift × 3 chunks = 18 边界。

早期分析用了 `actions_planned`，错误；已改为：

```text
actions_executed
executed_steps
executed_mask
```

pilot 结果（仍只有 3 个 state）：

| 模型 | Clean train | Shift test |
|---|---:|---:|
| P0 | 0.134 | 0.347 |
| P1 | 0.008 | 0.608 |

结论：

> P1 在 9 条 Clean 样本上严重过拟合；B3-1 仍是 instrumentation pilot，不能证明视觉历史有效/无效。完整 B3-1 需要 10 states × 3 seeds、state-held-out、P0/P1-agent/P1-wrist/P1-multiview/P2。

产物：

```text
scripts/b3_collect.py
scripts/b3_analysis.py
docs/data/b3_boundary.json
```

---

### 2.14 B3-2：Clean-reference flow residual 诊断（3-state 与 10-state）

定义：

```text
同一物理 snapshot
同一 z0
同一 flow state x
同一 tau, k
r* = v_C(x) - v_S(x)
```

3-state 修正版（4 z0 × 10 steps × 12 positions = 1440 rows）：

| 模型 | rel | agg | cos | mag | G_row | G_agg |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| M0 | 1.712 | 1.358 | 0.688 | 1.785 | -0.712 | -0.358 |
| M1a (+agent) | 1.780 | 1.386 | 0.543 | 1.766 | -0.780 | -0.386 |
| M1w (+wrist) | 1.380 | 1.055 | 0.635 | 1.255 | -0.380 | -0.055 |
| M1both | 1.384 | 1.064 | 0.589 | 1.387 | -0.384 | -0.064 |

结论：

> wrist pooled 特征相对 M0 有小幅改善，但所有模型仍不如零修正；agentview 在当前 pooled 形式下有害。

10-state 毕业考（10 target-dev × 4 z0 × 10 × 12 = 4800 rows，leave-one-state-out）：

> **重要更正：`docs/data/b3_oracle10.json` 的 Constant 和 scalar-calibrated 字段已标记 `partially_retracted`。**
> 旧 Constant 使用了 held-out state 标签，旧 beta 使用了 held-out predictions/labels；
> 不能引用旧的 `0.233 / 0.309` 或 `0.195 / 0.252`。
> 正确账本：`docs/data/b3_oracle10_leakage_fixed.json`。

无泄漏账本（equal state mean；G_agg 为每 state G_agg 的均值）：

| 模型 | G_row(all) | G_agg(all) | G_row(unseen7) |
|---|---:|---:|---:|
| Zero | 0 | 0 | 0 |
| Constant（outer-train state mean，无 held-out 标签） | -0.090 | 0.058 | -0.208 |
| M0 | -0.047 | 0.127 | -0.162 |
| M1a | 0.016 | 0.158 | -0.065 |
| M1w | -0.330 | -0.002 | -0.474 |
| M1both | 0.001 | 0.150 | -0.090 |

无泄漏账本中 `cal_all == raw_all`：Ridge 在训练折内已贴合，`beta_train_in_sample` 被 clip 到 1.0；
不能再报告 0.195/0.252 的校准增益。

结论：

> **无泄漏 Constant 的 G_agg(all)=0.058 仍优于所有 learned 模型（M1a 0.158、M1both 0.150）。**
>
> **unseen-7 上所有 learned G_row ≤ 0；Constant G_agg 仅 0.003，仍不劣于 learned。**
>
> **直接回归 7 维 flow residual 触发停止条件，正式封存。**

重要限定：

- 这是 **privileged target-dev oracle diagnostic**，不是 source-only；
- 34/43/45 是 pilot 用过的 3 个 state，10-state 结果不能当完全独立确认；
- 校准使用了训练折 Clean/Shift oracle residual，不是部署可用信号；
- 完整 Euler / Clean No-Op 尚未完成；但 Constant 基线已经足够触发停止。

产物：

```text
scripts/b3_flow_pairs.py
scripts/b3_oracle2.py
scripts/b3_oracle10.py
docs/data/b3_oracle2.json
docs/data/b3_oracle10.json
```

原始 32MB flow 数据在服务器：

```text
/hy-tmp/patch-diag/runs/patch_diag/b3_flow_pairs_10.json
```

---

## 3. 当前研究结论分级

### 已确认（强证据）

1. Clean 21/30，Shift 11/30；Lighting 2292 有真实闭环损害。
2. P1.1 控制器物理映射正确。
3. velocity head W 满行秩（rank 7, cond 1.5），无约束 Oracle 无诊断意义。
4. B-Flow-2/3：bounded hidden correction 在特权 Oracle 下有充足 headroom：
   - source 8.642 → 0.151 mm；
   - target core-13 5.94 → 0.020 mm。
5. 当前协议 H=E=12，L=0，无 leftover。
6. Gate 1 失败异质性：core13 为 7/3/3。
7. B2：相同动作下 10 个 target-dev state 已测物理 maxdiff=0；agentview 视觉偏移明显，eye-in-hand 稳定。
8. B0 完整 Euler：验证集 9.013→7.843 mm，Clean drift ~3 mm，不安全。
9. B3-2 10-state：Constant baseline 超过所有 learned 模型；直接 7D residual regression 停止。

### 未确认

1. 多帧 target-only 信号是否可用于 TTA；
2. selective abstention / Clean No-Op；
3. 完整 Euler 后 predicted correction 是否改善；
4. 闭环成功率；
5. Background/Camera/Sensor 等新扰动泛化；
6. target-test 不能用于选型，最终确认需要新扰动配置。

---

## 4. 已排除/不应再走的方向

1. **DINO block0.norm1 小范围更新**：held-out 仅 0.245%，弱。
2. **单帧 pooled visual mean + ridge**：跨 state 不够，Clean No-Op 失败。
3. **固定随机 token retrieval**：没有稳定超过 B0。
4. **直接 7D flow velocity residual regression**：10-state Constant baseline 胜出，停止。
5. **E=6 / SALT leftover 作为主方案**：当前无 leftover，改 E 改变 baseline；r=6 历史更差。
6. **训练 Adapter / FAVR gate / VANE rollback**：没有可靠 readout 前不启动。
7. **target-test 选型**：禁止。

---

## 5. 与相关工作的边界（不能重复包装为创新）

- **TTA-VLA**：已经做 target trajectory 上 Δz / Δu 时序对齐，更新 visual projector；我们不能再把“多帧视觉—动作 transition 对齐”本身当创新。它的轨迹来源消融显示 Random/Policy/Demo 接近，但冷启动第一 chunk 仍没有历史。
- **FAVR**：已经做 flow-aware visual residual + action-conditioned visual retrieval + bounded gate；LIBERO-Plus 使用 14,347 条扰动 demonstrations、5000 steps。我们与它的区别只能是严格无目标域演示、无 target reward、选择性修正/Clean No-Op，但必须用实验证明，不能只换 backbone。
- **SALT**：核心是 action-chunk leftover；我们 H=E=12，L=0，当前不兼容。
- **VANE**：核心是 candidate/shadow + future validation + delay commit；适合作未来安全机制，但不能替代修正目标本身。
- **G-PSMR**：扰动应只在受影响输入流上修；但“agentview 漂移大”不等于“eye-in-hand 可替代其任务信息”，不能简单做跨视角一致性 gate。

---

## 6. 下一步建议（给接手 AI）

### 不要做

- 不要重新跑 B0/B2/Gate 1；
- 不要启动完整 10-state B3-1 作为新方法；
- 不要再调 ridge / 加 DINO patch / 训练 Adapter；
- 不要碰 target-test；
- 不要为了 SALT 改 E=12。

### 应该做

1. **封存 B3-2 负结论**：写入 ledger，保留 10-state 汇总与原始服务器数据。
2. **转向 TTA-VLA 式 target-only 时序诊断**：
   - 在 CoRP 的 `H=E=12` 协议下采集 Frozen 轨迹的 chunk 边界；
   - 记录 per-view Resampler tokens、action hidden、实际 clipped actions、state8；
   - 先做 feature/loss diagnostic，不更新参数；
   - 明确 environment time 与 flow time 的区别，不能把 10 个 flow step 当 10 个观测。
3. **selective abstention 作为核心问题**：
   - 不是“检测到 Shift 就修正”；
   - 而是估计当前视觉条件变化是否值得修正；
   - 先做 Clean No-Op / 拒绝更新诊断，再谈 bounded correction。
4. **完整 Euler + 物理指标**：
   - 任何新 residual predictor 必须经过 10-step Euler、unnormalize、clip、OSC_POSE 物理指标；
   - 不能只看局部 velocity MSE。
5. **最终统计**：
   - 以 initial state 为 cluster；
   - 10 states × 3 seeds 不是 30 个独立场景；
   - 报告 per-state，不只平均值；
   - target-test 封存，新扰动做最终确认。

---

## 7. 关键文件与复现入口

```text
docs/PATCH_DIAGNOSTICS.md                 # 实验日志总账
docs/data/baseline_ledger.json            # P0 配对账本
docs/data/rgb_oracle_pairs_fresh.npz      # P2/B-Flow fresh RGB
docs/data/bf3_target_pairs.npz            # target-dev 第一 chunk 配对
docs/data/b3_admission.json               # 10-state 物理准入
docs/data/b3_oracle10.json                # B3-2 10-state 汇总
docs/data/gate1_first_chunk.json          # Gate1 可恢复性
docs/data/gate0_chunk_audit.json          # H/E 审计
docs/data/b2_replay3.json                 # B2 物理 parity
dacsc/action_metrics.py                   # 动作物理度量
scripts/collect_rgb_oracle_pairs_v2.py    # fresh RGB collector
scripts/b_flow2_velocity_oracle.py        # B-Flow bounded Oracle
scripts/collect_bf3_target_pairs.py       # B-Flow-3 target pairs
scripts/b2_replay_control.py              # B2 replay
scripts/b3_flow_pairs.py                  # B3-2 same-x flow residual
scripts/b3_oracle10.py                    # B3-2 10-state 分析
```

远程分支：

```text
fix/patch-freshness-v1
最新提交：6fba252
```

服务器工作区：

```text
/hy-tmp/patch-diag
/hy-tmp/patch-diag/repos/CoRP
/hy-tmp/validation/multidomain_matrix.json
/hy-tmp/libero_plus
```

本地主仓库：

```text
/Users/wentiliangkaihua/dacsc-robot
```

---

## 8. 最重要的经验教训

1. **先修测量，再跑模型。** 我们的 controller scaling、freshness、action planned/executed、visual/velocity 变量覆盖，都曾造成假结论。
2. **Oracle headroom 不等于 deployable readout。** B-Flow 很强，但不能说明部署时可预测。
3. **Constant baseline 必须始终保留。** 10-state B3-2 就是被 constant baseline 一票否决。
4. **不要用平均值掩盖 state 异质性。** Gate1 7/3/3、B3-2 state 43 灾难性反跳都说明。
5. **target-dev 特权诊断必须标清楚。** 不能写成 source-only。
6. **target-test 不能用于方法选择。**
7. **Clean No-Op 是一票否决条件，不是事后指标。**
8. **H=E=12 是当前协议；不要为了 leftover 改协议。**

---

## 9. 当前一句话交接

> 冻结 CoRP 在 Lighting 2292 下确实存在视觉域偏移与闭环性能下降；特权 Clean/Shift Oracle 证明 `velocity_projection` 前的有界 hidden correction 有巨大 headroom；但单帧/池化视觉 readout 无法跨 target-dev state 预测 7 维 flow residual（Constant baseline 胜出），因此直接 residual regression 已正式封存。下一步应转向 target-only 时序诊断与 selective abstention，而不是继续训练 Adapter。
