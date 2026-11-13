# SALT + Action Budget：阶段一离线可行性审查

> 范围：只审查现有 B2/B1 数据与代码能否支撑阶段一判断；**不运行任何 rollout、不新增实验**。
> 结论先行：**现有数据不足，阶段一无法完成；需要一次“离线 policy 前向/梯度提取”，但仍不需要 simulator rollout。当前 GPU 主机不可达。**

## 1. 阶段一需要回答的四个问题

| 编号 | 问题 | 需要的量 |
|---|---|---|
| Q1 | 能否在同一观测、同一 action latent seed 下得到更新前后的动作块？ | `a(β)`, `a(β+Δβ)`, 同一 `seed` |
| Q2 | 能否可靠估计 `J_{a,β}Δβ` 并与实际 `Δa` 比较？ | 动作对 β 的 Jacobian/JVP，或固定 seed 有限差分 |
| Q3 | 在 SALT 更新上，预设 ρ 是否产生合理 α 分布？ | SALT 梯度 `∇L_SALT`、`Δa`、`α=min(1,ρ/‖Δa‖)` |
| Q4 | 与 A-null 相比，预算是否允许更多动作敏感方向变化？ | SALT+A-null 与 SALT+budget 的同状态动作块 / A 坐标变化 |

## 2. 现有数据审计

### 2.1 B2 正式 JSONL（165 条）

字段包括：
`candidate, branch, tau_prox, dp_*, action_changes, action_change, proposal_norm, applied_norm, predicted_a_change, alpha, beta_max, step_scale, accepted, backtracks, local_a_change, ...`

**缺失：**
- 不保存**动作块向量**（只有每段 `action_changes` 的标量均值）；
- 不保存**完整 β 向量**（只有 norm/max）；
- 不保存**动作 Jacobian/JVP**；
- 不保存**SALT 更新**：B2 的候选是 SG-CPA raw/prod，不是 SALT；
- 不保存**分支观测**（只有 `target_contact/d_branch` 等聚合量），无法离线复原同一状态。

→ **Q1/Q2/Q3 无法从 B2 JSONL 计算。**

### 2.2 B1 capture v2（667 replans）

字段包括：
`action_chunk_norm/ctrl, view_actions, action_seed, beta=0, state, images paths, body/geom poses, ...`

- 有同一观测下的冻结动作块（β=0）；
- 有 action seed；
- 有连续 replan，可构造 SALT leftover 目标；
- **没有**更新后的动作块/β/梯度。

→ 可用于 Q1 的“更新前”一侧，但**没有“更新后”一侧**。

### 2.3 图像

B1 图像只存在于服务器 `/hy-tmp/validation/b1_capture_v2/`，本地归档只有 `image_manifest`（sha256/路径），**没有 JPEG 本体**。
离线前向需要图像；当前 GPU 主机连接被拒。

## 3. 代码审计

| 组件 | 状态 |
|---|---|
| SALT loss / leftover 目标 | 有：`dacsc/salt_tta.py`, `dacsc/sg_salt.py` |
| β 适配器（K projection） | 有：`SaltKeyAdapter` |
| A-null 投影 `P_null` | 有：`SGCpa.nullspace` / `jacobian_a`（A 坐标） |
| 动作对 β 的 Jacobian/JVP `J_{a,β}` | **没有**；`predicted_a_change` 是 `‖J_AΔβ‖`，不是动作输出变化 |
| Action Budget 缩放 α | **没有**（阶段一本来就要验证） |
| 同 seed 更新前/后动作块采集 | B2 脚本内部算过 `ch0/ch1`，但**没有保存** |
| 离线 policy 前向脚本（不 step 环境） | 没有现成入口 |

## 4. 结论

**阶段一无法基于现有归档数据完成。** 不是统计口径问题，而是缺少三类必要量：

1. 更新后的动作块/β 向量；
2. 动作 Jacobian 或可控有限差分；
3. SALT 更新（现有 B2 是 SG-CPA 候选，不能替代）。

因此不能把 B2 的 `action_change`、`predicted_a_change` 当作 Action Budget 的 `Δa`/`J_{a,β}Δβ`——那会混用不同更新与不同层位。

## 5. 恢复 GPU 后需要的最小离线提取（不是 rollout）

只对已有 B1 的 branch 观测做前向，不调用 `env.step`：

```
输入：B1 的 (suite, task, episode, replan=8/12) 观测、state、instruction、上一动作块、seed
1. 载入 CoRP + β=0，固定 action latent seed ξ；
2. 计算 SALT leftover loss L_SALT，反传得到 g=∇_β L；
3. 采样 a0 = π_β(o, ξ)；
4. 取 Δβ = -η g（可先固定 η，仅用于可行性）；
5. 采样 a1 = π_{β+Δβ}(o, ξ)（同一 ξ）；
6. 记录 Δa = a1 - a0，β 向量，g 范数；
7. 计算/近似 J_{a,β}Δβ（JVP 或固定 seed 有限差分）；
8. 对 ρ 网格计算 α=min(1, ρ/(‖Δa‖+ε))，记录 α 分布、binding 频率；
9. 另做 A-null 版本：Δβ_proj = P_null g，重复 3–7；
10. 比较 budget 与 A-null 的动作变化/敏感方向变化。
```

输出 schema（建议）：

```
suite, task_id, episode, branch, seed, beta_before, beta_after,
action_chunk_before, action_chunk_after, delta_action,
pred_action_change (JVP), actual_action_change,
a_null_delta_action, a_null_pred_action_change,
rho, alpha, binding, g_norm, compute_ms
```

**这不是 rollout**：不推进环境、不产生新 episode，只复用已有观测做 policy 前向/梯度。但需要 GPU 与 B1 图像。

## 6. 当前阻塞

- GPU 主机 `i-2.gpushare.com:20119` 连接被拒；
- B1 图像只在服务器，本地无本体；
- 因此阶段一提取无法执行。

## 7. 下一步

1. 恢复 GPU 主机（或提供可达地址）；
2. 只运行上述离线提取（不跑 pilot、不跑正式实验）；
3. 根据提取结果判断 Q1–Q4 是否成立；
4. 若 Q2 的相关性差或 Q3 的 α 几乎不 binding，则停止 Action Budget，不进入 pilot；
5. 若通过，再单独申请阶段二 pilot。

**本报告不包含任何 Action Budget 有效性结论，也不把 B2 数据当作 SALT 更新数据使用。**
