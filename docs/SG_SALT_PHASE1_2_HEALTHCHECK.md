# 工单 1.2（精简版）：SALT 最小健康检查 + SG-SALT 接口准备

> 基于 `105aab6`。只做离线/小输入检查；**未改 SALT 算法、未跑 rollout、未实现其他机制**。

## 1. Flow Matching / leftover 实现是否仍正确

- `pytest tests/test_salt.py` → **21 passed**（含 CoRP-native 方向解析积分测试、leftover 对齐、mask reduction、
  B.4 gate、λ=0 之外的基础模块）。
- 结论：工单 1.1 的 CoRP-native Flow 方向与 leftover 实现未被本次诊断改动破坏。**PASS**

## 2. β 饱和是否影响基础 SALT 更新（用 Phase-1.2 日志复核）

| task | 首次饱和 replan | 饱和后 update 数 | 饱和后 Δβ_l2 均值 | min | max |
|---|---|---|---|---|---|
| 0 | 1 | 55 | 0.0966 | 0.0047 | 0.4153 |
| 1 | 3 | 70 | 0.1095 | 0.0279 | 0.2299 |
| 2 | 1 | 70 | 0.0671 | 0.0000 | 0.3669 |

**读法**：
- 饱和发生得很早（replan 1–3，`frac≥0.99η` 超 0.5）；
- 但**并非完全冻结**：饱和后仍有 Δβ（≈0.07–0.11，最大 0.42），说明部分维度仍在移动/在边界附近被裁剪；
- 饱和 β 相对 β=0 对动作的影响（小输入实验）：`mean|Δa|≈0.0082–0.0110`（动作尺度约 0.3，即 ~3–4%），
  **有影响但量级不大**。

**结论**：β 饱和确实存在且发生很早，但基础 SALT 更新并未完全失效；它更像"大部分维度贴边 + 少量维度继续更新"。
这足以作为对照基线使用，但**在 SG-SALT 实验中必须记录 β 饱和比例**，避免把"饱和造成的固定缩放"误判为创新收益。

## 3. A 子空间与当前 CoRP 接口一致性

使用文件：`/hy-tmp/validation/interface_stats_spatial_k32_big2.json`（trajectory-matched，87 frames；fixed RNG）。

| 检查 | 结果 |
|---|---|
| A shape | **(768, 5)**，rank=5 |
| site 记录 | `cross-attention write (last Euler step)` |
| 正交性 `max|AᵀA−I|` | **7.94e-09** |
| 当前 CoRP 写回维度 | **768**（一致） |
| 写回捕获方式 | `blocks[0].attention.attend` 最后 Euler 步、12 token 均值（与 A 构造一致） |

**接口小输入实测**（均匀 β=η vs β=0，同一观测/seed）：

| 量 | 值 |
|---|---|
| `‖Δz‖` | 0.6410 |
| `‖AᵀΔz‖` | **0.1238**（占 ‖Δz‖ 的 ~19%） |
| `‖Δz − AAᵀΔz‖` | 0.6290 |
| 动作变化 `mean|Δa|` | 0.00821 |

→ **A 与当前接口同层、同维度、同 token 布局；保护项在该空间有明确定义。**

## 4. A 方向保护项的离线数值验证

构造合成正交基 Q（768×5）与已知 Δz：

| 检查 | 结果 |
|---|---|
| 沿 A 方向 Δz = 2·q₁ → `‖QᵀΔz‖²` | **4.0000**（期望 4） PASS |
| 正交方向 Δz（投影为 0） → penalty | **6.8e-34**（期望 0） PASS |
| λ=0 时 SG-SALT 与 SALT 的 loss 差 | **0.00e+00** PASS |

→ 保护项数学实现正确；在构造上 `λ=0` 严格退化为 SALT。

## 5. 结论与建议

| 问题 | 结论 |
|---|---|
| 1.1 基础实现 | 仍正确（21 tests PASS） |
| β 饱和影响 | 存在且发生早，但更新未完全冻结；动作影响 ~3–4%；需在实验中记录饱和比例 |
| A 接口一致性 | **一致**（768 维、last Euler write、token-mean、正交） |
| 保护项数值 | **正确**（沿 A=4.0；正交≈0；λ=0 等价） |

**可以进入工单 2.0（SG-SALT 局部实现）**。实现时请注意：
1. 使用 `interface_stats_spatial_k32_big2.json` 的 A（并记录 checkpoint SHA / commit）；
2. 首轮验证按你的决定**固定触发条件**（不引入 feature gate 变量），只比较 Frozen / SALT / SG-SALT；
3. 保护项放在**写回空间**：`L = L_SALT + λ‖Aᵀ(z_φ − z_0)‖²`，其中 `z` 为同层 last-Euler token-mean 写回；
4. 每次实验记录 β 饱和比例（`frac≥0.95η/0.99η`），避免把饱和效应误读为创新收益。

## 6. 交付

- 脚本：`scripts/sg_salt_healthcheck.py`
- 测试：`pytest tests/test_salt.py` → 21 passed
- 原始诊断（Phase 1.2）：`/hy-tmp/validation/diag_clean_always/*.json`
- commit SHA：见本次提交
