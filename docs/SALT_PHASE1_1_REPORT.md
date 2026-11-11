# 工单 1.1 报告：CoRP-SALT 算法一致性修复（Phase 1.1）

> 基于 `e2233b7`。只修 SALT 基础实现与 CoRP 原生 Flow Matching / SALT Appendix B.4 的一致性；
> **未实现 SG-SALT、未改 CPIA/Oracle/checkpoint、未跑大规模矩阵。**

## 1. P0-1：Flow Matching 时间方向（已修）

CoRP 官方采样：`action = randn; for step: tau=step/N; action += v/N`（`action_generator.py:182-215`），
即 **τ=0 是噪声、τ=1 是数据**。

`dacsc/salt_tta.py::flow_interpolate` 已改为 CoRP-native：

```
X_tau = (1 - tau) * eps + tau * Y
v*    = Y - eps
```

并删除旧的 `(1-tau)Y + tau*eps / eps-Y` 约定（那是 SmolVLA 风格，与 CoRP sampler 相反）。
新增**解析积分测试** `test_corp_native_flow_integration_direction`：
恒定速度 `Y-eps` 从 `eps` 出发、按 CoRP 正向 Euler 积分 N 步必须得到 `Y`；
反方向 `eps-Y` 必须失败。

## 2. P0-2：SALT B.4 Gate（已按论文重写）

`FeatureChangeGate` 现在实现论文 Appendix B.4：

| 项 | 实现 |
|---|---|
| 变化量 | `d[r,i] = 1 - cos(F[r-1,i], F[r,i])`（cosine 前 token-norm floor=1e-6） |
| 中心 | token-wise **median** |
| 尺度 | `s_i = 1.4826 * MAD_i`，共享下限 `0.25 * median_i(s_i)` |
| 分数 | `S_r = mean_i clip((d[r,i]-m_i)/s_i, 0, 10)`（标准化后裁剪，取平均） |
| 阈值 | **leave-one-task-out 交叉拟合**：`0.9 * max(held-out episode max S)` |
| 部署统计 | 用全部 nominal 数据重新拟合 m/s |

Token 范围：`--gate-tokens 96`（visual_all，主设置）/ `48`（agentview_only 消融），显式参数，不静默替代。

**Calibration 实测**（5 任务 × 4 episodes = 20 条 nominal，331 个 transition）：

| 项 | 值 |
|---|---|
| threshold | **4.4280** |
| calibration JSON | `/hy-tmp/validation/p11_calib/gate.json`（含 tasks/seeds/frames/checkpoint） |

## 3. 其余修复

- **Loss reduction**：`_masked_mse` 改为只对监督元素求均值：`sum(diff²·M)/sum(M)`，文档与测试固定；
- **Noise-time draws**：每 episode 一个 `torch.Generator(seed+17)`，5 个 optimizer step 连续抽样（不再每步 `manual_seed(1234)` 复用同一组）；
- **β diagnostics**：每 replan 记录 `beta_l2 / abs_mean / abs_max / frac≥0.95η / frac≥0.99η / loss_first / loss_last`；
- 测试修复：`projection.weight` 不再与自身比较，改为适配前后逐项比较；补充 Q/K/V+bias、mask reduction、gate B.4 数值参考、cross-fit threshold、缺失 calibration 报错等。

## 4. 验收：`pytest tests/test_salt.py` → **21 passed**

## 5. Phase 1.1 诊断结果（仅流程诊断，不是性能结论）

| 运行 | 成功 | updates | gate_step | β_l2 | loss_first → loss_last |
|---|---|---|---|---|---|
| clean 0,1,2 salt + **feature gate** | 3/3 | **0** | -1（未触发） | 0 | — |
| clean 0,1,2 salt + **always**（消融） | 3/3 | 60–85 | 0 | **4.14–4.16** | 0.0008→0.101（任务1）等，整体上升 |
| camera 648,650,641 salt + feature gate | 0/3 | 0 | -1（未触发） | 0 | — |

**读法**：
1. feature gate 在 clean 上不触发（符合预期）；
2. camera 子集 frozen 本身 0/3，且 gate 未触发 → 该子集不能用于效果判断；
3. **always 消融暴露两个风险**：
   - β 饱和：`‖β‖₂≈4.14–4.16`，理论上限 `0.15√768≈4.157`，`frac≥0.99η≈0.999`；
   - 适配过程中 leftover flow loss **上升**（如 task1 0.0008→0.101），说明固定 lr=2η/K 在 CoRP 上可能过强/不稳定。

## 6. 剩余风险（未解决，需负责人决定）

1. **β 饱和与 loss 上升**：可能是 `lr=2η/K` 在 CoRP 上偏大或每 replan 重复 5 步导致；按任务书**未擅自调参**，只记录。
2. **feature gate 在 camera 子集不触发**：阈值 4.428 来自 clean 校准；camera 起始即扰动，相邻 replan 变化可能仍低于阈值 → 需专门分析 gate score 分布（clean vs camera）。
3. Calibration 细节：token-norm floor 的论文精确形式、tau 训练分布均未在论文中找到逐字定义，已标注为假设。
4. camera 子集 frozen 0/3 → Phase 3 必须换有基线成功的任务。
5. 仍未验证 SALT 在 CoRP 上的**性能收益**（本轮无任何性能结论）。

## 7. 交付

- 代码：`dacsc/salt_tta.py`、`scripts/salt_eval.py`
- 测试：`tests/test_salt.py`（21 passed）
- 校准文件：`/hy-tmp/validation/p11_calib/gate.json`
- commit SHA：见本次提交
