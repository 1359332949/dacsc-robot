# 工单 1 报告：CoRP 上的 SALT 基础复现（Phase 1）

> 范围声明：**SALT adaptation port to CoRP**。CoRP 与 SALT 原文 backbone 不同
> （CoRP 为 fused QKV 的 DiT，1 个 block），本实现是把 SALT 的 key-only SVD 缩放迁移到 CoRP，
> **不声称复现论文原始 backbone 的完全实现**。未实现 SG-SALT、未改 CPIA/Oracle、未改 checkpoint。

## 1. 架构迁移说明

| 项 | SALT 原文 | CoRP | 迁移决定 |
|---|---|---|---|
| Attention | 分离 Q/K/V | `SelfAttention.qkv` = Linear(768,2304) fused | `project_qkv` 按 `view(B,T,3,heads,hd)` 拆分：index 0=Q、1=K、2=V → **K 行 = [768:1536]** |
| 更新参数 | attention key projection | fused qkv 的 K 行 | **只更新 K 行**；Q/V/projection/其余全冻结 |
| 参数化 | `W(β)=U diag(σ⊙e^β) V^T`，β₀=0，|β|∞≤0.15 | 同式，实现为 `W0 + U diag(σ(e^β−1)) V^T` | β=0 时**逐位等于**原权重；β 处梯度非零 |
| 更新 | Adam, K=5 steps, lr=2η/K, 8 draws/step | 同 | 可配置，集中记录 |
| 生命周期 | episode 内保留 β/optimizer，episode 开始 reset | 同 | `SaltTTA.reset()` 每 episode 调用 |

## 2. 公式 ↔ 代码对照

| 论文公式 | 实现位置 |
|---|---|
| `W0 = U diag(σ) V^T` | `dacsc/salt_tta.py` `SaltKeyAdapter.__init__`（float64 SVD） |
| `W(β)=U diag(σ⊙e^β)V^T`，β₀=0 | `SaltKeyAdapter.effective_key_weight`（`W0 + U diag(σ(e^β−1))V^T`，数值等价且零初始化精确） |
| `X_τ=(1−τ)Ỹ+τε`，`v_target=ε−Ỹ` | `flow_interpolate` |
| `L=E‖M⊙(v_θ(X_τ,τ)−v_target)‖²` | `flow_matching_leftover_loss`（归一化动作空间；`M` 仅重叠窗口） |
| leftover 对齐 `chunk[0:H−E] ↔ prev[E:H]` | `build_leftover_target` |
| gate：相邻 replan 的视觉 token 变化，median/MAD 校准，触发后 latch | `FeatureChangeGate` |

## 3. 验收结果（`pytest tests/test_salt.py`：**15 passed**）

| # | 验收项 | 结果 |
|---|---|---|
| 1 | K/Q/V 拆分顺序 | PASS |
| 2 | 只有 K 的 singular-value scales 可训练（其余 requires_grad=False） | PASS |
| 3 | Q、V、output projection 不变 | PASS |
| 4 | β=0 时有效权重与原始权重完全一致 | PASS |
| 5 | β 更新符合奇异值上下界（|β|≤η） | PASS |
| 6 | `sample()` 与 `sample_with_grad()` 同 RNG 下输出逐位相等 | PASS |
| 7 | normalized/unnormalized 转换（含 gripper 维不变） | PASS |
| 8 | leftover `E` 与当前 `0` 严格对齐 | PASS |
| 9 | mask 只监督有效重叠区间（非重叠部分改动不影响 loss） | PASS |
| 10 | Flow τ/ε/velocity 方向正确（公式级） | PASS（注：未做"积分等价"端到端测试） |
| 11 | adapter 参数有梯度、冻结参数无梯度 | PASS |
| 12 | 更新后 prefix cache 重新生成（key 变化） | PASS |
| 13 | gate calibration / trigger / episode latch | PASS（合成数据） |
| 14 | episode reset 清空 β 与 optimizer 状态 | PASS |
| 15 | Frozen 与官方 `policy.predict()` 动作输出一致 | PASS |

## 4. 最小 rollout 日志（不是成功率结论）

**Clean 任务 0,1,2（exec_len=6，`--gate-mode always` 诊断）**

| 模式 | 成功 | updates | 备注 |
|---|---|---|---|
| frozen | 3/3 | 0 | 与 clean 基线一致 |
| salt | 3/3 | 75 | β_norm≈4.16，流程正常 |
| self | 3/3 | 75 | 对照流程正常 |

**Camera 任务 648,650,641（exec_len=6）**

| 模式 | 成功 | updates | 备注 |
|---|---|---|---|
| frozen | 0/3 | 0 | 该子集 frozen 本身即失败 |
| salt | 0/3 | 0 | **feature gate 未触发**（见风险 2） |
| self | 0/3 | 0 | 同上 |

结论：**流程可运行；clean 上未被适配破坏；但 camera 子集无法用于判断 SALT 效果（frozen 全失败），
且 feature gate 在该子集未触发，说明阈值/校准仍不可靠。这些都不是 Phase 2 的依据。**

## 5. 剩余风险（必须在 Phase 2 前解决/记录）

1. **gate 常数未与论文核对**：z=4、MAD floor=1e-3、score clip=5 目前是可配置默认值，
   SALT Appendix B.4 的准确常数/规则尚未逐项确认；校准仅用 2 个 clean episode（28 帧）。
2. **feature gate 在 camera 子集未触发**（updates=0）→ 需要按论文常数重校准，或明确记录为"gate 保守"。
3. camera 子集 648/650/641 的 frozen 本身 0/3，不能作为效果比较任务集；Phase 3 需换成有基线成功的任务。
4. τ/ε 与官方 `sample()` 的"积分等价"只做了公式级测试，端到端等价未单独验证（`sample_with_grad` 已 PASS）。
5. fused-QKV 的 K-only 是架构映射；与论文 backbone 不完全同构。

## 6. 交付物

- 代码：`dacsc/salt_tta.py`、`scripts/salt_eval.py`
- 测试：`tests/test_salt.py`（15 passed）
- 审计：`docs/SG_SALT_CODE_AUDIT.md`
- 本报告：`docs/SALT_PHASE1_REPORT.md`
- commit SHA：见本次提交
