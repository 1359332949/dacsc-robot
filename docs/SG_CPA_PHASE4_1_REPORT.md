# 工单 4.1 报告：SG-CPA 局部实现与单元测试

> 只做局部实现 + 单元测试：**不跑机器人 rollout、不改 Phase 3 数据、不调超参**。
> 交付：`dacsc/sg_cpa.py`、`tests/test_sg_cpa.py`、本报告。
> 测试结果：`pytest tests/test_salt.py tests/test_sg_salt.py tests/test_sg_cpa.py` → **42 passed**（21 + 11 + 10）。

## 1. 冻结配置（按 4.0b/审查决定实现）

| 项 | 值 |
|---|---|
| K / views | 4 = identity + 3 个独立 Gaussian-noise 视图 |
| Noise σ | **0.05**（像素空间 [0,1]，先反归一化再加噪、再归一化） |
| 聚合 | Anchored Mean：`A_α = A_id + α(mean(A_1..A_3) − A_id)`，**α = 0.1** |
| Loss | 动作块直接蒸馏（归一化动作空间，3 个增强视平均） |
| 参数化 | K-projection SVD scaling `β`（复用 `SaltKeyAdapter`） |
| 更新 | **Projected SGD**：`Δβ = −η P_null g`，η=0.05 |
| A | 真实 `interface_stats_spatial_k32_big2.json`（768×5） |
| J_A | 在**原始 identity 输入**的写回 `z(o_t, β)` 上计算（非增强视图） |
| 有限步检查 | 更新后重测 `‖Aᵀ(z'−z0)‖`，>tol 则步长减半，最多 3 次，仍超则回退 |
| Rollout | **禁止**（本工单未跑） |

## 2. 实现要点

- `SGCpa` 冻结全部 CoRP 参数，只训练 `adapter.beta`；
- `build_target()`：**先临时把 β 置 0** 得到 frozen teacher 的 `A_id` 与 3 个加噪视图动作，再恢复 β；
- 每个 camera 视图独立噪声：`seed + camera_index*1000`（避免两路相机共用同一噪声模式）；
- `write()`：最后 Euler 步、token 均值写回，保留计算图；
- `jacobian_a()`：对 5 个 A 基向量做 VJP，得到 `J_A ∈ R^{5×768}`；
- `nullspace()`：`P = I − J_Aᵀ(J_A J_Aᵀ)†J_A`（SVD 伪逆）；
- `step()`：teacher 目标 → 蒸馏 loss → ∇β → J_A → P_null → Δβ → 有限步校验/回退 → 更新。

## 3. 十项验收测试（全部 PASS）

| # | 要求 | 测试 | 结果 |
|---|---|---|---|
| 1 | Pixel normalization round-trip | `test_pixel_roundtrip` | `max|from_pixel(to_pixel(x))−x| < 1e-5` |
| 2 | 三视图确定性与独立噪声 | `test_noise_views_deterministic_and_independent` | 同 seed 逐位相同；不同 seed 不同；同 seed 下两路相机噪声不同 |
| 3 | Teacher/Student 独立前向 K=1 no-op | `test_teacher_student_noop` | **bitwise equal**；loss=0；β.grad=0；所有策略参数无梯度 |
| 4 | Anchored Mean 公式与 α 边界 | `test_anchored_mean_formula` | α=0 等于 A_id；α=1 等于 aug_mean（1e-6）；α=0.1 公式一致 |
| 5 | 直接蒸馏 loss 梯度正确性 | `test_distill_gradient_correctness` | 解析梯度 vs 有限差分（eps=1e-4）相对误差 ≤5% |
| 6 | 真实 A 的 Jacobian 与 SVD 投影 | `test_real_jacobian_and_nullspace` | J_A [5,768]；`‖J_A P‖/‖P‖ < 1e-10` |
| 7 | 投影 SGD 一阶零空间性质 | `test_projected_sgd_nullspace` | `‖J_A P g‖ < 1e-9·‖P g‖` |
| 8 | 有限步 A 变化检测与回退 | `test_finite_step_rollback` | 强制 tol=1e-15 → `accepted=False`、β 回退到 0 |
| 9 | Episode reset 与 RNG 配对 | `test_reset_and_rng_pairing` | step 后 β≠0；reset 后 β=0；同 seed target 逐位相同；不同 seed 不同 |
| 10 | 冻结参数无梯度、只有 β 更新 | `test_only_beta_changes` | 所有策略参数更新前后逐位相同；requires_grad=False；optimizer 只含 β |

## 4. 对审查意见的落实

| 审查问题 | 处理 |
|---|---|
| K=1 no-op 自我比较 | 改为**独立 teacher 前向 + 独立 student 前向**，验证 bitwise equal、loss=0、grad=0 |
| 投影验证与最终配置不一致 | `test_real_jacobian_*` 使用最终配置（noise 0.05、K=4、α=0.1），J_A 在 identity 输入上计算 |
| K 敏感性变量混杂 | 4.1 不再做 K 消融；在报告中注明：未来 K 消融必须固定单一变换族、只增加 seed 数 |
| D_fit 不是独立正确性证据 | 报告与设计文档明确：D_fit 只证明"锚定系数在保真与拟合间权衡"，不证明伪标签正确 |
| 两路相机噪声共用 seed | 已改为 `seed + camera_index*1000`，并有测试覆盖 |
| 缺少独立 pytest 文件 | 新增 `tests/test_sg_cpa.py`（10 tests），非实验脚本内嵌断言 |

## 5. 已知局限

- 单步成本：teacher 目标（4 次采样）+ 3 个增强视图 student 采样 + identity 写回 + 5 次 VJP + 有限步校验；
  实际部署需测单步耗时（4.1 未计时）。
- 信号量级：4.0b 显示 consensus 拟合信号约 1e-3 量级；能否改变成功率未知。
- 未验证：多 episode 稳定性、与执行循环的集成、β 饱和行为。
- **无任何成功率结论**（未跑 rollout）。

## 6. 结论与下一步建议

4.1 十项测试全部通过，方法三个部分（Identity-Anchored Counterfactual Consensus、直接动作蒸馏、A-nullspace projected update）
已按公式实现并可单元验证。**建议批准 4.2 小规模筛选设计**（Frozen / SALT / SG-CPA 配对），但**不自动启动 rollout**；
启动前需你确认实验规模、任务与停止条件。

_commit SHA：见本次提交。_
