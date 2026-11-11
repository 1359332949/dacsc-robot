# SG-SALT 工单 2.0 / 3.0（按多域评测更新）

> 依据：`docs/SG_SALT_DOMAIN_PLAN.md`（7 扰动因果分类与候选矩阵）、
> `docs/SG_SALT_PHASE1_2_HEALTHCHECK.md`（接口一致性与保护项数值验证）。
> 原则不变：一个创新点；局部验证通过才进入端到端；不允许实现错误进入性能结论。

---

## 工单 2.0：SG-SALT 局部实现（不跑 rollout）

### 目标
在已通过验证的 SALT 基础上，只增加一个保护项：

```
L_SG-SALT = L_SALT + lambda * || A^T (z_phi - z_0) ||_2^2
```

- `A`：`/hy-tmp/validation/interface_stats_spatial_k32_big2.json` 的 `action_subspace`（768×5，正交，rank=5）；
- `z`：`blocks[0].attention.attend` 最后 Euler 步、12-token 均值写回（与 A 构造一致）；
- `z_0`：同一观测、同一 RNG 下 **beta=0（等价 Frozen）** 的写回；
- `z_phi`：当前适配参数的写回；
- 触发条件固定为 **always-on**（首轮不引入 gate 变量）。

### 允许修改
- 新增 `dacsc/sg_salt.py`（复用 `SaltTTA`/`SaltKeyAdapter`，不复制 SALT 主体）；
- 新增/修改 `tests/test_sg_salt.py`；
- 可选：`scripts/salt_eval.py` 增加 `--lambda-protect` 与诊断字段（不改变 SALT 行为）。

### 禁止
- 改 lr / η / adapt steps / gate threshold / token 数；
- 改 CoRP 模型、checkpoint、SALT loss 本身；
- 跑成功率实验；
- 增加其他机制（verifier、状态网络、复杂 gate）。

### 实现要点
1. `z_0` 计算必须在 `torch.no_grad()` 下、adapter `beta=0`、与 `z_phi` 相同的 observation/RNG；
2. `z_phi` 的捕获必须保留计算图（保护项要能反传到 beta）；
3. 保护项归一化：`||A^T dz||^2` 用平方和（不是均值），lambda 默认候选 {0, 0.01, 0.1, 1.0}（仅离线/局部消融，不在 2.0 跑 rollout）；
4. 每次更新记录：SALT loss、penalty、lambda、β 饱和比例。

### 验收（全部 PASS 才允许进入 3.0）
| # | 检查 | 通过条件 |
|---|---|---|
| 1 | A 维度/正交性 | 768×5；`max|A^T A - I| < 1e-6` |
| 2 | λ=0 时 SG-SALT 与 SALT **完全等价** | 同一 batch/RNG 下 loss 逐位相等；β 更新轨迹逐位相等 |
| 3 | 沿 A 方向扰动被检测 | 合成 dz=2q₁ → penalty=4.0（±1e-9） |
| 4 | A 正交补方向不被误判 | 正交 dz → penalty < 1e-12 |
| 5 | 保护项梯度可回传 | beta.grad 非空且非零；冻结参数无梯度 |
| 6 | lambda 增大时 A 方向变化单调下降 | 固定观测/RNG：`||A^T dz||` 随 λ 增大不上升（容差 1e-6） |
| 7 | episode reset 正确 | 重置后 beta=0、optimizer 重建、状态清空 |
| 8 | 诊断字段齐全 | 每次 update 记录 SALT loss / penalty / λ / β 饱和比例 |
| 9 | 不改变 SALT 默认行为 | 不传 `--lambda-protect` 时，SALT 结果与 `30c2241` 逐位一致 |

交付：`SG_SALT_PHASE2_REPORT.md` + 测试结果 + 代码 diff + 未解决风险。

---

## 工单 3.0：多域小规模筛选（Frozen / SALT / SG-SALT）

### 实验条件（不再只看 Camera）

| 类型 | 域 | 套件 | 任务数 | episodes/任务 |
|---|---|---|---|---|
| 主条件 | Camera | spatial, object | 3+3 | 5 |
| 主条件 | Background | spatial, object | 3+3 | 5 |
| 主条件 | Noise | spatial, object | 3+3 | 5 |
| 对照 | Light（高基线） | spatial, object | 3+3 | 5 |
| 边界分析（不进主表） | Robot / Layout / Language | — | — | 仅文档说明 |

- 任务选择（避免选择偏差）：先用**独立的少量 pilot episodes**（独立 seeds，如 seed=101+）估计每个任务
  的冻结成功率，选成功率在 20%–80% 的任务；**正式配对比较使用与 pilot 不重叠的新 seeds**（如 seed=7+）。
  单个 episode 的 frozen 结果不能用来计算 20%–80% 成功率；
- 任务清单、pilot seeds、正式 seeds 在实验前固定并写入配置；
- 触发：**always-on**（静态扰动下 change-gate 不适用，声明为固定触发消融）；
- 方法：Frozen / SALT / SG-SALT（λ 取 2.0 中局部验证选定的一个值，不做大规模超参搜索）。

### 规模
`3 主域 × 2 套件 × 3 任务 × 5 episodes × 3 方法 = 270 rollouts`
（Light 对照另 90；总计 360，分 2–3 批并行）

### 验收 / 判定
| 指标 | 要求 |
|---|---|
| 配对设计 | 同任务/同初始状态/同 seed/同扰动 |
| 逐 episode 记录 | 成功、步数、更新次数、β 饱和比例、耗时 |
| 主指标 | SG-SALT − SALT 的配对成功率差（按域报告，不只报总平均） |
| 统计 | McNemar + 任务级 cluster bootstrap 90% CI |
| 高基线非劣效 | Light 上 SG-SALT 相对 Frozen 的非劣效（±5pp，paired） |
| 跨域声明 | A 来自 Camera/spatial：若只在 Camera 有效，必须写 "Camera-specific" |
| 失败分析 | 每域至少 2 个失败案例记录（gate/β/penalty 状态） |

### 停止/继续条件
- 若 SG-SALT 在 **≥2 个主域**上优于 SALT（配对差>0 且 cluster CI 不含 0 或方向一致），进入 3.1 扩样本；
- 若只在一个域有效 → 结论限定在该域，不扩样本；
- 若所有域都不优于 SALT → 停止 SG-SALT，回到机制分析或换创新点，不再调参；
- 任何情况下不允许为了"看到涨点"而修改 test 集或删除失败样本。

---

## 与前一版工单的差异

| 项 | 旧版 | 更新后 |
|---|---|---|
| 评测域 | 以 Camera 为主 | **Camera + Background + Noise 三个视觉外观域**，Light 对照 |
| 触发 | feature gate | 首轮固定 always-on；gate 另行研究 |
| 创新验收 | 成功率提升 | 先 λ=0 等价、保护项数值、A 方向单调性，再谈成功率 |
| 声明 | 无域限制 | 明确"非观测侧扰动（Robot/Layout/Language）不进入主结论" |
| 规模 | 未定 | 270 主 + 90 对照 rollouts，分批判 |
