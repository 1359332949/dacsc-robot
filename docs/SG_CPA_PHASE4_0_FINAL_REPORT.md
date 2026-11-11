# 工单 4.0（最终修订）离线验证报告

> 按审查意见完成：Medoid 主目标、4 视图（Identity + Noise + Blur + Photo）、
> 分组距离、真实 loss 的 K=1 no-op、Jacobian 有限差分、诊断修复与上游校验。
> 数据：4 个扰动域（Camera/Background/Noise/Light）× 2 套件 × 2 任务 × 4 chunks = **64 个观测**。
> 原始 JSON：`/hy-tmp/validation/sg_cpa_4_0_final/sg_cpa_4_0_final.json`；脚本 `scripts/sg_cpa_phase4_final.py`。

## 1. 最终协议（冻结）

| 项 | 定义 |
|---|---|
| 视图 | Identity、Gaussian noise σ=0.05、Gaussian blur k=3 σ=0.5、photometric 1.05（均在像素空间 [0,1] 加噪后再归一化） |
| 教师 | 4 视图 **共享同一动作噪声 seed** 采样 |
| 目标 | **Medoid**：`k* = argmin_k Σ_j d(A_k,A_j)`，`Ā⁰ = A_{k*}`（Mean/Anchored Mean 为消融） |
| 损失 | 直接动作块蒸馏 `L = (1/K) Σ_k ||Â_k^β − sg(Ā⁰)||²` |
| 距离 | 归一化动作空间；分组报告 XYZ / Rotation / **Gripper（原生 ±1 编码，连续比较）** |
| 更新 | 投影 SGD `Δβ = −η P_null g`，`P_null = I − V_r V_rᵀ`（对 J_A 直接 SVD） |
| J_A | 在**原始 Identity 输入**的写回 `z(o_t,β)` 上计算，r=5 VJPs |
| 参考 | 有限步检查对比**同一观测**下 β=0 写回（避免把观测变化误算成 A 漂移） |
| 边界 | |β|≤0.15，通过整体步长缩放（不逐维 clamp） |

## 2. 跨域离线证据（64 观测）

| 域 | D_view | Medoid 锚点偏移 | Mean(α=0.1) 锚点偏移 | **Medoid=Identity 比例** | β=0 蒸馏 loss |
|---|---|---|---|---|---|
| Camera | 0.0177 | 0.00086 | 0.00089 | **81.2%** | 4.1e-4 |
| Background | 0.0224 | 0.00120 | 0.00109 | **81.2%** | 7.6e-4 |
| Noise | 0.0289 | 0.00125 | 0.00177 | **75.0%** | 2.1e-3 |
| Light | 0.0223 | 0.00084 | 0.00130 | **75.0%** | 6.5e-4 |

分组偏移（medoid 相对 identity）：以 XYZ 为主（约 Rotation 的 1.5–2 倍），gripper 最小。

### 关键发现（决定性）

> **在 4 视图（含 Identity）上，Medoid 有 75%–81% 的概率就是 Identity 视图本身。**
> 这意味着大多数时刻教师目标 = 冻结策略在原始输入上的动作，**蒸馏信号接近 no-op**；
> medoid 目标的平均动作偏移只有 ~0.001（动作尺度 ~0.3 的 0.3%）。
> 这不是实现错误，而是"identity 是中心点"的几何后果（3 个轻度增强都围绕 identity 扰动）。

**结论**：按批准的"含 identity 的 medoid"协议，SG-CPA 的自监督信号很弱；正式实验必须把它当作主协议执行并如实报告，
同时用 **medoid-excluding-identity** 作为诊断变体（不是调参，而是检验"identity 主导"这一机制假设）。

## 3. K 敏感性（固定单一噪声族，identity + noise seeds 1..7）

| K | D_view | Medoid 锚点偏移 |
|---|---|---|
| 2 | 0.0348 | 0.0000（两视图平局退化） |
| 4 | 0.0294 | 0.0263 |
| 8 | 0.0271 | 0.0286 |

- K↑ → 视图间分歧小幅下降（0.0348→0.0271），说明更多视图让"共识"更集中；
- K=2 的 medoid 平局退化，不能作为证据；
- **D_view 下降只说明预测更一致，不证明动作更正确**（无专家动作参照）。

## 4. 真实 loss 的 K=1 no-op

| 项 | 结果 |
|---|---|
| `SGCpa.loss()`（identity-only 视图，β=0，共享噪声） | **loss = 0.0，grad = 0.0** |

## 5. 直接蒸馏 vs Flow Matching（K=4，同一教师目标）

| 目标 | loss (β=0) |
|---|---|
| 直接动作蒸馏 | **2.07e-4** |
| Flow Matching 一致性 | **1.35e-2**（约 65×） |

→ 直接蒸馏是首版主目标；Flow 版本不满足 K=1 no-op（此前已证），仅作离线对照。

## 6. Jacobian / A 基底 / 投影

| 检查 | 结果 |
|---|---|
| J_A shape | [5, 768] |
| 奇异值 | 0.986, 0.520, 0.381, 0.104, 0.087 |
| A 正交性误差 | 7.94e-09 |
| 投影零空间残差 `‖J_A P‖/‖P‖` | **5.3e-17** |
| **有限差分 vs VJP 相对误差** | h=1e-2: **2.5e-4**；h=1e-3: **4.3e-3** |

→ A 基底、hook 写回位置、Jacobian 在同一坐标系，VJP 数值正确。

## 7. 成本（单次，64 观测平均）

| 阶段 | 秒 |
|---|---|
| 教师 4 视图采样 | 0.080 |
| 学生 4 视图 loss | 0.075 |
| J_A（5 VJPs + 写回） | 0.104 |
| SVD + 零空间 | 0.007 |

总离线验证 135 s（含 64 观测采集 103 s）。

## 8. 诊断与工程修复

- β 饱和阈值改为 `adapter.eta`（0.15）；
- 记录 `attempted / accepted / rejected / skipped`；
- `D_A` 命名改为**当前观测**下的 `d_a_local` / `d_a_current_obs_total`（不再称 episode 累计）；
- 启动入口调用 `verify_upstream(["CoRP"])` + checkpoint 校验；
- 视图变换全部在像素空间，round-trip 测试 <1e-5；
- 每路相机独立噪声 seed；RNG 用 `fork_rng` 隔离；
- 每观测固定 β=0 参考；SVD 零空间；β 全局步长缩放。

## 9. 结论与下一步

1. 4.0 最终协议已实现并通过离线验证（17 项单元测试 + 本报告数据）；
2. **Medoid 含 identity 时 75–81% 选中 identity，教师信号很弱**——这是正式实验必须检验的核心机制问题；
3. 决策：按批准协议（medoid 含 identity）跑 **Frozen / SALT / SG-CPA / SG-CPA w/o projection** 多域对比，
   另加 **medoid-excluding-identity 诊断**（只用于机制分析）；
4. 不根据结果调参；不声称 medoid 更"正确"。

_commit SHA 见本次提交。_
