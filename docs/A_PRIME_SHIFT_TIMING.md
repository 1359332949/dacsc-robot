# A′ 原位相机切换（Shift Timing）与 Oracle 表达能力测试

## 1. 原位相机切换可行性（已通过 5 项验证）

实现：`scripts/midshift_eval.py`（`--validate`）+ `scripts/camera_shift_probe.py`。
在**同一个 MuJoCo 环境**内直接修改 `model.cam_pos`（相机为固定模式 `cam_mode=0`），
用 `env.env._get_observations(force_update=True)` 强制重新渲染；不重建环境、不迁移状态。

单任务验证（task 0，offset=+0.05m x）：

| 验证项 | 结果 |
|---|---|
| 物理状态（qpos/qvel/act/ctrl/time） | **不变** ✅ |
| 物理动力学（相同 ctrl 下 shift vs sham 的下一步状态） | **一致** ✅ |
| 图像变化 | 平均像素差 **16.5** ✅ |
| 可重复性（sham 重复应用） | 最大差 **0**（早前 probe） ✅ |
| 恢复原相机后图像 | 差 ≈0.007（浮点噪声） ✅ |

关键工程点：
- LIBERO 的 `agentview` 是**固定相机**（40 个相机全部 mode=0），可直接改 `cam_pos/cam_quat`；
- 修改后必须 `sim.forward()` + `_get_observations(force_update=True)`，否则拿到缓存图像（曾因此误判"图像没变"）；
- 物理校验用 `sim.step()`（不渲染）避免 EGL 渲染上下文问题；ctrl 维度 9，不能塞 7 维动作。

## 2. A′ 实验矩阵（进行中）

`scripts/midshift_eval.py` 支持：
- `frozen`：中途切换相机后正常重规划；
- `leftover`：中途切换后**直接执行切前 clean chunk 的剩余动作**（Direct Leftover baseline）；
- `cpia`：用切前 leftover 做一次接口自适应后重规划；
- `--sham`：同样的代码路径、零位姿偏移（控制渲染刷新/时间切换的影响）。

10 任务 × {frozen, leftover, cpia, sham} 已启动（每条约 1–2 分钟）。

## 3. Oracle 表达能力测试（LOOCV，20 任务，293 对样本）

CPIA 的形式是**共享低秩乘性**变换 `w' = w + (w V) Uᵀ`，
而 Oracle 是**逐状态加性**修正 `w' = w − Δw`。两者是否同一假设空间？

| 模型 | held-out cosine 均值 |
|---|---|
| Oracle additive | **1.000**（上界） |
| 全局均值偏移 | 0.145 |
| 步数条件化加性 | 0.131 |
| 低秩乘性 r=1 | 0.398 |
| 低秩乘性 r=2 | **0.533** |
| 低秩乘性 r=4 | 0.649 |
| 低秩乘性 r=8 | **0.692** |
| 低秩乘性 r=16 | 0.700 |
| 低秩乘性 r=32 | 0.706 |
| 低秩乘性 r=64 | 0.708 |

**解读**：
1. rank-2（我们实际使用的容量）只能表达 Oracle 修正方向的 **~53%**（cosine）；
2. rank 增大到 64 仍**停在 ~71%** → **共享乘性结构存在表达天花板**（Oracle 偏移是状态相关的）；
3. 因此 CPIA≈0 不能全部归因于"信号不足"，也有一部分是**参数化表达能力不足**；
4. 若要逼近 Oracle，需要**状态条件化的加性/乘性**参数化（如 `Δ = g(state)`），而不是共享 `UVᵀ`。

## 4. 对下一阶段的意义

| 观察 | 含义 |
|---|---|
| Oracle +33pp（60 任务，p=1e-5） | 接口方向存在、可纠正 |
| 共享低秩乘性上限 ~71% cosine | CPIA 参数化不足以复现 Oracle |
| rank-2 仅 53% | 我们此前主实验的容量偏小 |
| 全局/步数条件化加性 ≤0.15 | 简单无状态加性也不行 |

**建议**：下一步优先做"状态条件化加性校准"（用小网络或线性模型 `Δ(state)`），
并在 A′ 的 mid-shift 数据上验证；同时保留 Oracle 作为上界。
