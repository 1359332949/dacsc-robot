# 工单 2.0 报告：SG-SALT 局部实现（无 rollout）

> 基线：`30c2241`。**已用 git merge-base 验证 `30c2241` 包含 `7c5452a` 的 Flow Matching 方向与 loss reduction 修复**，
> 因此 30c2241 可以作为正确的 SALT 对照基线。
> 本工单只实现保护项与单元测试；**未跑 rollout、未改 lr/η/gate/token、未增加其他机制**。

## 1. 实现

新增 `dacsc/sg_salt.py`：

```
L_SG-SALT = L_SALT + lambda * || A^T (z_phi - z_0) ||_2^2
```

- `A`：`interface_stats_spatial_k32_big2.json` 的 `action_subspace`（768×5，正交，rank=5）；
- `z`：`blocks[0].attention.attend` 最后 Euler 步、12-token 均值写回（与 A 构造一致）；
- `z_0`：**detach、no_grad、beta=0**、与 `z_phi` 相同 observation / 采样 seed（`fork_rng` 隔离，不扰动全局 RNG）；
- `z_phi`：保留计算图的适配写回，梯度可回传到 β；
- **λ=0 显式分支**：直接返回 `L_SALT`，不做任何额外采样、不消耗 RNG：

```python
if lam == 0.0 or a_basis is None:
    salt = flow_matching_leftover_loss(...)
    return salt, salt.detach(), None
```

## 2. 验收结果（9/9 PASS，整体 30 passed）

| # | 检查 | 结果 | 证据 |
|---|---|---|---|
| 1 | A 维度/正交性 | PASS | 768×5；`max|AᵀA−I|=7.9e-9` |
| 2 | **λ=0 与 SALT 逐位等价** | PASS | loss `torch.equal`；β 3 步轨迹逐位相等；Adam state（exp_avg/exp_avg_sq/step）逐位相等 |
| 3 | 沿 A 方向被检测 | PASS | dz=2q₁ → penalty=4.0000（±1e-9） |
| 4 | 正交补不被误判 | PASS | penalty < 1e-10 |
| 5 | 保护项梯度方向正确 | PASS | 解析梯度 `2·2·q₁` 一致；正交方向梯度 <1e-5 |
| 6 | λ 对 A 方向变化的影响（**记录趋势，不强制单调**） | PASS | 见下表 |
| 7 | episode reset | PASS | β 清零、optimizer 重建 |
| 8 | 诊断字段 | PASS | 返回 `(total, salt, penalty)`；λ=0 时 penalty=None |
| 9 | 不改变 SALT 默认行为 | PASS | λ=0 时 loss/β/optimizer 状态与 SALT 逐位一致；无额外 RNG 消耗 |

### 第 6 项趋势记录（固定观测/RNG，3 步更新，λ 扫描）

| λ | ‖Aᵀ(z_φ−z_0)‖² |
|---|---|
| 0.0 | 8.159e-4 |
| 0.01 | 8.156e-4 |
| 0.1 | 8.128e-4 |
| 1.0 | **7.855e-4** |

趋势方向符合设计（λ↑ → A 方向变化↓），但按你的要求**只记录不强制严格单调**；
单元测试层面已单独验证"保护项梯度确实惩罚 A 方向变化"。

## 3. 其他验证

- 全局 RNG：λ=0 分支调用前后 `torch.get_rng_state()` 逐位相同；
- 冻结参数：`all(p.grad is None for p in policy.parameters())`；
- 写回维度：z0/z1 均为 (768,)；β=0 与 β=0.1 的写回不同（保护项有输入信号）。

## 4. 成本与局限

- **额外开销**：λ>0 时每次 update 需额外 2 次 10-step 采样（z0 与 z_φ）→ 适配成本约 +2×（仅时域计算，无额外前向缓存）；
- `protect_seed` 固定为 1234（同一 update 内 z0/zφ 用同一噪声起点，保证可比）；
- A 来自 Camera/spatial 的接口统计（跨域使用同一 A 是 3.0 中的可检验点）；
- β 饱和现象未改变（本工单不允许调参），3.0 中仍需记录饱和比例；
- **没有任何性能结论**。

## 5. 交付

- 代码：`dacsc/sg_salt.py`
- 测试：`tests/test_sg_salt.py`（9 passed）；连同 `tests/test_salt.py` 共 **30 passed**
- 修改文件：`dacsc/sg_salt.py`、`tests/test_sg_salt.py`
- commit SHA：见本次提交
- 未解决：跨域 A 适用性、protect sampling 成本、β 饱和、无 rollout（下一阶段）

## 6. 建议

按工单 2.0 验收结论：**允许进入工单 3.0 小规模多域筛选（Camera / Background / Noise，Light 对照）**；
但进入前需按你的意见把 3.0 的任务筛选改为"独立 pilot episodes 选任务 + 正式比较用新的配对 seeds"。
