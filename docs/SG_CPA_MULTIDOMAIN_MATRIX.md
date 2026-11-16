# 多视觉域实验矩阵与 Phase-A 数据验证

日期：2026-11-16  
状态：**矩阵骨架 + 4 个 Phase-A 单元已通过构造审计；未批量采集；未写新 Gate；未跑 rollout。**

## 1. 研究定位

> **Visual Sensitivity-Guided Test-Time Adaptation for VLA under Heterogeneous Domain Shifts.**

核心问题不是“找与任务无关的有害子空间”，而是：

> 给定任务、状态、域扰动和冻结策略，预测某个视觉特征干预是否带来**闭环条件收益**：
> \[
> \tau_{i,d,m}=\mathbb E_\epsilon[Y^{repair}_{i,d,m}-Y^{frozen}_{i,d}]
> \]

视觉敏感度和动作传播是机制证据，不能替代闭环收益。

## 2. 域分类

| 类别 | 第一轮 |
|---|---|
| Background / Appearance | 核心 |
| Lighting | 核心 |
| Sensor / Image Quality | 核心；第一轮限定 agentview motion-blur 子集 |
| Camera / View Geometry | 核心 |
| Object Layout | 后续 |
| Robot Initial State | 辅助评价 |
| Language | 暂不纳入 |

## 3. 矩阵结构

- 4 个 base task：
  - `libero_spatial` cookie-box（dev task）
  - `libero_object` alphabet soup（dev task）
  - `libero_goal` cabinet drawer（held-out task）
  - `libero_10` KITCHEN_SCENE3 moka-pot（held-out task）
- 4 个视觉域 × 每任务每域 2 个变体
- 共 **32 个 Clean/Shift 配对单元**
- 每单元状态划分：seed=1234，30 dev / 20 held-out
- 实际 `dev_indices` / `heldout_indices`、init 文件 SHA256 已写入矩阵 JSON。
- 变体选择：Sensor 固定为 motion-blur 5/10；Camera 排除已用于历史 Camera pilot 的 705；其余仍为 provisional，Background/Lighting 的参数编号不代表真实视觉距离，正式冻结前需人工确认。
- 复合键：`suite:classification_id:variant_name`，classification id 不单独当全局键。
- 状态划分的 `seed`、`dev_indices`、`heldout_indices`、init 文件 SHA256 已写入 JSON。

角色：

| 维度 | 角色 |
|---|---|
| spatial/object | dev_task |
| goal/libero_10 | heldout_task |
| Background/Lighting 第 1 变体 | dev_domain |
| Background/Lighting 第 2 变体 | heldout_same_domain |
| Sensor/Camera | heldout_unseen_category |

`cell_role = base_role + variant_role`。  
held-out task 的状态永远不能用于调参，即使它同时属于 dev_domain。

机器可读：`docs/data/multidomain_matrix.json`。

## 4. 上游实现审计

已核对 `env_wrapper.py`、`benchmark/__init__.py`、`libero_tabletop_manipulation.py`：

| 域 | 变体形式 | 数据路径 |
|---|---|---|
| Background | `_table_*` | 实际 variant BDDL |
| Lighting | `_light_*` | 实际 variant BDDL |
| Camera | `_view_*` virtual | env 从 name 解析相机参数，加载 base BDDL |
| Sensor | `_view_*_noise_*` virtual | env 解析 noise；**只污染 agentview_image** |

关键结论：

1. `set_init_state()` / `regenerate_obs_from_state()` **不会施加 Sensor Noise**；
2. `ControlEnv.step()/reset()` 才施加 noise；
3. 因此 Sensor 必须在恢复状态后显式调用 `apply_agentview_noise()`，不能直接 `sim.render()`，也不能为了噪声而 step 物理；
4. Sensor 第一轮只覆盖 `noise<=10` 的 motion-blur 子集；
5. Camera 在同一 env 内修改相机参数，避免多 env 静态物体 placement 差异；
6. 静态物体必须比较 body_xpos/xquat，不能只比 qpos；
7. 正式 success 的统计单位是 episode，不是 frame/state。

## 5. Phase-A 四域单元审计

每个域选 1 个 cell、2 个初始状态，检查：

- 状态恢复；
- 语言一致；
- 目标/控制视角图像变化；
- Camera 参数变化；
- Sensor 噪声机制和污染视角。

结果（`docs/data/phase_a_cell_audit.json`）：

| 域 | cell | 状态一致 | 语言一致 | agentview Δ | eye-in-hand Δ | 备注 |
|---|---|---:|---:|---:|---:|---|
| Background | 146 | ✅ | ✅ | 0.206–0.207 | 0.286 | 两视角都变 |
| Lighting | 2292 | ✅ | ✅ | 0.073 | 0.051 | 两视角都变 |
| Sensor | 1476 | ✅ | ✅ | 0.028–0.030 | 0.000 | motion_blur(5)，只污染 agentview |
| Camera | 704 | ✅ | ✅ | 0.127–0.129 | 0.000 | agentview 相机参数变，非目标相机不变 |

四个 cell 的 `model_compatible` 均为 True（严格 body/geometry；Camera/Sensor 同时 strict physics）。

结论：

> 四个域的数据构造路径已打通；可以进入 Phase-A 退化筛查，但仍不能批量采集 32 个单元。

### 5.1 E0 清理

- Camera 705 已从矩阵排除，标记为历史 Camera slice；
- Sensor 固定为 motion_blur 5/10，只污染 agentview；
- Noise helper 修复 `noise=0` 和全局 RNG 状态恢复；
- 状态审计增加 `model_compatible` 检查；
- Background/Lighting 变体选择仍标记 provisional，正式采集前需要人工按实际视觉差异复核。

### 5.2 E1 Frozen 退化筛查结果

**已完成：360 / 360 episodes，20 个 worker 并行，无 Traceback。**

配置：4 tasks × (1 Clean + 8 Shift) × 10 episodes；BF16；同一 init index 和 episode seed 配对；只跑 Frozen CoRP。

Clean baseline：

| task | Clean success |
|---|---:|
| spatial cookie-box | 10/10 |
| object alphabet soup | 9/10 |
| goal cabinet | 10/10 |
| libero_10 moka-pot | 9/10 |

Shift 结果（10 episodes/cell，筛查级）：

| task | domain | variant successes | Δ vs paired clean |
|---|---|---|---:|
| spatial | Background | 8/10, 9/10 | -0.2, -0.1 |
| spatial | Lighting | 9/10, 7/10 | -0.1, -0.3 |
| spatial | Sensor | 10/10, 6/10 | 0.0, -0.4 |
| spatial | Camera | 5/10, 7/10 | -0.5, -0.3 |
| object | Background | 6/10, 7/10 | -0.3, -0.2 |
| object | Lighting | **0/10**, 10/10 | **-0.9**, +0.1 |
| object | Sensor | 10/10, **1/10** | +0.1, **-0.8** |
| object | Camera | 10/10, 10/10 | +0.1, +0.1 |
| goal | Background | **0/10, 0/10** | **-1.0, -1.0** |
| goal | Lighting | 9/10, 9/10 | -0.1, -0.1 |
| goal | Sensor | 10/10, **0/10** | 0.0, **-1.0** |
| goal | Camera | 7/10, 4/10 | -0.3, -0.6 |
| libero_10 | Background | 1/10, 2/10 | -0.8, -0.7 |
| libero_10 | Lighting | **0/10**, 2/10 | **-0.9, -0.7** |
| libero_10 | Sensor | 9/10, **0/10** | 0.0, **-0.9** |
| libero_10 | Camera | 10/10, 2/10 | +0.1, -0.7 |

筛查结论：

- **最稳定的退化候选**：goal + Background；libero_10 + Background/Lighting；object + Lighting(2222)；Sensor noise=10 在多个任务上严重退化。
- **Camera 不一致**：spatial/goal/libero_10 有明显退化，object Camera 两个变体都 1.0；不能把 Camera 当统一退化域。
- **Sensor noise=5 多数不退化**，noise=10 才明显；Sensor 只污染 agentview，wrist 仍正常。
- Clean baseline 接近 1.0，存在明显天花板效应；10 episodes 只能筛查。

统计限制：

- 10 episodes/cell；
- 配对 bootstrap 的 95% 区间很宽，部分为 [-1,-1] 或 [0,0]；
- 不能据此宣称显著性或泛化；
- 这些 cell 是开发/筛查证据，最终确认必须换独立 episode seeds 和未用于选择的变体。

## 6. 阶段顺序（已修正）

1. **Phase A：数据与退化验证**
   - 四域单元构造（当前 4 个已过）；
   - 多域 Frozen CoRP 退化筛查；
   - 10 episode 初始筛查，不作为确认统计。
2. **Phase B：配对修复可行性**
   - Visual-subspace repair；
   - Action-aware repair；
   - Matched random / Reverse-sign / Clean-difference oracle；
   - Dev 选择 α，held-out 验证；
   - 动作指标拆 xyz/rot/gripper。
3. **Phase C：闭环修复确认**
   - 独立 episode seeds；
   - 配对成功率差、置信区间、Clean retention；
   - 统计单位 episode。
4. **Phase D：Target-only 收益预测器**
   - 只允许目标域无标签观测；
   - 在 held-out same-domain / unseen-category / held-out-task 上验证；
   - 只有 Phase B/C 证明有稳定修复收益后才做。

## 7. 下一批工作

不批量采集 32 单元。先做：

1. 每个域再验证 1–2 个 cell（不同 task / 不同 variant）；
2. Sensor 的 `apply_agentview_noise` 固定 seed 重复性测试；
3. 多域 Frozen CoRP 退化筛查（10 episodes，独立 seed）；
4. 退化筛查结果出来后再决定是否进入 Phase B。

## 8. 不做什么

- 不把 Camera 705 当主线；
- 不把 `s_aug` 当核心算法；
- 不在 proxy 未通过前启动 V2；
- 不在 held-out task 上调任何超参数；
- 不把 frame/state 数量当 episode 样本量。
