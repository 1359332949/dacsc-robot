# 多视觉域实验矩阵（冻结稿）

日期：2026-11-16  
状态：**仅建立实验矩阵；未写新 Gate，未跑新 rollout。**

## 1. 研究定位

正式定位：

> **Visual Sensitivity-Guided Test-Time Adaptation for Vision-Language-Action Policies under Heterogeneous Domain Shifts.**

研究目标不是 Camera adaptation，而是：

> 冻结 CoRP 面对异质视觉目标域时，能否用目标域无标签观测估计有害视觉敏感方向，并做选择性校正，提高 OOD 任务成功率且保持 Clean。

Camera 只是第一个受控切片。

## 2. 域分类

| 类别 | 是否主研究 | 说明 |
|---|---|---|
| Background / Appearance | 核心 | 背景、桌面纹理、场景主题 |
| Lighting | 核心 | 亮度、颜色、方向光、阴影 |
| Sensor / Image Quality | 核心 | 噪声、模糊、退化 |
| Camera / View Geometry | 核心 | 位置、角度、FOV |
| Object Layout / Scene Content | 单独阶段 | 任务相关空间状态，不能当 nuisance |
| Robot Initial State | 辅助评价 | 状态域，不是纯视觉 |
| Language | 暂不纳入主线 | 跨模态扰动，另设协议 |

第一轮矩阵只包含四类视觉域；Object Layout、Robot State、Language 不混入。

## 3. 矩阵结构

- Base tasks：4 个
  - `libero_spatial` / cookie-box（dev task）
  - `libero_object` / alphabet soup（dev task）
  - `libero_goal` / cabinet drawer（held-out task）
  - `libero_10` / KITCHEN_SCENE3 moka-pot（held-out task）
- Domains：Background、Lighting、Sensor、Camera
- Variants：每个 base task × 每个 domain 选 2 个变体
- Pairs：32 个 Clean/Shift 配对单元
- State split：每单元 30 dev / 20 held-out（固定 seed，episode/state 级）
- 不把同一 episode 连续帧拆到不同 split。

### 3.1 角色

| 维度 | 角色 |
|---|---|
| base task | spatial/object = dev_task；goal = heldout_task |
| Background / Lighting variant | 第 1 个 = dev_domain；第 2 个 = heldout_same_domain |
| Sensor / Camera | 全部 = heldout_unseen_category |
| state | 30 dev / 20 heldout，所有单元共用固定划分规则 |

`cell_role = base_role + variant_role`，不是所有交叉单元都独立。  
正式报告要明确每个对比属于哪一种泛化：

1. 同域新实例：Background/Lighting 的 heldout_same_domain；
2. 未见域类别：Sensor/Camera；
3. 未见任务：goal 的 heldout_task 单元。

## 4. 实际选中的变体

完整机器可读清单：`docs/data/multidomain_matrix.json`。

| Base task | Domain | Dev variant id | Held-out variant id |
|---|---|---:|---:|
| spatial cookie-box | Background | 138 | 139 |
| spatial cookie-box | Lighting | 2292 | 2293 |
| spatial cookie-box | Sensor | — | 1476 / 1477 |
| spatial cookie-box | Camera | — | 687 / 688 |
| object soup | Background | 1 | 2 |
| object soup | Lighting | 2222 | 2223 |
| object soup | Sensor | — | 1397 / 1398 |
| object soup | Camera | — | 647 / 648 |
| goal cabinet | Background | 1 | 2 |
| goal cabinet | Lighting | 2313 | 2314 |
| goal cabinet | Sensor | — | 1509 / 1510 |
| goal cabinet | Camera | — | 691 / 692 |
| long kitchen | Background | 1 | 2 |
| long kitchen | Lighting | 2246 | 2247 |
| long kitchen | Sensor | — | 1485 / 1486 |
| long kitchen | Camera | — | 683 / 684 |

注意：classification id 是 **1-based**；benchmark/task-map index 是 0-based。  
没有把 Camera 705 直接混入这个矩阵；Camera 705 的既有结果是独立 camera slice，保留为历史对照。

## 5. 配对协议

每个单元：

1. 固定 base task、instruction、物理初始状态；
2. Clean 观测用 base BDDL；
3. Shift 观测用变体 virtual BDDL 名（`_table_*`、`_light_*`、`_view_*`、`_noise_*`），由 `OffScreenRenderEnv` 解析；
4. 同一物理状态 restore 后渲染；
5. 保存 Clean/Shift 双视角图像、DINO patch features、8D policy state、state digest、camera params；
6. Clean/Shift 配对只用于离线 Oracle，不进入任何 proxy 估计器。

## 6. 分阶段执行

### Phase A：矩阵与数据构造（当前）
- 矩阵元数据；
- 配对合法性；
- 数据分布审计；
- **不写 Gate**。

### Phase B：多域敏感度地图
对每个域、任务、状态计算：

- signed feature displacement；
- `s_pair` 与 low-rank covariance；
- 跨状态稳定性；
- 跨域共享/特异方向；
- agentview / eye-in-hand 差异；
- 动作传播（仅推理，不 rollout）。

### Phase C：Target-only Estimator
只允许目标域无标签观测；额外结构需预先声明：

- 多视角一致性；
- 相机几何（若部署可用）；
- 时间连续性；
- 冻结源模型参考。

必须在：
- 同域新变体；
- 未见域类别；
- 未见任务；
上分别验证。

### Phase D：选择性适应与完整任务评价
- Channel suppression；
- Feature-space projection；
- Affine modulation（含 FTM-style baseline）；
- 完整 success + Clean retention；
- 组合域测试。

## 7. 不做什么

- 不继续增加 photo/crop/zoom 代理；
- 不在第一阶段混入 Object Layout / Robot State / Language；
- 不把 Camera 单域结论推广到所有视觉域；
- 不在 proxy 未通过前启动 V2；
- 不根据 held-out Clean 结果调 Gate。


## 8. 上游源码审计结论（执行前）

已核对 `LIBERO-plus` 的 `env_wrapper.py`、`benchmark/__init__.py`、`envs/problems/libero_tabletop_manipulation.py`：

1. **Background / Lighting**
   - 变体是真实 BDDL 文件（如 `_table_11.bddl`、`_light_1.bddl`）；
   - 使用实际 variant BDDL 建 env。

2. **Camera**
   - 变体 BDDL 文件不存在，名字是 virtual `_view_<params>_initstate_0`；
   - `OffScreenRenderEnv` 会 split `_view_` 并加载 base BDDL；
   - 同一 env 内修改相机参数比多 env 更可靠（多 env 的静态物体 placement 可能不同）。

3. **Sensor Noise**
   - 名字是 `_view_..._initstate_0_noise_N`，属于 virtual；
   - **noise 在 `ControlEnv.step()` 中施加到 observation**；
   - 直接调用 `sim.render()` 会绕过 noise，因此 Sensor 单元必须使用 `env.step` 返回的观测，或复现同样的 noise 函数，不能复用 Camera 的 render 路径。

4. **Init states**
   - `Benchmark.get_task_init_states()` 会把 variant 映射回 base init state；
   - classification id 是 **1-based**，benchmark/task-map index 是 0-based；
   - 同一 base task 的 Clean/Shift 必须共享同一个 base init state。

5. **静态物体**
   - 多 env 的 reset/placement 可能导致 body 几何不同；
   - 同状态配对必须使用同一 env 或显式恢复/固定静态 body 位置，并验证 body_xpos/xquat，不能只比较 qpos。

6. **动作/任务 rollout**
   - `evaluate_libero.py` 使用 episode-level success；统计单位是 episode，不是 frame/state；
   - 正式成功率必须用独立 episode seeds，配对状态实验只能作为机制诊断。
