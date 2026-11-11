# 工单 3.0 冻结实验协议（多域小规模筛选）

> 状态：**协议已冻结并开始执行**。正式结果出来前，不允许根据结果调整 λ / lr / η / A / 任务 / seeds。

## 1. 冻结配置

| 项 | 值 |
|---|---|
| Checkpoint | `repos/CoRP/checkpoints/corp_libero.pt`（官方 LIBERO） |
| A 子空间 | `/hy-tmp/validation/interface_stats_spatial_k32_big2.json`（768×5，rank 5） |
| **λ_protect** | **5.0** |
| λ 标定依据 | 真实 clean batch（8 个）上 `mean‖∇L_SALT‖/‖∇penalty‖ = 5.88`，snap 到网格 {0.1,0.5,1,5,10,50,100,500,1000} |
| η / K / draws / lr | 0.15 / 5 / 8 / 0.06（=2η/K） |
| exec_len / horizon / steps | 6 / 12 / 10 |
| 触发 | **always-on**（不引入 feature gate） |
| Pilot seeds | 202–205（4 episodes/任务，任务选择用） |
| Formal seeds | 7–11（5 episodes/任务，方法比较用） |
| 方法 | Frozen / SALT / SG-SALT |

## 2. 任务选择（Pilot 结果）

每个 cell（domain × suite）用 pilot 估计 Frozen 成功率，优先选 20%–80% 的任务：

| cell | 选中任务 | pilot 成功率 |
|---|---|---|
| Camera|spatial | 705, 620, 671 | 2/4, 1/4, 4/4 |
| Camera|object | 701, 901, 653 | 1/4, 3/4, 4/4 |
| Background|spatial | 11, 19, 18 | 3/4, 3/4, 1/4 |
| Background|object | 3, 8, 18 | 3/4, 1/4, 4/4 |
| Noise|spatial | 1384, 1375, 1392 | 2/4, 3/4, 4/4 |
| Noise|object | 1414, 1432, 1442 | **4/4, 4/4, 4/4**（无 in-band 任务，已记录局限） |
| Light|spatial | 2235, 2338, 2130 | 2/4, 3/4, 4/4 |
| Light|object | 2238, 2248, 2266 | 3/4, 4/4, 4/4 |

**局限（必须在报告声明）**：pilot 只有 4 episodes，部分 cell 的 in-band 任务不足；
Noise|object 全为 4/4（无提升空间），其结果只作"不伤害"参考，不用于"SG-SALT 优于 SALT"的证据。

## 3. 规模与执行

- 8 cells × 3 tasks × 5 episodes = 120 paired episodes / method
- × 3 methods = **360 rollouts**
- 分批：Frozen（快）→ SALT → SG-SALT；同一 cell 内三种方法使用**相同任务/初始状态/seeds**。

## 4. 统计与判定

- 主指标：`Δ = SR_SG-SALT − SR_SALT`（按域报告）
- 次级：SG-SALT − Frozen、SALT − Frozen
- 配对：McNemar；任务级 cluster bootstrap 90% CI
- Light（高基线）非劣效：SG-SALT vs Frozen，±5pp 操作判据
- 记录：逐 episode 结果、更新次数、β 饱和比例、penalty、耗时

## 5. 决策规则（预注册）

| 结果 | 决策 |
|---|---|
| ≥2 个主域 Δ>0（方向一致且 cluster CI 不含 0 或一致） | 扩大实验 |
| 仅 Camera 有效 | 保留创新，限定 Camera-specific |
| 仅 1 个主域有效 | 分析适用条件，不宣称跨域 |
| 三个主域均无改善 | 暂停 SG-SALT，不再调参 |
| Light 明显退化 | 记录 clean 行为影响 |

## 6. 禁止

- 根据正式结果中途调 λ/lr/η/token/A/任务/seeds；
- 删除失败 episode；
- 增加其他适配机制；
- 未经确认扩大规模。

_冻结时间：2026-11-11；commit 见本次提交。_
