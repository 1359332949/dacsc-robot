# 工单 4.2 报告勘误（代码审查后）

审查确认以下问题，**原报告的相关解释需要修正，原始 600 episodes 数据不变、保留为负结果**。

## 1. 随机数复用缺陷（影响实现，不推翻数据）

| 问题 | 事实 | 影响 |
|---|---|---|
| 视图噪声 seed 固定 | `build_views()` 把 seed 闭包在视图函数里；`reset(seed)` 只改 `self.seed`，不重建视图 | 同一进程所有 episode、所有 replan 复用同一噪声图案，不是独立随机反事实视图 |
| 动作初始噪声固定 | driver 每个 replan 调用 `tta.step(prepared, seed=episode_seed)` | 同一 episode 所有 replan 使用同一 latent；适配拟合固定噪声样本，执行却用新 latent |

修复（commit 见文末）：
- 变换函数签名改为 `fn(images, seed)`，每个 replan 传入新 seed；
- driver 每个 replan：`action_seed = episode_seed + 7919*replan`，`view_seed = action_seed+1`；
- 同一 replan 内 teacher/student 共享同一 latent（Common Random Numbers）；
- 新增测试覆盖"不同 replan 视图噪声不同"、"不同 replan action latent 不同"。

## 2. 无投影消融的混淆（必须收窄结论）

原实现 `use_projection=False` 只关闭一阶 Jacobian 投影，**有限步 A guard 仍然生效**。
因此 4.2 的消融实际比较的是：

- SG-CPA：投影梯度 + A guard
- 消融：原始梯度 + **A guard**

**修正后的结论**：在共享有限步 A guard 的前提下，没有观察到一阶投影带来的额外成功率收益；
不能据此说"A 方向保护完全没有贡献"。

修复：新增 `a_guard` 开关，三种配置明确区分：
| 配置 | 投影 | A guard |
|---|---|---|
| Full SG-CPA | 开 | 开 |
| No tangent projection | 关 | 开 |
| No A protection | 关 | 关 |
（`--sgcpa-no-projection`、`--sgcpa-no-a-guard`；对应单元测试 21 passed）

## 3. 成本比较口径错误

原报告：SG-CPA 2.15 s/replan vs SALT "0.1 s/update" → 20×，**单位不一致**。
SALT 每个 replan 有 5 个 adaptation steps，正确比较应为：
- SALT ≈ 0.1 × 5 = **0.5 s/replan**；
- SG-CPA ≈ 2.15 s/replan → **约 4.3×**，仍明显更慢，但不是 20×。

## 4. 实验规模与预注册不一致

原设计：Frozen 120 / SALT 120 / SG-CPA 120 / w/o projection 60 = **420**。
实际执行：前三组 120，w/o projection **120**，另加 medoid-excl-identity **120** = **600**。
→ 后两组应标注为**超出原计划的扩展诊断**，不能与预注册主比较混为一谈；
主比较（前三组各 120 配对）仍然有效。

## 5. 设计文档过期

`SG_CPA_PHASE4_2_DESIGN.md` 仍写"Identity + 3 Noise、Anchored Mean、α=0.1"，
实际执行 `cc5ab91` 已改为"Identity + Noise + Blur + Photo、Medoid、Mean 为消融"。
→ 以代码/最终协议为准；设计文档已加状态标注（见文末）。

## 6. 统计解释需软化

- 每域仅 **6 个任务簇**（3 任务 × 2 套件），cluster bootstrap 未按套件/域分层；
- Background：McNemar p=0.453，但任务级 CI 不含 0 → 只能说"负向趋势"，不能称显著退化；
- 未做多重比较校正。
- Rollout 未保存每次 replan 的 medoid index / loss / grad 明细（未开 `--diag-out`），
  因此不能用 4.0 离线 identity 比例直接当作 rollout 内证据；后续诊断实验必须开 `--diag-out`。

## 7. 未变的部分

- P_null 的 SVD 实现与方向正确；
- β 参数化（K-projection SVD）正确；
- 600 episodes 原始数据有效，保留为"当前实现的负结果"；
- 预注册暂停条件仍然成立（当前实现无主域改善）。

_代码修复 commit 见本次提交。_
