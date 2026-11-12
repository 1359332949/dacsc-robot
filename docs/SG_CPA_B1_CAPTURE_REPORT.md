# Stage B1 数据采集报告（小规模 pilot）

> 目的：补齐 B0 审计发现的缺失字段，为 B1 的前瞻性信号比较提供数据条件。
> **只采集观测/动作/图像/进度；不做适配、不评估成功率方法。**
> 数据：`/hy-tmp/validation/b1_capture/`；脚本：`scripts/sg_b1_capture.py`、`sg_b1_goal_map.py`、`sg_b1_progress.py`。

## 1. 执行

- 域/套件：Camera、Background × libero_spatial、libero_object；
- 任务：每格 2 个（Camera spatial 705/620；Background spatial 11/19；Camera object 653/701；Background object 3/8）；
- episodes：每任务 1 个，共 **8 episodes**；
- 结果：**222 个 replan 记录**（230 行含 episode summary），**848 张 JPEG**（双相机，含 next），
  **13 MB**，耗时 **75 s**；
- 策略：冻结 CoRP；每个 replan 使用独立 action seed（`episode_seed + 7919*replan`），
  K=4 视图（identity/noise/blur/photo）共享该 replan 的 latent。

## 2. 字段清单（每个 replan）

| 字段 | 说明 |
|---|---|
| domain/suite/task_id/episode/seed/action_seed/replan/steps | 身份与时间 |
| state | 8 维原始状态（policy_observation） |
| instruction | 任务语言指令 |
| beta | 当前适配参数（采集为 0，冻结） |
| action_chunk_norm / action_chunk_ctrl | 冻结策略动作块（归一化/控制单位） |
| view_actions_mean_distance | K=4 视图动作两两平均距离（多视图一致性信号） |
| history_leftover_distance | 上一动作块未执行尾部 vs 当前块头部（历史连续性信号） |
| images | `o_t` 双相机 JPEG 路径 |
| next_images | 执行 `exec_len=6` 后的 `o_{t+1}` JPEG 路径（未来视觉预测用，延迟信号） |
| body_names / body_pos | 38 个 body 名称与位置（计算任务进度） |
| goal_text | 基础 bddl 的 goal 文本（On/In 等谓词） |
| success_so_far | 当前时刻是否已成功（采集用，不用于适配） |

## 3. 字段完整性校验

| 检查 | 结果 |
|---|---|
| replans | 222 |
| next_images | 214 / 222（最后一步无 next，符合预期） |
| history_leftover_distance 非空 | 214 / 222 |
| 视图距离均值 | 0.031 |
| body 名称/位置 | 38 bodies，全部记录 |
| goal 文本 | 8/8 episode 通过 longest-prefix 恢复并解析出 On/In 谓词 |

## 4. 预注册的任务进度 p

定义（在任何信号分析之前固定）：

```
p_t = mean over goal literals (On/In obj target) of clip(1 - d_t/d0, 0, 1)
d_t = ||pos(obj_main) - pos(target_main)||, d0 = 第一个 replan 的值
```

pilot 结果（`progress.jsonl`）：

| task | suite | success | p_last | n replans | monotone_frac |
|---|---|---|---|---|---|
| 705 | spatial | False | 0.013 | 37 | 0.73 |
| 620 | spatial | False | 0.111 | 37 | 0.86 |
| 11 | spatial | True | 0.062 | 14 | 0.86 |
| 19 | spatial | True | 0.066 | 13 | 1.00 |
| 653 | object | True | 0.083 | 24 | 0.96 |
| 701 | object | False | 0.000 | 37 | 0.95 |
| 3 | object | True | 0.086 | 23 | 1.00 |
| 8 | object | False | 0.000 | 37 | 1.00 |

**观察（仅字段校验，不是 B1 结论）**
- 成功 episode 的 p_last 均 > 0.06；失败的 701/8 为 0，但 705（失败）为 0.013、620（失败）为 0.111；
- p 的量级偏小（距离下降 6–11%），说明该线性距离指标对"完成"的区分有限；
  B1 正式分析前需要确认 p 是否足够敏感（可考虑加入高度/朝向项，但必须在分析前固定）。

## 5. 限制

- 仅 8 episodes / 222 replans，**不能做 B1 统计结论**；
- p 为距离型粗指标；对"抓取-放置"的完成度区分有限；
- 成功标签仅用于未来分层/参考，不参与 p 计算；
- 图像存 JPEG（quality 92），会引入轻微压缩误差；
- 未采集中途视觉变化（episode 内静态扰动），与 always-on 协议一致。

## 6. 下一步（需授权）

1. **扩到 24 episodes**（2 域 × 2 套件 × 3 任务 × 2 episodes ≈ 300–600 replans），
   约 3–5 分钟采集 + ~40 MB；
2. 数据完成后，按 `SG_CPA_STAGE_B0_B1_REPORT.md` 的预注册设计做 B1 分析
   （LOO task、基线增量、任务簇 bootstrap、置换检验、分域报告）；
3. B2（更新/不更新成对反事实）仍需单独授权。

## 7. 交付

- 采集脚本：`scripts/sg_b1_capture.py`
- goal 恢复/进度：`scripts/sg_b1_goal_map.py`、`scripts/sg_b1_progress.py`
- 数据归档：`docs/data/b1_capture_pilot.tar.gz`（JSONL/goal_map/progress；图像保留在服务器）
- commit 见本次提交。
