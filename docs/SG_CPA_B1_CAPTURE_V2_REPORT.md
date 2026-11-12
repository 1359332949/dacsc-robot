# Stage B1 采集 v2 报告（24 episodes）+ 数据验收状态

> 依据审查意见对 `b753ce7` 的修正：固定任务列表、H=12、完整 view 动作/几何字段、
> 图像 manifest+sha256、严格映射、LOO 基线修正、成功动作标为 Oracle、未来视觉标为延迟信号。
> 数据：`/hy-tmp/validation/b1_capture_v2/`（64 MB，含图像）；归档：`docs/data/b1_capture_v2_meta.tar.gz`。

## 1. 采集执行

- 任务列表（扩采前固定，不基于 pilot success/p_last）：
  - Camera spatial: 705, 620, 671
  - Background spatial: 11, 19, 18
  - Camera object: 653, 701, 901
  - Background object: 3, 8, 18
- 每 task 2 episodes → **24 episodes**；
- 结果：**667 replan 记录**、**2620 张 JPEG**（双相机 o_t / o_{t+1}）、**64 MB**、耗时 **221 s**；
- 冻结 CoRP；每 replan 独立 action seed（`episode_seed + 7919*replan`）。

## 2. 对审查意见的落实

| 审查问题 | 处理 |
|---|---|
| 进度指标只测"接近" | 保留 `proximity_progress` 作为基线；新增基于 On/In 几何的 `geometric_progress`（见 §4，**当前未通过验收**） |
| 缺 body quaternion / 几何尺寸 | 已采集 `body_quat`、`geom_names/geom_pos/geom_size/geom_bodyid` |
| H=10 与 6 步执行不匹配 | **预注册修订为 H=12**（=2 个 replan × exec_len 6），`dp12_*` 字段 |
| progress 脚本 task_id 覆盖 | 全部改用 `(suite, task_id, episode)` 完整 key |
| 缺失映射静默填 0 | 改为**严格报错**；v4 中按语言指令固定 target 映射，解析失败即中止 |
| 只存平均视图距离 | 已存 **4 个完整 view action blocks** + 6 个 pairwise distance |
| 图像不可恢复 | 新增 `image_manifest.json`：相对路径 + sha256 + bytes（2620 条） |
| LOO 与 task-ID baseline 冲突 | 预注册修订：LOO 主 baseline 不含 task-ID；task-ID 仅用于同任务分析 |
| 成功动作在 LOO 下泄漏 | 标为 **Oracle 上界 / 不可估计**，不进入普通候选信号排名 |
| 未来视觉延迟 | 单独归类为延迟验证信号；预测器需在训练任务上拟合，每折独立 |

## 3. 字段 schema（每 replan）

`domain, suite, task_id, episode, seed, action_seed, replan, steps, state(8), instruction, beta,
action_chunk_norm(12×7), action_chunk_ctrl(12×7), view_actions(4×12×7),
view_pairwise_distances(6), history_leftover_distance, images{view:path+sha256+bytes},
next_images{...}, body_names, body_pos, body_quat, geom_names, geom_pos, geom_size,
geom_bodyid, goal_text, success_so_far`。

校验：667 replans；next_images 643/667（最后一步无 next）；history 643/667；manifest 2620 条、sha256 全记录。

## 4. 进度指标验收

### 4.1 proximity_progress（距离型，保留为基线）

`p = mean clip(1 - d_t/d0)`，成功 episode 的 p_last 通常在 0.5–0.9；**可作为基线**。

### 4.2 geometric_progress（On/In 几何关系）——**未通过验收**

当前实现：目标 body 最大 geom 的 inside-xy × height/depth 项。
验收结果：**object 任务（In basket）在多数成功 episode 上 p_geom_last = 0.0**（如 3/653/901），
仅 spatial 任务 19 成功（0.98）；说明该几何公式对容器内部/物体底部的关系建模不正确。

**处理：数据验收对 geometric_progress 判 FAIL；不进入 B1 信号分析。**
修复方向（需重新冻结后重算，不需要重采集）：
- On：用物体**底部**（中心 z − 物体 geom 半高）vs 支撑面**顶部**（目标 geom 顶部）；
- In：用物体中心是否位于容器**内壁包围盒**内（容器 geom 的 inside 体积），而不是与容器中心比较；
- 用机械判据（几何自洽，不看 success）做单元测试后重新验收。

## 5. B1 分析状态

- 数据字段/H12/严格映射/manifest 已通过结构验收；
- **进度指标（geometric）未通过**，因此 **B1 信号比较尚未运行**；
- 样本量：667 replans，但独立 task cluster 只有 **12 个**（每域约 6 个）；结论定位为探索性筛选。

## 6. 下一步

1. 修复 geometric_progress（力学几何公式 + 单元测试）并重新冻结；
2. 数据验收通过后再执行 B1 分析（四信号、LOO、cluster CI、permutation、基线增量）；
3. B2 成对反事实仍未授权；无 rollout。

_数据归档：`docs/data/b1_capture_v2_meta.tar.gz`（capture/progress/goal_map/manifest，不含图像；图像在服务器）。_
