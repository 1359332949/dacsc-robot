# LIBERO-Plus 公开数据：元数据审计计划（Gate 0 修订）

> 触发：本地 Gate 0 只证明“本地仓库没有专家 HDF5”，不能证明公开数据不存在。
> 本文件只定义元数据审计；**不下载 75.5GB RLDS、不训练 DRAIL、不跑 rollout**。
> 当前环境访问 HuggingFace API/文件超时，实际元数据核验尚未完成。

## 1. Gate 0 的正确表述

| 结论 | 状态 |
|---|---|
| 本地无 LIBERO 专家 demonstration | ✅ 已确认 |
| 公开 LIBERO-Plus 成功轨迹存在 | ✅ 用户提供信息：LeRobot 14,347 episodes / RLDS 20,000+ successful |
| 公开数据可直接训练 DRAIL | ❌ 未证明 |
| 有大规模 CoRP 自身失败轨迹 | ❌ 没有 |
| 成功/失败样本任务·阶段·域匹配 | ❌ 未完成 |

因此 Gate 0 从“数据不存在”改为：

> 专家正样本已有公开来源；但负样本、域标签、格式兼容性和匹配矩阵尚未核验，DRAIL 训练条件仍不成立。

## 2. 公开数据能解决/不能解决什么

| 缺口 | 公开成功轨迹 | 第三方失败轨迹 |
|---|---|---|
| 专家正样本 | ✅ 可解决 | — |
| 视觉扰动下成功行为 | ⚠️ 部分，类别需核验 | — |
| 连续状态/动作 | ✅ 可解决 | ✅ 可能 |
| 当前 CoRP 策略负样本 | ❌ 仍缺 | ❌ 是其他策略 |
| 任务/阶段/域匹配 | ⚠️ 需要匹配矩阵 | ⚠️ 需要匹配矩阵 |

**关键限制**：第三方失败轨迹不是 CoRP 当前策略分布；用它们训练判别器，得到的是
“成功示范 vs 其他策略失败”的分数，不是严格的在线 DRAIL 奖励。

## 3. 纯元数据审计项

### 3.1 LeRobot 官方镜像
优先检查，不下载视频/parquet 内容：

- `meta/info.json`：`total_episodes`, `total_frames`, `fps`, `robot_type`, `features`；
- `meta/tasks.jsonl`：40 个基础任务及 task_index；
- `meta/episodes.jsonl`：每个 episode 的 length、tasks；
- `meta/episodes_stats.jsonl`（若存在）：state/action 统计；
- 抽样 1–2 个 parquet 的 schema（不读全量）。

### 3.2 RLDS
- 通过 dataset card / builder 元数据确认 trajectory 数、shard 数、features；
- 用 `episode_metadata/file_path` 恢复扰动类别；
- 不把第三方恢复的类别直接当官方标签，必须对照 RLDS 原始路径。

### 3.3 第三方失败数据
- 确认 repo、robot_type（只取 Franka Panda）、episodes/frames；
- 确认失败判定来源、生成策略、相机/压缩/控制频率；
- 只作为候选负样本，不假设与 CoRP 同分布。

## 4. 匹配矩阵（审计输出）

行：`suite × task × perturbation_category`；列：success/failure episode 数、长度分箱、机器人、action/state 版本。

| suite | task | perturbation | success eps | failure eps | length bins | robot | format |
|---|---|---|---|---|---|---|---|
| libero_spatial | ... | camera_pose | ... | ... | ... | Franka | LeRobot |
| ... | ... | ... | ... | ... | ... | ... | ... |

同时输出：

- 成功/失败交集为 0 的单元；
- 只有成功、没有失败的单元；
- 域标签不确定的单元；
- action/state 语义不兼容的单元。

## 5. 必须先核验的兼容性

1. **机器人**：只保留 Franka Panda；
2. **action**：7 维坐标系、旋转表示、gripper 编码、控制模式；
3. **state**：8 维定义、归一化；
4. **相机**：Front/Wrist 对应关系，256×256，压缩差异；
5. **时间**：20 FPS 与 CoRP 12-step chunk / 6-step execution 的对应；
6. **扰动参数**：训练与评测 camera angle 差 5°，需按参数留出而不是按类别留出。

## 6. 修订后的 Gate 0/Gate 1

| Gate | 内容 | 当前判断 |
|---|---|---|
| Gate 0a | 专家正样本存在 | 公开数据支持，待元数据核验 |
| Gate 0b | 负样本存在 | 第三方候选，待核验 |
| Gate 0c | 任务/阶段/域匹配矩阵 | 未完成 |
| Gate 0d | 格式/动作语义兼容 | 未完成 |
| Gate 1 | 成功 vs 失败轨迹/transition 区分 | 数据通过后再做 |
| Gate 2 | 候选动作 → 完整 success | 仍需新配对数据 |

## 7. 当前阻塞

- HuggingFace API/文件在当前环境超时（curl 20s timeout；web_fetch fetch failed）；
- 无法读取 `info.json` / `episodes.jsonl` / RLDS metadata；
- GPU 主机仍不可达，但这不影响纯元数据审计。

## 8. 下一步

1. 恢复 HuggingFace 访问（或由用户提供 `info.json`、`tasks.jsonl`、`episodes.jsonl`、失败数据元数据）；
2. 只跑元数据审计，输出匹配矩阵；
3. 矩阵显示有足够 task×perturbation×stage 的成功/失败交集，才考虑小分类器；
4. 小分类器通过后才考虑 DRAIL；仍不训练大数据、不跑 rollout。

## 9. 结论

公开数据**部分**修复 Gate 0：专家正样本缺口有了候选来源；但：
- 负样本不是 CoRP 分布；
- 扰动类别/格式/动作语义未核验；
- 匹配矩阵未建立。

所以 **DRAIL 训练条件仍未满足**；下一步是元数据审计，而不是下载全量数据或实现 IR-ASTTA。
