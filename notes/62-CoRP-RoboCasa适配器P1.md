# 62 - CoRP→RoboCasa 适配器（P1）

日期：2026-10-24
状态：本地已实现并 smoke 通过（随机权重 CPU）；待服务器恢复后接真实 checkpoint。

## 1. 结论

P1 的适配器代码已写好，官方 CoRP 架构**原样保留**，只改接口：

| 字段 | 官方 CoRP | CoRP-RoboCasa |
|---|---|---|
| views | 2 (agentview + eye_in_hand) | 3 (robot0_agentview_left / robot0_eye_in_hand / robot0_agentview_right) |
| state_dim | 8 (pos3+rotvec3+gripper2) | **16** (base_pos3+base_rot4+eef_pos_rel3+eef_rot_rel4+gripper2) |
| action_dim | 7 | 12 |
| action_horizon | 12 | 10 |
| hidden_dim | 768 | 768（不变） |
| resampler | K=48/view, 8 heads, FiLM rank 64 | 不变 |
| action head | 1-layer DiT 768, 12 heads, flow 10 步 | 不变 |

命名必须是 **"CoRP adapted to RoboCasa / our reproduction"**，不能写成官方 CoRP 在 RoboCasa 上。

## 2. 新增文件

- `dacsc/corp_robocasa.py`：适配器模块
  - `ROBOCASA_CORP_CONFIG`（RoboCasa 接口配置）
  - `CorpRoboCasa`：包官方 `RepresentationExtractor` + `DiTActionGenerator`
  - `flow_matching_loss`：补上官方缺失的训练路径（rectified-flow velocity loss）
  - `sample_action`：推理采样
  - `load_corp_robocasa`：从 `corp_libero.pt` 迁移共享权重，返回可审计报告
- `scripts/test_corp_robocasa.py`：随机权重 CPU smoke

## 3. 权重初始化策略（可审计）

- **共享层从 checkpoint 拷贝**：DINOv2、T5、resampler 的 input_projection / cross-attention / FiLM、state token、DiT block、adaLN、time embedding、action position。
- **view query / view_embedding 部分迁移**：`agentview_left <- agentview`、`eye_in_hand <- eye_in_hand`；`agentview_right` 随机初始化。
- **接口层随机初始化**：state projection (8→9)、action_projection (7→12)、velocity_projection (7→12)。这些语义不一一对应，不能拷贝，论文里必须明确标注。

## 3.1 真实 RoboCasa 接口（来自 meta/modality.json，61/61 task 一致）

```text
observation.state 16D =
    base_position(0:3) + base_rotation(3:7)
  + end_effector_position_relative(7:10) + end_effector_rotation_relative(10:14)
  + gripper_qpos(14:16)

action 12D =
    base_motion(0:4) + control_mode(4:5)
  + end_effector_position(5:8) + end_effector_rotation(8:11) + gripper_close(11:12)

语言：extras/episode_*/ep_meta.json 的 `lang` 原始文本；
      meta/tasks.jsonl 也有 {task_index, task} 文本。
```

**修正**：早期笔记写的 state 9D 是错的；真实是 16D。adapter 已按 16D 修正。

## 4. Smoke 结果（随机权重，CPU）

```text
params = 48,735,244
官方 = 48,689,159
差值 = 46,085（= 第3 view query/embedding + state 8→9 + action/velocity 7→12）

extract tokens = (B, 177, 768)   # 3×48 + 32(T5) + 1(state)
flow_matching_loss = 2.40（有限，可反向）
sample_action = (B, 10, 12)

权重合并（合成官方 checkpoint）：
  total=258  copied=254  random_init=4
  partial: queries + view_embedding（view0/1 拷贝，view2 随机）
  random_init: state_proj.weight / action_proj.weight / velocity_proj.weight / velocity_proj.bias
```

## 5. 本地 checkpoint 问题

本地 `repos/CoRP/checkpoints/corp_libero.pt` 只有 401KB（Git-LFS 截断占位），sha256 `6bc48ad0...`，与官方 `f933ab03...` 不符。真实 227MB checkpoint 在服务器 `/hy-tmp/models/corp/CoRP/checkpoints/`。

## 6. 待办（需要服务器）

1. 拉取真实 `corp_libero.pt`（227MB）到本地或直接在服务器跑真实权重 smoke。
2. RoboCasa 真实观测键 / action / state 维度的最终确认（以数据集 metadata 为准）。
3. RoboCasa action/state normalizer 统计量。
4. 语言输入从 CLIP `lang_emb` 改为 T5 原始指令文本（需确认数据集里有原始 `lang` 文本，而非只有预计算 embedding）。
5. 图像预处理决策：官方 CoRP 用 render 256 → resize 224（无 center crop），RoboCasa 当前用 center crop 224；需显式选一个并写进配置。

## 7. 下一步（P2，先不装 gate）

真实 checkpoint 到位后：
1. 真实权重 smoke（load_corp_robocasa 报告核对）。
2. no-gate CoRP-RoboCasa baseline：51 步 smoke → 200 步 → val/test MSE。
3. 之后才做 P3 门控消融（no gate / random mask / domain gate）。
