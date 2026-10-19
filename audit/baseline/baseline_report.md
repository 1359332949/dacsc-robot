# 阶段 1 官方 Diffusion Policy 基线审计

## 1. 解析后的配置（源码级）

- 配置：`train_diffusion_transformer_bs192` + `task/robocasa/pretrain_human300`
- 输入 shape：obs 键 7 个，action shape = [12]

| 键 | 值 |
|---|---|
| horizon | 10 |
| n_obs_steps | 2 |
| n_action_steps | 8 |
| crop_shape | [224, 224] |
| batch_size | 192 |
| num_epochs | 1000 |
| max_train_steps | 500 |
| obs_encoder_group_norm | True |
| dit_n_layer | 12 |
| dit_n_cond_layers | 4 |
| dit_n_head | 8 |
| dit_n_emb | 512 |

| obs 键 | shape | type | lerobot 键 |
|---|---|---|---|
| robot0_agentview_right_image | [3, 256, 256] | rgb | video.robot0_agentview_right |
| robot0_agentview_left_image | [3, 256, 256] | rgb | video.robot0_agentview_left |
| robot0_eye_in_hand_image | [3, 256, 256] | rgb | video.robot0_eye_in_hand |
| robot0_base_to_eef_pos | [3] | low_dim | state.end_effector_position_relative |
| robot0_base_to_eef_quat | [4] | low_dim | state.end_effector_rotation_relative |
| robot0_gripper_qpos | [2] | low_dim | state.gripper_qpos |
| lang_emb | [768] | low_dim | - |

动作向量 lerobot 键：`['action.end_effector_position', 'action.end_effector_rotation', 'action.gripper_close', 'action.base_motion', 'action.control_mode']`

## 2. ResNet18ConvFiLM 的 block 结构

- 方法：**native_import**
- conv block 数：**8**，通道：`[64, 64, 128, 128, 256, 256, 512, 512]`
- 每个 block 后接 FiLM：**8** 个，语言维度 768
- 输入 224×224：base_block 输出 [64, 56, 56]，最终特征 [512, 7, 7]
- 输入 256×256：base_block 输出 [64, 64, 64]，最终特征 [512, 8, 8]

## 3. 每任务 horizon

- 任务数：**317**，范围 **300–7200** 步，全部为 150 的倍数：**True**
- 分布（步数: 任务数）：`{'300': 11, '450': 26, '600': 15, '750': 12, '900': 10, '1050': 9, '1200': 11, '1350': 16, '1500': 16, '1650': 14, '1800': 16, '1950': 10, '2100': 14, '2250': 12, '2400': 19, '2550': 10, '2700': 8, '2850': 7, '3000': 10, '3150': 5, '3300': 9, '3450': 5, '3600': 8, '3750': 6, '3900': 7, '4050': 7, '4200': 3, '4350': 4, '4500': 2, '4650': 2, '4800': 2, '4950': 2, '5100': 2, '5250': 2, '5400': 2, '5850': 1, '6750': 1, '7200': 1}`
- 注意：配置文件里的 `max_steps` 会被 `eval_robocasa.py` 的 `get_task_horizon(task)` 覆盖。

## 4. 显存与评测开销

- 官方建议：训练 ≥ **24 GB**（推荐 48 GB+）；推理 ≥ 8 GB
- 评测实例：16 core / 64 GB RAM（env_runner 注释）
- 本地实测：CUDA 不可用，未实测
- DP(DiT-512, bs192, 3×224×224 输入) 的显存主要由三段决定：①视觉编码器激活（3 路相机 × ResNet18，crop 224）；②DiT 12 层/512 维、horizon=10 的注意力激活；③EMA 影子参数 + Adam 状态（参数量的 3 倍）。官方建议 ≥24 GB、48 GB+ 更佳，可作为硬约束。

## 5. 判定

**基线配置已确认，可进入 dry run**