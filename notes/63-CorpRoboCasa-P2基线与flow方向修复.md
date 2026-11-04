# 63 - CoRP-RoboCasa P2 基线：flow 方向修复 + 51 步 smoke

日期：2026-10-24
状态：适配器真实权重 smoke ✅；5-task 51 步 smoke ✅；全量 61-task 运行中。

## 1. 关键 bug：flow 方向与官方 sample() 相反

官方 `DiTActionGenerator.sample()`：

```python
action = torch.randn(...)          # x(0) = noise
for step in range(N):
    tau = step / N                  # 0 -> 1
    action += v(action, tau) * 1/N  # x(1) = data
```

即 `x(0)=eps`、`x(1)=x0`、`dx/dtau = x0 - eps`。

第一版训练写成了 `x_tau=(1-tau)*x0+tau*eps`、`target=eps-x0`，方向相反。
**症状**：train loss 从 1.37 降到 0.59，但采样动作 MSE 从 3.9 升到 6.3。

修复为：

```python
x_tau = eps * (1-tau) + x0 * tau
target = x0 - eps
```

修复后同样 51 步（5 task）：loss 1.58→0.30，source MSE 0.52→0.13。

> 教训：训练 loss 下降不能证明采样正确，必须同时看 action MSE。

## 2. 5-task 51 步 smoke

```yaml
tasks: AdjustToasterOvenTemperature / AdjustWaterTemperature / CloseBlenderLid /
       CloseCabinet / CloseDishwasher
train styles 11-30 (90% episodes): 43466 frames
source holdout (10% episodes):      4900 frames
val styles 31-40:                  26810 frames
```

| step | source_mse | val_mse (held-out style) | gap |
|---|---|---|---|
| 25 | 0.5232 | 0.5501 | +0.0269 |
| 50 | 0.1217 | 0.1112 | -0.0104 |
| 51 | 0.1332 | 0.1103 | **-0.0230** |

初步信号：**51 步时未见 style 没有变差**。但这只是 5 task / 51 步，不能下结论。

## 3. 保真度决定（全部有依据）

| 项 | 决定 | 依据 |
|---|---|---|
| tau 分布 | `Beta(1.0, 1.5)` | 论文 |
| 图像 | 256 → bilinear resize 224，不 crop | 官方 `preprocess` |
| state 归一化 | train-only z-score | 官方 CoRP |
| action 归一化 | identity | RoboCasa DP 基线 `get_normalizer`（scale=1/offset=0）；部分维在若干任务是常量，min-max 会退化 |
| task-classification aux loss (λ=0.01) | **不复现** | 官方发布仓库没有训练代码/该损失；若将来加，必须标为“我们的补充” |
| 命名 | **CoRP adapted to RoboCasa / our reproduction** | 官方训练代码未发布 |

2026-11-04更新：上表“不复现”是P2当时的状态。此后已按论文Eq.7补齐为
“CoRP paper Eq.7 reproduction”；官方仓库仍无trainer，因此不是官方训练代码复现。
task head只在训练期使用，隐藏宽度256是论文未公布细节下的显式实现选择。

## 4. 接口（以 modality.json 为准）

```text
raw observation.state 16D =
  base_position(0:3)+base_rotation(3:7)
  +eef_position_relative(7:10)+eef_rotation_relative(10:14)+gripper_qpos(14:16)

策略输入 state 9D（与 DP 基线 shape_meta 一致）=
  eef_position_relative(3)+eef_rotation_relative(4)+gripper_qpos(2)

action 12D（config 拼接顺序）=
  eef_position(3)+eef_rotation(3)+gripper_close(1)+base_motion(4)+control_mode(1)
```

## 4.1 全量 61-task / 51 步结果

```yaml
n_train=501594  n_val=275479  n_test=265528
```

| step | source_mse | val_mse (held-out style) | gap | train_loss |
|---|---|---|---|---|
| 50 | 0.1140 | 0.1090 | -0.0049 | 0.4307 |
| 51 | 0.1175 | 0.1138 | **-0.0037** | 0.3412 |

`test_action_mse = 0.1215`（styles 41–50，只在最后跑一次；比 source 高约 3.5%）。

**初步结论**：51 步下 CoRP 对未见 style 几乎不退化（gap ≈ 0）。这与论文“CoRP 对视觉随机化鲁棒”一致，也意味着“style shift 伤害策略”这个 gate 前提目前**没有证据支持**。
仍需 200 步确认后期是否出现 gap。

## 4.2 全量 61-task / 200 步结果

| step | source_mse | val_mse (held-out style) | gap | train_loss |
|---|---|---|---|---|
| 100 | 0.1074 | 0.1102 | +0.0028 | 0.2147 |
| 200 | 0.1053 | 0.0874 | **-0.0179** | 0.2510 |

`test_action_mse = 0.1044`（styles 41–50）。

**结论（51 + 200 步一致）**：

```text
no-gate CoRP-RoboCasa 在 held-out style 上没有退化：
  step 51 : gap = -0.0037
  step 200: gap = -0.0179
  test (41-50) = 0.1044 vs source 0.1053
```

也就是说，在 RoboCasa atomic61 上，**“style shift 伤害策略”这个 gate 前提目前没有证据支持**。
这与论文“CoRP 对视觉随机化鲁棒”一致，也与 notes/43、47、59、60 的负结果一致。

需要注意的边界：
- 200 步仍远短于完整训练；
- 聚合 MSE 可能被 base_motion / control_mode 等常量维稀释；
- 但 source 与 held-out 用同一指标、同一模型比较，结论方向是可靠的。

## 5. 下一步

1. 全量 61-task / 51 步：确认 source vs held-out style gap。
2. 若仍无 gap → gate 方向不成立，按“负结果 + 机制发现”写。
3. 若出现 gap → 200 步确认，再考虑论文支持的杠杆（微调/压缩/语言），不是通道 hard gate。
