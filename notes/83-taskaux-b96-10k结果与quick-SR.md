# 83 - task-aux batch96 10k：baseline 与 random_stats 结果

日期：2026-11-05

> 后续说明：本记录对应 `corp_robocasa_v4` 初始化和 SR v6。note84 已改为部分复用官方
> 7D 动作头并升级为 checkpoint v5 / SR v7；本记录不能与新的 v5 初始化结果混表。

## 1. 实验身份

代码：`344bd0a6b3391bf3716e3efcc048c381aef9bff6`

两轮 run 使用同一协议：

```yaml
steps: 10000
batch_size: 96
lr: 5e-5
ema_decay: 0.99
seed: 42
n_obs_steps: 1
task_aux_lambda: 0.01
task_aux_depth: 1
task_aux_classes: 61
task_aux_head_hidden_dim: 256
train_augmentation: none
eval_shift: color_jitter
```

对比：

```text
baseline:    token_style_mode=none
control:     token_style_mode=random_stats
             token_style_alpha=0.5
```

结果目录：

```text
/hy-tmp/runs/robo_nogate_taskaux_b96_10k_seed42_v1
/hy-tmp/runs/robo_random_stats_taskaux_b96_10k_seed42_v1
```

稳定归档：

```text
/hy-tmp/archive/20261105_taskaux_b96_10k
```

归档保存两份最终 EMA 模型、`model_step10000.pt`、`result.json`、`run.log`、
quick-SR 完整目录、快速评测 manifest、实验摘要及 SHA256。最终模型在归档中使用
同文件系统硬链接，因此不会额外复制约 1.5 GB 权重；删除原 run 后归档仍保留数据。

## 2. 离线 MSE

| 条件 | Baseline source | Random source | Baseline val | Random val |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.06445 | 0.06800 | 0.06993 | 0.07336 |
| color_jitter | 0.12995 | 0.11431 | 0.12222 | 0.10949 |

Random stats 相对 baseline：

```text
clean source:       +5.5% MSE（变差）
clean val:          +4.9% MSE（变差）
color_jitter source: -12.0% MSE（变好）
color_jitter val:    -10.4% MSE（变好）
```

Clean source/val gap：

```text
baseline:    val - source = +0.00548
random_stats: val - source = +0.00535
```

当前 style split 的 clean MSE gap 仍然很小。

## 3. Color-jitter 分组变化

| 动作组 | Source 变化 | Val 变化 |
| --- | ---: | ---: |
| eef_position | -7.6% | -4.2% |
| eef_rotation | -0.1% | -0.4% |
| gripper_close | -18.4% | -18.2% |
| base_motion | +14.8% | +26.6% |
| control_mode | -13.8% | -13.1% |

因此 random_stats 在该次训练中的扰动改善主要来自 `gripper_close` 和末端位置，
旋转基本不变，同时 `base_motion` 变差。不能写成“所有动作维度整体更鲁棒”。

## 4. 快速 success rate

快速评测固定：

```text
tasks: OpenDrawer, CloseDrawer, OpenCabinet
episodes: 2 per task per split
splits: source, val
shift: none
```

| 方法 | Source | Val |
| --- | ---: | ---: |
| Baseline | 1/6 = 16.7% | 0/6 = 0% |
| Random stats | 1/6 = 16.7% | 0/6 = 0% |

唯一成功均来自 `CloseDrawer` 的 source 1/2。两组快速 SR 完全相同。

这是 go/no-go smoke，不是方法结论：每 split 只有 6 个 episode，
不能支持显著性、任务级泛化或方法优劣判断。

## 5. 可以得出的结论

- 10k batch96 baseline 已具备至少一次闭环成功，不再是完全不可执行的策略。
- CoRP Eq.7 task auxiliary loss 正常训练；task loss 从约 4.1 降到约 0.001--0.03。
- `random_stats` 在本 seed 下呈 clean 精度略差、color-jitter MSE 较好的 trade-off。
- 快速 SR 未观察到 random_stats 的成功率收益。
- 当前 style split 的 clean MSE gap 仍弱，不能仅凭此证明稳定 appearance gap。

## 6. 不能得出的结论

- 不能声称 random_stats 提升 success rate。
- 不能声称 random_stats 是有效的最终方法。
- 不能把随机统计混合当作 `cross_style_stats` 有效性的证据。
- 不能用 6+6 个 episode 做跨方法统计比较。
- 不能把离线 MSE 改善直接解释为闭环任务能力改善。

## 7. 下一步

使用同一冻结 manifest，对 baseline 与 random_stats 顺序运行：

```text
10 tasks x 5 episodes
source + val
shift=none
```

baseline 首轮使用 `manifest-mode=create`，random_stats 使用同一 manifest 的
`replay`。只有更大规模 SR 给出可解释结果后，才决定是否进入
`cross_style_stats` 同预算训练。
