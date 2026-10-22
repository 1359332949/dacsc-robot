# 59 - 统一 center-crop 的 flow no-mask / random-mask 5-seed baseline

日期：2026-10-22
服务器：`i-2.gpushare.com:40424`
代码提交：`a3ab79b`
结果目录：`audit/strict_flow_centercrop_baseline_51/`

## 1. 统一协议

```text
train styles: 11-30
val styles:   31-40
test styles:  41-50
crop:         224 center crop in both train and eval
random_crop:  false
backbone:     DINOv2 ViT-S/14 + resampler K48 + flow matching
steps:        51
seeds:        42 / 43 / 44 / 45 / 46
val:          16 batches × 8 = 128 samples
test:         16 batches × 8 = 128 samples
```

`scripts/test_crop_protocol.py` 验证同一输入在 train/eval mode 下的 crop 输出完全一致。

Paired 审计：

```text
同 seed 下 Hydra 配置只有 policy.enable_gating 不同   ✅
training.seed 和 task.dataset.seed 相同                ✅
DomainBalancedSampler 使用同一显式 seed                ✅
val/test 均为 128 样本                                 ✅
```

## 2. 结果

| run | seed | val_action_mse | test_action_mse |
|---|---:|---:|---:|
| no-mask | 42 | 0.766393 | 0.773379 |
| no-mask | 43 | 0.730130 | 0.725989 |
| no-mask | 44 | 0.655733 | 0.659773 |
| no-mask | 45 | 0.645081 | 0.645647 |
| no-mask | 46 | 0.662850 | 0.658736 |
| random-mask | 42 | 0.882766 | 0.905463 |
| random-mask | 43 | 0.665392 | 0.692160 |
| random-mask | 44 | 0.656803 | 0.673334 |
| random-mask | 45 | 0.660171 | 0.681191 |
| random-mask | 46 | 0.625736 | 0.647838 |

均值：

```text
no-mask:
  val  0.692037 ± 0.047654
  test 0.692705 ± 0.049130

random-mask:
  val  0.698174 ± 0.093329
  test 0.719997 ± 0.093875
```

Paired delta（random - no-mask）：

```text
val  mean +0.006136, sample std 0.069195, 95% CI [-0.079767, +0.092039]
test mean +0.027292, sample std 0.064094, 95% CI [-0.052278, +0.106863]
```

## 3. 结论

```text
paired protocol 成立                                  ✅
random-mask 均值更高                                  ✅ 数值事实
random-mask 稳定差于 no-mask                          ❌ 区间跨 0
random-mask 方差与 no-mask 相当                       ❌ random 方差明显更大
51 步足够支撑 DomainDrop 归因                         ❌ 模型仍处于训练早期
可以进入 DomainDrop                                   ❌
```

seed 42 的 random-mask 是主要异常来源；其余 4 个 seed 的差异都很小。当前证据仍不足以把 random-mask 的均值差解释为稳定效应。

## 4. 下一步

升级到 200 步、相同 center crop、5 seeds：

```text
no-mask × 5 seeds
random-mask × 5 seeds
val/test 各 32 batches = 256 samples
```

若 200 步后 paired delta 仍跨 0：

```text
继续增加 seeds 或评估样本
不实现 DomainDrop
不跑 success rate/2k
```

只有 random-mask 的基线方差稳定后，才实现原生 source styles 11-30、20 类、DINOv2 微调、多 block GRL/WRS 的 DomainDrop 消融。
