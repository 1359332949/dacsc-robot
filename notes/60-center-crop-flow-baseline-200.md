# 60 - 统一 center-crop 的 200 步 no-mask / random-mask 5-seed baseline

日期：2026-10-22
服务器：`i-2.gpushare.com:40424`
相关提交：`27682fc`（center crop）、`a3ab79b`（参数化 baseline）
结果目录：`audit/strict_flow_centercrop_baseline_200/`

## 1. 协议

```text
train styles: 11-30
val styles:   31-40
test styles:  41-50
crop:         224 center crop, train/eval identical
random_crop:  false
steps:        200
seeds:        42 / 43 / 44 / 45 / 46
val:          32 batches × 8 = 256 samples
test:         32 batches × 8 = 256 samples
```

Paired 审计：

```text
同 seed 配置差异只有 policy.enable_gating
初始化 seed、sampler seed、val/test 样本相同
```

## 2. 结果

| run | seed | val_action_mse | test_action_mse |
|---|---:|---:|---:|
| no-mask | 42 | 0.304463 | 0.302424 |
| no-mask | 43 | 0.308010 | 0.302020 |
| no-mask | 44 | 0.278079 | 0.267780 |
| no-mask | 45 | 0.318473 | 0.316419 |
| no-mask | 46 | 0.290396 | 0.284320 |
| random-mask | 42 | 0.302520 | 0.310839 |
| random-mask | 43 | 0.299366 | 0.302327 |
| random-mask | 44 | 0.298229 | 0.303435 |
| random-mask | 45 | 0.314340 | 0.323497 |
| random-mask | 46 | 0.303890 | 0.306576 |

均值：

```text
no-mask:
  val  0.299884 ± 0.014128
  test 0.294593 ± 0.016836

random-mask:
  val  0.303669 ± 0.005716
  test 0.309335 ± 0.007673
```

Paired delta（random - no-mask）：

```text
val  mean +0.003785, sample std 0.012370,
     95% CI [-0.011572, +0.019142]

test mean +0.014742, sample std 0.014150,
     95% CI [-0.002825, +0.032309]
```

逐 seed test delta：

```text
42: +0.008415
43: +0.000307
44: +0.035655
45: +0.007078
46: +0.022256
```

## 3. 与 51 步对比

| 指标 | 51 步 | 200 步 |
|---|---:|---:|
| no-mask val std | 0.047654 | 0.014128 |
| random-mask val std | 0.093329 | 0.005716 |
| test paired mean | +0.027292 | +0.014742 |
| test paired std | 0.064094 | 0.014150 |

200 步显著降低了训练早期方差，使 paired 比较更稳定。

## 4. 结论

```text
200 步 baseline 方差明显下降                    ✅
paired 协议成立                                ✅
random-mask test 均值高约 5.0%                  ✅ 数值事实
5 个 seed 的 test delta 全部非负                ✅
test delta 95% CI 仍轻微跨 0                    ⚠️
random-mask 显著有害                            ❌
可以声称 DomainDrop 有效                         ❌
可以进入 2k/success rate                        ❌
```

最准确表述：

> **在统一 center-crop、200 步、5 seeds 的受控比较中，random 33% mask 的 test action MSE 平均高约 5.0%，方向在 5 个 seed 中一致，但 95% 区间仍轻微跨 0。因此这是一个稳定基线和不利趋势，不是一个已证明的显著效应。**

## 5. 结果定位与下一步

这组 200 步 flow 结果仅用于工程验证和噪声估计：

```text
验证 paired protocol / center crop / val-test 链路
确认 200 步明显比 51 步稳定
为迁移到 CoRP 前的工程风险提供参考
```

它不是论文主 baseline。主实验切换到：

```text
Official CoRP inference baseline on LIBERO
→ CoRP-RoboCasa interface adaptation
→ no-gate CoRP-RoboCasa baseline
→ 再比较 random visual mask / domain-sensitive visual gate
```

P0 官方保真审计见 `notes/61-corp-official-p0-audit.md`。
