# 58 - 严格 split 下的 flow no-mask / random-mask 51 步基线

日期：2026-10-22
服务器：`i-2.gpushare.com:40424`
代码提交：`dc4aec4`
结果目录：`audit/strict_flow_baseline_51/`

## 1. 实验设置

```text
train styles: 11-30
val styles:   31-40
test styles:  41-50
backbone:     DINOv2 ViT-S/14 + resampler K48 + flow matching
seeds:        42 / 43 / 44
steps:        51
```

评估：

```text
max_val_steps:  16 batches × 8 = 128 samples
max_test_steps: 16 batches × 8 = 128 samples
```

- val 只使用 styles 31-40；
- test 只使用 styles 41-50，训练结束后运行一次；
- checkpoint 不保存；
- `train_action_mse_error` 只作为训练 batch 诊断，不作为验证结果。

## 2. 逐 seed 结果

| run | seed | val_action_mse | test_action_mse |
|---|---:|---:|---:|
| no-mask | 42 | 0.761169 | 0.769135 |
| no-mask | 43 | 0.732050 | 0.726994 |
| no-mask | 44 | 0.657006 | 0.654269 |
| random-mask | 42 | 0.887538 | 0.908093 |
| random-mask | 43 | 0.662196 | 0.688616 |
| random-mask | 44 | 0.650933 | 0.670007 |

均值与 seed 标准差：

| group | val_action_mse | test_action_mse |
|---|---:|---:|
| no-mask | 0.716742 ± 0.043880 | 0.716799 ± 0.047445 |
| random-mask | 0.733556 ± 0.108979 | 0.755572 ± 0.108116 |

## 3. Paired delta

定义 `delta = random - no_mask`，正值表示 random 更差。

| seed | val delta | test delta |
|---:|---:|---:|
| 42 | +0.126369 | +0.138958 |
| 43 | −0.069853 | −0.038378 |
| 44 | −0.006073 | +0.015739 |

均值：

```text
val delta:  +0.016814 ± 0.081726
test delta: +0.038773 ± 0.074207
```

按 n=3、t(2)=4.303 粗略计算 95% 区间：

```text
val delta  CI ≈ [-0.186, +0.220]
test delta CI ≈ [-0.146, +0.223]
```

区间均跨 0。

## 4. 结论

```text
严格 split 的 no-mask 和 random-mask 都能完整跑通       ✅
val/test 各 128 个样本，指标来自正确 loader              ✅
random-mask 均值更差                                     ✅ 数值事实
random-mask 稳定差于 no-mask                             ❌ 3-seed 方差过大
可以据此判断 DomainDrop 是否优于 random                  ❌ 当前噪声水平不允许
可以进入 DomainDrop / 200 步                             ❌
```

因此：

> **这轮建立了干净 baseline 的链路和初步数值，但没有建立足以做方法归因的统计稳定性。下一步应先降低 baseline 噪声，而不是接 DomainDrop。**

## 5. 下一步

优先级：

1. 保持完全相同的初始化和数据顺序，增加 seeds（至少 5 个）确认 random-mask 的均值/方差；
2. 检查 random-mask 在 seed 42 的异常是否来自数据顺序或随机 mask RNG；
3. 若 5-seed 仍不稳定，再扩大 val/test 支持或改为固定验证集评测；
4. 只有 no-mask/random 的 paired delta 稳定后，才实现原生 source-style 20 类的 DINOv2-DomainDrop 消融。

不要：

```text
不要用 test 做模型选择
不要把 train_action_mse 当验证
不要把这轮 random 的均值差写成显著结论
不要直接接 DomainDrop 或跑 200/2k
```
