# 49 - 官方 CNN baseline 上的 DomainDrop 忠实复现与诊断

日期：2026-10-20  
代码：`dacsc/cnn_dropout.py`, `dacsc/cnn_dropout_policy.py`  
配置：`train_official_cnn_dropout.yaml`  
诊断目录：`/hy-tmp/runs/cnn_dropout_diag`

## 1. 这次改成了什么

官方 DomainDrop 的核心结构已经按 `resnet_domain.py` 对齐：

```text
ResNet stage-end 特征（layer1/2/3/4，对应 FiLM 索引 1/3/5/7）
→ 每个 stage 一个 domain discriminator
→ 每一步所有 stage 的 domain loss 都参与训练
→ 随机选一个 stage 做 WRS channel mask
→ layer_wise_prob=0.8（有 20% 概率这一步不做 mask）
→ 只有被选中的 stage 乘 mask
```

同时支持：

```text
domaindrop       原版 raw Linear(GAP(feature)) 域头
domaindrop_norm  LayerNorm + Linear（稳定性修补，不是官方）
fixed / random / stage_end_random / stage_end_multi / last4_random
```

每个域头记录：

```text
domain_loss
domain_acc
feature_norm
logit_norm
grl
keep_frac / ratio
```

## 2. 诊断 A-E（51 步，1 seed，batch 96，3 个粗域 × 32 样本）

| 诊断 | 配置 | domain acc | domain loss | logit norm | action MSE |
|---|---|---:|---:|---:|---:|
| A | fixed layer7 + norm + GRL 0.25 | 0.333 | 2.58 | 3.22 | 0.2805 |
| B | fixed layer7 + norm + GRL 0 | 0.318 | 1.13 | 0.36 | 0.2808 |
| C | 所有 stage-end + GRL 0 + 不 mask | 0.26–0.33 | 1.10–1.14 | 0.33–0.58 | 0.2938 |
| D | 官方随机 stage + 所有层 loss + norm + GRL 0.25 | 0.333 | 1.58–3.13 | 1.72–3.91 | 0.2806 |
| E | 官方随机 stage + 所有层 loss + 原版 raw Linear | 0.27–0.34 | 1.41–4.12 | 3.47–10.18 | 0.2803 |

3 类随机水平：`1/3 = 0.333`。

### A → B：GRL 是 logits 膨胀的主因，但不是域头不学的主因

```text
GRL 0.25 → 0：logit_norm 3.22 → 0.36
              domain_loss 2.58 → 1.13
              domain_acc 0.333 → 0.318（仍然接近随机）
```

关闭 GRL 后域头稳定了，但**没有变得可分**。

### C：所有 stage-end 层都不能线性分开 3 个粗域

| layer | domain acc | domain loss | feature norm | logit norm |
|---|---:|---:|---:|---:|
| 1 | 0.271 | 1.131 | 12.21 | 0.578 |
| 3 | 0.297 | 1.138 | 16.31 | 0.403 |
| 5 | 0.260 | 1.123 | 24.43 | 0.329 |
| 7 | 0.328 | 1.104 | 38.11 | 0.508 |

结论：

> **不是 layer 7 选错了；所有 stage-end 层都接近 3 类随机水平。**
> **问题不在 dropout 实现，而在“3 个粗域标签在 ResNet18 GAP 特征里不可线性分离”。**

### D/E：官方随机层 + 所有层 loss 也不能救回域头

D（LayerNorm）和 E（raw Linear）的 domain acc 仍然在 0.27–0.34。
E 的 feature_norm 达到 26–78，logit_norm 达到 3.5–10.2，说明 raw Linear 确实会放大 logits；
LayerNorm 能把 logits 压回 1.7–3.9，但不会创造域可分性。

## 3. 1-seed screen 的 action MSE

同一 batch 96、3 粗域 × 32 样本：

| group | action MSE |
|---|---:|
| none | 0.2935 |
| random 33% | **0.2690** |
| DomainDrop (A/D/E) | 0.2803–0.2808 |
| PLACE | 0.2808 |
| DgCD | 0.2809 |

只能说明：

- 33% random channel dropout 有短期正则收益；
- PDomainDrop/PLACE/DgCD 的收益都接近，且都小于 random；
- 1 seed 下差异不足以做统计结论。

## 4. 最关键的科研判断

当前 DomainDrop 失败的原因是：

```text
不是固定 layer 7
不是 GRL 太强
不是 LayerNorm 不够
不是 domain loss 没有覆盖所有层
而是：3 个粗域标签在 ResNet18 GAP 特征里不可线性分离
```

这与之前 DINOv2 风格审计一致：

```text
style 是真实存在的
但 style 与 layout/task 强耦合
GAP 特征被 layout/fixture 主导
连续 style ID 分组没有视觉依据
```

因此，继续跑 3 seeds / 200 步没有意义：域头在训练 50 步后仍然接近随机，
mask 的“域敏感性”不可信。

## 5. 下一步

在重新跑 DomainDrop 长实验之前，必须先解决域定义/表示：

1. **视觉聚类分域**：用 DINOv2/ResNet 特征对 style 做聚类，而不是连续 ID 分组；
2. **layout 残差化域特征**：domain head 输入先减掉 layout/view 均值；
3. **单 style 域**：用 50 个 style 的小样本域，或对比学习式 domain head；
4. **sanity check**：用 task/layout 作为域标签，先验证 DomainDrop 实现能学到可分域；
5. 只有 domain acc 稳定显著高于 1/3，再跑 3 seeds / 200 步。

## 6. 当前定位

```text
官方 CNN baseline：已能跑通
DomainDrop 结构：已对齐（stage-end 多头、所有层 loss、随机层、layer_wise_prob）
DomainDrop 域头：❌ 不可分
DomainDrop 动作收益：⚠️ 接近 random dropout，不能归因于域选择
PLACE：✅ 渐进 ratio 正常
DgCD：✅ WRS 正常
token-style canonical normalization：旁支，201 步 -9.8%（见 note 48）
```

一句话：

> **这次不是“DomainDrop 代码又没写对”，而是域头在所有 stage-end 层都学不到 3 个粗域。
> 在域定义/特征表示修正之前，继续跑 DomainDrop 的 3-seed/200 步只会得到不可解释的结果。**
