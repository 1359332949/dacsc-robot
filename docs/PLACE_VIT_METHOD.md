# PLACE-ViT for CoRP

更新时间：2026-11-09

状态：**500-step 筛查完成；PLACE progressive 未超过 fixed-random，停止扩 5k。**

## 1. 固定上游

```text
paper:      https://arxiv.org/abs/2112.03676
repository: https://github.com/lingeringlight/PLACEdropout
revision:   d17f0a79dc2795f315a68ab745aa98b231000f2b
```

官方仓库只有 ResNet/CNN 实现，没有 ViT 实现，也没有检索到仓库级 LICENSE 文件。
因此本实现命名为：

```text
PLACE-inspired ViT progressive channel dropout for CoRP
```

不能称为官方 PLACE ViT 复现。

## 2. 官方公式

官方 `models/resnet_PLACE.py` 的 progressive ratio 为：

```text
p_t = p_max * (2 / pi) * atan(t / velocity)
```

默认：

```text
p_max = 0.33
velocity = 4
```

每个 batch 随机选一个候选 stage；在该 stage 内，每个样本随机选择
`floor(C * p_t)` 个 channel 置零，并使用：

```text
x' = x / (1 - p_t)
```

恢复尺度。推理时 dropout 关闭。

## 3. CoRP 映射

```text
[views * batch, tokens, 384]
→ [batch, 384, views, tokens]
```

384 是 DINOv2 block-output channel；三视角和所有 token 共享同一个
per-sample channel mask。

当前候选层：

```text
2, 5, 8, 11
```

这是四个 ViT block 位置的项目适配，不是官方 ResNet stage 的逐字复制。

## 4. 进度定义

官方 `t` 是 stage-2 epoch 序号。CoRP pilot 使用固定优化步预算，因此适配为：

```text
equivalent_epoch = reference_epochs * current_step / total_stage2_steps
```

默认 `reference_epochs=30`，与官方 stage-2 默认 epoch 数一致。
这把官方 30-epoch 的 arctangent 课程映射到预注册的 stage-2 step budget；
该映射写入 checkpoint，且 fixed-random 对照使用同一个 stage-2 预算。

## 5. fixed-random 对照

PLACE 的对照不是 DomainDrop 的 `random`，因为后者还带 domain head、GRL、
domain loss 和 action consistency。PLACE fixed-random 使用：

```text
同一随机层选择
同一候选层集合
同一 channel budget
同一 1/(1-p) recovery
无 domain head / GRL / domain loss / consistency loss
```

唯一区别是：

```text
fixed-p:  p_t = 0.33
place:    p_t = 0.33 * 2/pi * atan(t/4)
```

## 6. 500-step 实验

```text
baseline_30k
place_random_500
place_500
```

所有条件固定：

- 同一初始 checkpoint；
- 同一 source train selection；
- 同一 batch size；
- 同一 seed；
- 同一 style-balanced batch（12 styles × 4 samples）；
- 同一 eval support 与 color_jitter；
- 同一 optimizer budget。

PLACE 主实验只使用原 CoRP flow loss，不混入 DomainDrop 的 consistency。

## 7. 验收规则

必须同时满足：

1. `p_t` 单调增长并符合官方 arctan 公式；
2. eval 严格 no-op；
3. actual drop ratio 与 `floor(C*p_t)` 一致；
4. 三视角共享 mask；
5. fixed random 与 PLACE 只有比例调度不同；
6. clean 不明显下降；
7. PLACE 相对 fixed random 有稳定改善。

如果 `place ≈ place_random`，结论只能是“渐进调度没有额外价值”，不能声称
PLACE 的方法特异性成立。500-step 通过前不扩 5k，不进入闭环，不碰 target 1–10。

## 8. 500-step 实际结果

| condition | source clean | source jitter | val clean | val jitter |
| --- | ---: | ---: | ---: | ---: |
| 0-step baseline | 0.03175 | 0.08689 | 0.02962 | 0.06208 |
| place_random 500 | 0.03243 | 0.07980 | 0.03028 | 0.06066 |
| place 500 | 0.03213 | 0.08411 | 0.03067 | 0.06169 |

PLACE 相对 fixed-random：

```text
source jitter: 差约 5.4%
val jitter:    差约 1.7%
source clean:  略好
val clean:     略差
```

因此 arctangent 渐进调度没有表现出额外收益。当前不扩 5k、不跑闭环、不做超参搜索；
完整 support、hash 和决策见[place_pilot_500_20261109.json](../audit/place_pilot_500_20261109.json)。
