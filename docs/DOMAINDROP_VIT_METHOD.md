# DomainDrop-ViT for CoRP

更新时间：2026-11-09

状态：**官方逐位置 normalization 已修正；旧 500-step 结果标记为 provisional/invalidated，
等待 plain balanced control、修正后 random/DomainDrop 和 style-probe 门禁。**

本文固定当前通道方法的来源、CoRP 张量映射、训练目标和实验门禁。它不是论文结果，
也不能把 ResNet 上的 DomainDrop 数字直接迁移到 CoRP/RoboCasa。

## 1. 为什么选 DomainDrop

DomainDrop（ICCV 2023，arXiv:2308.10285）直接研究**训练期抑制域敏感 channel**：

1. 在多个中间层放置 domain discriminator；
2. discriminator 与 backbone 之间使用 gradient reversal；
3. 用真实 domain 对应的 classifier 权重计算 channel score；
4. 用 weighted random selection（WRS）选择要丢弃的 channel；
5. 只随机激活一个中间层做 dropout；
6. 对同一输入的两次 dropout 预测加 dual-consistency loss；
7. 推理时关闭 dropout。

这与当前目标一致：方法只在 source training 中使用，最终 CoRP 推理仍是普通前向，
不读取 clean reference、GT action、test domain label 或 target fixture。

固定上游：

```text
repository: https://github.com/lingeringlight/DomainDrop
revision:   a2d9607d4f1fb9ba35a9fa48fbceff82fd67d15f
paper:      https://arxiv.org/abs/2308.10285
```

官方仓库只有 ResNet/CNN 实现，没有 ViT 实现，也没有仓库级 LICENSE 文件。因此本实现只能命名为：

```text
DomainDrop-inspired ViT channel dropout for CoRP
```

不能写成“官方 DomainDrop 的 ViT 复现”。

## 2. 官方公式到 CoRP 的逐项映射

### 2.1 Channel score

官方 `LayerDiscriminator.get_scores` 对真实 domain class 的 FC 权重 `W[domain]` 和特征相乘：

```text
score = normalize(feature * W[true_domain])
```

这里的第一次 normalize 必须在每个 `(H,W)` 位置沿 channel 维做 min-max：

```text
scores.max(dim=1, keepdim=True)
scores.min(dim=1, keepdim=True)
```

不能把 `C×V×T` 全部 flatten 后做一次全局 min-max。CoRP 的映射是：

```text
按 C 在每个 (view, token) 位置归一化
→ adaptive average pool views/tokens
→ 再按 C 做一次 min-max
→ WRS
```

修正后的实现已在 50 组随机 feature/weight/label/seed 上与官方 WRS mask 逐元素一致。

对 CoRP，ViT block 输出为：

```text
[views * batch, CLS+patch, 384]
→ [batch, 384, views, CLS+patch]
```

前两维被当作官方 CNN 的 `[B,C,H,W]` 使用。三个相机和所有 token 共用同一个
per-sample、per-channel mask。384 维 block-output channel 是当前方法对象；
它不是 1536 维 GELU 中间神经元，也不是 attention head。

### 2.2 WRS

官方对 channel score `s_j` 和随机数 `r_j` 计算：

```text
key_j = r_j ** (1 / s_j)
```

选择 key 最大的 `M = int(C * P_drop)` 个 channel。分数越高，越可能被丢弃。
本实现保留官方阈值和 `>` 行为；由于连续值理论上可能相等，实际丢弃数会被记录，
不能假设每个 batch 都严格等于 nominal `M/C`。

### 2.3 Layer-wise dropout

官方默认：

```text
P_drop = 0.33
P_active = 0.8
GRL lambda = 0.25
```

CoRP ViT-S/14 有 12 个 block。当前 pilot 选四个等距中间层：

```text
layers = 2, 5, 8, 11
```

这是把 ResNet 四个 stage 的 discriminator 位置映射到 ViT 深度，属于适配参数，不是官方
ViT 最优配置。每个 step 仍只允许一个层实际 mask；其他层只计算 domain logits。

### 2.4 Scale recovery

官方实现把 binary mask 乘以 `mask.numel() / mask.sum()`，本实现逐项保留。
该恢复按整个 `[B,C]` mask 统计，不是逐样本重新归一化；这是官方行为，不应擅自改成
常见的 inverted-dropout per-sample scaling。

### 2.5 Domain loss and GRL

每个配置层的 discriminator 输出都参与 domain cross-entropy。分类器学习 source style，
backbone 经 GRL 接收反向梯度。`random` 对照也保留完全相同的 discriminator、GRL、
domain loss、consistency 和 dropout budget，只把 channel score 换成随机 score。

因此最重要的消融是：

```text
random: same training graph, random channel selection
domaindrop_vit: same training graph, domain-sensitive channel selection
```

如果没有这个 matched random 对照，不能把收益归因于“域敏感 channel 选择”。

## 3. 动作模型中的 dual-consistency 适配

DomainDrop 原论文比较同一图像两次 dropout 后的 softened class logits，用对称 KL。
CoRP 输出的是 action velocity 回归，没有 class logits。当前适配为：

```text
同一 batch 复制两份
同一 flow eps / tau
一个 active layer 的两个不同 channel mask
velocity_1 与 velocity_2 的 masked action MSE
```

这部分是**明确的项目适配**，不能称为官方 DomainDrop KL loss 的等价复现。
最终权重和它是否真的改善闭环，必须由 pilot 与 matched control 决定。

## 4. 已实现的安全与审计条件

- controller 只在 `model.train()` 时改 block output；
- `model.eval()` 时 hook 严格 no-op；
- 只改当前激活层，其他层输出保持不变；
- 三视角与所有 token 共用 per-sample channel mask；
- random 与 DomainDrop 使用同一 domain head 和 domain loss；
- flow `eps`/`tau` 在两份复制 batch 中完全相同；
- discriminator 参数进入 optimizer、checkpoint 和 resume state；
- 每个日志 step 记录 active layer、实际 channel drop fraction、domain loss、
  consistency loss 和 domain balanced accuracy；
- `channel_diagnostics` 记录 per-style recall 与 top confusions；
- test/target 1–10 不参与本方法选择。

## 5. 近期相关方法为什么暂不混入同一实现

### DgCD（CIKM 2024）

官方代码：https://github.com/gyeomo/DgCD

固定 revision：

```text
56f65c33f1f61a4539d65784f2ecc2c8c299eb61
```

DgCD 用 environment-wise 与 total normalization 的 discrepancy 给 channel 打分。
其官方代码按固定 32 个同环境样本分组。当前训练 batch 是 mixed-style，单个 batch 内很多
style 只有很少样本，直接套用会产生不稳定的环境统计。要作为 faithful baseline，
必须先设计并验证 style-balanced batch；当前不伪造 DgCD 数字。

### PLACE（arXiv:2112.03676）

PLACE 是 layer-wise + random channel dropout，并用 progressive drop ratio。
当前 `random` 只是 DomainDrop 图上的 matched random control，不应叫 PLACE 复现。

### 近期引用工作

DomainDrop 的后续工作提供了方法边界，但当前没有直接可移植的 CoRP/VLA 官方代码：

- DgCD：discrepancy-guided channel dropout；
- CLEAR（THMS 2026）：channel filtering + ensemble attention；
- DSP-Reg（arXiv:2601.19394）：domain-sensitive **parameter** regularization，
  不是 activation channel dropout；
- Meta/gradient-guided channel masking：cross-domain few-shot 任务，目标与闭环 VLA 不同。

这些工作不能只读摘要就补写成官方实现。DgCD 是下一个最相关的 channel-level baseline，
但必须先解决 environment-wise batch 构造。

## 6. Pilot 协议

从同一个 30k baseline checkpoint 初始化，所有条件固定：

```text
tasks: OpenDrawer, OpenCabinet, TurnOffStove,
       TurnOnSinkFaucet, TurnOnElectricKettle
train styles: 11-50
val styles: 51-60
source holdout residue: 9
train augmentation: none
steps: 5000
batch size: 48 (logical batch; channel methods use two copies)
samples per native style per batch: 4
styles per batch: 12
lr: 1e-5
domain discriminator lr: 1e-3
seed: 42
eval shift: color_jitter
```

`StyleBalancedBatchSampler` 固定每个 batch 内 12 个 style、每个 style 4 个样本。
这是为了满足 domain discriminator 的多域监督前提：40 个 style 在一个普通随机 batch 中
通常每个只有约 2 个样本，无法稳定比较 domain-wise 行为。三个条件必须共用同一个
balanced sampler；它改变的是训练 batch 组成，不是通道方法本身。

主模型从 30k checkpoint 微调，因此 backbone 使用 `1e-5`；domain discriminator 是随机
初始化且只有一层，继续使用 `1e-5` 在 100-step 诊断中无法有效学习。pilot 给随机 head
单独的 `1e-3` 参数组。这是 fine-tuning 适配，不是官方“所有参数同 LR”的复现；
两组 LR 都写入 training origin 和 checkpoint。

三个条件：

```text
control:        channel_method=none
matched_random: channel_method=random
domaindrop:     channel_method=domaindrop_vit
```

入口：

```bash
OUT=/hy-tmp/runs/v10_channel_pilot_control \
INIT_MODEL=/hy-tmp/runs/v10_baseline_30k_oss_cache4_f2/model.pt \
CHANNEL_METHOD=none \
bash scripts/run_v10_channel_pilot.sh
```

## 7. 进入下一阶段的 gate

先看训练正确性和 clean 保持，再看 shift 收益：

1. loss、domain/consistency 项有限；
2. clean source/val MSE 不出现明显崩溃；
3. DomainDrop 不能只优于 `none`，还必须优于 matched `random`；
4. color_jitter MSE 的改善必须和 clean 代价一起报告；
5. 只有 5k pilot 出现可重复方向，才扩到更长训练或闭环 SR；
6. 闭环 SR 仍是最终任务判据，离线 MSE 不能替代。

如果 DomainDrop 与 random 无差异，应停止继续调通道 gate，转向训练数据覆盖、
domain randomization、像素增广或更上游的表示学习，而不是继续堆通道技巧。

## 8. 旧 500-step 结果（provisional，公式修正前）

**该结果使用过错误的全局 flatten normalization，不能作为 DomainDrop 方法结论。**

固定同一 baseline、同一 support、同一 seed、同一 balanced batch 和相同训练图：

| condition | source clean | source jitter | val clean | val jitter |
| --- | ---: | ---: | ---: | ---: |
| 0-step baseline | 0.03175 | 0.08689 | 0.02962 | 0.06208 |
| matched random 500 | 0.03239 | 0.07363 | 0.03003 | 0.05776 |
| DomainDrop-ViT 500 | 0.03257 | 0.07282 | 0.03007 | 0.05712 |

DomainDrop 相对 matched random：

```text
source jitter: 仅好约 1.10%
val jitter:    仅好约 1.10%
clean:         略差
```

它可以保留为工程 short-screen 记录，但结论必须改写为：

```text
公式修正前的 500-step screen 没有显示明显特异性；
需要 plain balanced control、修正后匹配对照和 style-probe 后才可判断。
```

当前仍不扩到 5k，不启动 gate 或通道扫描。

完整的 support、hash、per-task 结果与决策见
[channel_pilot_500_20261109.json](../audit/channel_pilot_500_20261109.json)。
