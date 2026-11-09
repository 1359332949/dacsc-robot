# DgCD-ViT for CoRP

更新时间：2026-11-09

状态：**官方公式适配、environment-balanced sampler 和本地数值测试已完成，等待 GPU smoke/500-step。**

## 1. 固定上游

```text
paper:      https://doi.org/10.1145/3627673.3679539
repository: https://github.com/gyeomo/DgCD
revision:   56f65c33f1f61a4539d65784f2ecc2c8c299eb61
license:    MIT
```

官方仓库只有 ResNet/CNN 实现，没有 ViT 实现。因此当前名称固定为：

```text
DgCD-inspired ViT discrepancy-guided channel dropout for CoRP
```

## 2. 官方核心

官方 `domainbed/networks/DgCD.py` 对每个 environment 和整个 batch 做不同标准化：

```text
environment-wise alignment
vs
total-batch alignment
```

然后用两者的 log-softmax KL discrepancy 对每个 channel 求局部 scaler 梯度，并按梯度方向做一次固定范数更新。更新后的 scaler 再用于计算 channel discrepancy：

```text
score = scaling((aligned_total - aligned_environment)^2)
```

scalar：

```text
rho = 0.05
drop ratio = 0.33
```

dropout 仍使用 weighted random selection，并执行官方：

```text
mask = mask * mask.numel() / mask.sum()
```

## 3. CoRP 适配

ViT block output：

```text
[views * batch, tokens, 384]
→ batch, channels, views, tokens
```

384 维 channel 在 `V×T` 位置上做平均后进入 DgCD，mask 对三视角和全部 token 广播共享。

官方 ResNet 在每个 step 选择：

```text
一个随机中间层 + 最后一层
```

当前 ViT 候选层：

```text
2, 5, 8, 11
```

保留该调度：

```text
random layer from [2, 5, 8]
+ layer 11
```

## 4. Environment-balanced batch

官方 DgCD 代码固定每 32 个样本为一个 environment。当前训练改成：

```text
3 styles × 32 samples = batch 96
```

每个 32-sample 连续 group 只含一个原生 style；group 内尽量在任务间均匀分配，避免 task shortcut 被当成 style discrepancy。

matched random 使用：

- 同一 style groups；
- 同一 two-layer schedule；
- 同一 channel drop budget；
- 同一 recovery；
- 只把 DgCD discrepancy score 换成随机 score。

## 5. 数值与设备边界

- DgCD feature 和 scaler 固定 FP32；
- 主 CoRP 训练仍使用 BF16 autocast；
- DgCD 的局部 backward 只更新临时 scaler，不进入主 optimizer；
- 主模型梯度只能通过 `x * mask` 路径产生，不能让 discrepancy 路径反向污染主模型；
- 推理时 DgCD hook 严格 no-op。

DgCD 主实验不加入 DomainDrop 的 domain discriminator、GRL 或 consistency loss。

## 6. 实验顺序

```text
dgcd_random_500
dgcd_500
```

两者必须使用同一 environment-balanced batch、同一层调度、同一 seed、同一 drop ratio 和同一 optimizer budget。

只有 DgCD 稳定优于 matched random，且 clean 不明显下降，才可扩到 5k；否则停止该路线。
target styles 1–10 仍封存。
