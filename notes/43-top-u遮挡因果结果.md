# Top-u 遮挡因果结果（2026-10-20）

## 目的

固定 flow K48 checkpoint，用同一 batch、同一 flow noise，比较：

- full
- 遮挡 top-u 33% 维度
- 遮挡 top-v 33% 维度
- 遮挡 random 33% 维度

## 结果

```yaml
full:    1.1390
top_u:   1.1418
top_v:   1.1706
random:  1.1415

top_u  - full:   +0.0028
top_v  - full:   +0.0316
random - full:   +0.0025

top_u  - random: +0.00035
top_v  - random: +0.0291
```

## 解释

- `top-u` 的遮挡影响几乎等于 random；
- `top-v` 的遮挡影响约为 random 的 12 倍；
- 所以 domain-sensitive 维度（u）不是动作因果通道；
- 当前 embedding-dimension gate 的选维目标本身不对。

## 决策

停止在 flow 的 DINOv2 token 特征维度上继续调 gate。  
不再跑 200/2k/2万步，也不把当前 gate 直接搬到 CoRP。

下一步应转向：

```text
token-level style statistics
TFS-ViT style token stylization
resampler 输出上的 token gate
domain-aware probabilistic mask
```

其中最优先的是 **token-level style statistics**，因为它针对 style 的统计量，而不是假设某个 384 维坐标就是域通道。
