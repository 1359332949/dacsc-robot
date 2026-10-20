# 51 - ResNet18 域头：离线 probe 可学，在线 DomainDrop 头不可学

日期：2026-10-20  
审计：`notes/50-CNN-domain-audit.md`  
结果：`audit/cnn_domain_audit_summary.csv`

## 1. 同一特征空间的离线线性 probe

ResNet18 stage-end GAP，3840 样本，3 视角 × FiLM 1/3/5/7：

| label | balanced acc |
|---|---:|
| style 连续 3 组 | **0.593** |
| style 连续 3 组 + layout 残差 | 0.610 |
| layout coarse10 | **0.538** |
| task coarse10 | **1.000** |
| style exact 50 类 | 0.801* |
| style KMeans K=3 | 0.794* |

`*`：KMeans 用同一特征空间聚类，存在循环性；50 类 probe 也可能过拟合。  
关键结论仍然成立：

```text
style 3 粗域不是完全不可分，离线线性 probe 可以到 0.593；
layout 0.538；task 1.000。
```

## 2. 在线 DomainDrop 域头（51 步）

| 诊断 | style acc | layout acc | 说明 |
|---|---:|---:|---|
| fixed7 + LayerNorm + GRL 0.25 | 0.333 | — | A |
| fixed7 + LayerNorm + GRL 0 | 0.318 | — | B |
| 所有 stage-end + GRL 0 + 不 mask | 0.26–0.33 | — | C |
| 官方随机 stage + 所有层 loss + norm + GRL 0.25 | 0.333 | — | D |
| 官方随机 stage + 所有层 loss + raw Linear | 0.27–0.34 | — | E |
| global norm + GRL 0 + head LR 1e-3 | 0.281 | 0.156 | 固定 layer7 |
| global norm + GRL 0 + head LR 1e-2 | 0.344 | 0.115 | 固定 layer7 |
| frozen backbone + global norm + LR 1e-2 | **0.432** | — | 固定 layer7 |

## 3. 关键判断

### 3.1 标签对齐没有问题

- style: min 11, max 60, 50 个唯一值；
- layout: min 11, max 60, 50 个唯一值；
- style 粗域映射 `(id-11)//20` 与审计脚本一致；
- 域头在 step0/10/20 的 loss/acc 都有变化，说明 mask/标签/特征顺序没有明显错位。

### 3.2 离线可学，在线不可学

```text
style 离线 0.593  vs  在线 0.28–0.43
layout 离线 0.538  vs  在线 0.11–0.16
```

这不是“标签完全没有信号”，而是：

> **在线域头的优化条件/样本效率/特征漂移，远差于离线 probe。**

### 3.3 冻结 backbone 有改善，但不够

```text
style 在线：
moving backbone + LR 1e-2: 0.344
frozen backbone + LR 1e-2: 0.432
```

冻结能提升，说明 backbone 的 DP 训练确实在改变特征分布；  
但 0.432 仍低于离线 0.593，说明在线 head 的优化/样本效率本身也不足。

### 3.4 不是单一归一化问题

- LayerNorm、raw Linear、global standardization 都试过；
- GRL 0/0.25 都试过；
- head LR 1e-3/1e-2 都试过；
- 固定 layer7、所有 stage-end、随机 stage 都试过。

所以不能简单归因于某一个超参。

## 4. 下一步：先 warm-start 域头，再谈 style 分域

最直接、最省算力的修法是：

```text
用离线特征缓存预训练每个 stage-end 的 domain head
→ 保存 head 权重
→ 训练时以较小 LR fine-tune
→ 再决定是否加 GRL
```

这相当于给 DomainDrop 一个“可用的域头初始化”，而不是让它在 51 步内从零学。

验收：

```text
layout warm-start:
  domain_acc 明显高于 1/10
  domain_loss 明显低于 ln(10)

style warm-start:
  domain_acc 明显高于 1/3
  domain_loss 明显低于 ln(3)
```

如果 warm-start 后 layout 能学、style 只能到 ~0.5，  
再做视觉聚类/残差化 style 域；不要在域头还没建立时继续调 dropout。

## 5. 当前不做

```text
不跑 3 seeds
不跑 200 步
不跑 2k
不继续调 GRL/lambda/LR/layer
```

一句话：

> **离线 probe 证明标签有弱线性信号；在线域头失败是优化/漂移问题，不是标签完全无效。
> 下一步应先 warm-start 域头，再用 layout sanity check 验证整条链路。**
