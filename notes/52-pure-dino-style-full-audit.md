# 52 - 冻结 DINOv2 原生标签表征审计

日期：2026-10-22
服务器：`i-2.gpushare.com:40424`
数据：RoboCasa365 atomic，61 tasks
脚本：`scripts/audit_pure_dino_style.py`
结果：`audit/pure_dino_style_native/`

## 1. 审计目的与边界

这不是策略泛化实验，也不是 DomainDrop 有效性证据。它只回答：

> 冻结 DINOv2 特征中，RoboCasa 原生 `style_id` 是否有可解码信息？

特征为 DINOv2 ViT-S/14：

- left agentview；
- right agentview；
- wrist；
- 三视角 concat；
- patch-token mean 与 CLS 两种表示。

标签只使用数据原生的：

```text
style_id
layout_id
task
```

实验按 episode 划分 train/val，同一 episode 的相邻帧不会跨 split。没有 FiLM、语言条件、策略梯度、门控或随机 crop。

## 2. 样本与支持

- 扫描到 6935 个带标签与视频的 episode；
- 选择 400 episodes；
- 50 个 style，每个 style 6 train / 2 val episode；
- 300 train episodes 与 100 val episodes；
- 每 episode 采 3 帧，三视角合计 3600 个 view-frame 特征。

抽样图：`audit/pure_dino_style_native/samples/contact_sheet.png`。图中固定 `CloseMicrowave` 与原生 `layout_id=13`，展示 styles 11 / 29 / 37 各 3 帧；外观差异可直接检查。

style 的原始类别是 50 类，随机平衡准确率为 `0.02`。

## 3. 原生 style 结果

### 3.1 最佳结果

| 特征 | 分类器 | episode balanced acc | frame balanced acc |
|---|---|---:|---:|
| left agentview patch mean | linear | **0.520** | 0.463 |
| right agentview patch mean | linear | 0.510 | 0.487 |
| right agentview patch mean | MLP | 0.510 | 0.443 |
| concat | linear / MLP | 0.420 | 0.380 / 0.383 |
| left agentview CLS | linear | 0.330 | 0.313 |
| right agentview CLS | linear | 0.350 | 0.283 |
| wrist patch mean | linear | 0.120 | 0.117 |
| wrist CLS | linear | 0.100 | 0.077 |

结论：

1. `style_id` 不是纯噪声；agentview patch mean 明显高于 50 类随机水平；
2. wrist 几乎没有可解码的 style；
3. 三视角 concat 没有超过单个 agentview；
4. 这是“可解码性”证据，不是“style 域已被干净分离”的证据。

### 3.2 固定 layout 后

在选定样本中，选择包含最多 style 的单一原生 `layout_id=12`，只用该 layout 的 episode 重新做 style 线性 probe：

```text
类别数:                 22
train episodes:          61
val episodes:            30
episode balanced acc:    0.250
```

这个控制集的类别 support 很小：22 个类别中多数只有 1–2 个 val episode。因此它不能证明稳定的 style 域结构，也不能反过来证明 style 完全不可学。正确结论是：

> **全局 style probe 有信号；固定 layout 后样本量不足，当前证据不足以证明存在跨 layout 稳定、可用于域门控的 style 表示。**

## 4. 原生 layout/task 结果

这些结果只作为 sanity reference，不用于策略模型选择。

| 标签 | 最佳特征 | 分类器 | episode balanced acc | 类别/支持说明 |
|---|---|---|---:|---|
| layout | right agentview patch mean | MLP | 0.490 | 7 类，100 val episode |
| task | concat | MLP | 0.579 | 45 类，support 高度不均（多数 1–4） |

task 的多类 support 很不均衡，0.579 不能作为强可分性结论。layout 的 7 类结果也高于随机，但本轮没有把它作为域头训练 gate。

## 5. 能得出的结论

```text
冻结 DINOv2 agentview 中存在可解码的 style 信号   ✅
wrist 不是主要 style 视角                          ✅
固定 layout 后仍有稳定 style 域结构                ❌ 当前证据不足
可以把 style classifier 当作已验证的 DomainDrop 域头 ❌
channel dropout 能提升动作                         ❌ 本审计没有测
```

因此：

> **本审计只支持“style 在冻结视觉特征中可解码”，不支持把 DomainDrop 升级为具身策略主路线。**

DomainDrop 只能作为严格 split 下的受控消融；主基线仍是微调 DINOv2 + resampler + flow/DDPM + 动作头。

## 6. 删除的错误做法

本脚本已删除所有人为标签分组，只保留原生的 `style_id / layout_id / task`。
凡是用人为分组训练域头、并把其 accuracy 当作主 gate 的旧结果，均不再引用。
