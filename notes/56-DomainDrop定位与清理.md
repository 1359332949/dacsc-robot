# 56 - DomainDrop 在具身策略中的定位与代码清理

日期：2026-10-22

## 1. 清理内容

已删除以下没有证据支持的做法：

- 冻结 DINOv2 诊断旁路接入主 policy；
- 用人为分组标签训练域头；
- 用 held-out style 的域分类 accuracy 作为主方法 gate；
- 与上述做法对应的离线 warm-start、校准、在线验证脚本和结果。

当前 active 代码只使用数据原生字段：

```text
task
style_id
layout_id
episode
```

## 2. 为什么 DomainDrop 只能做消融

相关证据：

1. `notes/13-精读2610.08183-CorP论文.md`：冻结 DINOv2 与微调 DINOv2 的动作性能差距很大；
2. CoRP 的无差别信息瓶颈消融会伤害动作性能；
3. `notes/43-top-u遮挡因果结果.md`：域敏感维度遮挡接近 random，而动作重要维度遮挡明显更伤动作；
4. DomainDrop / PLACE / DgCD 的原始验证主要是图像分类 DG，不是具身动作策略；
5. `notes/52-pure-dino-style-full-audit.md`：冻结 DINOv2 的 style 可解码性审计只能证明“可解码”，不能证明 channel dropout 会提升动作。

这些证据合在一起只支持一个结论：

> **DINOv2-DomainDrop 是合理的 ViT 适配消融，但当前不是具身策略的已验证主路线。**

## 3. 如果继续做 DomainDrop，允许的最小实现

```text
DINOv2: 微调，不冻结
domain label: source style_id 11-30
domain head: 每个选定 DINOv2 block 一个
所有 head: 正常更新，并通过 GRL 向 backbone 反传
mask: D维 min-max -> T维 GAP -> WRS
层选择: 多 block 随机选一个执行 mask
```

评估：

```text
domain head 监督/验证: source styles 内部 episode
策略 val: styles 31-40
策略 test: styles 41-50，仅最后一次
指标: val_action_mse / test_action_mse / success rate
```

## 4. 必须先成立的验收

```text
DomainDrop val_action_mse 不差于 random mask
DomainDrop test_action_mse 有改善趋势
success rate 不下降
3 seeds 方向一致
```

否则应把 DomainDrop 写成负结果，而不是更换标签或继续调 gate。

## 5. 主基线

```text
Official CoRP on LIBERO
→ CoRP-RoboCasa interface adaptation
→ no-gate CoRP-RoboCasa baseline
```

旧自定义 flow 只保留为工程验证，不作为论文主结果。DomainDrop / PLACE / DgCD 只作为受控消融。
