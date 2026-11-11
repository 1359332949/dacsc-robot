# 项目章程（2026-11-11 起生效）

> 以预训练 **CoRP** 为固定 VLA 基线，不重新训练整个模型，研究**低成本的推理阶段
> Test-Time Adaptation**，通过复用已有论文的自监督信号与轻量参数更新，在保证
> 代码正确、实验可复现、clean 性能不明显下降的前提下，提高目标域任务成功率。

## 1. 目标与边界

| 维度 | 约束 |
|---|---|
| 基础模型 | CoRP 官方 checkpoint，结构与配置不改 |
| 训练 | 不做离线全模型训练；允许测试时少量参数更新（Adapter / 低秩 / 注意力投影 / prompt） |
| 研究类型 | Test-Time Adaptation / Test-Time Training |
| 目标 | 目标域任务成功率（优先相机/噪声等确有下降的域） |
| 方法来源 | 优先复用论文与公开实现；创新为小幅、单因素改动 |
| 代码 | 正确性、可复现性优先于效果 |
| 实验 | 最少实验先验证有效性，不堆矩阵 |

## 2. 架构事实（必须按官方代码说话）

```
多视角图像 + proprioception + 语言
   → DINOv2 (ViT-S) + T5-tiny
   → Cross-Attention Resampler（语言条件化 learned queries，48 tokens/view）  ← "Cross-Attention"在这里
   → 条件 tokens（视觉 96 + 语言 + 状态）
   → DiT Action Generator（Flow Matching，num_layers=1）
        SelfAttention: fused qkv Linear(hidden→3*hidden) + projection Linear(hidden→hidden)
   → action chunk（horizon=12，10 Euler steps）
```

**纠正**：动作生成器里没有"视觉 Cross-Attention"；条件 token 以 prefix 进入 DiT。
任何"接口适配"必须明确：改哪个矩阵、数据怎么流过、如何影响最终动作。

## 3. 技术路线（四步）

1. **固定 CoRP 基线**：复核 checkpoint/结构/Flow-Matching 推理/Frozen 评测可信。
2. **复现 SALT 风格 leftover 适配**（优先）：冻结主干，只更新选定注意力投影；
   bounded multiplicative singular-value scaling + Adam + 少量步数 + leftover 自监督 + 门控。
3. **一个最小创新**（SALT 基线出来后再定）：动作敏感性指导的参数选择 / 可靠性门控 /
   clean 保持约束，一次只改一个因素。
4. **验证成功率**：目标域为主，同时报 clean、适配开销、失败类型；只保留可复现改进。

## 4. 每次新增方法必须通过的检查

| 检查项 | 要求 |
|---|---|
| 模型正确 | 官方 checkpoint/配置/结构一致 |
| 位置正确 | 挂载位置与论文定义一致 |
| 参数正确 | 只有计划参数有梯度，其余冻结 |
| 公式正确 | loss/mask/动作单位/Flow 时间方向正确 |
| 初始化正确 | 初始化后输出与 Frozen 逐条一致 |
| 重置正确 | 每 episode 重置参数与优化器 |
| 随机性正确 | 配对预测固定采样噪声 |
| 评测正确 | 相同任务/初始状态/seed/成功判定 |
| 可复现 | checkpoint、配置、commit、日志、逐 episode 结果 |

> **跑通、数学正确、协议有效、性能提高，是四个不同结论，不能混为一谈。**

## 5. 保留资产

通道敏感性分析、Cross-Attention 分析、CPIA、Oracle 实验、相机扰动实验：
保留为**诊断工具 / 候选方法 / 已有结果**，不再决定研究方向。
