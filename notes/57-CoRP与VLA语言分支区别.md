# 57 - CoRP 与 VLA 的语言分支区别

日期：2026-10-22  
参考：CoRP arXiv `2610.08183`；本地 `notes/13-精读2610.08183-CorP论文.md`  
目的：澄清 CoRP 是否使用 CLIP、是否属于 VLA，以及语言分支在两类模型中的不同角色。

## 1. 一句话结论

> **CoRP 有语言分支，但它不是 VLA；CoRP 论文用的是 T5-Efficient-Tiny，不是 CLIP。**
>
> **VLA 的核心是“预训练视觉语言大模型/LLM 作为策略大脑”；CoRP 的核心是“DINOv2 视觉表示 + 轻量语言条件 + 紧凑动作头”。**

---

## 2. CoRP 的真实组成

CoRP 论文的四个组件：

```text
1. DINOv2-initialized ViT-S/14 视觉编码器
2. T5-Efficient-Tiny 语言编码器 + adapter
3. learned-query cross-attention resampler
4. single-block DiT + flow matching 动作生成器
```

总参数约 **48.9M**。

数据流：

```text
多视角图片
  → DINOv2 ViT-S/14
  → 每视角 256 个 patch token
  → learned-query resampler
  → 每视角 48 个 token

语言指令
  → T5-Efficient-Tiny
  → LayerNorm + linear adapter
  → pooled language embedding
  → FiLM 调制 resampler query / 动作条件

视觉 token + 语言条件
  → DiT + flow matching
  → 动作序列
```

关键点：

- **视觉编码器**：DINOv2，联合微调；
- **语言编码器**：T5-Efficient-Tiny，联合微调；
- **没有 VLM**；
- **没有 LLM decoder**；
- **没有世界模型/未来画面预测**。

论文原文要点：

> "a T5-Efficient-Tiny language encoder ... fine-tune it jointly with the policy"
>
> "Despite having neither a VLM nor a video-generative prior, CoRP reaches 97.0% average success on LIBERO"

---

## 3. 我们代码里的 CLIP 是什么

我们的 RoboCasa 配置里有：

```text
lang_emb: 768 维
CLIPTextModelWithProjection
```

它的作用是：

```text
语言指令
  → CLIP text encoder
  → 768 维文本 embedding
  → 作为 lang_emb 输入策略
```

它不是：

```text
CLIP 视觉编码器当策略大脑
CLIP 和动作 token 一起做自回归
```

也就是说：

```text
我们的 CLIP 只是“文本编码器”
不是 VLA 里的“策略大脑”
```

而且：

- CoRP 论文用 T5-tiny，不用 CLIP；
- 我们用 CLIP text embedding，是本地 RoboCasa 实现的差异；
- 不能把“代码里有 CLIP”说成“CoRP 是 VLA”。

---

## 4. VLA 的真实结构

VLA = Vision-Language-Action。

典型结构：

```text
图片 + 语言指令
       │
       ▼
预训练 VLM / LLM
（PaliGemma、Qwen-VL、SmolVLM、Llama 等）
       │
       ▼
动作专家
（flow matching / 自回归 action token）
       │
       ▼
动作
```

代表：

- π0 / π0.5：PaliGemma + flow matching action expert；
- SmolVLA：SmolVLM + flow matching action expert；
- OpenVLA：Llama/Prismatic VLM + action token。

VLA 的特点：

```text
视觉语言大模型是核心
语言理解/视觉语义在同一个大模型里
模型规模通常几亿到几十亿
动作头通常挂在大模型后面
```

---

## 5. CoRP vs VLA 的核心区别

| 维度 | CoRP | VLA |
|---|---|---|
| 视觉编码器 | DINOv2 ViT-S/14 | VLM 的视觉塔（SigLIP/PaliGemma/Qwen-VL） |
| 语言编码器 | T5-Efficient-Tiny | VLM/LLM 本体 |
| 语言角色 | 轻量条件向量，FiLM 调制 | 大模型里的语义推理 |
| 是否有 VLM/LLM 大脑 | ❌ 没有 | ✅ 有 |
| 是否有世界模型 | ❌ 没有 | 通常也没有（WAM 才有） |
| 动作头 | DiT + flow matching | flow matching / 自回归 token |
| 参数量 | ~48.9M | 通常 0.5B–几十B |
| 训练方式 | 视觉/语言/动作联合微调 | 预训练 VLM + 动作专家微调/LoRA |
| 定位 | 紧凑模仿学习策略 | 视觉语言动作大模型 |

---

## 6. CoRP 和 DP/ACT 的区别

| 维度 | DP | ACT | CoRP |
|---|---|---|---|
| 视觉 | ResNet18/CNN | ResNet18/CNN | DINOv2 ViT-S/14 |
| 视觉预训练 | 通常无 | 通常无 | ✅ DINOv2 |
| 语言 | 通常无 | 通常无 | T5/CLIP 轻量条件 |
| 动作头 | DDPM | 动作块回归 | DiT + flow matching |
| 视觉 token | 全局特征 | 全局特征 | 256 patch → 48 token |
| 规模 | ~100M | ~80M | ~49M |
| 视觉随机化鲁棒性 | 很差 | 很差 | 强很多 |

CoRP 不是“DP 换了个名字”，而是：

```text
预训练视觉表示
+ learned-query token 压缩
+ flow matching 动作头
```

---

## 7. CoRP 和 WAM 的区别

| 维度 | CoRP | WAM |
|---|---|---|
| 是否预测未来画面 | ❌ | ✅ |
| 是否有世界模型 | ❌ | ✅ |
| 视觉骨干 | DINOv2 | 视频生成模型（Wan2.x 等） |
| 模型规模 | 小 | 很大 |
| 算力 | 单卡可跑 | 通常多卡/H100 |
| 核心优势 | 紧凑、视觉表示强 | 能预演未来 |

一句话：

```text
CoRP：看好现在，直接出动作
WAM：先想象未来，再出动作
```

---

## 8. 语言分支到底重不重要

CoRP 论文的结论是：

```text
语言不是永远需要；
当视觉已经能唯一确定任务时，语言作用很小；
当视觉有歧义（例如 LIBERO Goal）时，语言非常关键。
```

所以 CoRP：

```text
保留轻量语言条件
但不把语言模型做成大 VLM 大脑
```

VLA：

```text
把语言理解放进大模型
让语言和视觉一起做语义推理
```

这就是两者最本质的区别。

---

## 9. 对我们的方法的含义

我们的视觉门控应该：

```text
作用在 DINOv2 的视觉 token / feature 维度上
不作用在语言 token
不作用在动作头
```

训练时：

```text
domain label → 域敏感分数 u
action loss → 动作重要性 v（只作为训练信号）
```

推理时：

```text
视觉特征 → 已学好的门控 → 下游
不需要语言输入或动作输入来算门控
```

因此：

> **CoRP 可以作为我们视觉门控的主基线；VLA/WAM 是更大的对照背景，不是我们的主实验平台。**
>
> **CLIP 出现在代码里不等于 CoRP 是 VLA；CoRP 论文的语言编码器是 T5-tiny，且是联合微调的。**

---

## 10. 最终总结

```text
CoRP：
  DINOv2 好眼睛
  + T5-tiny 轻量语言条件
  + resampler 压缩
  + flow matching 动作头
  没有 VLM/LLM 大脑
  没有世界模型

VLA：
  预训练 VLM/LLM 当大脑
  + 动作专家
  语言和视觉语义在大模型里融合

DP/ACT：
  小模型从零学
  视觉和动作头都简单

WAM：
  世界模型预测未来
  + 动作头
```

一句话：

> **CoRP 是“好视觉 + 轻量语言条件 + 紧凑动作头”的模仿学习策略；VLA 是“大视觉语言模型当策略大脑”的模型家族。两者都有语言分支，但语言分支的规模、角色和训练方式完全不同。**
