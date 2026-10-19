# 方法家族与 backbone 调研（基于 XPolicyLab 68 个策略）

> 2026-10-16。数据来自本地 `repos/XPolicyLab/policy/*/README.md` 的机器抽取。

## 第 0 层：先回答"XPolicyLab 是不是已经有调研了？"

**没有。它只做了接口，没做调研。**

| 指标 | 实测 |
|---|---|
| 策略总数 | **68 个** |
| 有 arXiv 号的 | **39 / 68**（其余是 `TBD` / `not released`） |
| 有原始代码链接的 | **31 / 68** |
| 它提供的分类 | 只有 4 个家族标签（VLA / world-action / imitation-learning / memory-augmented），**没有逐个标注** |
| 有没有 backbone 对比表 | ❌ 无 |
| 有没有横向评测结果 | ❌ 无（只有各家的排行榜链接） |

**结论：XPolicyLab 是"工程框架"，不是"综述"。** 它把 68 个模型接进了同一套接口，
但没有告诉你它们各自的技术路线、骨干网络、谁比谁强。**这份调研得我们自己做**（就是本篇）。

---

## 第 1 层：三大家族 + 一个横切维度

用一句话讲清每个家族"在做什么"：

| 家族 | 一句话 | 代表 | 直觉类比 |
|---|---|---|---|
| **模仿学习（IL）经典** | 直接从示范数据学"图像→动作"，模型小、从零训 | **DP**、ACT、RDT-1B | 手写规则 + 小网络 |
| **VLA** | 拿一个**预训练视觉语言模型**当大脑，加个动作头 | **π0 / π0.5**、GR00T、SmolVLA、OpenVLA-OFT、TinyVLA、Evo-1 | 让一个读过万卷书的模型学开机器人 |
| **World-Action Model（WAM）** | 先**预测未来画面**（世界模型），再据此出动作 | **Awomo-0.5**、OpenWAM、FastWAM、AHA-WAM、PatchWAM、X-WAM、DreamZero、GigaWorld-Policy | 先"在脑子里预演一遍"再动手 |
| *（横切）memory-augmented* | 不是独立家族，是"带记忆"的功能标签 | SimpleMemVLA、Mem-0、HWM-CogWAM | 给上面三家加个记事本 |

**注意**：第 4 类不是并列的家族，它是横切维度。

---

## 第 2 层：家族 → 代表方法 → backbone

从 68 个 README 里机器抽取的骨干关键词覆盖：

| 骨干 | 出现次数 | 代表策略 |
|---|---|---|
| **Qwen-VL / LLM** | **21** | Abot-M0, Awomo-0.5, Being-H05, EventVLA, InternVLA-A1/A1.5, HWM-CogWAM … |
| **Wan2.x（视频扩散）** | **11** | AHA-WAM, DreamZero, FastWAM, LingBot-VA, ME-Dex-1.0, ME-U0 … |
| PaliGemma / SigLIP | 4 | π0 系（Dexora-1B, GalaxeaVLA, RDT-1B, Simate-beta） |
| ViT | 4 | HWM-CogWAM, LDA-1B, ME-U0, VPP2 |
| DiT | 4 | H-RDT, InternW0-Δ, StarVLA, Xiaomi-Robotics-1 |
| LoRA（微调方式） | 4 | Mem-0, OpenWAM, TinyVLA, X-VLA |
| DINOv2 | 2 | HWM-CogWAM, LDA-1B |
| **ResNet18** | **只有 DP** | Diffusion Policy（我们用的那个） |

**关键读数**：
1. **Qwen-VL 已经是最常见的 VLM 骨干（21/68）**
2. **Wan2.x 成了 WAM 家族的标配生成骨干（11/68）** —— 这是一条全新的技术栈
3. **ResNet18 只剩 DP 一家在用** —— 我们是这条线上唯一还在用 CNN 的

---

## 第 3 层（最有价值）：技术演进的三个轴

| 轴 | 2023（起点） | 2024–2025 | **2026（现在）** |
|---|---|---|---|
| **① 视觉骨干** | ResNet18 从零训 | DINOv2 / R3M 自监督 | **预训练 VLM（Qwen-VL / PaliGemma）** |
| **② 动作头** | DDPM 去噪（DP） | + Flow matching | **Flow matching（π0）/ 自回归 action token（FAST）/ 视频扩散（WAM）** |
| **③ 训练范式** | 从零训 | 预训练 + 全量微调 | **预训练 + LoRA 微调（视觉塔基本冻结）** |

### 证据

**轴 ②**：XPolicyLab 里 π0 的 README 论文标题原文就是
> "Pi0: A **Vision-Language-Action Flow Model** for General Robot Control"

而 **DDPM 扩散**在 68 个 README 里只被 **DP 自己**提到。
arXiv 趋势也印证：`flow matching policy` 2024→2025→2026 = 14 → 93 → **389**，增速比 `diffusion policy`（93→240→302）还快。

**轴 ③**：只有 4 个策略明说用 LoRA，但这是**低估**——很多厂商 README 写得很简（就一句 "See"）。
实际上 π0/GR00T 的官方微调流程就是 LoRA 或冻结视觉塔。

---

## 第 4 层：这对我们（DACSC）意味着什么

我们的方法作用在**视觉编码器**上。三个轴对我们是三种不同性质的影响：

| 轴 | 对我们 | 说明 |
|---|---|---|
| ① **视觉骨干** | ⚠️ **挑战** | ResNet18 → 预训练 VLM。CNN 的通道语义清晰、可解释；预训练 ViT 的通道语义模糊得多，"通道域敏感性"的论证会变弱 |
| ② **动作头** | ✅ **无影响** | 门控在编码器上，head 是 DDPM / flow matching / 视频扩散都一样 |
| ③ **训练范式** | ⚠️ **约束** | 预训练 + LoRA 意味着视觉塔基本冻结 → 通道级"重训"不现实，只能做"推理时/轻量适配"的门控 |

### 所以结论是（和上一轮的判断一致，但现在有了数据支撑）

1. **DP 是唯一能做"通道级机制研究"的地方** —— 因为它是唯一保留清晰 CNN 编码器 + 从零训练 + 能跑几十组消融的骨架。
2. **但这既是机会也是风险**：
   - 机会：**68 个策略里没一个在做"通道级域敏感性"，这片是空的**
   - 风险：如果用 DP 当唯一平台，会被质疑"骨架过时"
3. **正确的解法**（上一轮已定）：实验分两层
   - L1 机制验证在 DP（便宜、可解释、能做全部消融）
   - L2 挑 **SmolVLA** 做迁移验证（68 个里唯一"现代 VLA + 消费级硬件可跑"的）

---

## 附：68 个策略完整清单（按是否可复现排序）

**有 arXiv + 原始代码（可直接复现，最值得对比）**：
ACT · DP · AHA-WAM · DreamZero · EgoVLA · FastWAM · H-RDT · KinRT · MolmoAct2 · OpenWAM ·
OpenVLA-OFT · RISE · StarVLA · TinyVLA · VPP2 · ViGAR · X-WAM · Xiaomi-Robotics-0 · Dexora-1B ·
ME-U0 · ME-Dex-1.0 · MoPA · SimpleMemVLA · InternW0-Δ · WorldScape-Policy-2 · GPT-6-Astra …

**只有技术报告 / 未发表（TBD 或 not released）**：
Awomo-0.5 · GR00T-N1.7 · π0.5 · π0-Fast · SmolVLA · ABot-M0 · Being-H05 · LDA-1B ·
LingBot-VA/VLA · Mem-0 · X-VLA · Spirit-v1.5 · GalaxeaVLA · GigaWorld-Policy · GO-1 …
