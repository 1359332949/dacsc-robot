# GPU 规格调研：DACSC-Robot 到底需要什么卡（2026-10-19）

> 结论先行：**第一轮实验 1 张 24–48GB 卡即可；主实验推荐 1×A100 40GB 或 1×L40S 48GB；
> 要完整复现官方 batch 192，推荐 1×A100/H100 80GB。当前代码没有 DDP，2 张卡主要用于并行跑不同实验，
> 不会自动让单次训练快一倍。**

---

## 一、我们自己的模型到底多大

| 项 | 数值 | 来源 |
|---|---:|---|
| 总参数 | **85.84M** | 本机真实实例化（`notes/18`） |
| 视觉主干 | DINOv2 ViT-S/14，22.1M | CoRP 4.3 |
| 相机 | 3 路 | RoboCasa 官方配置 |
| 观测窗口 | 2 帧 | `n_obs_steps=2` |
| 动作头 | DiT 12 层 / 512 维 | 官方 DP 配置 |
| 动作窗口 | H=10 | 官方 DP 配置 |

**关键：参数不是显存大头。** 85.84M 参数在 BF16 下约 0.17GB；加上 AdamW 状态、梯度、
EMA 影子参数，大约 1.3–1.7GB。真正吃显存的是**视觉激活**。

我们的编码器每一步处理的图片数是：

```text
B × n_obs_steps × n_views = B × 2 × 3
B=32  → 192 张/步
B=64  → 384 张/步
B=192 → 1152 张/步
```

ViT-S 会把每张 224×224 图切成 256 个 patch token，并保留 12 层激活，
所以显存会随 batch 近似线性增长。

---

## 二、最相关的三份公开依据

### 1. RoboCasa 官方 Diffusion Policy 文档

官方原文：

> For training we recommend a GPU with **at least 24 GB** of memory, but **48 GB+ is preferred**.
> For inference we recommend a GPU with **at least 8 GB**.

同一页还写明官方配置是 `train_diffusion_transformer_bs192`，即 **batch=192**。

来源：
- https://robocasa.ai/docs/benchmarking/policy_learning_algorithms.html
- 本地配置：`repos/diffusion_policy/diffusion_policy/config/train_diffusion_transformer_bs192.yaml`

### 2. CoRP（arXiv 2610.08183）——结构最接近我们

CoRP 的视觉主干同样是 **DINOv2 ViT-S/14**，动作头是紧凑的 DiT + flow matching，
总参数 48.9M。论文训练配置：

> AdamW，学习率 5e-5，**batch size 392**，**BF16**，gradient clipping 1.0，EMA 0.999。
> **We use 2×A800 for training**；LIBERO 约 **8 小时**，RoboTwin 2.0 约 **13 小时**。
> 训练步数：LIBERO 80k，RoboTwin 120k。

来源：
- https://arxiv.org/abs/2610.08183
- https://corp-policy.github.io/

注意：
- 论文没有写 A800 的显存；A800 常见版本是 80GB。
- 这是 ICLR 2027 投稿，尚未开源，硬件数据只能作为参考。
- 我们的模型 85.84M，约为 CoRP 的 1.75 倍；但动作序列很短，增量显存主要在 DiT 和 3 视角。

### 3. Diffusion Policy 原论文

原论文图像实验使用 **batch size 64**，推理在 RTX 3080 上约 0.1s/次。
它没有给出训练 GPU 的统一规格，所以不能拿它反推 24GB/48GB；
RoboCasa 官方文档才是我们基线最直接的硬件依据。

来源：
- https://arxiv.org/abs/2303.04137

### 4. 不要被 DINOv2 原论文的预训练规模吓到

DINOv2 原论文预训练用了几十张 A100 80GB。但我们是**加载已经训练好的 ViT-S 权重再微调**，
不是从零预训练 DINOv2，所以那张“32×A100”的账单与我们无关。

---

## 三、推荐档位（从省钱到稳妥）

### A. 烟测 / 调通代码：1×RTX 4090 24GB

- batch：**8–16**，必要时 gradient accumulation
- 用途：确认数据加载、前向、反向、门控梯度、显存峰值
- 不建议直接用它跑完整 61 任务消融：不是不能跑，而是 batch 小、速度慢

### B. 第一轮正式 pilot：1×A100 40GB 或 1×L40S 48GB（推荐）

- batch：A100 40GB 用 **32–48**；L40S 48GB 用 **48–64**；再 gradient accumulation 到有效 batch 128–192
- 用途：20k–50k 步的短训练、5–10 个任务的趋势对比
- 这是**性价比最高**的档位

### C. 完整 batch / 追求速度：1×A100 80GB 或 1×H100 80GB

- batch：**96–192**
- 用途：完整复现官方 batch 192，或需要大量消融时
- 如果预算有限，不必一上来就租

### D. 多卡：2–4 张 24GB/40GB，优先并行跑不同实验

当前训练配置是 `training.device: cuda:0`，没有 DDP。也就是说：

- **2 张卡不会自动把一次训练加速 2 倍**
- 但可以同时跑“baseline / KL-only / gate / full”四组实验
- 只有把 DDP 接进来后，多卡才会加速单次训练

CoRP 用 2×A800 是因为它要跑 global batch 392；我们可以用**小 batch + 梯度累积**替代，
第一轮没必要追求同样的 global batch。

---

## 四、显存与时间估算（待 GPU 实测校正）

以下是工程估算，不是实测：

| batch | 每步图片数 | 预计显存 | 适合的卡 |
|---:|---:|---:|---|
| 8 | 48 | 8–12GB | 4090 24GB |
| 16 | 96 | 12–18GB | 4090 24GB |
| 32 | 192 | 18–28GB | 4090 可能紧 / A100 40GB |
| 48 | 288 | 24–35GB | A100 40GB |
| 64 | 384 | 30–45GB | A100 40GB 边缘 / L40S 48GB |
| 128 | 768 | 55–80GB | A100 80GB |
| 192 | 1152 | 80GB+ | A100/H100 80GB |

时间上，按 CoRP 的公开吞吐外推：

- 20k 步 pilot：大约 **4–10 GPU-hours**
- 50k 步：大约 **10–25 GPU-hours**
- 250k 步完整训练：大约 **50–100 GPU-hours**
- 真正的瓶颈还包括闭环评测：50 rollouts × 多任务 × 3 相机渲染

这些数字必须在租到卡后用真实数据跑一次 memory probe 才能定死。

---

## 五、AutoDL 选卡建议

### 最省钱路线

1. 无卡模式：上传代码、装环境、下载数据
2. 切 **RTX 4090 24GB**：跑 1 小时显存探针
3. 如果 batch 16 能稳定跑，先用 4090 做小规模趋势实验
4. 如果 batch 32 就 OOM，换 **A100 40GB / L40S 48GB**
5. 只有确认方法有效后，才租 **A100 80GB** 跑完整 batch

### 直接做主实验

- **首选：A100 40GB 或 L40S 48GB**
- **更稳：A100 80GB**
- CPU/内存：至少 **16 vCPU / 64GB RAM**
- 数据盘：至少 **50GB**（atomic 全套 5.3GB，环境 + checkpoint 另算）
- 不需要 NVLink；单卡足够。多卡并行时 PCIe 也够用。

### 不建议

- 8×H100 / H200：我们的 85.84M 模型用不上
- 为了“复现 CoRP 的 2×A800”而直接租两张 80GB：第一轮没必要
- 在 4090 上强行跑 batch 192：大概率 OOM，浪费调试时间

---

## 六、下一步的实际动作

租到卡后先跑 **10 分钟显存探针**，不要直接开 20k 步：

```bash
# 先测 batch=8/16/32/64，记录 torch.cuda.max_memory_allocated
# 找到“峰值显存 < 卡容量 × 0.85”的最大 batch
```

然后用该 batch 跑 500–1000 步，确认：

1. loss 正常下降
2. 门控参数真的更新
3. DINOv2 梯度非零
4. 单步时间可接受

再扩展到 20k 步。这样不会把租卡预算浪费在 OOM 和错误配置上。
