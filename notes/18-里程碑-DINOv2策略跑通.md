# 里程碑：DINOv2 策略在本机跑通（2026-10-16）

> **v1 目标达成**：把官方 DP 的视觉编码器换成 DINOv2 ViT-S/14，
> 插进官方训练/评测管线，**官方文件零修改**，全部在本机 CPU 验证通过。

---

## 一、实测结果

### 我们的 DINOv2 策略

| 项 | 值 |
|---|---|
| hydra 实例化 | ✓ 成功（2.1 s） |
| **总参数** | **85.84 M** |
| ├ 观测编码器 | 22.63 M（其中 **DINOv2 22.06 M**） |
| └ DiT | 63.21 M（12 层 / n_emb 512，与官方一致） |
| obs_feature_dim | 256 |
| compute_loss | ✓ loss 1.2553，前向 0.3 s，反向 0.6 s |
| **梯度流到 DINOv2** | ✓ **174/175 个参数有梯度** → **联合微调生效** |
| predict_action | ✓ `action [1,8,12]`、`action_pred [1,10,12]`（与官方格式**完全一致**） |
| 推理耗时（CPU，100 步去噪） | 1.6 s |

### 与官方基线对比

| | 官方 DP | 我们（DINOv2 v1） | 变化 |
|---|---|---|---|
| 总参数 | 106.03 M | **85.84 M** | **−19%** |
| 观测编码器 | 42.45 M | **22.63 M** | **−47%** |
| obs_feature_dim | 969 | 256 | − |
| 编码器是否预训练 | ❌ 从零/ImageNet | ✅ **DINOv2 自监督预训练** | ★ |
| 动作头 | DDPM + DiT | 同左（未变） | — |

**关键**：我们的模型**更小**，但编码器是**预训练**的。
这正是 CoRP 4.3 的结论（预训练 ViT 高 ≥14.9 分）与 4.4 的结论（必须联合微调）的组合。

---

## 二、交付物

```
dacsc/                                     ← 我们自己的包（官方仓库零修改）
├── __init__.py
├── compat.py              兼容 shim（diffusers 版本差异，内存补符号）
├── dinov2_encoder.py      172 行  DINOv2 观测编码器（多视角共享权重）
└── policy.py              147 行  RoboCasaDINOv2Policy（继承官方策略，只换编码器）

repos/diffusion_policy/diffusion_policy/config/
└── train_dinov2_bs192.yaml    ← 新配置文件（官方配置一字未改）

scripts/
├── setup_env.sh           ★ 一键建环境（venv + .pth + 依赖）
├── fetch_dinov2.py        ★ 拉取+验证 DINOv2 backbone
└── test_dinov2_policy.py  ★ CPU 冒烟测试（4 项检查 + 官方对比）
```

**复现**：
```bash
bash scripts/setup_env.sh
python3 scripts/fetch_dinov2.py
./.venv/bin/python scripts/test_dinov2_policy.py --compare-official
```

---

## 三、设计要点（为什么这么做）

### ① 官方文件零修改

只在 `__init__` 里换掉编码器，其余方法（`conditional_sample` / `predict_action` /
`compute_loss` / `get_optimizer` / `set_normalizer`）**全部原样继承**。

原因：官方策略的这些方法都通过 `self.obs_encoder` **泛化调用**，不关心编码器具体是什么。
所以继承 + 重写 `__init__` 就够了。

**好处**：随时能用官方配置复现基线；我们的改动集中在 2 个文件里。

### ② 编码器接口只需 2 个方法

```python
def output_shape(self): return [self.feature_dim]   # → obs_feature_dim
def forward(self, obs_dict): ...                    # → (B, feature_dim)
```

官方脚手架用 `obs_encoder.output_shape()[0]` 决定 DiT 的 `cond_dim`，所以**天然支持任意编码器**。

### ③ 踩过的 4 个坑（都已在代码里解决）

| # | 坑 | 解法 |
|---|---|---|
| 1 | `torch.hub.load` 走 github.com（被拦） | `load_dinov2()` 本地优先（代码从 codeload、权重从 Meta CDN） |
| 2 | **DINOv2 要求输入是 14 的倍数**，256 会报错 | `_Crop224` 把 256 → 224（训练随机裁剪、推理中心裁剪） |
| 3 | 官方配置里的 `crop_shape` 等参数漏进 `**kwargs` → 传给 scheduler 报错 | `__init__` 显式接收并忽略这几个官方专属参数 |
| 4 | diffusers 新版缺 `Union`/`Optional`（官方 pin 0.11.1） | `dacsc/compat.py` 在内存里补符号，不改官方源码 |

### ④ 为门控预留的挂载点

```python
def gate_modules(self):    return []          # v3 会返回所有门控模块
def gate_parameters(self): return [...]       # 用于单独设置 weight_decay / lr
```
`get_optimizer` 已经返回 3 个参数组（DiT 的 decay / no-decay + 编码器），
v3 加门控时只需再插一组。

---

## 四、下一步（v2 / v3）

| 版本 | 加什么 | 为什么 |
|---|---|---|
| **v2** | **learned-query resampler**：把 256 个 patch token 压成 K=48 个（CoRP Eq.2，含语言 FiLM） | 保留细粒度空间信息（现在 v1 是均值池化，丢了空间细节）；CoRP 4.6 证明"指令感知的硬压缩"比全量 token 高 +13.8 分 |
| **v3** | **通道级域敏感性门控** ← **我们的核心方法** | 找"爱看装修的员工"并关掉 |

**v2 需要动的地方**：DiT 的 `cond` 从「每观测步 1 个 token」变成「K 个 token」，
需要调整 `TransformerForDiffusion` 的 `n_obs_steps` 语义（或新写一个 cond 通路）。

**v3 不需要动 DiT**：门控插在 DINOv2 的各层/各维度上，输入输出形状不变 → 现有管线直接可用。
