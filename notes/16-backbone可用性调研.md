# CoRP 的 backbone 我们能直接用吗？（已实测）

> 2026-10-16。**结论：能，而且已经在你这台电脑上跑通了。**

---

## 直接回答你的四个问题

| 你的问题 | 答案 |
|---|---|
| **哪个代码库有这个 backbone？** | **StarVLA**（在 XPolicyLab 里）有一个现成的 `DINOv2BackBone` 封装（123 行），用的正是 `dinov2_vits14`。另外 `OpenVLA_OFT` 有 `dinosiglip_vit.py`，`HWM_CogWAM` 有 `dino_v3.py`（更新的 DINOv3） |
| **能直接用吗？** | **能。我已经在你电脑上实测跑通了**（见下方实测结果） |
| **论文给 backbone 代码了吗？** | **没给。** CoRP 全文无 GitHub/HF 链接（它是 ICLR 2027 投稿）。**但不需要**——因为 backbone 是 Meta 开源的 DINOv2 |
| **直接用 XPolicyLab 行不行？** | **不行，也不需要。** 见下方第 4 节 |

---

## 一、实测结果（决定性）

```
✓ 代码本地加载成功（绕过被拦的 github.com）
✓ 权重加载：missing=0  unexpected=0      ← 完全匹配
✓ 参数量 = 22.1 M                        ← 与 CoRP 论文写的 22.1M 一致
✓ patch_size=14   embed_dim=384
✓ 输入 224×224 → patch tokens (1, 256, 384)   ← 正是 CoRP 的 P=256
```

**复现命令**（已封装成脚本）：

```bash
python3 scripts/fetch_dinov2.py            # 下载 + 验证，约 1 分钟
python3 scripts/fetch_dinov2.py --verify   # 只验证
```

**权重体积/速度实测**：

| 型号 | 参数量 | 权重体积 | 是否适合我们 |
|---|---|---|---|
| **dinov2_vits14** | **22.1 M** | **84 MB** | ★ **CoRP 用的，就是它** |
| dinov2_vitb14 | 86.6 M | 330 MB | 备用（更大） |
| dinov2_vitl14 | 300 M | 1161 MB | 太大 |

---

## 二、怎么绕开被拦的 github.com（重要经验）

`torch.hub.load("facebookresearch/dinov2", ...)` **在本环境会失败**，因为它默认走 github.com。
但我们有两条替代路径：

| 需要什么 | 从哪拿 | 本环境实测 |
|---|---|---|
| **代码** | `codeload.github.com/facebookresearch/dinov2/zip/refs/heads/main` | ✅ 通（3 MB） |
| **权重** | `dl.fbaipublicfiles.com/dinov2/...`（**Meta 官方 CDN**） | ✅ **4.9 MB/s**，84MB 仅 **17 秒** |

**🔑 一个改变认知的发现**：
> **GitHub 的 38 KB/s 是 GitHub 特有的，不是网络整体慢。**
> Meta 的 CDN 快到 4.9 MB/s（快 130 倍）。
> 所以以后下模型权重（HF 镜像、各厂商 CDN）**可能都很快**，不要以 GitHub 的速度推断全部。

**本地加载的正确写法**（StarVLA 的封装早就写好了这个回退路径）：

```python
HUB = os.path.expanduser("~/.cache/torch/hub")
m = torch.hub.load(f"{HUB}/facebookresearch_dinov2_main",
                   "dinov2_vits14", source="local", pretrained=False)
m.load_state_dict(torch.load(f"{HUB}/checkpoints/dinov2_vits14_pretrain.pth"))
out = m.forward_features(x)["x_norm_patchtokens"]     # [B, 256, 384]
```

---

## 三、⚠️ 一个必须知道的尺寸约束

**DINOv2 要求输入尺寸是 `patch_size=14` 的整数倍。**

| 输入尺寸 | 224/14=16 ✓ | 256/14=18.28 ✗ |
|---|---|---|
| 结果 | 256 个 token，正常 | **报错** |

**好消息**：我们的 RoboCasa DP 基线用的是 **256×256 → crop 224×224**。
**224 = 16 × 14，正好整除** → **不需要改任何尺寸**，直接用 crop 后的 224 喂 DINOv2。
（CoRP 也用 224，完全对齐。）

---

## 四、为什么"直接用 XPolicyLab"不行

| 原因 | 说明 |
|---|---|
| **CoRP 不在 XPolicyLab 里** | 我核对了 68 个策略的完整清单，**没有 CoRP** |
| **XPolicyLab 是评测/部署框架** | 它管"怎么把策略接进环境"，**不管"怎么改视觉编码器内部"**——而后者正是我们方法的全部 |
| **它的 env client 不对口** | 只接了 RoboTwin / RoboDojo，**没有 RoboCasa365** |
| **它的 DP 不是我们的 DP** | `XPolicyLab/policy/DP/` 是它自己那份通用 DP，我们的基线是 RoboCasa 官方 fork（3 相机 + `ResNet18ConvFiLM` + 768 维语言） |

**但它有价值**：给我们提供了一个**现成的 DINOv2 封装可以抄**（`policy/StarVLA/.../dino_model/dino.py`，123 行）。

---

## 五、所以我们的 L1b 该怎么做

**目标**：把官方 DP fork 里写死的 `ResNet18ConvFiLM`，换成 / 并列为 DINOv2 ViT-S/14。

```
现状（L1）：
  3 相机 256×256 → crop 224 → ResNet18ConvFiLM（8 个 conv block）→ SpatialSoftmax → 64 维
                                      ↑ 我们的门控插在这里

新增（L1b）：
  3 相机 256×256 → crop 224 → DINOv2 ViT-S/14 → 256 patch token × 384 维
                                      ↑ 门控改插在这里（按 token 的隐藏维度）
                              → （需要一个投影/resampler 接到 DiT 的 conditioning）
```

**需要写的东西**（估计 1–2 天）：

| 要写什么 | 说明 |
|---|---|
| `DINOv2ObsEncoder` | 包一层 DINOv2，输出 patch token，投影到 DiT 的宽度 d=512 |
| 复用 StarVLA 的加载逻辑 | 直接抄它的 `DINOv2BackBone`（含本地加载回退） |
| 改 DP fork 的 policy 构造 | 现在由 robomimic 的 `get_robomimic_config` 硬编码 ResNet18ConvFiLM，需要加一条 DINOv2 分支 |
| 门控插入点调整 | ResNet：8 个 conv block 的 1920 通道；ViT：12 层的 384 维隐藏（或只插后几层） |

**不需要写的**：CoRP 的 resampler / flow matching 头（那是它的贡献，不是我们的必需；可作可选 L3）。

---

## 六、一句话结论

> **CoRP 的 backbone（DINOv2 ViT-S/14）是公开的，我已经在你电脑上跑通了。**
> 不需要 CoRP 的代码，也不需要 XPolicyLab——**只需要抄 StarVLA 那 123 行的加载封装**，
> 然后把它接到我们的 DP fork 上（1–2 天工程量）。
> 而且我们的 224 crop 尺寸**天然适配** DINOv2 的 14 的倍数要求，不用改数据管线。
