# 54 - Frozen right-agentview DINOv2 域头在线验证

日期：2026-10-22  
服务器：`i-2.gpushare.com:40424`  
数据：61/61 atomic tasks  
主策略：flow K48  
域分支：冻结 DINOv2 ViT-S/14，右侧 agentview，patch mean  
mask：关闭  
训练：51 步，batch 48（3 粗域 × 16），seed 42

结果：

- `audit/domain_head_validation/domain_head.jsonl`
- `audit/domain_head_validation/summary.json`
- 服务器：`/hy-tmp/runs/domain_head_validation/domainhead_s42`

## 1. 域头配置

- coarse3 head：style-balanced 400-episode 审计缓存训练；
- layout10 head：layout-balanced 200-episode 缓存训练；
- 右侧 agentview，固定 center crop 224，冻结 DINOv2；
- 主策略的 DINOv2/flow loss 不反传到域分支；
- 域头先 warm-start，再做 temperature scaling：
  - coarse3: T=5.981，val CE 0.850，val acc 0.64；
  - layout10: T=20.0，val CE 2.287，val acc 0.222。

## 2. 在线结果（step 10–50 均值）

| head | episode acc | CE loss | chance |
|---|---:|---:|---:|
| **coarse3** | **0.544 ± 0.068** | **1.020 ± 0.100** | 0.333 |
| layout10 | 0.169 ± 0.054 | 2.283 ± 0.025 | 0.100 |

step50：

```text
coarse3: acc 0.625, loss 0.835
layout10: acc 0.104, loss 2.292
```

## 3. 验收

```text
coarse3 acc > 0.50        ✅ 0.544
coarse3 loss < 1.10       ✅ 1.020
layout10 acc > 0.30       ❌ 0.169
layout10 loss < 2.30      ✅ 2.283（但 acc 不可用）
```

结论：

> **冻结右侧 agentview DINOv2 + warm-start + temperature scaling 的 coarse3
> 域头在线有效：0.544，明显高于 0.333。**
>
> **layout10 在线仍然只有 0.169，说明 layout10 不是可靠的 sanity gate；
> 这不是 frozen branch 的接口问题。**

## 4. 工程细节

- batch 96 时，主 flow K48 + 第二份冻结 DINOv2 在 4090 上 OOM（47.2/47.4GB）；
- 改为 batch 48（3 粗域 × 16）后正常；
- CLIP-large 已下载到 `/hy-tmp/models/clip-vit-large-patch14`；
- 温度缩放把 coarse3 在线 CE 从 ~2.99 降到 ~1.02，准确率不变。

## 5. 下一步

coarse3 已经通过 no-mask 在线验收。接下来才加 mask：

1. 用冻结右 agentview DINOv2 域头作为**域 scorer**；
2. 在主策略的右侧 agentview DINOv2 patch tokens 上做 domain-sensitive channel WRS；
3. 先做 51 步 × 3 seeds：
   ```text
   no mask
   random mask
   domaindrop (coarse3 scorer)
   place
   dgcd
   ```
4. 通过后再 200 步 / 2k。

不要再：

```text
不要用 layout10 作为唯一 sanity
不要继续调 ResNet18 GAP 域头
不要用 wrist 单独定义 style
不要直接跑 200 步 / 2k
```

一句话：

> **frozen right-agentview DINOv2 + warm-start coarse3 域头在线达到 0.544 / CE 1.02，
> 这是当前第一个通过 no-mask 在线验收的域头；下一步是用它给主策略的右侧 patch token 做域敏感 mask。**
