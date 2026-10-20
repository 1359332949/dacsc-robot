# 48 - Token-style 旁支：canonical normalization 有正信号

日期：2026-10-20  
相关提交：`2d8e1ed`、`6336d63`、`b56979f`、`f7ea532`、`8877d6b`  
结果表：`audit/token_style_201_3seed.csv`

## 1. 方法

在 flow K48 的 resampler 路径上实现 token-level style statistics：

```text
DINOv2 patch tokens
→ [可选 patch-stage style stats]
→ learned-query resampler
→ [可选 resampler-stage style stats]
→ DiT flow
```

支持：

- `random_stats`：随机 donor 的 mean/std；
- `cross_style_stats`：不同 style donor 的 mean/std；
- `mean` / `meanstd` / `std`；
- `cross_style_stats_action_aware`：按 resampler token 的 action importance 降低调制；
- `normalize`：canonical token normalization
  ```text
  z_norm = (z - mu) / sigma
  out    = gamma * z_norm + beta
  ```
- `normalize_action_aware`：对 action-important token 保留原始统计量；
- patch stage（resampler 之前）和 resampler stage 两种位置。

## 2. 51 步 3-seed smoke（flow K48）

| group | 配置 | action MSE |
|---|---|---:|
| none | 无调制 | 0.7503 ± 0.0833 |
| random | 随机 donor | 0.7999 ± 0.0455 |
| cross | 跨 style donor，meanstd | 0.8293 ± 0.0415 |
| action20 | 跨 style + top-20% token 保护 | 0.7950 ± 0.0569 |
| action30 | 跨 style + top-30% token 保护 | 0.7953 ± 0.0566 |
| mean | resampler mean-only | 0.8293 ± 0.0415 |
| patch25 | patch stage meanstd，alpha=0.25 | 0.8251 ± 0.0393 |
| patch50 | patch stage meanstd，alpha=0.5 | 0.8298 ± 0.0412 |
| patchmean | patch stage mean-only，alpha=0.25 | 0.8243 ± 0.0388 |
| **norm_res** | resampler canonical normalization | **0.6069 ± 0.0669** |
| norm_patch | patch canonical normalization | 0.7169 ± 0.0714 |
| norm_action | canonical + action-aware blend | 0.6933 ± 0.0681 |

结论：

1. donor statistics replacement 全部比 no modulation 差；
2. mean-only 与 meanstd 几乎一样，说明主要伤害来自 mean shift；
3. top-k action protection 只能小幅降低调制幅度，不能恢复动作性能；
4. **canonical normalization** 是第一个明显优于 none 的 token-style 变体；
5. action-aware normalization 反而比纯 normalization 差。

## 3. 201 步 3-seed 确认

| group | action MSE | 相对 none |
|---|---:|---:|
| none | 0.3199 ± 0.0274 | — |
| norm_res | 0.2982 ± 0.0200 | **-6.8%** |
| **norm_patch** | **0.2884 ± 0.0107** | **-9.8%** |

逐 seed：

```text
none:      0.2993 / 0.3016 / 0.3587
norm_res:  0.3012 / 0.2723 / 0.3210   (2/3 better)
norm_patch:0.2912 / 0.2741 / 0.2998   (3/3 better)
```

`norm_patch` 在 201 步的均值最低、方差最小，并且 3 个 seed 全部优于对应的 none。

## 4. 解释

- donor replacement 会把 **另一个样本的内容偏置** 一并注入，伤动作；
- canonical normalization 只去掉当前样本自己的 token mean/std，不注入 donor 内容；
- patch-stage normalization 在 201 步更稳，可能是因为它在 resampler 之前去掉了更接近外观的统计量；
- action-aware protection 会把部分 token 拉回原始统计量，反而削弱 normalization 的收益。

## 5. 当前定位

按主线程决定：

```text
token-style 标记为旁支
不继续扩大到 2k
不把 token-style 当作当前主方法
```

但这是一个有价值的旁支：

```text
canonical token normalization
→ 201 步 3 seed，-9.8% action MSE
→ 3/3 seed 方向一致
```

等 faithful DomainDrop / PLACE / DgCD 的 CNN baseline 复现完成后，再决定是否把 normalization 作为 ViT/flow 适配的候选重新接回主线。

## 6. 服务器产物

- 51 步：`/hy-tmp/runs/token_style_51`
- top-k protection：`/hy-tmp/runs/token_style_action_51`
- mean/patch：`/hy-tmp/runs/token_style_stage_51`
- normalization：`/hy-tmp/runs/token_style_norm_51`
- action-aware normalization：`/hy-tmp/runs/token_style_norm_action_51`
- 201 步：`/hy-tmp/runs/token_style_201`
