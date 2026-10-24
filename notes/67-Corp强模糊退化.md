# 67 - CoRP 在强 Gaussian blur 下显著退化（前提证实）

日期：2026-10-24
checkpoint：官方 CoRP（LIBERO 训练）`corp_libero.pt`
suite：`libero_spatial`
采样：每扰动类别 2 个 task × 1 episode（14 tasks）
环境：CoRP 自带 venv（robosuite 1.4.1）+ LIBERO-Plus assets

## 结果

| 条件 | 成功 | 成功率 |
|---|---|---|
| clean (σ=0) | 14/14 | **100%** |
| Gaussian blur σ=2 | 7/14 | **50%** |
| Gaussian blur σ=4 | 0/14 | **0%** |

分类别（σ=2）：

| 类别 | 成功 |
|---|---|
| Camera Viewpoints | 2/2 |
| Robot Initial States | 1/2 |
| Language Instructions | 2/2 |
| Light Conditions | 0/2 |
| Background Textures | 0/2 |
| Sensor Noise | 1/2 |
| Objects Layout | 1/2 |

## 结论

```text
CoRP 在 clean / LIBERO-Plus 原生扰动下：强鲁棒（14/14）
CoRP 在强 Gaussian blur 下：显著退化（σ=2: 50%, σ=4: 0%）
```

**“大偏移下 CoRP 会退化”这一前提首次得到证实。**
这为 token 级 / 表征不变性方法提供了真实的问题空间。

## 边界

- 只测 `libero_spatial`，每类 2 task × 1 episode，样本小；
- blur 施加在观测图上（post-render），不是改仿真渲染；
- 需要扩大样本（每类 20–50 task）并覆盖 4 个 suite 才能定量。

## 工程

- `scripts/eval_corp_libero_plus.py --blur-sigma`：对 agentview / eye_in_hand 做
  `scipy.ndimage.gaussian_filter`，再走官方 CoRP 预处理。
- 3 组并行评测：`/hy-tmp/runs/libero_blur_{0,2,4}/summary.json`。

## 下一步

1. 扩大 blur 评测（每类 20–50 task，4 suites）；
2. 在 CoRP 上做 token_style 对照：
   `none / random_stats / cross_style_stats / cross_style_stats_action_aware`；
3. 指标：clean 与 blur 下的 success，以及 per-dim / per-task MSE。
