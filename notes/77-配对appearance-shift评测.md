# 77 - 同一进程配对 appearance-shift 评测

日期：2026-10-24
checkpoint：官方 CoRP（LIBERO 训练）
suite：`libero_spatial`，每类 2 task（14 tasks）
协议：**同一个 env、同一个初始状态、同一个 seed，5 个 shift 在同一进程内顺序跑**
输出：`/hy-tmp/runs/libero_paired_smoke/`（JSON 自带 `shift`）

## 结果

| shift | 成功 | 成功率 |
|---|---|---|
| none | 14/14 | **100%** |
| grayscale | 14/14 | **100%** |
| color_jitter | **3/14** | **21.4%** |
| low_contrast | 11/14 | 78.6% |
| hue_shift | 10/14 | 71.4% |

## 与上一轮对比

上一轮是跨进程、不同批次；本轮是**同一进程配对**。结果完全一致：

```text
none 100% / grayscale 100% / color_jitter 21.4% / low_contrast 78.6% / hue 71.4%
```

这排除了“跨进程/环境差异”的解释。

## 解读

```text
CoRP 对灰度不敏感
CoRP 对轻度对比度/色调有韧性
CoRP 对强 color jitter 极敏感
-> A 路线前提成立（强 appearance shift 会打崩 CoRP）
```

## 边界

- 14 task × 1 episode，小样本；
- 仍建议扩到 20 task/类 × 4 suite；
- color_jitter 是强增广（b1.6/c1.6/s2.2），不是自然分布内的轻微变化。
