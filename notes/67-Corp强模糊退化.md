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

## 可靠性 caveat（必须写进论文）

方向可靠，但绝对数值不可与论文 97% 直接比较：

1. **"clean" 不是标准 LIBERO clean**：σ=0 仍使用 LIBERO-Plus 原生扰动 task
   （Camera/Robot/Light/Background/Noise/Layout 已内建）。所以 σ=0→σ=2 的退化
   包含 blur 与原生扰动的交互，不是纯 blur 效应。
2. **1 个初始状态/task**：官方 LIBERO 用 50 个；这里只用 `initial_states[0]`。
   配对比较有效，但绝对值方差大。
3. **task 是每类前 20 个**，不是随机抽样；若 task 难度与 ID 相关，绝对值有偏。
4. **7 个 category 是不同 task family**，不是同一 task 的受控 matched pair；
   by-category 数字不能横向比较（"Camera 0.35 vs Layout 1.0"不代表 Camera 更难）。
5. **blur 是 observation-side**（传感器退化），不是 simulator-side 扰动。

**可支撑的表述：**

> 在 LIBERO-Plus 的 140 task/suite 任务集上，额外施加 observation-side Gaussian blur
> 会让 CoRP 成功率从 72.9%（σ=0）降到 36–39%（σ=2），σ=4 时归零。
> 这是方向可靠、量级很大的退化现象；绝对数值不能与官方 LIBERO 的 97% 直接比较。

**不可支撑的表述：**

> "blur 让 CoRP 从 97% 掉到 39%"（clean 不是标准 LIBERO，初始状态/采样也不同）

### 若要更强的受控验证

1. 每类随机抽 20 个 task（固定 seed），而不是前 20；
2. 每 task 跑 5–10 个 initial state；
3. 标准 LIBERO（无原生扰动）上做 σ=0/2/4，隔离 blur；
4. RoboCasa-Shift 的 matched-pair 设计（同 task / 同初始状态 / 只改一个 factor）。

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
