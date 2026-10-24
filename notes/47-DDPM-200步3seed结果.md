> ⚠️ **范围限定（2026-10-24）**
> 本结论来自 **legacy DDPM channel gate**，**不是 CoRP 结论**。
> 引用时必须写明 "legacy DDPM path"。

# 47 - DDPM 200 步 3-seed 结果：51 步正信号是瞬态的

日期：2026-10-20  
矩阵目录：`/hy-tmp/runs/ddpm_f2_200`  
结果表：`audit/ddpm_f2_200_3seed.csv`

## 1. 实验设置

- 主干：DDPM 受控基线（`train_dinov2_bs192`）
- batch：30（10 style × 3 samples）
- 实际更新：201 步（index 0..200），确保 step200 probe 落盘
- seeds：42 / 43 / 44
- 关闭 checkpoint 保存（`save_last_ckpt=false`, `topk.k=0`）
- 每组配置：

| group | gating | domain score | action importance | selection |
|---|---|---|---|---|
| no_mask | off | — | — | — |
| random | on | random | off | linear |
| D | on | DgCD | off | linear |
| F2 | on | DgCD | RSC clean Taylor | two-stage |

F2 其他配置与 51 步版本一致：

```yaml
hard_mask_mode: gumbel_ste
hard_mask_domain_shared: true
dgcd_group_smooth: 1.0
dgcd_domain_center: true
layout_residual: true
hard_candidate_ratio: 0.6
hard_action_protect_ratio: 0.2
hard_clean_importance_every: 4
gate_diagnostics_every: 50
```

## 2. gate on/off 轨迹（3 seed 均值）

正值表示 gate 打开后动作 MSE 变差。

| group | step50 | step100 | step150 | step200 | 最终 action MSE |
|---|---:|---:|---:|---:|---:|
| no_mask | — | — | — | — | 0.201 ± 0.004 |
| random | -15.0% | +8.5% | -20.8% | +8.3% | 0.207 ± 0.013 |
| D | -21.7% | +24.8% | +14.5% | **+22.6%** | 0.210 ± 0.009 |
| F2 | -22.1% | +25.0% | +16.3% | **+22.5%** | 0.194 ± 0.016 |

关键变化：

```
step50:  D/F2 约 -22%
step100: D/F2 约 +25%
step150: D/F2 约 +15%
step200: D/F2 约 +23%
```

所以 51 步矩阵看到的 `-14% ~ -22%` 不是稳定收益，而是**早期训练阶段的瞬态效应**。
到 step100 就已经反号，step200 仍然明显为正。

## 3. step200 机制指标

| group | drop gap | sample drop gap | v_raw_rel_std | dp_loss |
|---|---:|---:|---:|---:|
| random | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.000 | 0.260 ± 0.030 |
| D | 0.649 ± 0.063 | **0.804 ± 0.003** | 0.000 | 0.236 ± 0.043 |
| F2 | 0.470 ± 0.010 | **0.741 ± 0.002** | **0.339 ± 0.008** | 0.234 ± 0.015 |

说明：

1. 域选择性仍然很强：D/F2 的 sample gap 分别为 0.804 / 0.741，random 接近 0。
2. 动作重要性没有塌缩：F2 的 `v_raw_rel_std=0.339`。
3. 但强域选择性并没有转化为动作收益；gate on/off 在 step200 仍是 +22.5%。

## 4. 动作保护的效果

F2 相对 D：

```
sample gap: 0.804 → 0.741（仍然很强）
gate on/off: +22.6% → +22.5%（几乎没变）
action MSE: 0.210 → 0.194（均值略好，但方差大）
```

F2 的最终 action MSE 比 no_mask 低约 3.7%，但：

- F2: 0.194 ± 0.016
- no_mask: 0.201 ± 0.004
- n=3，差异远未达到统计显著。

因此不能据此宣称“动作保护带来了动作收益”。更准确的结论是：

> **两阶段动作保护保住了 v 的非塌缩，也在均值上略微改善了 action MSE，
> 但没有修复 gate on/off 的符号问题；域门控仍然在伤害同一 batch 的动作 loss。**

## 5. 与 51 步结果的对比

51 步 DDPM 3-seed：

```
random: -8.63% ± 0.27%
D:      -15.28% ± 1.59%
F2:     -14.58% ± 0.53%
```

200/201 步：

```
random: +8.27% ± 4.06%
D:      +22.57% ± 1.86%
F2:     +22.53% ± 1.82%
```

这不是 seed 波动能解释的：D/F2 在 3 个 seed 上方向一致，且 step100/150/200 均为正。
51 步时的负值更像是 warmup 后短窗口内的 dropout 正则效应。

## 6. 解释

这轮结果和 top-u 遮挡实验一致：

```
域敏感维度 u：能识别 style，域选择性很强
动作重要维度 v：能识别动作关键通道，且不塌缩
但 u 与 v 在 layer 11 的 384 维 embedding 上不是同一组通道
```

因此当前 hard gate 实际上在做：

```
丢掉“看起来域敏感”的通道
→ 这些通道对动作因果影响很小
→ 早期像 dropout 正则，后期逐渐破坏动作相关表征
```

动作保护（two-stage）只能在候选集里排除 top-v 通道，不能改变“域分数选错了干预对象”这一根本问题。

## 7. 决策

```text
不进入 2k
不继续调 WRS 的 lambda / drop ratio / layer
不继续把当前 dimension gate 当作主方法
```

下一步应转向 token-level：

1. token-level style statistics / TFS-ViT 风格调制；
2. token-level top-style / top-action / random 遮挡诊断；
3. 在 resampler 输出 K token 上做 gate，而不是在 384 维 pooled embedding 上做 gate；
4. 只有在 token-level 出现稳定的非正 gate on/off 后，才考虑 CoRP 主基线。

一句话：

> **200 步结果推翻了“51 步 F2 已有稳定动作收益”的判断。
> 域选择是真的，v 也是真的，但当前 dimension gate 选错了因果对象。
> 方法应从 embedding-dimension gate 转向 token-level style statistics。**
