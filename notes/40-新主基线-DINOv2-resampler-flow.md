# 新主基线：DINOv2 + resampler + flow matching（2026-10-20）

## 定位

保留三层基线，避免把“基线”混成一个概念：

| 角色 | 模型 | 状态 |
|---|---|---|
| 官方参照 | RoboCasa ResNet18ConvFiLM + DDPM/DiT | 历史对照，不删 |
| 受控基线 | DINOv2 + DiT + DDPM | 已有，用于阶段一门控验证 |
| 新主基线 | DINOv2 ViT-S/14 + learned-query resampler + DiT flow matching | 已实现并 smoke 通过 |

## 代码

- `dacsc/resampler.py`
  - learned-query cross-attention
  - 每个视角 256 patch tokens → `[K, token_dim]` 的 K 个 token
  - 语言 FiLM 调制 learned query
  - **不 flatten 成单个 256 维向量**
- `dacsc/dinov2_encoder.py`
  - `use_resampler`
  - 输出 `[1 + V*K, token_dim]` 的条件 token 序列
  - 保留 `last_patch_tokens["patch_tokens"]` 供诊断
- `dacsc/policy.py`
  - `flow_matching`
  - rectified-flow velocity loss：`v = x1 - x0`
  - Euler flow sampling
  - DDPM 和 flow matching 共用同一 DINOv2 / 门控 / 动作保护链路
  - resampler 分支把 `To*(1+V*K)` 个条件 token 送入 DiT
- `train_dinov2_flow.yaml`
  - `use_resampler: true`
  - `flow_matching: true`
  - `num_inference_steps: 10`
  - 安全默认：`batch_size=30`、`num_epochs=1`、`max_train_steps=51`

门控仍然插在 DINOv2 block 输出上、在 resampler 之前；这里的“channel”应表述为 **ViT token 特征维度门控**。

## 21 步 flow/resampler smoke

```text
step20
drop-IoU same:        1.000
drop-IoU diff:        0.472
drop-IoU gap:         0.528
sample drop gap:      0.738
score cosine gap:     0.325
eligible_frac:        0.541
layout_resid_frac:    1.0
v_raw_rel_std:        0.373
```

动作头 loss 与 DDPM 不可横比。此实验只证明：

- resampler 可以训练；
- flow matching 可以训练和采样；
- 门控/动作保护链路与 flow matching 兼容。

## 3-seed DDPM 阶段一结果

| group | drop gap mean ± std | gate on/off mean ± std | v raw rel std |
|---|---:|---:|---:|
| random | -0.0003 ± 0.0007 | -8.63% ± 0.27% | 0 |
| D | **0.6886 ± 0.0309** | -15.28% ± 1.59% | 0 |
| F2 | **0.5185 ± 0.0320** | -14.58% ± 0.53% | **0.3643 ± 0.0216** |

结论：

- 域结构稳定优于 random mask；
- F2 在保留动作保护的同时仍保留稳定域差异；
- 但 51 步截断不能作为性能结论。

## 下一步

1. 对 flow/resampler 跑 51 步，拿 step50 gate on/off；
2. 对比 `flow + resampler` 无门控 vs 随机门控 vs domain gate；
3. 仍不进入 2k。
