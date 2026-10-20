# 单 style 域与 layout 残差（2026-10-20）

## 1. 背景

在连续 `style 11–15` 粗分组下，DgCD 的域原型平均余弦约 0.80，mask 近似静态。  
D/E/F 证明 `domain center` 与 `two-stage 动作保护` 有效，但连续 ID 分组缺乏视觉依据。

本轮先不跑长训练，只修正域定义和 domain score。

---

## 2. 代码改动

### 2.1 单 style 域

- `domain_group_size = 1`
- `num_domains = 50`
- `domain_max_groups_per_batch = 10`
- 每个 batch：随机 10 个 style × 3 样本 = 30
- 不再把相邻 style ID 合并

### 2.2 layout 残差 domain score

新增 `layout_residual`：

```text
pooled = DINOv2 patch-token mean
layout_mean[l, v, c] = EMA(pooled for layout l, view v)
domain_input = pooled - layout_mean[自己的 layout, view]
```

- 按 `(layout, view)` 分别统计，避免把左/右/腕部相机混在一起；
- 只用于生成 domain score，不改视觉特征；
- 训练时更新 EMA；验证/探针时不更新；
- 统计状态保存在 buffer 中，可随 checkpoint 恢复。

### 2.3 validation

- 恢复 workspace 的 epoch-end validation；
- `DomainLerobotCotrainingDataset.get_validation_dataset()` 返回 style 41–50 的 `Subset`；
- `no_grad` 验证阶段只计算 DP loss，不生成 mask、不推进 gate_step。

验证 smoke：

```text
val styles 41-50: 16 steps
val_loss = 1.7504
```

---

## 3. 21 步短诊断

配置：

```yaml
domain_group_size: 1
domain_max_groups_per_batch: 10
num_domains: 50
dgcd_group_smooth: 1.0
dgcd_domain_center: true
layout_residual: true
hard_mask_domain_shared: true
hard_selection_mode: two_stage
use_action_importance: true
```

step20：

```text
drop-IoU same: 1.000
drop-IoU diff: 0.510
drop-IoU gap: 0.490
sample drop-IoU gap: 0.732
score cosine gap: 0.252
eligible_frac: 0.577
layout_resid_frac: 1.0
layout_seen_count: 150
v_raw_rel_std: 0.355
```

结论：

- 单 style 域和 layout 残差都正常工作；
- 域 mask 选择性仍然明显；
- 动作保护没有让 v 塌缩；
- 但这只是 21 步短诊断，不能替代 60 步/多 seed 验证。

---

## 4. 环境

新实例：

```text
GPU: RTX 4090 49GB（nvidia-smi 实际结果，不是 A100）
CUDA: 可见
数据: /hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic
任务: 61
episodes: 6935
styles: 50
```

审计 PASS，CLIP 和 DINOv2 已本地化。

seed 矩阵在用户要求“先解决问题再长跑”后主动终止，避免继续消耗费用。

---

## 5. 下一步

1. 用当前单 style + layout 残差配置跑 60 步 F2；
2. 若 step50 仍满足：
   - drop-IoU gap > 0.3；
   - `v_raw_rel_std > 0.2`；
   - gate on/off 不伤害动作；
   - val_loss 有记录；
   再做 random/D/F2 的 3-seed 对照；
3. 仍然不要直接进 2k。
