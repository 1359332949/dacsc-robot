# 机制验收与 style 可分性（2026-10-19）

## 结论先行

这一轮完成了从“门控能跑”到“门控机制是否成立”的验收。最终判断：

> **原始 DgCD mask 的域选择性很弱，主要来自所有 style 共享的通道排序。**
> **跨域中心化（domain center）是目前最关键的修复；两阶段动作保护是下一步方向。**

当前不进入 2k/20k。

---

## 1. 实验矩阵（均为截断探针，不是完整 epoch）

统一设置：

- 数据：RoboCasa365 pretrain atomic，61 tasks
- batch：domain-balanced 30（10 粗域 × 3）
- 截断步数：60 或 200
- mask：per-sample hard mask，33% drop
- Gumbel-STE
- gate on/off 探针：同 batch、同 noise、同 timestep

| 组 | 配置 | 步数 | 同域 drop-IoU | 异域 drop-IoU | gap | gate on 相对 off |
|---|---|---:|---:|---:|---:|---:|
| A | DgCD，无动作项，独立噪声 | 60 | 0.802 | 0.793 | 0.009 | -19.5% |
| B | A + group smooth=1.0 | 60 | 1.000 | 0.786 | 0.214 | -18.8% |
| C | B + 独立 Gumbel 噪声 | 60 | 1.000 | 0.956 | 0.044 | -20.2% |
| R | 随机 33% mask | 60 | 0.199 | 0.199 | 0.0002 | -12.3% |
| D | B + domain-shared 噪声 + domain center | 60 | 1.000 | 0.296 | **0.704** | -18.7% |
| E | D + clean RSC 动作保护（线性相减） | 60 | 1.000 | 0.735 | 0.265 | **-22.4%** |
| F | D + 两阶段动作保护 | 60 | 未跑完（step20: 1.000 / 0.517） | — | 0.483 | 未跑完 |

补充：

- A/B/C2/D/E/R 的 `step0` 结果也记录了，说明 mask IoU 指标本身能区分随机与域结构。
- 随机 mask 的 drop-IoU 同/异域都约 0.199，符合 33% 独立随机选择的期望，验证了指标实现。
- C 的 200 步中，`v_ready=true` 后 `v_raw_rel_std` 稳定在 0.33 左右，动作重要性塌缩问题已修复。

---

## 2. 关键发现

### 2.1 动作重要性修复有效

`ActionImportanceRSC` 现在：

```text
raw_c = mean |f_c · dL_DP/df_c|   # clean pre-gate Taylor
v_c   = quantile_rank(raw_c)       # RSC 式通道归一化
```

- 只在 clean forward / pre-gate 上更新；
- 有效梯度不足时不更新 EMA；
- 不再把 hard-masked 分支的零梯度写进 v。

C 组 200 步：

```text
v_ready: true
v_raw_rel_std: ~0.33
```

### 2.2 DgCD 的原始分数不是 style-specific

A 组：

```text
proto_cos_mean: ~0.80
drop-IoU gap: 0.009
score_cos_gap: 0.037
```

10 个粗域的原型高度相似，top-k 通道排序几乎共享。  
因此 A 组的 gate 提升主要来自 33% dropout 正则，而不是域选择。

### 2.3 group smooth 恢复了同域一致性

B 组：

```text
drop-IoU same: 1.000
drop-IoU diff: 0.786
gap: 0.214
```

说明“同域分数共享”有效，但不同域仍高度重叠。

### 2.4 独立 Gumbel 噪声会打散域结构

B → C：

```text
gap: 0.214 → 0.044
```

Gumbel 噪声逐样本独立采样，导致同一 style 的 mask 也被打散。  
改成 domain-shared noise 后，同域 mask 严格一致。

### 2.5 domain center 是最有效的修复

D 组去掉跨域共享通道成分：

```text
drop-IoU same: 1.000
drop-IoU diff: 0.296
gap: 0.704
score_cos_same: 1.00
score_cos_diff: -0.11
score_cos_gap: 1.11
```

这是本轮唯一同时满足：

- 同域稳定；
- 异域明显不同；
- 随机基线（0.199 / 0.199）被清楚区分；
- gate 不伤害动作。

### 2.6 动作保护存在明显 trade-off

E 组用 `u - λv`：

```text
gap: 0.704 → 0.265
gate on/off: -18.7% → -22.4%
v_raw_rel_std: 0.348
```

动作保护提高了 gate 收益，但显著削弱域选择性。  
因此不建议继续手调 `λ`，应改为两阶段：

```text
1. 按 domain score 选候选通道
2. 在候选集内排除高动作重要性通道
3. 在剩余候选中按 domain score 选 top-k drop
```

F 组就是这个实现；step20 为：

```text
drop-IoU same: 1.000
drop-IoU diff: 0.517
gap: 0.483
eligible_frac: 0.606
v_raw_rel_std: 0.350
```

F 的 step50 尚未跑完，因实例在 22:00:45 被平台关机。

---

## 3. style 视觉可分性审计

脚本：

```text
scripts/audit_style_separability.py
```

数据：

```text
OpenDrawer + CloseDrawer + PickPlaceCounterToMicrowave
322 帧，50 styles，50 layouts
冻结 DINOv2 ViT-S/14，mean patch token
```

结果：

| 指标 | 数值 |
|---|---:|
| 同 style 余弦 | 0.671 |
| 异 style 余弦 | 0.624 |
| 同异差 | 0.047 |
| style 1-NN | 18.6% |
| style chance | 2.3% |
| layout 1-NN | 21.7% |
| layout chance | 2.4% |
| style silhouette | -0.183 |
| layout 残差后同异差 | 0.126 |
| layout 残差后 style 1-NN | 17.1% |
| layout 残差后 silhouette | -0.139 |

解释：

1. **style 信号真实存在**：1-NN 比随机高约 8 倍；
2. **layout 信号更强**：layout 1-NN 高于 style；
3. **style 在原始 DINOv2 空间高度重叠**：silhouette 为负；
4. 去掉 layout 均值后，同异 style 余弦差从 0.047 增加到 0.126。

因此：

> style_id 不是噪声，但当前表示主要被 layout/task 主导。  
> 直接按连续 style ID 分组没有视觉依据。

---

## 4. 当前代码改动

- `dacsc/hard_gate.py`
  - domain-shared Gumbel noise
  - drop-mask IoU
  - domain score cosine separation
- `dacsc/policy.py`
  - 当前 batch 内生成 per-sample mask，不再使用上一 batch 的 mask
  - domain center
  - group smooth
  - two-stage 动作保护
  - gate on/off 同噪声探针
  - RSC/clean Taylor 设备修复
- `dacsc/losses.py`
  - RSC 通道重要性改为 Taylor + 通道分位数
- `dacsc/sampler.py`
  - 采样器索引缓存
- `dacsc/dataset.py`
  - style 41–50 的 held-out validation Subset
- `scripts/audit_style_separability.py`
  - DINOv2 style 可分性/聚类审计
- `scripts/summarize_gate_log.py`
  - GATE_METRICS → CSV

---

## 5. 下一步（按优先级）

1. **恢复实例后先跑完 F 的 60 步**，确认 step50 的 gate on/off；
2. 用随机 mask、D、F 三个对照做 3 个 seed，确认 gap 和 gate 收益不是单次波动；
3. 用 `style_separability` 的 K-means 结果替代连续 ID 分组；
4. 在 domain score 里加入 layout 残差（按 layout 去均值），区分 style 与 layout；
5. 只有 F 在 200 步仍保持 `gap > 0.3`、`gate on/off <= 0`、`v_raw_rel_std > 0.2`，才进入 2k；
6. 2k 验证集必须用 style 41–50，不能用 target style 1–10 做模型选择。

## 6. 当前阻塞

- 官方 `BaseImageDataset.get_validation_dataset()` 默认返回空数据集；
- 官方 workspace 的 validation 代码块被注释掉；
- 因此目前 `val_every` 不会真正计算 val loss。

已补：

```text
DomainLerobotCotrainingDataset.get_validation_dataset()
→ style 41–50 的 Subset
```

但还需要在 workspace 中恢复/接入 validation loop，才能进入 2k。

---

## 7. 结果文件

```text
audit/mechanism_diag_20261019.csv
```

每个实验的完整逐 step 指标可用：

```bash
python scripts/summarize_gate_log.py <train.log> --out <gate.csv>
```
