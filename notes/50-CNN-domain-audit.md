# 50 - ResNet18 stage-end 域标签线性 probe 审计

日期：2026-10-20  
脚本：`scripts/audit_cnn_domains.py`  
结果：`audit/cnn_domain_audit.json`, `audit/cnn_domain_audit_summary.csv`  
特征空间：ResNet18ConvFiLM stage-end GAP（FiLM 1/3/5/7，3 个视角）  
样本：20 × 96 × To2 = 3840

## 1. 结果（每类最佳 view/layer）

| label | balanced acc | macro F1 | silhouette | cos gap |
|---|---:|---:|---:|---:|
| style 连续 3 组（11–30 / 31–50 / 51–60） | **0.593** | 0.591 | -0.003 | +0.000 |
| style 连续 3 组 + layout 残差 | **0.610** | 0.610 | -0.004 | +0.001 |
| style KMeans K=3 | 0.794 | 0.663 | -0.011 | +0.001 |
| style KMeans K=5 | 0.696 | 0.576 | -0.022 | -0.000 |
| style KMeans K=10 | 0.705 | 0.659 | -0.029 | +0.001 |
| style exact（50 类） | 0.801 | 0.800 | -0.054 | +0.003 |
| layout coarse10 | 0.538 | 0.537 | -0.016 | +0.001 |
| task coarse10 | **1.000** | 1.000 | +0.046 | +0.024 |

注意：

- KMeans 的 label 是用同一特征空间聚类出来的，因此 KMeans probe 的 0.70–0.80 **有循环性**，不能当作独立证据；
- style exact 50 类 / 512 维的线性 probe 也可能过拟合，需要 held-out style 验证；
- silhouette 和 cos gap 都接近 0，说明特征簇高度重叠；线性 probe 仍然能利用微弱的全局方向。

## 2. 对 note 49 结论的修正

note 49 说“3 个粗域不可线性分离”，这句话过强。

同特征空间审计显示：

```text
连续 3 个 style 粗域：balanced acc = 0.593
随机水平：0.333
```

所以正确说法是：

> **3 个连续 style 粗域有弱线性信号，但不是干净可分。**
>
> **DomainDrop 在线域头掉到 1/3，不能简单归因于“标签完全不可学”。**

更可能的原因：

1. `domaindrop_norm` 的 **per-sample LayerNorm** 会移除样本之间的全局 style 差异；
2. raw Linear 的 feature scale 很大（norm 12–78），logits 容易膨胀；
3. 在线域头只看到 96 样本/batch，并且 backbone 同时在更新；
4. GRL 与域头联训让优化更不稳定；
5. 线性 probe 用 3840 样本 + 全局标准化，和在线域头的训练条件不同。

## 3. task/layout sanity check

```text
task coarse10: 1.000
layout coarse10: 0.538
```

task 特征非常可分，layout 中等可分。  
因此下一步应该用 **layout 作为 DomainDrop 的 sanity check**：

```text
如果 layout 域头能学到明显高于 1/10，
说明 DomainDrop 代码链路正常；
如果 layout 也学不到，
才继续查标签对齐/优化器。
```

## 4. 下一步

1. **layout sanity check**：DomainDrop 用 layout 标签，51 步，1 seed；
2. 把 per-sample LayerNorm 换成 **global feature standardization**（逐维 batch/running mean/std），保留样本间差异；
3. 用 DINOv2 等**另一个特征空间**定义 style 聚类，再在 ResNet18 上 probe，避免循环性；
4. 只有域头明显高于随机，再跑 51 步 × 3 seeds → 200 步。

一句话：

> **style 不是完全不可分，而是“弱可分 + 高度重叠”。  
> 当前 DomainDrop 失败更像是“域头归一化/优化条件不对”，而不是标签完全没信号。**
