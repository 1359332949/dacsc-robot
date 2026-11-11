# 工单 4.0b 报告：弱变换 + Identity Anchoring 离线验证

> 只做离线验证：**不跑 rollout、不改算法、不动 Phase 3 数据**。
> 原始 JSON：`/hy-tmp/validation/sg_cpa_4_0b/sg_cpa_4_0b.json`；脚本 `scripts/sg_cpa_phase4_0b.py`。
> K=4（identity + 同一种变换的 3 个变体）；每种变换单独测试；18 个观测 batch；
> 每组共享 3 个动作噪声 seed（1234/2345/3456）配对比较。

## 0. 先修正 4.0 的一个实现错误（重要）

检查代码时发现：`prepare_observation()` 返回的是 **ImageNet 归一化后的图像**（`corp/preprocessing.py:53-65`），
而 4.0 的变换直接在归一化张量上做（并 clamp 到 [0,1]），**破坏了输入**。
→ **4.0 的变换敏感性 / 共识漂移结论无效**；4.0 的 Jacobian、投影、loss、成本结论不依赖变换，仍然有效。
4.0b 已改为：先反归一化到像素 [0,1] → 做变换 → 再归一化（`to_pixel/from_pixel`）。

## 1. 弱变换下的动作影响（修正后）

| 变换族 | D_view（3 个增强视图间） | aug_dev（增强均值 vs identity） | 说明 |
|---|---|---|---|
| noise σ=0.01 | 0.00314 | 0.00159 | 很温和 |
| noise σ=0.02 | 0.00638 | 0.00382 | 温和 |
| blur k3 σ=0.4–0.6 | 0.00636 | 0.00704 | 温和 |
| brightness 1.05–1.07 | 0.00085 | 0.00373 | 很温和 |
| contrast 1.05–1.07 | 0.00049 | 0.00222 | 很温和 |
| saturation 1.05–1.07 | 0.00018 | 0.00082 | 几乎不变 |
| **strong noise 0.05** | 0.01697 | 0.01758 | 4.0 级别 |
| **strong blur k5** | 0.00443 | 0.02703 | 4.0 级别 |
| **strong photo 1.2** | 0.00602 | 0.02467 | 4.0 级别 |

## 2. Identity Anchoring（Ā_α = A_id + α(Ā_aug − A_id)）

代表配置（anchor = `D_anchor_all`，fit = `D_fit`）：

| 配置 | α=0 | α=0.1 | α=0.3 | α=1.0 |
|---|---|---|---|---|
| noise 0.02 | fit 7.4e-5 | anchor 3.8e-4 / fit 6.7e-5 | anchor 1.2e-3 / fit 5.6e-5 | anchor 3.8e-3 / fit 3.8e-5 |
| blur k3 | fit 4.8e-4 | 7.0e-4 / 4.2e-4 | 2.1e-3 / 3.4e-4 | 7.0e-3 / 2.0e-4 |
| **strong noise 0.05** | fit 1.46e-3 | **1.76e-3 / 1.29e-3** | 5.27e-3 / 1.00e-3 | 1.76e-2 / 5.6e-4 |
| strong blur k5 | fit 2.69e-3 | 2.70e-3 / 2.21e-3 | 8.11e-3 / 1.40e-3 | 2.70e-2 / 1.7e-4 |

**分解（XYZ / Rotation / Gripper）**：所有配置下锚点偏移以 **XYZ 为主**（约为 rot 的 1.5–2 倍），gripper 最小。
例如 strong noise α=0.1：xyz 2.70e-3 / rot 1.32e-3 / grip 2.4e-4。

**结论**：Anchoring 能连续控制“行为保真 vs 共识拟合”的权衡；α=0.1–0.3 给出可控漂移与更低的 fit 误差。

## 3. 修正后的 K 敏感性（混合弱视图，K=2/4/8）

（K 池：identity + noise0.02 + blur + brightness + contrast + saturation + noise seed2 + blur σ0.6）

| K | D_view | D_anchor Mean | D_anchor Medoid |
|---|---|---|---|
| 2 | 0.00492 | 0.00246 | 0.0000†（平局） |
| 4 | 0.00629 | 0.00231 | 0.00035 |
| 8 | 0.00674 | 0.00262 | 0.00027 |

**重要更正**：修正实现后，**没有出现"K 越大锚点越漂移"的现象**（anchor 基本持平 0.0023–0.0026）。
4.0 中的"稳定性↑则漂移↑"红旗是**变换实现错误造成的伪结论**，已撤回。
（K 增大 D_view 略升，说明弱视图之间的差异反而被更多样的视图暴露。）

## 4. K=1 no-op（直接动作蒸馏）

- distill loss = **0.0**；distill grad = **0.0** ✅（K=1、identity、β=0、共享噪声时严格 no-op）

## 5. 投影更新在 anchored-consensus 梯度下的有效性

（family=noise 0.01，α=0.3，真实 A、真实 Jacobian）

| 项 | 结果 |
|---|---|
| 原始梯度 `‖g‖` | 2.12e-3 |
| 投影后 `‖P_null g‖` | 1.62e-3（**retained 76.4%**） |
| `‖J_A P_null g‖` | **0.0**（相对 0） |
| 有限步 A 方向变化 | raw 5.22e-3 → projected **1.80e-5（≈290× 降低）** |

**结论**：投影在保护 A 的同时保留了约 76% 的更新量，不是“投影后什么都不动”。

## 6. 验收结论

| 验收条件 | 结果 |
|---|---|
| 1. 至少一种弱配置在有限动作偏移下产生可测的视图一致性改善 | **PASS**（noise 0.02：D_view 6.4e-3，aug_dev 3.8e-3；α=0.1 时 fit 从 7.4e-5 降到 6.7e-5，anchor 3.8e-4） |
| 2. Anchoring 控制 identity 偏移且不退化为 no-op | **PASS**（α 连续调节；α=0.1–0.3 有非零 anchor 与更低 fit） |
| 3. 投影后仍保留非零、有意义的更新 | **PASS**（retained 76.4%，有限步 A 变化降低 290×） |
| 4. K=1 直接蒸馏仍满足 no-op | **PASS**（loss=0，grad=0） |

**结论：4.0b 四项验收全部通过，可以进入 4.1 局部实现。**

## 7. 但必须诚实说明信号量级

弱变换的可用信号（D_fit）只有 **1e-5–5e-4** 量级；strong（noise0.05/blur k5）也只有 **1e-3–3e-3**。
这意味着：
- 4.1 首版建议采用 **strong noise σ=0.05 或 blur k5 + α=0.1**（信号较大、anchor 仍受控 1.8e-3–2.7e-3）；
- 若 4.1 训练后动作变化仍小于 ~1e-3，几乎不可能改变成功率 → 应及时判定“共识信号太弱”，不要硬调。

## 8. 建议的 4.1 冻结配置（待确认）

| 项 | 建议 |
|---|---|
| 视图 | identity + noise σ=0.05（3 个噪声 seed）；或 identity + blur k5（σ=1.0/1.1/1.2） |
| 聚合 | anchored Mean，α=0.1 |
| 损失 | 动作块直接蒸馏（K=1 no-op 已验证） |
| 更新 | 投影 SGD 起步（Adam 需投影实际更新向量），η 小步 + 有限步 A 校验 |
| 不跑 rollout | 4.1 只做局部实现 + 单测；rollout 需单独批准 |

_commit 见本次提交。_
