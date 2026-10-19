# SWAD 对比记录（后续可选）

来源：Cha et al., **SWAD: Domain Generalization by Seeking Flat Minima**, NeurIPS 2021  
论文：https://arxiv.org/abs/2102.08604  
官方代码：https://github.com/khanrc/swad （MIT）

## 它做什么

- 在训练轨迹上**密集平均模型权重**（Stochastic Weight Averaging Densely）；
- 用训练域验证 loss 的“谷底”选择平均区间（LossValley）；
- 目标是寻找**平坦极小值**，提升域偏移下的稳健性。

## 它不是什么

- 不是通道门控；
- 不是域敏感特征选择；
- 不会显式区分“风格通道”和“动作关键通道”；
- 不能替代 DomainDrop / DgCD / 我们的门控。

## 和我们的关系

适合作为：

1. **强 baseline**：`ERM + SWAD`；
2. **组合增强**：`我们的 hard gate + SWAD`；
3. **训练稳定器**：减少 checkpoint 噪声、改善模型选择。

如果使用，必须公平地应用到所有对比方法，不能只给我们的方法加 SWAD。

## 迁移到本项目的注意点

- 用验证 style 41–50 的动作 MSE 或少量闭环成功率选谷底；
- **不能用 target style 1–10 做选择**；
- SWAD 与现有 EMA 不是同一个东西：先分别做 `EMA` 与 `SWAD` 两组，再考虑组合；
- soft gate 参数可被 SWAD 平均；hard mask / training-only dropout 需要单独决定推理时是否保留 mask；
- 一个平均模型副本约 0.17GB（85.84M 参数，BF16），显存开销可接受。

## 当前决策

**暂不执行。** 先把 v3 的门控机制验证通过，再在后续对比实验中加入 `ERM + SWAD` 作为 baseline。
