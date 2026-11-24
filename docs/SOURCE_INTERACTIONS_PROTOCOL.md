# 独立 Source Clean 连续交互数据

本轮范围：历史时间审计与源域采集。禁止训练 Adapter、风险分类器或四套并行信号，禁止访问 target-test 环境。正常观测器的拟合留到数据验收之后。

## 冻结数据划分

依据 `docs/data/tta_split_libero_spatial_perturbation.json`：

- Source Fit：0、1、2、4、6、8、10、13、14、17。
- Source Validation：18、20、21、22、25、27、29、31、32、33。
- 本轮固定一个 evaluation seed：2014。独立统计单位为初始状态，不以 chunk 放大样本量。
- CoRP 官方 checkpoint 的预训练状态重叠未经独立核查；上述独立指本次观测器 fit/validation 划分，不声称相对策略预训练数据未见。
- 每个状态从独立 Clean 环境重置，使用原生返回观测和冻结 `policy.predict()`；本次数据不是 target-dev 配对回放中的 Clean 半边。
- H=E=12、action_dim=7、10 Euler steps、FP32、OSMesa、最多 220 个执行步，10 个 settling steps。
- 全部预定状态保留，不根据最终成败丢弃轨迹。Clean 表示环境条件，**不等于健康、正确或无风险标签**。

## 数据字段

每条轨迹为独立 `episode.npz` + `metadata.json`。保存：

- N+1 个连续边界的 state8、两路 RGB、各自 48×768 visual tokens；无跨视角平均。
- 每个 chunk 的计划动作、实际裁剪动作、执行 mask / 步数、z0；执行命令必须等于计划动作裁剪后的前缀。
- 每个执行步的 state8 和模拟时间；边界下一状态必须等于最后实际执行步的状态。
- success、legacy done、terminated、env_truncated、collector_truncated、合并 truncated 分开。
- 部分 chunk 保存，但 `fixed_horizon_eligible=false`；不混入完整 12 步预测任务。
- 本轮不提取固定初始状态探针来冒充实时动作意图，也不填造任何 probe 字段。
- 每个 episode 初始实际观测进行一次官方采样路径一致性检查。
- checkpoint、split、BDDL、初始状态文件、代码、NPZ 的 SHA256，角色、状态、seed 和真实 outcome。

## 防泄漏与验收

- `role_ids` 同时拒绝 Fit/Validation 重叠、Source/Target 重叠、错误角色状态和重复状态。
- Source Fit 只能用于后续模型、PCA、标准化、残差尺度拟合；Source Validation 用于独立评价及阈值确定。
- dataset validator 重新打开 NPZ，验证文件 hash、完整角色覆盖、无重复 episode、代码和采样协议一致、时间和下一状态对应。
- 采集分组只影响进程调度，不修改初始状态、语言、z0 或数据角色。每个 episode 使用独立固定种子。
- 部分采集验收会显式输出 `coverage_complete=false`，不得作为完整数据集。
- 源域失败轨迹不能直接按整个 episode 给所有 chunk 加危险标签；也不能把所有 Clean chunk 自动作为健康负例。

## 下一阶段的三个关口

1. 正常预测：先固定低维 per-view 表示；PCA、标准化、对角/收缩残差尺度只在 Fit 拟合。比较状态+动作、视觉+状态+动作、视觉不带动作，保持预测目标相同。
2. 无害变化：冻结模型和阈值，在已有受控回放检验仅光照变化的响应。原始 delta_z 差不是观测器预测误差。
3. 任务风险：最后才在自然 Lighting 内部检验未来事件；与状态+动作基线比较，检查输入状态/动作支持范围，按初始状态分组。

三个关口不能相互替代。采集通过只意味着具备拟合条件，未获得风险检测能力。

## 文献核对范围

2026-11-21 重新读取 PhyLatent 作者的 arXiv 摘要：
`https://arxiv.org/abs/2608.05720`。

该工作明确讨论外观变化下的物理不变性、不同物理状态的可区分性、以及不同动作条件未来的区分问题。本轮借鉴这个问题分解；不将其训练 JEPA 的效果移作冻结 CoRP 风险检测证据。未在本轮验证其代码或全文实验，也未据此宣称本项目创新性。

原始视觉变化在两状态受控回放中不具备外观不变性，并不能证明所有原始特征统计在任何任务上都不可能提供风险信息；能确认的是当前这些量不能未经任务风险验证就直接作为报警器。

## 复现与数据验收

服务器目录：`/hy-tmp/patch-diag`。保持 `MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，并使用原有 LIBERO/CoRP PYTHONPATH。

```bash
python scripts/collect_source_interactions.py --matrix /hy-tmp/validation/multidomain_matrix.json --checkpoint repos/CoRP/checkpoints/corp_libero.pt --split docs/data/tta_split_libero_spatial_perturbation.json --seed 2014 --role source_fit --out runs/source_interactions/NEW_RUN/fit
python scripts/collect_source_interactions.py --matrix /hy-tmp/validation/multidomain_matrix.json --checkpoint repos/CoRP/checkpoints/corp_libero.pt --split docs/data/tta_split_libero_spatial_perturbation.json --seed 2014 --role source_validation --out runs/source_interactions/NEW_RUN/validation
python scripts/validate_source_interactions.py --root runs/source_interactions/NEW_RUN --split docs/data/tta_split_libero_spatial_perturbation.json --out runs/source_interactions/NEW_RUN/dataset_validation.json
```

输出目录必须不存在。`--states` 只允许选择声明角色内部的状态，用于分组采集；最终验证必须覆盖两个角色全部预定状态，不得使用 `--allow-incomplete` 作为最终验收。
