# 阶段 1 / 2 完成：历史时间审计与独立源域数据

日期：2026-11-21。本轮按限定范围完成零 GPU 历史审计、独立 Source Fit / Source Validation 采集与验收。未训练观测器、Adapter 或风险分类器，本轮未访问 target-test 环境。

## 历史时间审计

- Clean 成功轨迹：21 条，结束步数范围 75–131，中位数 96。
- Lighting 2292 成功轨迹：11 条，范围 85–212，中位数 104。
- 两域全部 28 条未成功基线记录都在 220 步截断。结束时间不是风险发生时间。
- H=E=12，第一笔 transition 残差最早在第 12 步完成后可用。当前基线没有足够的逐步事件标签来定位首次风险或最后可恢复时刻，因此有效提前量仍未知。
- 旧 state37 的 Clean/Shift frozen 都成功，但第 1 步已出现明显命令分歧；这不能作为危险标签。
- Gate 1 的 7/3/3 保留为历史探索记录。续跑上限为 300，而原基线为 220；另有三个 Shift 续跑在 90/127/113 步成功，却与原基线失败不一致。因此不能直接把 7/13 当作已验证可干预比例。

完整证据见 `FAILURE_TIMING_AUDIT_20261121.md` 和 `docs/data/failure_timing_20261121/`。审计只有 CPU 计算，源文件指纹、CSV、图和可公开复现的最小旧日志摘录均已保存。

## 独立 Source Clean 数据已采完

保持冻结 CoRP / FP32 / OSMesa / H=E=12 / 最大 220 步。固定 seed=2014，仅使用原 split 的 Source Fit / Source Validation 初始状态。

| 数据角色 | 独立状态 / episode | 策略 rollout 成功 | 执行步数 | 完整 12 步 chunk | 部分 chunk |
|---|---:|---:|---:|---:|---:|
| Source Fit | 10 | 8/10 | 1247 | 101 | 8 |
| Source Validation | 10 | 10/10 | 945 | 74 | 10 |
| 合计 | 20 | 18/20 | 2192 | 175 | 18 |

这些成功率只是采集轨迹的原策略结果，**不是观测器性能、泛化收益或新的成功率对比实验**。

- Fit 状态：0、1、2、4、6、8、10、13、14、17。
- Validation 状态：18、20、21、22、25、27、29、31、32、33。
- 两组无交叉，也未复用 target-dev 配对回放中的 Clean 轨迹。
- Fit 的状态 4、17 在 220 步截止时未成功，原样保留；未按最终结果挑样本。
- 共 193 个 transition，主固定时间跨度预测最多使用其中 175 个完整 chunk。它们仍只有每角色 10 个独立初始状态，不能当成 175 个独立场景。
- Clean 是环境条件，不能把所有 Clean chunk 自动标为健康负例；也不能把两条失败轨迹所有 chunk 自动标成风险。

## 采集及验收内容

每条轨迹保存边界观测 state8、两路 RGB、分开的 48×768 visual tokens；每个执行步保存 state8、模拟时间、实际裁剪动作、mask 和 success / done / terminated / truncated。采集截止与环境截断分开标注。

- 每个 episode 的初始实际输入上，官方 predict 与显式 Euler 检查最大误差均为 **0**。
- 边界下一状态与最后一个实际执行步状态一致；模拟时间与执行步数一致。
- 文件哈希、完整角色覆盖、无重复 state/seed、代码与协议指纹一致性全部通过。
- 原始 NPZ 下载到本地后重新验证，服务器/本地的全部 episode 验收结果一致。
- **11 项测试通过**；此前 target-dev 配对回放在共享契约扩展后仍通过回归验收。
- 此次 Source 数据没有伪造固定 probe 字段，没有将初始状态锚点探针当成真实动作意图。

## 数据位置与证据

服务器原始数据：

`/hy-tmp/patch-diag/runs/source_interactions/20261121/`

本地原始副本（NPZ 总计 85,027,816 字节）：

`/Users/wentiliangkaihua/Documents/ChatGPT/专利论文/remote_artifacts/source_interactions_20261121/`

仓库中保留轻量审计产物，未把 85 MB 原始 NPZ 放入 Git：

- `docs/data/source_interactions_20261121/dataset_validation_server.json`
- `docs/data/source_interactions_20261121/dataset_validation_local.json`
- `docs/data/source_interactions_20261121/collection_metadata.json`
- `docs/data/source_interactions_20261121/episodes.csv`
- `docs/data/source_interactions_20261121/evidence_summary.json`
- `docs/data/source_interactions_20261121/local_tests.txt`、`server_tests.txt`

核心实现：`scripts/audit_failure_timing.py`、`scripts/collect_source_interactions.py`、`scripts/validate_source_interactions.py`、`dacsc/source_interactions.py`。

## 下一关口与尚未证明的事项

下一步可以按 `SOURCE_INTERACTIONS_PROTOCOL.md` 预先冻结低容量模型及低维 per-view 表示，只在 Source Fit 拟合、Source Validation 验证。预处理/PCA/残差尺度只从 Fit 得到；禁止用 Lighting 结果选择特征维度或模型。

本轮没有证明：下一帧视觉变化可被可靠预测、预测误差对无害光照稳健、Lighting 内部未来风险可预测，或风险出现后仍有可用干预窗口。尤其不能从晚于第 12 步的 episode 结束时间推出足够提前量。

数据采集和时间审计完成，为正常交互预测实验提供了独立输入；后续仍需依次验证正常预测、无害变化、任务风险三个关口。
