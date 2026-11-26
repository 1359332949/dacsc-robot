# CoRP 三套件主表收口清单（2026-11-26）

这是对仓库已有报告/审计的清单复核，不是本轮重新跑出的 CoRP 胜率。当前 GPU 任务仅为 SmolVLA severity3 的 40 局。

| 套件 | 已有可追溯证据 | 不能直接填入统一五臂主表的部分 |
|---|---|---|
| Spatial | 原冻结 Block3 全轨迹参考的三种子配对报告；该固定参考未被后续 shared calibration 公式错误一并推翻 | 该报告不是五臂正式比较；Pixel、Deep、ActMAD 必须在相同正确参考预算/条件/种子下验收并实测 |
| Object | 官方物理三臂、三种子跨套件报告，明确区分 Spatial 零样本参考与 Object 校准 | 不是五臂；standard100 的 +9.0pp 包含单眼 Sensor，不能直接写成无健康眼兜底的总体收益；双眼 Sensor 单独仅 +2/90 |
| libero_10 | 历史五臂、后续本套件校准双臂报告，以及错误/权限审计 | 旧 Lighting/Background 五臂已被后续报告标为污染/失效；旧共享 tau 公式错误；98/100 为同任务不同起点的校准权限，不能冒称严格任务 LOTO；修正版本须重新取得对应闭环数据 |

共同缺口：

1. **对照身份。** `actmad_true` 是旧移植标识，存在多帧双相机排列错误。`actmad_inspired_viewwise_v2` 修正排列并通过真实模型执行链检查，但未重新取得完整闭环成绩；它与作者原算法的保空间位置、均值/方差 L1 目标仍有差异。旧 3.22× 只能归属于当时的内部实现，不能据此宣称优于忠实原版 ActMAD。
2. **参考版本与隔离。** `aggregate_calib.py` 的旧阈值括号错误已修复，输出新文件而不覆盖旧参考。读过错误 shared reference 的后续实验与原冻结 V1 不能混为同一实现。需要给每臂记录 checkpoint、参考 hash、实际源任务与初始状态权限；套件内校准仍须按本项目要求外9/内8+1排除目标任务。
3. **计分口径。** 通用主问题用 Clean、双目受扰 Sensor、Lighting、Background。单目/反向压力只作诊断；不能把 standard100 含单目收益和另一套双目总分拼成一行。桌面/场景分组按真实资产，不按索引伪造。
4. **闭环证据。** 每份可比主表需要同期 Frozen、预定多策略种子、包含原成功样本、救回/误伤及任务聚类区间。显著性不足、地板、Clean 损失均如实保留。旧报告不因修代码而自动变成修复后结果。

优先顺序是先冻结正确参考与对照定义，再验收执行链/恒等性，最后只补缺失的同期闭环。不得在未经检查的旧五臂脚本上直接扩大实验，也不得为完成主表打开保护池。

依据（仓库已有文件）：

- `BASELINE_IMPLEMENTATION_AUDIT_V2.md`
- `SMOLVLA_REPRODUCTION_DIAGNOSIS_V1.md` 第5节
- `EGL_BLOCK3_FULL_TRAJ_REF_V1.md`
- `OFFICIAL_OBJECT_PHYSICAL_RESULTS.md`
- `PHASE2_LIBERO10_REPORT.md`
- `PHASE2_LIBERO10_INSITU.md`

本清单不把两个骨干的差异归因为“SSL vs VLM”的因果规律；精度、预训练、策略和套件等混杂因素尚未被控制实验拆开。
