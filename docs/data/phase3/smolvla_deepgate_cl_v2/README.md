# SmolVLA DeepGate闭环证据入口

主表与分组救伤/区间：../../../PHASE3_SMOLVLA_DEEPGATE_CL.md；规范结果：../smolvla_deepgate_cl.json。

方法代码c183679，配对渲染防护修订ee6ec5f；正式结果只来自v2整批重跑。edbac60阈值与原校准包保持不变。

门控模块dacsc/smolvla_deepgate.py不读取环境几何/域标签/成功标记，只读取原策略图像和固定源参考；实验runner的物理/渲染状态哈希仅用于配对输入一致性控制。

旧严格字节试跑归档排除。renderer_recovery_review.json如实说明原失败像素未保留、原差异量未知；正式重跑实际捕获的微差量见final_audit.json，不反推成原事件的实测值。

统计只比较本轮同期Frozen与DeepGate；历史V1不是同期第三臂，历史Frozen结局变化保存在审计记录中。Clean无伤只限本次有限配对，逐帧95%门控也不是整局95%安全保证。

复核：先在原环境运行scripts/audit_smolvla_deepgate.py --run原运行目录，再运行scripts/finalize_smolvla_deepgate.py --run原运行目录，可重新导出带审计的表。任务Bootstrap为10,000次整任务重采样；p值直接用binomtest().pvalue。

完整51配对轨迹、原始/审计报告、源代码、标定包、门控、延迟与恢复记录保存在永久remote_artifacts/smolvla_deepgate_cl_v2/smolvla_deepgate_cl_v2_evidence.tar.gz；本目录收据记录SHA256。HF权重与完整环境资产需另按记录准备。
