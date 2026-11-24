# 本轮原始文献核查与可用边界

本轮从作者项目页、arXiv 摘要和 ACPC HTML 正文核对。检索工具未返回可用内容，改用网络读取原始页面；不把会话转述作为论文内容的唯一依据。

1. **ACPC**：*Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency*，arXiv:2608.12939v1（2026-08-13）。阅读摘要、正文 §3 与 limitations。
   - 对同一历史的 Clean/扰动版本使用同一动作序列，比较世界模型的多步预测；IR 与 SR 同时使用以防止常量表示被当成稳健。
   - 文中区分预测一致性和预测准确性，理论有关联预测误差/规划代价而非对真实环境 return 的普遍保证。
   - 本轮 CoRP 是动作策略，不是被该论文定义的潜在动力学预测器。固定 flow x、测策略输出只是相关机制控制，不复现 ACPC 的 IR/SR，不继承其界限。
   - 原文：`https://arxiv.org/html/2608.12939v1`；摘要：`https://arxiv.org/abs/2608.12939`。
2. **PhyLatent**：*Learning Dynamics-Relevant Representations for JEPA World Models*，arXiv:2608.05720。重新核对摘要。
   - 重点区分外观不变性、不同物理状态可分性、不同动作条件未来可分性。该问题分解提醒我们：表征距离变化、任务行为变化和风险不是同一命题。
   - 本轮没有训练 JEPA，也没有完整物理状态标签和同动作跨状态未来，不能声称消除其定义的坍缩。
   - 原文入口：`https://arxiv.org/abs/2608.05720`。
3. **TTA-VLA**：*Can Vision-Language-Action Models Adapt to Visual Perturbations Using a Test-Time Trajectory?*，作者项目页。
   - 再核对其利用测试轨迹的视觉/动作表征转移与语言语义锚点。它是适配方法；本轮仅冻结策略诊断，没有优化 projector。
   - 本轮不复现其成功率或把“使用时序关系”当作本项目独立创新点。
   - 作者页：`https://tta-vla2.github.io/`。

不扩展未经本轮核验的其他论文效果或创新性结论。采用的距离、双视角混合、固定 probe 和命令差为本项目预先说明的描述性指标；不能把“类似 ACPC 的思想”写成“通过 ACPC 理论验收”。
