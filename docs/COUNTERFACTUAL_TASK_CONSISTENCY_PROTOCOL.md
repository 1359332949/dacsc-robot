# 反事实策略一致性诊断：首次执行前冻结的范围

先复核 Source provenance；通过后仅以现有配对 RGB/state8 在服务器执行冻结 CoRP 推理，不重训 V1、不启动 V1 B/C、不新建环境或新增 rollout、不访问 target-test。

## P0 来源复核与 loader 修复

逐一核对 Source 全部 20 条轨迹的 collector、数据契约、共享推理代码、checkpoint、split、base BDDL、初始状态文件、上游模型代码、渲染器、精度、库版本和执行协议。group 名称不参与判定。fit_smoke/state0 和 val_smoke/state18 都必须与其他分组具有同一 collection signature；核对原 V1 model.json 保存的逐文件 hash。缺失/不同指纹在 loader 读取 NPZ 之前拒绝。

同组一致性并不等于对未记录外部行为的证明；本轮结论基于持久化元数据、原始数据及源代码对应。原 V1 模型、配置和报告继续冻结。loader 的加强属于未来输入校验修复，不改变数值算法；验证相同数据预测不变，不绕过原 V1 的代码漂移检查或伪造其历史 hash。

## 反事实输入

现有初始状态只有 34、36。state34 使用 36 步 pilot，state36 使用 106 步 terminal 记录；state36 pilot 仅验证重复前缀，不重复计入。共 14 个观测边界（不是 14 个独立初始状态），两个 rollout 都是 Clean 命令驱动的受控回放。

每个边界：

1. 检查 Clean/Lighting 当前 state8 相同、已保存实际执行动作相同及时间对齐、此前 measured-state 配对检查通过。
2. 使用该边界**当前 state8**、相同语言，构造 CC/SC/CS/SS 四个输入。第一个字母代表 agentview，第二个代表腕部。混合视角是离线归因控制，未部署。
3. 在相同 x_probe、tau=0.5、k=0 上测量 DiT velocity 和 hidden；x_probe 不沿不同观测条件积分。
4. 完整动作使用另一固定 z0（seed=20261122，仅探针随机张量，不新增环境 seed），运行官方 policy.predict。相同 z0 只保证初始 noisy action 相同，不声称中途 x 相同；完整动作与 same-x probe 两类结果分开。
5. 固定初始 proprio 的历史 probe 不用于本次策略一致性指标。仍保留历史结果并解释其不同意义。

## 表征与动作指标

特征是语言条件 Resampler 的每视角 48×768 tokens，另报告 token 内平均后的 768 维 pooled 特征。不是 DINO 原始 patch dynamics，不反推上游 TTA 有效性。

每个 CC 对照报告 SC、CS、SS 的 per-view token RMS / pooled L2，same-x velocity RMS / k0 hidden L2，以及完整 12×7 controller action 的 preclip / postclip RMS、XYZ 命令向量差、旋转/夹爪命令差、饱和比例。控制命令差不是实际执行轨迹误差或任务伤害。物理尺度若缺少本次元数据核验不报告毫米误差。

同时报告 SS−SC−CS+CC 的向量范数，避免将非线性下游两路影响简单相加。没有动作风险阈值、任务成功改善或判定“有害视角”的规则。

## 可区分性必须并列保留及其局限

比较不同初始状态在共同边界 0/12/24/36 的 Clean 距离和 Shift 距离，记录 state8 差与此前动作序列是否一致。相同固定 x_probe/z0 支持比较策略条件响应，不构成执行相同物理动作的反事实未来实验。

现有跨初始状态动作历史不同，缺少完整物体物理状态标签；因此不能计算或宣称 ACPC 原论文意义的 Invariance Radius / Separation Rate，也不能据两状态断定未发生任务信息坍缩。额外报告同一轨迹各边界间距离，仍只作描述，不当独立样本。

同时报告原始距离和 Shift/Clean 比值；分母近零时拒绝比值。常量表征必须产生零分离而不能被称为“完美稳健”。这些检查是机制诊断，不是新的观测器或风险评分器。

## 验收与停止边界

- 重读原始 NPZ hash、验证契约；重复前缀一致；CC/SS 重提 token 与保存 token 一致。
- 单视角切换不改变另一视角 Resampler token；当前 state8 被实际传入，初始锚点不混入。
- 所有四种输入做官方采样路径比对；固定 probe 重复；无参数更新；保存 checkpoint/输入/config/代码/输出 hash。
- 重新用保存 z0 预测 CC，核对它与当时保存 actions_planned 一致。完整 12 步预测与实际执行 mask 分开，部分/终止边界不伪装成执行完整动作。
- 仅两状态，不给跨状态总体置信区间或风险分类指标。若策略输出变化非零，只能说明光照影响输出，不能认为它导致失败。
- 保留 STOP_V1。此诊断是另一个研究问题，不能事后改写 V1 gate 成功或充当被禁止的 B/C。
