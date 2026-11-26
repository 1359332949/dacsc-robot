# Phase3A SmolVLA 原版路径与 V1 移植协议

目标是构建隔离源参考、检查实现和开发Clean误伤、记录浅深层Clean漂移与工程开销；不把本轮当作三域主表或独立确认。上游roboeval核心未修改，新增外层instrumentation委托其load/act和suite.reset/step，避免重写像素、状态、动作转换。

固定兼容环境、原始Hub配置load_vlm_weights=True、BF16视觉，模型参数冻结。原生n_action_steps=1、chunk50、Euler10；EGL、env.noise=0、256像素、10步settling、max220。SmolVLA不套用CoRP H=E12。

## 数据与方法

- source：十个Spatial官方基础任务的init0，policy seed0。所有实际Frozen推理帧参加，不按任务成功筛选。
- source每帧记录SigLIP residual layers3/6/9/12的逐视角patch均值、二阶矩。完整1024patch，无CLS。两视角由真实输入预检确认顺序，各层每replan独立计数恰好2次。
- 每个目标任务T：mu/ss仅由其他9个任务全轨迹帧等权拟合；tau由这9任务内部8+1留任务评分，汇总全部留出帧取95%逐通道分位。T不出现在任一内层折。
- 主方法仅layer3、alpha=.75、有界增益[.5,2]、硬死区；与原V1公式相同，内部FP32，返回前转回原BF16。alpha0直接返回原特征；正常通道在BF16字位保持恒等。无EMA、梯度、外部模型、域标签或目标数据拟合。
- guard：十任务init1，Frozen/V1各一局，policy seed0。init1此前已用于诊断，必须称校准状态不重叠的开发Clean；不能标为未见或独立确认。未打开保护的Blind/Held-out清单。
- 外层每局reset/settling后显式固定Torch CPU/CUDA；共同replan前RNG hash前缀、首帧物理状态、双眼RGB都要求精确相同。原版reset(seed)自身不控制模型采样，该外层修改单独披露。

## 验证和判决

六项CPU测试：BF16恒等、首patch参与、LOTO内外折排除、相机参考不混、缺/多相机调用拒绝、错误dtype拒绝。真实接口预检四层alpha0逐位动作恒等；每个guard起点再独立做四层alpha0与固定alpha.75扫描。扫描仅代表Clean t0动作漂移，不证明物体位移响应或Shift修复。

完成所有10个配对，不因中途成败删例、重挑seed或调alpha。基础设施错误记录为错误，不算任务失败。必须报告Frozen/V1每例成功、救回/误伤/净数、实际激活率及精确McNemar。Clean护栏要求观察到0误伤，且总成功不下降；10例和1个策略种子不能证明总体零风险。95%源阈值不保证LOTO新任务/新状态恰好5%激活，3%-6%只作原请求的诊断期望，不反向调tau凑数。

源和guard使用4进程EGL并行。全部完成后单GPU隔离测延迟：2对预热，5轮AB/BA交错，CUDA前后同步，记录每轮端到端native act、峰值allocated/reserved；含CPU输入处理和动作回传，不含env.step。V1测量包含诊断moment capture，不能把5次样本当精确分布。

Phase3A结束后停下给用户审阅，不自动启动主表。SALT Appendix B.6 Table13描述首replan起施扰，但其Table17设置LIBERO10、H50/E8；本轮Spatial/E1不构成该表的公平直接比较，主表须另行锁定任务套件、执行频率、双视角扰动定义、严重度、同期Frozen及多种子。

SALT原文：https://arxiv.org/html/2610.07946v1 。本轮预检/协议/源码快照、每源轨迹及参考SHA、逐例动作/RNG/激活/EEF轨迹、每轮计时均保留在runs/phase3_smolvla；大型数据不入Git。
