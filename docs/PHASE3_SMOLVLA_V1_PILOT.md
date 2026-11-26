# Phase 3A：SmolVLA V1 校准、Clean 护栏与延迟

**本轮判决：NO-GO：Clean 出现误伤，暂不进入主表。** 同期 Frozen 7/10，V1 6/10；救回 0、误伤 1、净 -1。不是三域效果或独立盲测。

## 1. 实现与隔离

- 使用已复现的兼容环境，原版 roboeval 的 SmolVLA.load / act / 输入归一化 / 动作后处理，以及原版 suite.reset / step。外层仅选择指定init、控制配对RNG和注册hook。未修改上游源码或权重。
- 原始Hub配置load_vlm_weights=True，视觉BF16；hook内mean/std/d2/gain/out均FP32，返回BF16。无CLS，完整1024patch；四层均校验每次replan恰好两次串行调用，真实像素输入顺序已核对。
- 模型保持原生n_action_steps=1、Euler10、chunk50；EGL、env.noise=0、settling10、max220。没有套用CoRP H=E12。
- 环境Python/NumPy/libc srand及上游env.seed固定为task ID；每个回合在环境reset/settling后显式重置Torch CPU和CUDA seed0。两臂首帧物理state/RGB逐位一致，所有共同replan的RNG状态hash前缀相同。
- source仅十任务init0全部实际Frozen轨迹，包含失败轨迹、不按成功筛选；全轨迹帧等权，与冻结V1定义一致。目标任务T的mu/ss来自其他9任务，tau由这9任务内部8+1留任务评分后取95%分位。
- **init1此前已被诊断访问**，本轮称“校准状态不重叠的开发Clean”，不能称未见/独立确认。每任务source/guard的实际初始模拟状态hash亦不同；不打开Blind/Held-out清单。
- 校准共 1312 个实际replan；文件 spatial_init0_calib.npz，SHA e573729434d6cdba404d887dc3f685d33b6c67fd6901cd487c512a18fa0ef606。

## 2. 配对Clean结果

| Task | Frozen | V1 | 救/伤 | V1主眼/腕眼平均激活率 |
|---:|---:|---:|---|---|
| 0 | 1 | 0 | 0/1 | 2.6% / 5.3% |
| 1 | 1 | 1 | 0/0 | 4.8% / 0.8% |
| 2 | 1 | 1 | 0/0 | 4.3% / 1.3% |
| 3 | 0 | 0 | 0/0 | 5.5% / 2.1% |
| 4 | 1 | 1 | 0/0 | 13.4% / 14.3% |
| 5 | 0 | 0 | 0/0 | 3.3% / 14.7% |
| 6 | 0 | 0 | 0/0 | 6.0% / 5.8% |
| 7 | 1 | 1 | 0/0 | 11.4% / 1.7% |
| 8 | 1 | 1 | 0/0 | 3.2% / 3.2% |
| 9 | 1 | 1 | 0/0 | 10.9% / 2.6% |

精确McNemar p=1。本轮只有一个策略种子、10个任务/状态；即使零误伤，也只能说本样本未观察到误伤。 Frozen宏平均激活率：主眼6.65%、腕眼4.95%；V1：主眼6.54%、腕眼5.18%。

## 3. 浅/深层单层扫查

固定alpha=.75，四层分别独立干预，同一缓存Clean t0和同一随机数；不是多层叠加，不据此挑选新主方法。这里只测动作漂移，不能称为真实移物体的几何保真或三域修复。

| 层 | XYZ平均漂移L2 | Rotation平均漂移L2 | 动作平均漂移L2 |
|---:|---:|---:|---:|
| 3 | 0.008242 | 0.000672 | 0.008291 |
| 6 | 0.010201 | 0.001106 | 0.010395 |
| 9 | 0.018655 | 0.001725 | 0.018815 |
| 12 | 0.022611 | 0.001802 | 0.022849 |

全模型alpha0精确恒等共 44 次：真实接口预检4层，10个开发起点各4层。所有动作finite，六项CPU测试通过；4路源/评测worker均校验参数版本未改变、梯度关闭。

## 4. SmolVLA延迟/显存表

单GPU空闲后测；预热2对，5轮交错、CUDA前后同步。包含原版CPU预处理、完整replan及动作CPU回传；V1包含激活统计开销。不包含环境step和加载模型。

| 方法 | replan中位ms | 均值ms | Frozen倍数 | 峰值allocated MB |
|---|---:|---:|---:|---:|
| frozen | 608.14 | 605.95 | 1.000 | 1206.9 |
| v1 | 607.24 | 607.39 | 0.999 | 1206.9 |

5个样本用于工程量级估计，不包装为精确性能分布；原始逐轮数据见JSON。

## 5. 推进边界

NO-GO: Clean guard harmed a previously successful case; do not launch main table
SALT原文Appendix B.6 Table13确实从首个replan施扰，但其LIBERO10使用H50/E8（Table17）。本轮Spatial/E1和Clean护栏不构成同协议对比；后续主表需先锁任务套件、执行频率、双视角扰动定义、严重度、同时Frozen、多种子及source-data预算。
SALT来源：https://arxiv.org/html/2610.07946v1 。不把离线Clean漂移或启动成功率当成TTA闭环收益。

## 补充复核（不并入主批次计数）

另两项独立oracle测试通过：直接执行冻结CoRP V1源码中的align/fit/score；串行BF16视角输出与原FP32公式再转BF16逐位相同，全轨迹LOTO的mu/ss/tau也逐位相同。加上原六项共八项测试。没有修改本轮算法或调参。

仅对首个误伤task0额外回放一次，动作/初始状态/RNG/步数/成败全部精确复现：True。
该回放属于诊断复核，不是新的独立样本，不替换原结果。

证据包 SHA-256：`a76a718c56e2ac8c9925ed27b7d40df4d5ff2052b524c60fa18369ad9c8c3573`；代码实现提交 `850dcac`。模型权重和原始大文件不入Git。

扫层中的XYZ/Rotation距离是控制器动作分量差，不是末端位移或真实物体几何误差。本轮未做Shift修复和物理移物体响应测试。
