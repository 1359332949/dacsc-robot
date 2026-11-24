# 第 3 层全轨迹对齐：方法冻结、官方留出与跨套件确认协议 V1

**决定：结束 Spatial Dev 选参，跳过 EMA。** 下一步验证既有方法，不再根据桌面、灯光、种子或新套件分数修改它。方法冻结与评测名单获准是两个独立事件：前者已完成，后者不能由前者推导。当前 `history_audit.json` 为 `HISTORY_INCOMPLETE_NO_SCORING`，本次没有生成可执行目标留出名单，也不改变 V2 Blind 锁。

## 冻结对象与输入权限

`protocols/block3_confirmation_v1/method_lock.json` 绑定 `8576589` 里的实际执行路径、CoRP checkpoint、LIBERO-Plus 版本、wrapper、哈希化的源参考。单个 DINOv2 residual block 3 输出，只处理 patch，保留 CLS；两视角同公式、同 α=0.75、各用自身通道统计；95% 死区、有界缩放 [0.5,2]，逐 chunk 前向，无 EMA、无跨局记忆、无梯度、无权重更新、无域标签。FP32/EGL，H=E=12，10 步 settling，220 步上限，语言来自官方任务解析。冻结的是一个开发后选定的候选，不将旧 Dev 收益视为独立证据。

- Spatial：原封复用 `spatial_loto.npz`。目标任务 T 只能取其他 9 个 Spatial Clean 任务的折；资料来自 101 个实测推理块，外 9 内 8+1。哈希与已完成 Dev 的全轨迹参考完全相同。
- Object 零目标套件校准：使用单一 `transfer_spatial10.npz`，不按物体名挑 Spatial 折。由同一份封存的 10 个 Spatial 任务 101 块一次生成：均值、二阶矩按源块等权，轮流留出一个 Spatial 任务、其余 9 个拟合，汇集留出帧得 95% 分位阈值。没有用 Object 图像、标签、轨迹或 smoke 结果拟合；“零样本”仅指适配阶段，不宣称 CoRP 训练时从未见过 Object。
- Object 允许校准：**另列次要组**。从 Object 各任务各 1 条独立 Clean Frozen 轨迹（预定 index 2，seed 0，无成功过滤）建立与 Spatial 相同的外 9 / 内 8+1 折。对目标任务 T 不得含 T 自身 Clean。校准轨迹及其物理状态不得用于最终 Clean 或 Shift 计分。与零目标校准组对同一名单配对；不得二选一后只报较好者。

## 资格审计：只有 Frozen 跑过 ≠ 只被用过做 Frozen

论文名称采用 **project-defined method-held-out subset of official LIBERO-Plus assets**。它不是官方发布的划分，不是零接触 Blind。历史全 10,030 条 Frozen 均已访问。准入要求：历史接触仅限未用于逐例开发的 Frozen 批处理；不仅排除运行过候选方法的记录，也排除看过单例失败图、排名、轨迹、oracle/adapter/monitor 诊断、按成败选样或参与特征/阈值选择的实例及等价配置。研究者看过全池汇总须披露。未知不能按阴性处理。

实际初审入口 `scripts/audit_block3_method_holdout.py`。哈希核对服务器 P0 registry、当前 Dev，合并旧账本隔离和 90 个 Shift Dev；对项目历史中的压缩会话作字面身份交叉核对，并补查之前因体积跳过的 `b3_boundary.json` 的本地副本。日志提到名称可能仅因枚举，不声称每次均有 TTA。重复日志或源候选别名按 suite、逻辑基础任务、实际 BDDL 内容和 runtime 配置合并，跨 seed / init index 不重新获得资格。本次核对 80 份项目工作区会话档案及一个历史大文件；这些档案可能有继承与重复，不是 80 次独立实验。901 个等价组中 156 个明确排除或保守隔离（Lighting 65/292、Background 53/258、Sensor 38/351）；余下 227/205/313，共 745 个均为待证，不是可抽池。旧日志含 `/hy-tmp/validation/`、`topk10_l3_extra3` 与 `debias_cam_obj_big` 等批次线索；本次在新机 `/hy-tmp` 未找到这些对应原目录。确切计数、输入哈希、引用范围见 `history_audit.json`。确认身份不需要也不允许先渲染候选帧。

**补证条件（不看成功标签）：** 恢复历史非 Frozen 批次的任务入口、过滤器、已展开 ID/逻辑名、分类表版本、轨迹/人工诊断取样记录，包括 top-k/L3 与旧套件作业。将 1 基 classification ID 与 0 基 evaluator index 在固定版本上双向映射；把 job 通配范围、按等级/索引筛选和间接等价组纳入，无法缩小则隔离整作用域。列明覆盖的历史机器、run roots 与未覆盖来源；此前已按单例 Frozen 结果开发的也须排除。只在可证明完整的作用域生成带证据的 allowlist。不要简单排除被字符串命中的名字就抽取剩余；也不要用换状态、换种子、双眼重放来洗白历史。官方池条件无法补齐时，这一路保持未计分；另行预注册新的合成/外部候选，不混名、不替代或改写本次主结果。

## 冻结计分设计（先有资格，再抽样）

1. Spatial 标准：Lighting、Background、Sensor 各 100 个**合格等价组**，按 suite × factor 自然池等概率无放回，不看 Frozen 成败。固定 SHA-256 排序键 `block3-confirm-v1|suite|factor|equivalence_group`，取最小的 100 个，碰撞用逻辑名打破；不足 100 则该批不启动，不临时改配额。双眼压力集使用同一 100 个 Sensor 身份、相同 code/severity，两眼独立噪声；含雾等非模糊子型时注明。原始标准条件和人工双眼条件分开报，不描述为 400 个独立身份。Spatial Shift 取原始 index 0；与 Dev/源的相同基础物理起点是已知且允许的，因为验证的是新扰动实例，而非新初始状态。不可复用被隔离的扰动配置。Object 的 Shift 禁用本轮 smoke 与新域校准的状态。无法物理配对的记录在冻结前排除，不翻看结果重抽。
2. Clean 保持性单列：每任务 10 个独立、经历史审计的状态，按上述哈希原则以 `suite|task|physical_state_hash` 排序取样，排除 source fit、smoke、校准、Dev、逐例诊断及评测 Shift 初始化的相同物理状态。若无法获准，独立 Clean 结论保持缺失；不得用 Dev 30 局代替。名单冻结前同时核对源参考、渲染 wrapper、归一化、执行器/汇总代码指纹。缺名单或校验失败不得进入目标运行。
3. 策略种子 0/1/2；环境、传感器、策略采样及方法 RNG 分流；同一实例三组策略种子仅改变 policy 流。每臂真实闭环同期 Frozen 对照。沿用已核验执行器的首帧双眼、初始状态及共同块 z 哈希核验；同一物理状态同一步只接受原协议限定的舍入差异。冻结名单、计划总组数、统计实现 SHA 后用**独立正式执行入口**追加访问账本。不得把测试角色伪装成 `target_dev`，不得放宽当前 `authorize()`。不因分数决定继续、删案或重跑。基础设施失败保留收据，按同一 ID/seed 补齐，重复只计一次。
4. 主指标：Spatial **标准三因素等权**的 Method−Frozen 成功率差；列原始成功数、恢复/损害及逐 seed 净数。100/因素、3 seeds 下等价于三因素合并配对均值，不含 Clean。双眼相同身份为预定关键压力指标；单眼、纹理两类、照明另列。固定 30,000 次按基础任务整簇有放回 bootstrap（seed 20261124），一个任务的所有身份/种子/臂一起抽；每次重算因素等权均值，空因素的抽样跳过并报告数目。同报逐任务计数；只有 10 簇，区间不支持跨机器人普适定律。
5. 提前确定确认判据：标准主指标双侧 95% 配对区间下界 >0；独立 Clean 单侧 95% 配对下界 >−3 pp（预设工程容忍界，不作功效保证，另报任何误伤及不一致数）；两项均满足才作“标准集改善且 Clean 非劣”的主张。双眼单独报告其双侧区间与净数，不用它挽救主指标，若欲称双眼确认性改善须在主关口通过后其下界亦 >0。不得把 0 次观察误伤推广为零风险。如果所有配对差均为零导致 bootstrap 退化为 [0,0]，只报告本样本逐例一致，不单凭退化区间出具总体非劣效证明。未达到界限报未证实，不以其他套件或某个种子替代。

## Object：兼容性检查与迁移结果分表

优先 `libero_object`。既有 Object heldout 已退役，因此它是相对 Spatial 方法开发的跨套件，不自动成为历史未见测试数据。`libero_goal` 历史 0/20 不能直接归为适配失败；不因 Object 成绩换套件。Object 只运行 `scripts/block3_object_clean_compat.py` 里的十个规范 Clean 篮筐任务，indices 8/9 各一次，Frozen；完全独立于方法与源参考拟合。不使用扰动或候选方法，不读最终测试名单。兼容性关口在启动前固定：总成功至少 14/20 且**每个**任务至少 1/2，无接口错误。失败则整套件停在兼容性诊断，不挑能成功的任务。通过也不等于训练覆盖、跨套件有效或留出资格通过。smoke 的 raw/settled 状态哈希永久排除计分。

Object 正式目标名单同样先过方法接触审计，沿用 Spatial 各因素 100、双眼 100 个复用身份、独立 Clean、相同 seed/采样/统计与运行约束；标准和双眼均采用同期 Frozen、仅 Spatial 参考、允许 Object Clean 校准三臂。另报新域校准对零目标校准的配对差。仅 Spatial 参考为主要迁移组；校准组和跨套件总体结论不冒充 Spatial 主确认。方法冻结后如果补录历史改变了资格，只能在任何计分前按书面规则和新哈希重建名单；若已经计分则披露受影响运行，不能删除后继续称为同一次独立确认。

**本次执行记录：** `libero_object` Frozen 在 20 个预定 Clean 起点取得 20/20，各任务 2/2，通过兼容性关口。20 个起点的资产、动作及物理状态哈希见 `protocols/block3_confirmation_v1/object_clean_smoke.json`。本轮无任何 Object Shift 或正式 Clean 计分，无 EMA。
