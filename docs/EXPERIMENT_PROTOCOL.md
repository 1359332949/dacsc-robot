# 当前实验协议：CoRP-RoboCasa / appearance shift

2026-11-02更新（始于2026-10-30）。本文件是唯一当前执行协议，替代旧 notes/71、72；与其他历史 note
（包括 notes/75 的旧“权威索引”）冲突时，以本文件的证据范围和执行顺序为准。

## 研究问题和证据边界

研究定位：**外观变化怎样通过视觉FFN通道影响机器人动作，以及能否有选择地削弱这种影响，同时保留任务所需的视觉信息。**
channel是研究对象，动作鲁棒性是目的；gate只是后续可能采用的工具，不预设一定需要mask或token条件化网络。
工作题目可写成“外观变化如何经由视觉FFN通道影响机器人动作：机制分析与选择性干预”；这是待验证目标，不是已成立的贡献。
允许未观察到动作影响、影响被后续计算消化、或主要故障位于融合/动作路径等结果；不预设视觉FFN是唯一根因。

任务目标：**紧凑机器人策略在未见外观变化下保持任务成功率，同时保持正常条件性能。**
策略需要对物体、把手和目标位置等动作相关变化保持敏感，减少对无关外观变化的依赖。
不能预设颜色、材质或域信息全部无关，也不能预设视觉信息压制越强越好；颜色可能参与目标消歧，
严重 clipping 或模糊可能直接删除完成任务的可见线索。A 路线保持为 appearance shift，
blur/noise 是辅助动机，不据此改称已经验证的普遍 corruption/DG 方法。

基线固定为 CoRP-RoboCasa。先建立有效 clean 闭环能力，再在同一模型、任务、初始状态和
策略噪声下测 appearance shift，最后评估标准像素增广与通道/token-channel干预的各自增量。
**架构基线已实现；当前协议下稳定完成任务的闭环实验基线尚未建立，候选方法没有已确认的收益。**
**方向澄清：用户暂不选择AdaIN式统计混合作为主方法，主研究候选回到通道/token-channel选择。
具体粒度、插入点和训练/推理口径尚未定案，不据此移植旧gate或启动训练。**
当前只实现/本地验证只读FFN采集，服务器验证暂缓；后续真实环境验证顺序仍为视频读取→CUDA/BF16→MuJoCo重放。

### 外观变化的范围

主问题只讨论任务目标、语言指令、几何/机器人状态与正确动作保持不变的外观变化。
同一状态的配对图像可以用于诊断，但另一个style的另一条episode不能自动当作同状态反事实。
每个条件需记录配对依据、目标/把手等关键线索的可见性、clipping/过曝/饱和情况及样图。
即使名为ColorJitter，也可能删除动作所需线索；变换名称不保证标签不变或信息仍足够。
严重模糊、噪声或过曝等信息退化条件分列报告，不与保持任务线索的外观变化合并作shortcut解释。
诊断可用原动作标签核对一致性，不能因此允许最终方法在推理时读取GT动作。

### 三个研究问题

| 问题 | 需要的证据 | 当前状态与限制 |
| --- | --- | --- |
| Q1：外观变化在哪里被编码？ | 同状态正常/外观配对，按层、视角、CLS/patch位置记录FFN响应；保留episode和帧身份 | 只读观察器及本地非侵入性测试已具备，真实样本尚未采集；激活变化不等于有害依赖 |
| Q2：哪些变化真正传到动作？ | 有效同模型策略下，少量指定单元的clean→shifted替换、反向替换与同规模随机对照；观察resampler前后及动作/任务结果 | 尚未实现或执行这组干预；梯度、attention、可解码性和单纯动作差异都不能替代干预证据 |
| Q3：能否选择性调整并保留任务信息？ | 同候选集合下比较共享通道、限定token位置的通道及各自匹配随机对照；同时检查clean代价与扰动收益 | 尚无选定粒度或有效方法；不能先认定二维mask优于共享通道，也不以旧F2组合充当新贡献 |

resampler是Q2的重要观察边界：需要检验哪些视觉变化被压缩/融合过滤，哪些仍影响下游动作，
而不是仅按视觉响应大小排序。256个patch与48个learned queries没有预设一对一空间对应，
不能把query索引直接解释为像素区域；当前采集入口只观察视觉FFN/raw block，尚未实现resampler或动作路径的新采集/替换。
这些都是研究目标，不宣称该接口从未被研究或当前已完成故障定位。

### 证据等级与干预边界

原有三项前提检查继续保留，与Q1–Q3分开，不能互相替代：

| 问题 | 当前证据 | 仍不能推出 |
| --- | --- | --- |
| style 是否可解码？ | 纯视觉、episode-level 的原生标签 probe 有信号 | 策略依赖了有害 shortcut；域敏感维度应该被删除 |
| 外观变化是否伤害策略？ | 旧官方 CoRP/LIBERO-Plus 在部分强扰动下退化 | 当前 CoRP-RoboCasa 闭环必然退化；所有自然外观变化都困难 |
| 候选方法是否缓解退化？ | 当前只有实现与局部工程验证 | token statistics 或 action-aware 有效/无效 |

后续“机制发现＋最小方法”须逐级取证，不在本轮执行：

| 阶段 | 验收依据 | 不能冒充的结论 |
| --- | --- | --- |
| 有效基线与前提 | 同一个CoRP-RoboCasa在clean上有可重复闭环能力，选定外观条件下确有配对退化 | 官方LIBERO模型的退化或弱模型的动作差异不能替代 |
| 指定条件下的作用证据 | 少量单元替换与同规模随机对照有可重复差异，并有反向/clean对照；动作偏移、恢复和任务结果分别报告 | 变化更大不等于恢复更好，也不等于普遍“动作因果通道”或删除这些单元必然有益 |
| 粒度依据 | 共享通道与位置条件通道的clean/shift权衡有可重复差异 | 把mask从一维改成二维本身不构成贡献 |
| 可部署方法 | 比较无干预、各粒度随机干预、无保护版本与标准Torchvision像素增广；报告clean代价及扰动成功率 | 只优于无增广或只改善MSE不够；不能把使用特权信息的诊断当部署能力 |

激活替换必须匹配层、token/相机身份与调用时刻，固定state、指令、预处理及flow采样噪声；
只改声明的单元，检查替换是否形成异常表示。整份clean视觉输出恢复后续动作，首先是数据流恢复的正对照，
不能据此唯一定位FFN故障；“接近clean动作”也不代表clean动作正确，最终仍需动作标签/任务结果的独立依据。
共享通道与位置限定处理要报告候选集合、实际改动的token×channel单元数、强度和层位；
随机对照在各自粒度内匹配预算，无法同时匹配的因素单列说明，不能仅用相同drop比例宣称公平。

最终方法须明确是训练期正则还是推理期选择，以及部署时实际可用输入。
clean参考图像与GT动作可用于预先定义的source/val诊断，不能成为最终推理输入；依赖这些参考的实验单列为特权信息诊断。
使用更多参考信息不保证结果优于所有方法；没有另行证明时，不称为数学上的性能上界。
test域标签不得用于排名、候选筛选或方法选择，test继续封存至最终评测；不能借“机制诊断”提前接触它。
外观敏感性与动作影响的关系、resampler过滤作用、位置条件干预的必要性均为待验证贡献，
“FFN通道＋机器人＋mask”或“域分数与梯度加权”本身不作为新颖性/有效性证据。

当前不能写成结论的内容：

- 旧 CoRP-RoboCasa 离线 MSE 无明显 source/val gap，不能推出闭环成功率无 gap。
- CloseFridge 的 0/10 只能描述该 checkpoint 在该协议下失败，不能自动归因为训练不足。
- 旧官方 CoRP/LIBERO-Plus 的 14 个任务 paired 结果是动机证据，不能替代 RoboCasa 模型的验证。
- CoRP 论文 RoboTwin Randomized 训练包含随机化示范，不能用两组 success 差证明零样本外观泛化。
- legacy DP 的 u/v 分析、DDPM gate 和 custom flow 结果不是 CoRP token 方法结果；梯度分数不等于因果证明。

## 基线、对照与候选方法

基线名称固定为 **CoRP adapted to RoboCasa / our training adaptation**，不称为官方 RoboCasa 训练复现。
模型和初始化的来源见后文“上游保真与必要适配”；不再以早期 CLIP + 12-layer DiT 自定义 flow 为主基线。

| 部件 | 当前实现与接口 |
| --- | --- |
| 视觉观测 | 当前单帧 × 左 agentview / wrist / 右 agentview；224 RGB，无 crop |
| 视觉编码器 | DINOv2 ViT-S/14，256 patch tokens/view，384维；训练适配中参与微调 |
| 语言与压缩 | T5-Efficient-Tiny 参与微调；语言条件 FiLM resampler，48 tokens/view（三视角共144个视觉token） |
| 策略 state | 原始16D字段中选末端相对位置3D、相对旋转四元数4D、gripper qpos 2D，共9D |
| 动作生成 | 1-layer DiT，hidden768 / 12 heads；flow matching，10 Euler steps |
| 动作接口 | 未来10步 ×12D；动作键拼接与官方 DP reader 一致，闭环实际执行步数单独记录 |

单帧观测、较小 batch 或不同于原论文的训练预算是需披露的设计/预算差异，不自动构成实现错误。
它们也不允许省略公平对照或把短预算结果称为官方训练复现。

下面是研究候选/实现盘点，**不是本轮获准启动的训练矩阵**。
先测单项贡献；标准像素增广仍是必要对照，已有统计混合模块降为旁支，暂不扩展donor或风格归一化。

| 组 | 开关 | 作用与要回答的问题 |
| --- | --- | --- |
| 无干预 CoRP | `train-augmentation=none`、`token-style-mode=none` | 基础学习与 clean 闭环能力 |
| 标准像素增广 | `train-augmentation=color_jitter`、`token-style-mode=none` | 通用输入增广能解决多少外观退化？ |
| 随机通道干预 | 计划对照，尚无当前CoRP入口 | 在相同粒度/位置/预算下区分普通正则与选择机制 |
| 通道/token-channel选择 | 研究候选，尚未接入当前CoRP | 区分embedding坐标、FFN神经元和按token条件化的通道选择；先做上游审计 |
| 动作感知通道保护 | 研究候选，未实现当前CoRP版本 | 外观响应与实际动作影响是否相关，怎样指导选择，是否优于随机和无保护？ |
| 随机统计混合（旁支） | 已有`token-style-mode=random_stats` | 现有实现留作历史/备选，不是用户当前选择的主方法 |
| 跨 style 统计混合（旁支） | 已有`token-style-mode=cross_style_stats` | 同上，暂不继续扩展或训练 |
| 统计混合的动作梯度保护（旁支） | 已有`token-style-mode=cross_style_stats_action_aware` | 仅保护混合强度，不能冒称已实现action-aware channel gate |

像素增广直接使用 Torchvision；特征模块是受 TFS-ViT 启发的项目适配，不能冒用完整上游方法名。
复杂候选需要与标准像素增广及相同干预粒度的随机/无保护对照比较，不能只优于无增广就宣称域选择或动作保护有效。
channel hard gate、DomainDrop/PLACE/DgCD 的旧迁移实验保留为限定范围的历史分析，不是当前 CoRP 主方法证据。

通道方向的文献依据与边界：旧方案的“域敏感性+动作重要性+软门控”是待验证组合，不等于已知算法的忠实复现。
区分`Z[B,V,T,C]`上的token轴`T`、embedding-channel轴`C`与FFN中间神经元，不能把它们混称为一种通道。
[Test-time Registers](https://github.com/nickjiang2378/test-time-registers)与
[NeuronPath](https://github.com/foundation-model-research/NeuronPath)已有作者代码，可优先用于机制审计；
它们分别研究outlier机制/分类影响路径，不证明当前ViT-S/14中的风格信息可以安全删除。
[MCM](https://www.mdpi.com/1099-4300/27/8/794)是通道重建预训练，
[DisentangleFormer](https://arxiv.org/abs/2512.04314)是空间/通道并行架构，均不是现成的机器人DG gate。
另可核对已有作者代码的[DaViT](https://github.com/dingmyu/davit)：其spatial/channel-token attention
提供两条轴分开建模的实现参考，但它是骨干架构，不是DINOv2上可直接挂入的DG gate；不替换CoRP骨干后仍称同一基线。
DBCT/SCS与CLEAR仅有部分文献证据，未核验完整方法与可用官方实现；不按摘要自行补全后挂官方名字。
旧综述中的“只有两篇”“机器人领域空白”“通道有语义因此门控前提成立”不作为当前结论。

各项已核对代码的固定revision、实际符号与适配边界见[上游映射表](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md)。
当前技术建议是先审计1536维FFN神经元，以384维embedding通道为低成本对照；token-channel是位置相关作用方式，
不是与两种神经元对象互斥的第三种骨干。此为适配判断，不是效果结论。
`CoRPFFNCapture`现已用于本地CPU只读验证，直接调用固定Registers源文件的logger；
原detector通过单视角校准适配调用，不改排名计算。本轮不加门控、不连接服务器、不启动训练。
CLS/patch/三视角排列、hook生命周期及真实权重下输出一致性的范围见[采集说明](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md#只读采集实现与本地验证)。
三篇主读的[五行卡片与具身代码映射](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md#三篇主读论文五行卡片)现覆盖Registers、FFN steering和Action Atlas。
FFN steering的VLM常数覆写、Action Atlas的完整残差/MLP/SAE干预与当前DINO视觉GELU日志不是同一操作；
不把它们的层号、词表语义、系数恒等条件或pooling约定直接搬入CoRP。
这些研究已涉及具身内部特征干预，不能再以“机器人领域无人研究通道/神经元”宣称新意。
新增[ToaSt映射](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md#toast通道选择参考与源码适配边界)作为FFN选择/压缩参考，
与Registers的异常响应定位、NeuronPath的输出影响分析、具身干预工作分工阅读。
已读公开实现的跨batch聚合、CLS依据、attention改写、fc1残差与fc2中间通道选择均需单独适配；
公开仓库与作者/论文的关联尚未闭环核实，不能先标为已验证的官方复现，也不能从分类压缩收益推导机器人鲁棒性。

## 已有证据、作废结果与未验证问题

以下记录只在其原始模型、数据、代码和评测协议内有效；历史图像处理、loss、EMA、donor 或
结果 schema 与当前不一致时，不混入新比较，也不通过转换旧文件补齐当前溯源。

| 证据 | 可以保留的判断 | 边界/来源 |
| --- | --- | --- |
| 原生 style 的纯 DINOv2 probe | agentview 中存在可解码信息；最佳50类 episode balanced accuracy约.52 | [note52](../notes/52-pure-dino-style-full-audit.md)；固定layout后support不足，不能证明干净的域分离；不再引用coarse3/layout10 |
| legacy 通道分析、DDPM gate、custom flow 对照 | 特定旧模型、预算和干预下没有按预期获益，部分变差 | [note43](../notes/43-top-u遮挡因果结果.md)、[note47](../notes/47-DDPM-200步3seed结果.md)、notes59/60；不是CoRP结论，不证明所有通道方法不可能 |
| 旧官方 CoRP/LIBERO-Plus 强扰动 | 在已记录条件下出现明显退化，是动机证据 | [note67](../notes/67-Corp强模糊退化.md)、[note77](../notes/77-配对appearance-shift评测.md)；LIBERO-Plus原生条件不等于标准LIBERO clean；14任务配对color stress仅为小样本，不能替代RoboCasa验证 |
| hook未注册时的token-style run | 不能用于判断token-style效果，作废 | notes66/70已删；修复前实际没有执行干预，修代码不能追溯使旧结果有效 |
| 修复后的短token-style smoke | 支持代码路径会执行干预 | [note76](../notes/76-token_style-51步smoke.md)；旧短预算/少量任务不构成正向或普遍负向方法结论 |
| 旧CoRP-RoboCasa MSE与GT replay | 当时版本的数值/管线记录 | [note64](../notes/64-CorpRoboCasa-per-dim-per-task与success-rate阻塞.md)、[note68](../notes/68-RoboCasa-SR管线验证.md)；不替代当前协议下真实数据与闭环复核 |
| 此前像素增广工程验证 | `7efc94a`快照86项回归通过；官方CoRP CPU demo通过 | [像素增广验证记录](../audit/pixel_augmentation_local_validation.json)；只覆盖记录声明的本地范围，不是CUDA、真实视频或MuJoCo通过证明 |
| 此前只读FFN采集验证 | 当时本地96项回归通过；真实发布权重适配器在受控输入下挂hook前后表示/动作逐元素一致 | [采集验证记录](../audit/corp_ffn_capture_local_validation.json)；输入不是新解码视频，不证明outlier/style/动作机制或方法收益 |
| 只读验收入口与交付包准备 | `d51a043`版本108项本地回归通过，入口/包与出处可追溯 | [准备记录](../audit/readonly_validation_local_preparation.json)；fixture不是硬件/真实视频证明，视觉验收入口不执行完整策略/动作检查 |

旧的过强表述统一修正：

- “style可分，所以域门控应有效”改为“可解码性不证明有害依赖，也不决定干预对象”。
- “CoRP对外观没有退化”改为“部分历史条件下没有观察到退化，其他强扰动出现退化；当前RoboCasa闭环待测”。
- “u≠v判定channel gate失败”改为“legacy观察削弱了该具体实现的动机，梯度与遮挡分析不能证明普遍因果分离”。
- “改成token就适合ViT、能够保留动作信息”改为“当前统计混合是待验证假设，统计量也可能含任务/布局信息”。
- “CloseFridge 0/10必然因为模型弱”改为“只证明该checkpoint在当时协议下失败；预算、管线、控制与分布因素仍需排查”。
- “满200/2k步、gap超过5个百分点或CI跨零就自动决定去留”改为“结合学习能力、support、配对结果和不确定性判断，不以单阈值代替证据”。

当前未验证问题按依赖排序：

1. **真实执行链**：当前版本真实视频解码、CUDA/BF16及MuJoCo状态/动作重放是否按官方接口正确工作？
2. **有效基线**：当前协议下无干预CoRP是否能在冻结source/val场景稳定完成任务？旧0/10不能替代诊断。
3. **同模型退化**：同一个RoboCasa策略在保留任务线索的外观条件下是否出现可重复闭环退化？
4. **机制与粒度**：Q1响应变化是否在Q2指定干预下影响动作，resampler传递了什么，Q3共享通道是否过粗？统计混合/donor问题仅属旁支，不作为当前FFN主线的已定评分规则。
5. **方法增量**：选定粒度的通道干预是否优于匹配随机干预，动作保护是否优于无保护，复杂方法是否优于标准像素增广且不损伤clean能力？

前三项没有完成前，不把已存在的候选模块等同于已成立的方法；当前只推进本地只读FFN采集和回归，不启动训练。

## 数据和选择协议

使用 RoboCasa365 human pretrain/atomic；只接受原生 style/layout/task/episode。

| split | 条件 | 用途 |
| --- | --- | --- |
| train | style 11–30，episode index `%10 != 9` | 参数学习、normalizer 拟合、训练 donor |
| source holdout | style 11–30，episode index `%10 == 9` | 源域离线诊断 |
| val | style 31–40 | 预算、checkpoint、方法选择 |
| test | style 41–50 | 冻结选择后显式最终评估 |
| reserve | style 51–60 | 预留，不参与当前选择 |

episode 标识为 `(dataset/task, episode_index)`；不同任务的相同数字不代表同一 episode。
这是一套**自定义 held-out-style validation 协议**，不是 RoboCasa 官方 pretrain/target split，
也不能称为只用 source 内验证的 DomainBed 标准协议。

数据读取直接使用官方 `LerobotDataset`。extras 必须与 `episodes.jsonl` 对齐；缺失、
重复、非法字段或损坏 JSON 报错，禁止跳过后继续。每个样本是当前单帧 ×3 相机、9D state、
原始语言和未来10步12D动作。mask 从真实 episode 长度计算，末尾重复最后有效动作；
补齐位置不参与 loss 或 MSE 的分子/分母，遵循 CoRP Eq.5 / Appendix A.3。

离线评测按 `(task, style)` 分层，以固定 seed=123 选帧，所有方法/训练 seed 共用。
总帧数不等于独立 episode 数。输出独立 episode、每 task/style 帧数和 episode 数、
完整 metadata digest 和选中样本 digest；source holdout 个别任务只有1个独立 episode，必须如实报告。
不能把不存在的 style support 补成有样本，也不因某个 split 较难而用 test 调整划分。

## 上游保真与必要适配

复用顺序固定为：已确认来源和适用条件的上游原函数/官方入口 → PyTorch、Torchvision、SciPy等官方基础库 → 必要的模型/数据接口与协议校验。
保留上游算子与评分定义，不为适配工具而更换CoRP的attention/骨干。不因现成代码存在就略过权重、张量轴、输入权限和随机性检查。
没有可直接复用的实现时先记录缺口；如未来确需研究适配，另行明确差异与对照，不能按摘要猜测补全后称为官方复现。
本轮不新增选择器、评分、指标或兼容分支。

| 环节 | 来源 / 接口 | 边界 |
| --- | --- | --- |
| 模型/加载 | [官方 CoRP](https://github.com/corp-policy/CoRP/tree/52445740aa9517ae396cc8fb14bae8fe5922f118) `RepresentationExtractor`、`DiTActionGenerator`、`load_policy` | 保留 DINOv2/T5/K48/1-layer DiT；仅适配3 views、9D state、12D action、H10；12D动作接口重新训练 |
| 数据 | [官方 RoboCasa DP reader](https://github.com/robocasa-benchmark/diffusion_policy/blob/41212698b6f481ed92a55f0d5f1778ec56bea417/diffusion_policy/dataset/lerobot_dataset.py) | 不重写视频读取、动作键拼接和时间索引；在外层过滤原生标签/补 mask |
| 重放 | [官方 reset_to](https://github.com/robocasa/robocasa/blob/dc73a8ce18e4b8947e8df2ae2958245ca823f3b4/robocasa/scripts/dataset_scripts/playback_dataset.py) | 不调用 label_subtasks 中的修改副本，不修补 XML、不偷偷 retry |
| 布局/时限 | 官方 `LayoutType.TRAIN`、`unpack_layout_ids`、`get_task_horizon` | layout 11–60；时限按任务 registry，删除固定600步和旧 override |
| 图像 | Pillow bilinear + [Torchvision functional](https://docs.pytorch.org/vision/main/transforms.html) | 三条路径使用同一图像契约；保留 LIBERO 官方观测旋转，不给 RoboCasa添加该旋转 |
| EMA | [PyTorch AveragedModel / get_ema_multi_avg_fn](https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema-for-training) | 不重写 EMA 算术；适配参数名/共享权重和非持久 donor buffer |
| task 辅助损失 | CoRP论文Eq.7；PyTorch `LayerNorm`、`Linear/GELU`、`cross_entropy`；官方`DiTBlock.encode_prefix` | 只训练视觉前缀；任务头不进入推理；task label不进入策略输入 |
| 互斥 | [filelock](https://py-filelock.readthedocs.io/en/latest/how-to.html) | 输出所有权和日志创建位于同一锁内 |

共享权重从**已训练的 LIBERO CoRP policy checkpoint**迁移，不只是 DINOv2 视觉预训练。
先用官方 loader 完整加载源模型；DINOv2/T5/resampler/FiLM/DiT 共享层复用官方权重。
RoboCasa 与 LIBERO 的 action normalization 不同（identity vs motion min-max），
12D action projection/velocity projection必须随机初始化并训练；不能复制前7维或把新增维置零。
state projection 因 axis-angle→quaternion 语义不一致也保持随机；新 camera query/embedding
按声明规则初始化。
其他 missing/unexpected/shape mismatch 报错。报告中的单位是 state-dict tensor，需区分共享参数别名。

必须使用名称 **CoRP adapted to RoboCasa / our training adaptation**：已锁定的官方发布版本只提供推理，
我们的训练循环并非官方训练代码。task-classification auxiliary loss 已按论文Eq.7补齐：
对指定DiT前缀深度逐token做LayerNorm，在`VK`个视觉token上取均值，经跨深度共享的小MLP输出任务logits，
再用PyTorch `cross_entropy`；默认`lambda=.01`、`J={1}`，训练目标为`L_flow + lambda * L_task`。
官方仓库没有发布该训练头，因此这是公式复现而非官方trainer代码；论文未公布MLP隐藏宽度，
当前固定为256并写入checkpoint/result/SR训练身份。task head只在训练期使用，推理不读取task label或logits。
RoboCasa task class按排序后的61个atomic task名称映射，映射随checkpoint保存；不能把style/layout当作task标签。
RoboCasa state 使用 train-only z-score，action identity 沿用官方 DP reader；它不同于 LIBERO 的动作 min/max。

`upstream.lock.json` 固定三个源仓库 revision、关键执行文件 SHA256 和官方 checkpoint checksum。
校验失败直接退出；它不覆盖未列出的整个第三方仓库或仿真依赖，不能替代真实环境 smoke。

## 图像契约

统一解码 uint8 RGB → Pillow bilinear resize 224（无 crop）→ 外观条件 → ImageNet normalize。
DP reader 的 float `/255` 先还原 uint8，再走同一 resize，离线/仿真不分叉。
LIBERO 直接复用官方 observation conversion，外观 adapter 作用在其 resized、unnormalized RGB。

| 条件 | 直接调用 Torchvision |
| --- | --- |
| none | 恒等 |
| grayscale | `rgb_to_grayscale(..., num_output_channels=3)` |
| low_contrast | `adjust_contrast(..., .3)`，使用上游亮度均值定义 |
| color_jitter | 顺序 `adjust_brightness(1.6)` → `adjust_contrast(1.6)` → `adjust_saturation(2.2)` |
| hue_shift | `adjust_hue(.5)`，HSV 定义；删除 RGB 旋转矩阵 |
| blur（辅助） | `gaussian_blur`，sigma显式指定，kernel=`2*ceil(3*sigma)+1`，作用于224像素图像 |

这里的 color_jitter 是**固定参数 photometric stress**，不是随机 ColorJitter，也不是官方 benchmark severity。
强度、顺序、位置、Torchvision 版本全部写入结果；严重 clipping 可能丢失任务线索，须检查样图和任务语义。
新旧 PIL/逐通道 contrast/hue matrix/pre-resize blur 结果不能合并。blur/noise 只作辅助，A问题保持外观变化。

### 标准像素训练增广基线

直接使用 [Torchvision `ColorJitter`](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)：
`ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=0)`。
各非零因子从 `[.6, 1.4]` 均匀抽样，运算顺序也由上游随机抽样；不重写因子采样或颜色运算。
`.4/.4/.4` 无 hue 的配方参考 [timm `transforms_imagenet_train`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms_factory.py)
的通用 `color_jitter=.4` 设置。它是独立对照，**不是 CoRP 官方训练配方，也不是整套 timm 增广的复现**。
不复制分类用 random crop、flip 或标签混合，保持动作/state/语言及时间索引不变。

| 入口 | 随机性和作用范围 |
| --- | --- |
| `--train-augmentation none` | 默认；训练图像不做像素增广 |
| `--train-augmentation color_jitter` | 仅训练 styles11–30、episode `%10 != 9`；随机 ColorJitter |
| `--eval-shift color_jitter` | 前表的固定 `1.6/1.6/2.2` stress，不是训练增广 |

训练顺序为 Pillow resize224 → RGB `[0,1]` → 随机 ColorJitter → ImageNet normalize。
启用时 dataset 只在 source-training 角色输出带 `image_space=rgb01` 的图像，实际 ColorJitter 在
训练主进程中调用；不在 DataLoader worker 内持有随机增广对象。其余路径输出
`image_space=imagenet_normalized`，`move_batch` 拒绝对已归一化图像二次增广，也拒绝将 raw RGB 直接送入评测。
normalizer 只读取训练 state/action，不调用随机图像增广。

每个观测单独调用一次 ColorJitter，三个视角作为其 leading dimension 共用同一抽样；不同观测独立抽样。
这是本项目明确的多视角适配规则，不宣称是 CoRP 官方设置。因子/顺序使用上游 Torch CPU RNG，
通过 PyTorch `torch.random.fork_rng(devices=[])` 和私有 CPU generator 隔离全局 flow RNG，
run seed 初始化、完整保存/恢复其状态。CUDA 像素运算和 BF16 管线仍需真实环境验证。

首轮只比较单项干预：像素增广启用时要求 `--token-style-mode none`，组合直接报错，不自动回退。
runner 可选 `TRAIN_AUGMENTATION=color_jitter MODES=none`，输出
`$OUT/$TRAIN_AUGMENTATION/$MODE/seed_$SEED`；默认 `TRAIN_AUGMENTATION=none`。
训练结果、checkpoint 及评测的 `training_origin.pixel_augmentation` 保存算子、强度、顺序抽样规则、
位置、Torch/Torchvision 版本和 seed；无增广也必须显式记录 null。评测只读取该溯源，不重建或执行训练增广。
方法配对允许像素/特征干预不同；同一策略的条件配对要求该字段完全相同。
不得把不同库版本或缺少该字段的旧结果混入当前比较。

## 旁支实现：统计混合的donor与动作保护契约

本节只描述已有旁支代码，不将其指定为当前主方法。训练期 statistics mixing：eval 返回原视觉特征，不需要 style、task 或动作标签计算 donor。
CoRP插入点是DINOv2输出的归一化patch tokens之后、resampler之前，通过hook委托官方forward，
不复制官方提取器。输入为`[B,V,T,D]`，其中`T=256`、`D=384`；不是在48个resampler query上执行当前混合。
对每个样本/视角沿`T`轴计算逐特征维mean/std，再使用对应donor统计量构造：

```text
z_style = (z - mean_T(z)) / std_T(z) * std_T(z_donor) + mean_T(z_donor)
z_out   = (1 - alpha_k) * z + alpha_k * z_style
```

std有下限保护，mean/std detach。这个实现是patch-token序列上的逐特征维统计混合，
不是token删除、逐token域分类或已经证明的内容/风格解耦；从channel改名为token不构成有效性证据。

- `random_stats`：`torch.randperm(B)`，允许 self-pair/同 style；沿用 TFS-ViT 样本置换思想。
- `cross_style_stats`：每个 recipient 在 batch 不同原生 style 的样本中均匀抽一个。
  直接用 `torch.multinomial`；recipient 之间可复用 donor。缺标签或只一个 style 立即报错，无 fallback。
- `cross_style_stats_action_aware`：donor 规则不变，动作重要性仅调整调制强度。
- 一对样本共用 donor，左对左、腕对腕、右对右，不跨相机；官方 view-major 输出先转换再逆转换。
- **不限制同 task/layout**，不能称为“同任务、只变外观”的受控训练干预。
- mean/std detach；独立 donor RNG 由 run seed 初始化并完整恢复，不消耗 flow RNG。
- action-aware 为先前训练梯度构造的 patch 位置 EMA/min-max 权重，跨样本/视角统计；不是对象追踪或因果重要性。

动作分数来自先前训练迭代的`mean_{B,V,D}(|z * dL_flow/dz|)`及EMA，重要位置降低`alpha_k`；
启用task aux时显式只对`L_flow`做该梯度，不复用总损失梯度，避免任务分类目标污染“动作保护”名实。
指定top位置可不混合。它是当前训练目标下的梯度启发式，不是独立clean反事实测量；
跨task/相机平均的位置重要性不保证对应同一物体，不能写成已验证的“动作因果保护”。

来源：[TFS-ViT FeatureStylization](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/blob/main/domainbed/pretrained_models/DeiT_models/models.py)。
当前固定 alpha、单个 pre-resampler 插入点、多视角及异 style 规则都属于适配，
不是完整 TFS-ViT 的 Beta mixing、多层/token 策略。仅凭该模块不能把完整 TFS-ViT 填入比较表。
`sample_pair_v2` donor 契约不匹配的 checkpoint 拒绝加载。
CoRP checkpoint只接受上述三种训练期mixing模式，拒绝已退役的normalize分支或未声明的模式。
SR命令中的style范围必须逐段升序、唯一且落在原生pretrain ID 11–60内，不能静默跳过反向范围。

## 统一预算与保存

当前统一 runner 使用 batch30、lr5e-5、pilot EMA .99、grad clip1、BF16、单帧/视角、Euler10。
默认同时启用CoRP Eq.7 task aux（lambda=.01、depth1、shared head hidden256）；
用`--task-aux-lambda 0`明确关闭，并把“无辅助损失”保留为单独ablation，不能与新默认混表。
原训练适配中的 AdamW betas=(.9,.95)、weight_decay=1e-4、eps=1e-8和前10%预算的线性warmup
保持数值不变，完整写入结果的 `optimization`。这些细节是项目配方，不是官方CoRP公布的训练配置。
`STEPS` 必须显式声明；默认只跑 none。正式预算、EMA、任务集、训练 seeds 必须在比较前冻结，
不能根据 test 结果反复调整。51/200步只用于工程检查，2k/10k也不自动构成学会任务的证据。

CPU/CUDA RNG、optimizer、scheduler、EMA、donor、importance、像素增广RNG、DataLoader位置完整保存；
checkpoint原子替换，保存独立于评测周期。精确续训目前要求 workers=0、同数据路径/metadata/预算/架构。
不能把一个2k预算改写成10k后称为同预算的完整 resume；新的预算需另立实验。
已完成预算的 checkpoint 不再走训练 resume；显式使用独立 evaluator，避免将保存点之前的 metrics 当作最终评测。
模型格式为 `corp_robocasa_v6`、训练状态版本3，保存 raw/EMA/实际完成步数、初始化权重 digest、
`initialization_policy`、图像/上游契约、task-aux配置/类别映射和预算。
SR协议为`corp_robocasa_sr_v9`，训练身份包含初始化策略和task-aux配置。
旧格式不迁移。test 默认不执行。

## 评测与判定

主指标：闭环 success rate；辅指标：masked per-dimension/per-task MSE。
所有方法同一初始化、数据顺序、预算、训练 seeds、评测帧及仿真 scene manifests。
SR source 为训练 styles 下新生成的冻结场景，不是数据中的 `%10==9` episodes。
scene manifest 保存模型 XML、仿真状态、metadata 和 RNG；之后调用官方重放，再核对 state/style/layout。
每个 episode 的 style 由冻结 schedule 指定，`episode % len(styles)` 轮转，所有方法必须一致；
一个 run 内固定同一 pretrain layout，避免把 layout 变化误读成 style 变化。
创建 manifest 前必须清掉 GymWrapper 构造期残留的 `_ep_meta`，否则 RoboCasa 会复用初次场景，
看似运行 20 个 episode，实际可能始终只有同一个 style。

聚合必须检查任务清单、episode数、manifest digest、checkpoint SHA256、条件、推理预算、source/val范围和完整性。
schema `corp_robocasa_sr_v9` 保存实际训练seed、预算、优化配方、像素/特征增广、task-aux配置、
初始化策略及训练数据digest，不接受旧JSON冒充完整配对记录。
写入前锁文件，日志禁止截断；中断/失败不能产生“全完成”的汇总。
旧进程不遵守新锁，以后部署前须确认旧 writer 退出，再使用新目录；本轮未远程部署。

实验顺序固定：

1. 官方 demo 和本地回归通过，再严格按真实视频读取 → CUDA/BF16 → MuJoCo重放验证；细节见下节。
2. 训练 no-intervention baseline，在冻结的 source/val 场景验证 clean 闭环能力。
3. 同 checkpoint 的 clean/appearance 配对评测，确认问题及任务语义仍成立。
4. 建立标准像素增广对照；在同一有效策略上按Q1/Q2做描述性观察、局部替换及反向/随机对照，区分视觉响应与动作作用。
5. 有定位依据后，按Q3验证共享通道与位置条件粒度的必要性；在实现前明确输入权限、干预位置、调用时序和预算。
6. 证据支持时才定义最小方法，再分解选择/动作保护增量，与无干预、同粒度随机/无保护及标准像素增广比较。
7. 只有验证集证据支持才扩预算；所有选择完成后再显式最终 test。

上述是未来依赖顺序，不授权本轮新增干预、训练或方法矩阵。当前仍仅本地文档整理；下一执行阶段是既定只读验收，
不是直接部署方法。诊断可读clean配对信息与最终方法可用输入分开管理。

报告训练seed和episode/task support，使用适当的配对差值/置信区间（可复用 SciPy），
不能把相邻帧当独立episode，也不能仅以5个百分点或一次51步结果下结论。
如果 baseline 未学会，或 ours 不能优于合理对照，记录失败范围和不确定性，停止扩大方法；不据此宣称普遍不可能。

## 下一步环境验证：严格顺序，不启动训练

可执行入口与逐步命令见[只读验收操作说明](READONLY_SERVER_VALIDATION.md)。
`python -m dacsc.readonly_validation`提供video、官方CUDA demo委托、视觉FFN采集及只生成命令的replay-plan。
当前采集入口仅执行视觉encoder，完整策略/动作明确为`not_run`；MuJoCo仍直接用官方入口、单独验收。
本地测试使用的fixture不是实际服务器视频/CUDA/MuJoCo结果。

这里验证的是“当前代码能否正确读取、推理和重放”，不是方法效果。
本轮不调用训练入口main、不启动runner、不执行真实数据上的backward/optimizer更新，不跑方法矩阵。
失败时停止后续阶段并保留原始错误；未执行标为`not_run`，不能用fixture测试替代真实环境PASS。

部署前使用新的commit命名目录和独立验证输出目录，不覆盖旧代码/日志/模型；核对包SHA256、
`upstream.lock.json`、官方权重checksum、实际import路径和环境版本，确认没有竞争writer。
优先复用对应官方环境，CoRP与RoboCasa解释器/依赖可分开记录；不混装或升级共享环境来掩盖错误。
源码契约不匹配时先恢复锁定revision，不加monkeypatch、XML补丁或兼容shim绕过失败。

### 1. 真实视频读取与增广边界

官方入口：固定revision的`diffusion_policy.dataset.lerobot_dataset.LerobotDataset.__getitem__`，
其视频后端由官方`LeRobotSingleDataset`处理。外层只调用现有`CorpRoboCasaDataset`、
`collate_corp`和`move_batch`；不重写解码器、不用假reader、合成图或仅metadata检查代替本阶段。
具体只读执行方案见[真实样本R0/R1计划](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md#下一步真实样本只读验证方案尚未执行)：
先冻结少量source episode与三视角头/中/尾帧，完成每条件的非侵入性验收，再记录配对外观响应。
本轮只明确方案，尚未运行真实视频采集；无有效normalizer/checkpoint时，完整策略与动作验证必须单独记`not_run`。

验收：

- 从真实source-training episode读取头/中/尾帧的三个视角；记录task、episode、frame、原生style/layout及后端/import路径。
- 原图为有效RGB；resize后尺寸224且数值有限；state9D、action `[10,12]`、末尾有效mask与官方reader/episode长度一致。
- 核对source训练和source holdout/val的episode隔离。仅训练角色可输出`rgb01`；holdout/val维持归一化输入，不读test/reserve作本轮smoke。
- 训练增广在resize后、normalize前直接调用Torchvision；三视角共享抽样、观测间独立。与同RNG的上游算子对照，state/action/语言/标签不得变化。
- 固定评测扰动走独立API；在同一输入/同一评测条件下重复输出一致，评测不得调用训练ColorJitter或消耗其私有RNG。
- raw RGB缺少显式训练增广、对已归一化图像再次增广、混合image-space必须报错；不通过反归一化猜测数据空间。

保存读帧记录、shape/range/mask检查及可检查样图。本阶段不加载策略训练任务，不把读帧成功当作学习成功。

### 2. CUDA/BF16：官方demo在前，真实输入适配器在后

先记录GPU、driver、Torch/CUDA/Torchvision版本，确认CUDA与BF16支持。
在已核对的CoRP环境中直接运行官方入口；下面`python`指该环境解释器，`repos/CoRP`是已验证源码目录：

```bash
PYTHONPATH=repos/CoRP python repos/CoRP/scripts/smoke_test.py --device cuda:0 --precision fp32
PYTHONPATH=repos/CoRP python repos/CoRP/scripts/smoke_test.py --device cuda:0 --precision bf16
```

官方验收：checksum匹配、严格加载通过、48,689,159参数、有限动作`[1,12,7]`，两种精度分别成功退出。
它验证官方LIBERO模型，不等于三视角RoboCasa适配器通过。

之后用第一阶段真实解码图像调用现有适配器，以`eval()`/`inference_mode()`做CUDA前向和采样，
验收三视角/state9D接口、有限动作`[B,10,12]`以及实际autocast路径，不更新模型参数。
单独用真实图像验证CUDA上的训练增广与同device/同RNG的Torchvision结果一致，
调用前后全局CPU/CUDA flow RNG不变，增广RNG可恢复；不能以不同精度结果必须逐元素相同作为标准。
模型评测仍不得自动执行训练像素/特征增广。保存两次官方demo及适配器/增广检查日志，注明device/dtype。

### 3. MuJoCo：先状态重放，再GT动作重放

官方入口：`robocasa/scripts/dataset_scripts/playback_dataset.py`；它直接调用官方
`reset_to`和LeRobot XML/state/action读取函数。先检查所选数据集首个episode确为source style，
再在合适的RoboCasa环境中对同一个真实episode顺序运行：

```bash
MUJOCO_GL=egl python repos/robocasa/robocasa/scripts/dataset_scripts/playback_dataset.py \
  --dataset SOURCE_DATASET --n 1 --camera_height 224 --camera_width 224 \
  --video_path NEW_OUTPUT/state_replay.mp4 --verbose
MUJOCO_GL=egl python repos/robocasa/robocasa/scripts/dataset_scripts/playback_dataset.py \
  --dataset SOURCE_DATASET --n 1 --use-actions --camera_height 224 --camera_width 224 \
  --video_path NEW_OUTPUT/action_replay.mp4 --verbose
```

`SOURCE_DATASET`/`NEW_OUTPUT`须替换为已核对的真实路径和新输出目录；正确官方参数是`--use-actions`，不是`--use_actions`。
不使用`--first`截成一帧后声称完整重放通过。记录robocasa/robosuite/MuJoCo版本、资产路径、
episode metadata及初始XML/state身份；两次完整重放不得出现缺失资产、非法参数、NaN或EGL错误，视频须非空且可解码。
官方动作重放的state-divergence warning必须保留并检查，存在偏差就单独报告，不能隐藏后宣称确定性重放通过。
退出码0/生成视频不等于任务成功；若检查终态，沿用环境官方success判据，并与“重放完成”分开记录。
GT动作成功也不等于学习策略成功。本阶段不跑学习策略的success-rate矩阵。

阶段记录至少包含commit、上游/权重digest、实际命令、环境、原始日志/产物路径、PASS/FAIL/not_run和未解决问题。
三个阶段及训练/评测增广隔离均通过后，另行确定基线训练预算；不自动接着启动训练。

## 官方 demo 与本地测试分工

模型：直接运行 CoRP `scripts/smoke_test.py`，不再维护重复模型 demo。
标准 LIBERO：直接运行其 `scripts/evaluate_libero.py`。
RoboCasa GT replay：直接使用其 `scripts/dataset_scripts/playback_dataset.py --dataset ... --use-actions`。
外观扩展需要的 adapter 保留在本仓库；LIBERO 扩展直接调用官方 `evaluate_suite`，不复制 rollout/success/preprocessing。

`tests/` 的 fixture 是可控依赖叶模块，回归实际项目接口及实际官方函数；它们不能输出科研 success 结论。
本地真实官方 checkpoint smoke/适配器检查通过，也不能替代仿真闭环、CUDA、真实视频reader和长期训练验证。

## 标准指标与统计口径（2026-10-30）

不创建自定义加权总分、robustness index 或自动“方法有效”判定。

| 内容 | 直接复用的开源计算 | 报告口径 |
| --- | --- | --- |
| action squared errors | [PyTorch `F.mse_loss(reduction='none')`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html) | padding不参与任何分子/分母，原始未裁剪动作命令 |
| micro MSE | NumPy `sum` | 全部有效预测位置 ×12维的均值，延续原MSE定义 |
| task-macro MSE | NumPy `mean` | 每个task的MSE等权平均，与micro并列，不另设动作组权重 |
| 分组/12维MSE | 同一份逐维平方误差的标准均值 | 五个动作组与12个真实分量分开命名 |
| success rate | 官方success判定；官方相同的成功次数/episode次数 | 主指标为每task成功率的等权平均；不改成功判据 |
| 跨训练seed的mean/SD | NumPy `mean` / `std(ddof=1)` | 独立单位为训练seed；不将帧/episode重复当作模型seed |
| 均值与配对差值的区间 | [SciPy `sem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html) + `stats.t.interval` | 默认95%，近似正态独立seed估计假设；不手写t临界值或bootstrap循环 |

单seed没有样本SD/CI，输出null；观察到零seed变异（含机器精度舍入）不输出假精确CI。
CI只描述固定有限评测集条件下的训练seed变异，不代替场景抽样CI，也不证明总体无风险。
Student-t近似区间可能超出rate的[0,1]边界，不自行截断，也不将它冒充二项episode概率区间。
小样本的t近似有局限，不能把CI单独当成论文有效性门槛；既定3–5训练seed要求不因episode很多而降低。

CoRP论文以success rate为主；其NMAE、R²来自另外训练的冻结表征probe。本项目的策略MSE是工程辅助，
不是该probe的复现，也不在本轮新增probe训练。

### 明细和命名

action指标schema为 `corp_action_mse_v1`，删除歧义字段 `overall` / `per_dim`。
新字段为 `micro_action_mse`、`macro_task_action_mse`、`per_action_group_mse`、
`per_action_dimension_mse`、`per_task_action_mse`；12维名字使用原始命令分量编号，不假定厘米或角度单位。
保存每帧 `(task,episode,frame,style_id)`、12维平方误差和、有效预测位置数，并按episode聚合。
相邻帧和重叠未来动作位置不是新的独立episode。H10预测误差不同于实际执行8步后的控制效果；
保留原始未裁剪MSE，不偷偷用裁剪后/离散化后的误差替代。

训练中的完整明细写入 `action_metrics_stepN.json`，checkpoint仅保存摘要和明细文件digest，
避免每个checkpoint重复累积大量帧记录。正式比较使用保存checkpoint的独立MSE评测文件，
不能把无checkpoint身份的训练诊断文件当作最终评测。

### 严格配对的离线汇总入口

`scripts/compare_corp_evaluations.py` 只读取已完成的文件，不执行推理、仿真或训练。
比较方向固定 **right − left**：SR正值表示right更好；MSE正值表示right误差更大。
SR差值同时报告百分点（fraction差值×100）。MSE相对变化分母为0时输出null，不加epsilon制造百分比。

方法比较必须同一评测条件；条件比较必须同一个checkpoint。
SR逐一核对 `(task, manifest_sha256, policy_seed)` 与原生style/layout；
MSE逐一核对 `(task,episode,frame)`、style、有效位置、metadata/选样digest、eval seed、batch size、Euler步数和精度。
同一个comparison所有训练seed固定相同评测样本集；预算、优化配方、训练数据、初始化必须一致，
方法/参数配置不允许随seed改变。训练seed从checkpoint元信息获取，不从目录名猜测。
source和val场景不同，不能用相同episode编号作配对；它们之间只保留描述性分布差值。
缺样本、错seed、换场景、不同预算直接报错，不取交集、不静默丢seed。

SR比较manifest示例（路径相对manifest目录；数据必须已按SR v5生成）：

```json
{
  "kind": "sr", "comparison": "methods", "split": "val",
  "tasks": ["CloseFridge", "OpenDrawer"], "episodes_per_task": 10,
  "runs": [
    {"training_seed": 42, "left": "baseline/seed42/val", "right": "candidate/seed42/val"},
    {"training_seed": 43, "left": "baseline/seed43/val", "right": "candidate/seed43/val"},
    {"training_seed": 44, "left": "baseline/seed44/val", "right": "candidate/seed44/val"}
  ]
}
```

MSE使用 `kind: "mse"`，left/right指向独立评测JSON文件；增加 `left_condition` / `right_condition`。
方法比较两条件相同；clean-vs-shift比较使用 `comparison: "conditions"`，可指向同一评测文件的不同条件。
例如none和color_jitter分别报告，不能只用缩小的gap掩盖clean性能损失。

```bash
python scripts/compare_corp_evaluations.py --manifest comparison.json --out comparison_summary.json
```

本工具当前验证RoboCasa输出格式；LIBERO原生报告继续沿用官方入口，不用旧legacy parser猜读当前CoRP文件。
