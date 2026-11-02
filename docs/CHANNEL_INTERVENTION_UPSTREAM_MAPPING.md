# CoRP通道干预：上游代码映射与最小分析方案

2026-10-30。**上游映射部分是静态源码分析，不是方法效果验证。**
此前静态审计未执行模型；后续只读采集的本地CPU实现/验证单列于末节，不能混同为上游完整方法或闭环验证。
只读采集没有运行作者notebook、安装依赖、下载新权重、连接GPU服务器或进行反向/参数更新。
完整回归中的既有合成fixture断点测试包含微型参数更新，不是新训练或真实数据训练；本轮没有启动训练任务。
当前实验约束仍以[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)为准；AdaIN式统计混合是旁支。

## 结论与比较维度

**优先把FFN中间神经元作为机制审计对象，embedding通道作为低成本对照；先不做动态token-channel网络，也不向CoRP移植DaViT骨干。**
这个排序基于对象定义、源码对应关系与适配成本，不是收益排序，不能写成“FFN方法优于embedding方法”。

“embedding通道/FFN神经元”选择的是特征对象，“token-channel”选择的是位置相关作用方式，三者不是互斥架构：
既可有token×embedding坐标，也可有token×FFN神经元。DaViT的channel-token attention又是另一种通道间交互算子。

## 来源固定与证据范围

从作者仓库页面核对revision，并下载固定commit的原始文件到临时分析目录；Git克隆网络失败，未取得完整checkout。
下表和JSON中的固定URL/逐文件SHA256足以定位本次读过的源码，不声称完整仓库或运行环境已经验收。
没有把这些来源加进生产`upstream.lock.json`，也没有把上游代码复制成新的项目方法。

| 来源 | 固定revision | 本次主要检查对象 |
| --- | --- | --- |
| [Registers作者仓库][R-repo] | `860df43515c8d8e9e90952af25a46c26e4469570` | detector、MLP hook、hook manager、model state、额外token构造/切片、hub入口、原notebook配置 |
| [NeuronPath作者仓库][N-repo] | `5f93d0b9f2bcd2061640a3b90117f0087f068734` | joint attribution、CLS-FFN干预、分类概率目标、层级搜索和官方数据/模型入口 |
| [DaViT作者仓库][D-repo] | `fc79470d99babab23e45e0087790fa23cfd10a02` | `ChannelAttention`、`ChannelBlock`、空间/通道block组合、依赖与权重边界 |
| [FFN steering作者仓库][S-repo] | `559c0f25a3cc5a20fc8b804774a88415404e0d22` | VLM value-vector提取/词表投影、OpenVLA down-projection覆写hook、π0投影权重键 |
| [Action Atlas作者仓库][A-repo] | `6025c47658d4f63891daf4be66757489db0e0949` | 模型适配器的真实注册点、capture/replay、残差位置、SAE pooling/write-back协议 |
| 当前CoRP底座 | `52445740aa9517ae396cc8fb14bae8fe5922f118` | vendor DINOv2、visual encoder、extractor、resampler；项目代码基于`7efc94a` |

文件hash、符号行号、读取范围、下载失败记录与未执行项见[静态审计记录](../audit/channel_upstream_static_audit.json)。
NeuronPath和DaViT根LICENSE均为MIT；Registers的整体授权本轮不作推定，不把其DINOv2子树的许可自动扩展为整个仓库许可。
新增两套具身源码的固定URL、hash、符号行号及静态核验范围单列于[具身FFN审计记录](../audit/embodied_ffn_upstream_static_audit.json)。
本轮没有安装、导入或运行它们，没有将其加入分析运行依赖锁，也没有更换当前采集器。

## 三篇主读论文：五行卡片

阅读主线为“通道是什么→如何定位→是否影响动作→如何受控干预”。
Registers负责结构观察；FFN steering与Action Atlas负责对照行为干预的证据与接口。
NeuronPath、DaViT保留为相邻工具/结构参考，不再把“具身模型内部特征干预没人做过”当作研究缺口。

| 五行问题 | Registers（2025） | FFN steering（CoRL2025） | Action Atlas（2026，v5） |
| --- | --- | --- | --- |
| 操作什么？ | DINOv2/CLIP的FFN中间激活及高范数token位置 | VLA的VLM骨干FFN中间激活与固定输出方向；不是DINO视觉FFN | 多路径layer输出/MLP写入量及其SAE特征；F.2词表关联分析另属于PaliGemma FFN权重 |
| 为什么选它？ | outlier位置上持续强的FFN响应 | value vectors投影到词表后的语义关联，再组织候选cluster | 层/路径分组、跨任务注入、探针与任务选择性SAE等；不是统一的视觉神经元打分 |
| 怎么动它？ | 将激活搬到额外token并处理原位置 | OpenVLA把指定中间激活覆写为常数，再重算down-projection；不等于乘开度 | 按实际注册点整层归零、激活回放或SAE重建/特征修改；不同路径操作范围不一 |
| 怎样验证？ | outlier/特征图变化及下游视觉任务 | 行为方向、轨迹与任务表现的受控干预实验 | task override、轨迹和闭环结果；单元关联与完整路径干预不能混为一项证据 |
| 哪些不能搬到CoRP？ | ViT-L阈值/索引、额外token与日志节点 | VLM词表语义、动作token与模型专用hook；不能给视觉单元直接贴速度/方向标签 | 现成VLA adapter、SAE权重、层号、pooling及调用次数语义；CoRP没有对应的大VLM/多层expert拓扑 |

原文：[Registers](https://arxiv.org/html/2506.08010v5)、
[FFN steering](https://arxiv.org/html/2509.00328v1)、
[Action Atlas v5](https://arxiv.org/html/2603.19233v5)。
论文报告是作者证据，不是本项目复现。仅把VLA换成CoRP或重复内部干预，不足以自动构成新意。

## CoRP实际调用链：先确定对象

源码：[CoRP DINOv2][C-dino]、[extractor][C-extract]、[resampler][C-resample]。

| 位置 | 224输入、3视角时的张量/操作 | 干预含义 |
| --- | --- | --- |
| 每个DINO block输入/输出 | `[3B,257,384]`；首位置是CLS，其余256个是patch | 残差流的384维embedding坐标，不是MLP中间神经元 |
| `backbone.blocks[l].mlp.act`输出 | GELU后`[3B,257,1536]`，下游是`fc2(1536→384)` | 与Registers的普通MLP neuron hook对应的对象 |
| FFN分支之后 | `fc2` → `ls2` → 与残差相加 | 衰减FFN神经元只改变该分支，不等于直接删除整个残差特征 |
| visual encoder输出 | 最终LayerNorm后丢弃CLS，`[3B,256,384]` | 现有token-style旁支hook在这里；不能观测原始残差流的高范数分布 |
| extractor相机排列 | 先整批左视角，再整批腕部，再整批右视角，即view-major | 任何多视角适配须显式转换，不能把相邻行当作同一样本的三个相机 |
| resampler输入/输出 | `384→768`投影，语言FiLM作用于queries；48 tokens/view | 后面还有attention与LayerNorm，不能宣称post-norm干预再也不会被归一化影响 |

CoRP的attention用PyTorch SDPA，没有显式的attention softmax identity模块。
最后一个block的FFN是逐token运算，之后仅逐token归一化并丢弃CLS：
**仅改变最后block的CLS-FFN激活，无法影响该调用链中的patch输出。**
这是静态调用图推论，不是本轮运行测量；更早层的CLS仍可能通过后续attention影响patch，不能笼统说CLS全部无用。

## Registers：能借鉴神经元诊断，不能原样挂入当前CoRP

| 上游文件/符号 | 实际做法 | 对CoRP的映射与复用边界 |
| --- | --- | --- |
| [shared/algorithms.py: find_register_neurons][R-alg] | 在选定block输出上以范数阈值找outlier位置，再按这些位置的FFN激活绝对值聚合，形成layer/neuron排名 | 排名计算可保持上游定义；需要提供当前CoRP权重的model-state/数据适配，不是style分类分数 |
| [dinov2_hook_manager.py][R-hook] | 普通MLP的neuron对象是`.mlp.act`，SwiGLU走`hidden_identity` | CoRP的GELU模块可对应；不能把排名索引乘到384维输出上 |
| [shared/hook_manager.py][R-manager] | ANALYSIS同时注册attention、block、neuron日志；attention日志依赖显式post-softmax组件 | 当前SDPA底座不能直接运行这个manager；需另行提供只读block/FFN日志适配，不能为满足工具而替换attention实现 |
| [shared/hook_fn.py: activate_on_registers][R-fn] | 将选定FFN激活转移到末尾额外token，并按配置处理patch激活；源码对`output[0]`操作 | 是带额外token的重定向，不是普通channel gate；单图逻辑不能直接用于`3B`多视角batch |
| [dinov2_state.py][R-state]、[配置][R-config] | 原state入口加载作者版DINOv2并调用CUDA；示例为ViT-L/14、阈值150、detect layer=-2、最高层17 | 不加载它来替换当前CoRP模型；不能照搬ViT-L的阈值、层号或神经元索引到12层ViT-S/14 |
| [作者hub入口][R-hub] | 带预置test-time register neuron列表的封装是`dinov2_vitl14_tt_reg` | 有普通vits14入口不等于已有与当前CoRP等价的vits14 test-time-register策略 |

额外适配问题：官方logger逐次转CPU/NumPy，索引/拼接依赖单图用法；BF16日志转换和batch轴要单独核验。
这些诊断不应放进每个训练step。探测不到outlier时报告没有观察到，不能为必然得到一个排名而调阈值后称为论文默认。

**上游patch切片的静态风险：**固定revision的[模型文件][R-vit]在`prepare_tokens_with_masks`中
把register追加到序列末尾，但`forward_features`的reg/patch字段仍按“register紧接CLS”切片。
当有一个额外register时，直接取该`x_norm_patchtokens`字段可能漏掉首个patch并混入末尾register，
长度仍可为256；shape检查不足以确认patch身份。这个观察针对把该字段对接CoRP的用法，
不意味着作者使用CLS/原始block输出的全部demo无效。本轮不修改该上游文件，也不采用增加register的路线。

## FFN steering：输出方向与运行期激活不是同一个量

本次静态代码核验重点是OpenVLA/PyTorch路径；π0只追到value-vector提取及其权重键，未审完完整JAX运行期干预链。
不声称作者所有分支/实验都已核验，更没有运行其模型或词表投影。

| 作者源码/符号 | 实际位置和操作 | CoRP边界 |
| --- | --- | --- |
| [extract.py: extract_value_vectors][S-extract] | 遍历`model.language_model.model.layers`，用one-hot输入调用各层`mlp.down_proj`，得到固定输出方向 | 是VLM层权重的方向，不是从图像采集到的激活；索引对应其MLP中间维，不是CoRP384维坐标 |
| [project_to_vocab_batched][S-project] | 用`language_model.lm_head.weight`把上述方向投影到词表 | CoRP视觉FFN没有对应词表head；T5词表与DINO残差流不在同一已证明对齐的空间，不能借T5给视觉方向贴标签 |
| [hooks.py: apply_gate_proj_hooks][S-hook] | 在`layer.mlp.down_proj`注册forward hook；读取并原位覆写`input[0][..., neuron_ids]`为`coef`，再用原weight/bias做`F.linear`并返回新输出 | 这是主动干预，不是观察器，也不是普通乘法gate。默认中间维11008、扁平layer/neuron索引不能搬到1536维CoRP |
| 同上hook的恒等语义 | `coef=1`表示把选定激活设为1，不保证原输出不变 | 不能用`coef=1`作为无操作对照；当前只读hook返回None，与此机制不同 |
| [π0投影提取][S-pi0] | 从`PaliGemma.llm.layers.mlp.linear`取固定权重并reshape，再投影到token embedding | 不是DINOv2权重键；这部分代码不能视为已核验的完整π0干预实现 |

对当前CoRP，GELU后激活`a`与固定方向应分开理解：`FFN(x)=W2*a+b2`。
现有采集器记录`a`；若研究“写入残差的方向”，还需观察`W2[:,i]`及原`ls2`缩放，而不是只按激活大小解释。
**CoRP的fc2有bias**，直接照搬one-hot调用fc2会把同一bias加进每个候选向量；不能直接将该输出当成纯权重列。
本轮只记录这个适配要求，不实现新的贡献打分或语义投影。

作者提取脚本还按缓存路径是否存在来跳过计算；未来复用产物必须绑定checkpoint/source digest并使用独立目录，
不能在同一文件名下混用模型或旧缓存。其模型loader、量化、attention fallback、图像裁剪和动作token解码不进入CoRP管线。

## Action Atlas：先映射注册点，再解释干预结果

| 作者源码/符号 | 具体hook位置与行为 | 对CoRP的适配边界 |
| --- | --- | --- |
| [model_adapters.py][A-adapters] | Pi0.5分`gemma_expert.model.layers`和`paligemma...language_model.layers`；OpenVLA-OFT为`language_model.model.layers`；GR00T分Eagle/VL-SA/DiT | 分组便于定位路径，但这些模型路径和层数不对应CoRP。不能只换类名后称为支持CoRP |
| 同文件SmolVLA group | expert/VLM group直接返回每层`.mlp`，因为交错推理会绕过DecoderLayer wrapper | 此时通用capture/zero/injection改变MLP写入量，不是完整残差；与其他模型的整层操作不能同名并列解释 |
| [hooks.py: ActivationCaptureHook][A-hooks] | forward hook复制模块`output`或tuple的`output[0]`到CPU，逐调用追加，返回原output | 是最接近我们只读记录的思想；本项目保留单帧、显式CLS/patch/view轴与FP32日志，不需要替换当前已测采集器 |
| [ActivationInjectionHook][A-hooks]、[cross_task_injection_v2][A-inject] | 按逐调用list回放；存储耗尽或shape不合时返回原输出；统计注入次数/shape mismatch | 同shape不等于同物理时刻或同token身份；若未来适配，必须核对调用键/时序/次数，不能把未注入或静默跳过当作无效机制 |
| [ActivationCollector][A-hooks] | 可保留每token，也可`mean(dim=1)`；`gated=True`时每环境step只收一次，避免DiT多次去噪重复记录 | CoRP视觉编码每次extract调用一次，DiT每个Euler步都会调用；不能用同一个step计数合并两者，不能无条件平均patch/action token |
| [ResidualSite / SmolVLAResidualSite][A-sites] | 常规路径在完整层output capture/edit；Smol专用路径读post-attention norm输入、结合MLP输出重建残差，并通过MLP写回差量 | 这些是不同于裸MLP日志的专用算子。CoRP无同样wrapper接口，不复制该写回公式来改视觉层 |
| [reconstruction_hooks.py][A-recon] | 明确区分per-token替换、mean-pool广播替换、mean-pool差量写回；工厂函数检查pooling匹配 | SAE训练位置、归一化与重建协议必须一致；不存在可直接用于CoRP视觉FFN的作者SAE，当前不训练SAE或接入重建 |

[论文D.4](https://arxiv.org/html/2603.19233v5#A4.SS4)将整层残差和MLP写入区别报告；
[F.2](https://arxiv.org/html/2603.19233v5)的motor-primitive词表关联分析针对PaliGemma FFN权重，
不能直接称为当前DINO视觉FFN的动作神经元。本轮未定位/核验该词表统计的完整代码链，不按其章节标题补写实现。
部分回放源码基于逐调用list而非显式`episode/frame/tau`身份；这是迁移时需要补充的协议校验，不是本轮已经实现的功能。

现有CoRP可区分三个观察层次：视觉DINO FFN、语言条件resampler、DiT动作生成。
当前只读采集器只覆盖第一层次，不能据此排除融合/动作路径的故障。把整份clean视觉输出写回扰动运行，
在其余条件相同的确定性数据流中本就可能恢复clean动作；那只是未来patch实现的正对照，不能单独证明视觉FFN是故障根因。
当前不新增模块级或局部激活替换，更不使用这些主动hook来做只读验收。

## NeuronPath：相邻分析工具，CLS与输出目标必须适配

| 上游文件/符号 | 实际做法 | 对CoRP的映射与复用边界 |
| --- | --- | --- |
| [models/modeling.py: Mlp.forward][N-model] | 在GELU后替换/消融`x[:,0]`，返回该CLS位置的FFN向量 | 不是对所有patch的FFN神经元做归因；不能将CLS神经元结果直接当作动作patch通路 |
| [VisionTransformer.forward][N-target] | CLS接分类head，目标取类别softmax概率，对替换的激活求梯度 | CoRP没有该分类head；需要明确标量动作目标，是研究适配，不是参数名替换 |
| [path/joint_attr.py][N-joint] | 零基线插值、对已选路径的干预、分层搜索与joint score；批内使用首个类别标签 | 数学helper/路径搜索思路可借鉴；真实CoRP图、token范围、标量目标与混合任务batch语义不能原样复用 |
| [path/influence_pattern.py][N-pattern] | 调用作者改写的modeling_ip以沿路径计算影响 | 当前CoRP不接受`tgt_layer/tmp_score/path_weights/tgt_label`这些forward参数；不能复制整套分类模型替换CoRP |

官方脚本按ImageNet/CIFAR标签输出路径；本项目的动作误差包含horizon/mask、state、指令以及flow噪声/时间。
未来若做动作归因，必须冻结同一噪声/时间、正常/扰动观测的比较口径，并保留动作mask。
这会改变原论文的任务/目标与token范围，需独立数值核验，不称为官方“机器人NeuronPath”。
“模型对哪个单元敏感”也不自动意味着“抑制该单元会提升闭环表现”。
当前最小方案不移植完整JAS搜索、不增加逐训练step的积分梯度计算，也不实现新的动作归因目标。

## DaViT：可读到实际通道交互算子，但不是对角门控

| 上游文件/符号 | 实际做法 | 对CoRP的映射与复用边界 |
| --- | --- | --- |
| [timm/models/davit.py: ChannelAttention][D-attn] | 对`[B,N,C]`做QKV投影；每组通道产生`[B,heads,Cg,Cg]`交互矩阵，再投影回原shape | 可在原上游环境复用该类；输入shape相同只证明接口形状，不证明可无损替换CoRP |
| 同上forward | 源码使用缩放后的`KᵀV`和softmax，再作用于`Qᵀ` | 不是`g_c*Z`或`g_{t,c}*Z`，不能按教科书token注意力或SE门控自行重写后声称复现 |
| [ChannelBlock][D-attn] | 包含ConvPosEnc、norm、attention残差、可选FFN；还接收二维空间size | 不能只凭384维相同就替换CoRP block；额外参数、位置机制和权重来源都会改变基线 |
| [DaViT主体][D-model] | 官方默认依次组合spatial与channel block，并使用分层patch embedding | 不是保持CoRP原架构的轻量通道选择；本轮不搬入整个骨干 |

`ChannelAttention`的QKV/proj没有可从当前CoRP逐元素对应迁移的已声明权重。
新增它就增加容量/训练因素，须另立“新增随机模块”控制，不能将收益归给通道选择本身。
该文件还依赖作者仓库内的旧timm相对模块；当前pip timm中的同名实现不自动等于此固定revision。

## 哪些可以直接复用，哪些需要适配

这里“直接复用”表示保留原函数/算子定义，不表示本轮已经执行通过。

| 内容 | 分类 | 本轮决定 |
| --- | --- | --- |
| 当前CoRP的GELU、fc1/fc2、LayerScale、原forward和权重 | 保持现有官方底座 | 不修改，作为唯一主干 |
| Registers的原始outlier/激活排名计算 | 计算逻辑可保留，模型/数据/日志边界需适配 | 优先审计；与域敏感评分分开命名 |
| Registers的现成HookManager、追加register干预 | 不能直接用于当前CoRP | 缺显式attention节点、单图假设、token增删与切片契约；不纳入最小方案 |
| NeuronPath的插值、路径分数helper | 可借鉴原定义；整套动作应用需实质适配 | 当前只作源码参照，不实现积分归因或完整搜索 |
| NeuronPath原分类模型/脚本 | 不能直接用于当前CoRP | CLS/类别目标/forward接口不符，不移植模型 |
| DaViT ChannelAttention原类 | 可复用原算子，但放入CoRP是新的架构适配 | 不是当前最小gate，不加入模型 |
| FFN steering的权重/词表投影 | 可以参考固定输出方向的定义；VLM词表映射不适用 | 只建立映射，不导入作者模型，不给DINO单元贴语义标签 |
| Action Atlas的capture、分组及回放原型 | 只读记录思想可借鉴；主动回放、pooling和路径位置需要适配 | 保留现有已验证采集器，不新增SAE、注入或框架依赖 |
| 动态token-channel selector、动作保护评分或训练损失 | 此次固定版本源码没有提供当前CoRP可直接调用的完整方法 | 属于待设计项，本轮不自行补写 |

## 三种对象对CoRP的适合程度

| 候选 | CoRP中的确切位置 | 适配成本/优点 | 风险与用途 |
| --- | --- | --- | --- |
| embedding通道 | visual encoder最终norm后的384维输出，或block残差流坐标 | 输出hook最容易，维度和checkpoint不变 | 坐标混合、层位置差异；作为低成本同预算对照，不把384维自动解释为语义单元 |
| FFN神经元 | `blocks[l].mlp.act`的1536维GELU输出 | 与Registers/NeuronPath的中间对象最接近；现成可hook模块，无需替换attention | 仍无动作/域语义保证；前后还有fc2/残差；优先作为机制审计对象 |
| token-channel | 对embedding或FFN引入随token变化的选择 | 可表达不同位置的不同干预 | 不是前三篇提供的现成CoRP选择器；位置对齐、门控生成与参数容量引入额外问题，后置考虑 |

共享384维gate只需384个开度，共享FFN gate对应1536个开度；完整patch×embedding mask有256×384个元素，
patch×FFN mask有256×1536个元素。这是掩码元素数量，不等于一定新增同样多的可学习参数。
没有运行证据支持其中任一种在成功率上更优，也不能据此宣称FFN语义天然更干净。

## 最小方案：先只读审计，再讨论单层干预

### A. 只读适配规格（原静态审计建议，当前实现范围见下节）

1. 使用同一CoRP权重中的ViT-S/14，保留其resize/normalize，不另换作者ViT-L、分类checkpoint或任意神经元索引。
2. 只观察12层原始block输出与对应`mlp.act`。区分CLS和patch、view-major与sample-major；先用单图/单视角检查记录身份。
3. 保持原forward输出逐元素不变；不要求取attention矩阵，不替换SDPA，不追加register，不返回新token。
4. 在source训练校准样本上检查是否确有outlier现象，再按明确记录的阈值/层输出运行上游排名定义。
   固定正常/外观对、记录任务线索；没有outlier就报告没有，不能把高范数直接叫style shortcut。
5. 在有可信动作策略后才测patch神经元与动作误差的关系。随机初始化的RoboCasa接口层/未验证的弱策略不提供动作因果证据。

仅model-state、数据和只读logger接线属于必要适配；不能一并改归因目标、增加动态gate、加入新损失后称为“直接复用”。

### B. 后续最小干预候选（设计说明，不在本轮实现）

- 对象：只在**一个待校准选定的DINO block**，作用于GELU后、fc2前的patch位置FFN神经元。
- 形式：`H_patch[..., h]`乘一个共享开度`g_h`，`g_h=1`为恒等；保持CLS、原fc2/LayerScale/残差及T5/resampler/DiT不变。
- 先将其定义为离线、固定且确定性的衰减消融：同一开度应用于全部patch和三个视角；不读取测试style或GT action生成当次gate。
- 控制：恒等、同层同神经元数量/强度的随机衰减、明确来源的候选集合衰减；另保留384维embedding位置的匹配对照。
  outlier集合不称为domain集合；没有可信候选评分时不启动“domain gate”组。
- 初期不做可学习动态gate、不引入补偿分支、不追加register、不加DaViT模块、不做AdaIN、不联合完整NeuronPath搜索。
- 动作保护、训练期正则或可学习推理gate的具体评分/损失/预算仍待定义；不能将这个衰减消融规格称为已完成的新方法。

分析通过也不自动授权执行训练。真实视频→CUDA/BF16→MuJoCo验证与有效clean基线仍是后续方法实验的前提。

## 只读采集实现与本地验证

实现为`dacsc/corp_ffn_capture.py`，不修改CoRP、DINOv2或训练脚本。它不是门控、NeuronPath动作归因或新评分方法。

- `load_registers_reference(root)`校验`analysis_upstream.lock.json`的四个原文件，按隔离的Python命名空间加载原函数。
  不复制/重写logger和排名，不把作者`shared`包放进全局搜索路径，不自动下载或回退到其他源码。
- `CoRPFFNCapture`仅注册原block和`mlp.act`观察hook；返回None，既不修改激活也不替换返回值。
  要求policy及子模块eval、调用处`no_grad`/`inference_mode`，拒绝带token-style旁支的policy。
- 先detach并独立复制到CPU FP32，再调用原`log_internal`；保留原始dtype记录，不与模型输出共享存储。
  不替换SDPA来获得attention日志，不增加register或模型参数。
- 每次完整backbone forward只保留最新一帧：FFN为每层`[B,V,257,1536]`，原block输出为`[B,V,257,384]`。
  position0为CLS，`1:257`为256个patch；样本/相机轴由官方view-major次序转换，不按输入dict插入次序猜测。
- 上下文退出（包括失败）移除自身hook；清理未完成记录，防止把旧快照当成失败调用的新输出。
  同一采集器不可重复进入；不允许将单视角校准名用于完整三视角policy forward。
- `rank_register_images`只是原`find_register_neurons`的model-state/数据适配：同一CoRP backbone、指定单相机、项目固定预处理，
  沿用官方含CLS的`[layer,token,feature]`输入约定，不悄悄改为patch-only评分。阈值/检测层需显式指定，
  不搬用ViT-L配置；无outlier保留上游报错，无fallback。保留作者`random.sample`逻辑并在成功/失败后恢复调用方Python RNG。
  校准图片目录的source身份由调用方核对；此helper不判断图片style或自动建立训练样本manifest。它不是style评分或动作重要性。

可选分析源码需提前准备固定revision；实际运行校验逐文件hash，不要求临时源码目录具有完整Git checkout。
主policy加载、训练和评测入口都不导入此模块。完整矩阵采集有内存代价：B=2、三视角、12层的两类FP32快照约136MiB，
实际峰值更高；不把它放进训练step，不将作者逐图CPU/NumPy日志称为GPU向量化诊断。

API用法（现有eval模型和预处理后的输入由调用方提供，不是新训练/demo入口）：

```python
reference = load_registers_reference("/verified/test-time-registers")
with CoRPFFNCapture(model, reference) as capture, torch.inference_mode():
    representation = model.extract(images, state, instructions)
    frame = capture.last
cls_ffn = frame.ffn[0][:, :, 0, :]
patch_ffn = frame.ffn[0][:, :, 1:, :]
```

本地测试`tests/test_corp_ffn_capture.py`包括原Registers函数实际执行、CPU BF16日志、失败清理、
缺失/篡改源码、单视角误标三视角及相机顺序改变的拒绝路径。实际发布权重加载的三视角适配器使用B=2受控图像输入，
完整12层采集前后表示、pooled language、动作、Torch RNG、state_dict及hook数量均核对；数值比较`rtol=0, atol=0`。
这验证观察器不改变前向，不证明随机初始化的新RoboCasa接口层具有任务能力。
输入不是本轮新解码的RoboCasa视频，不能据此宣称真实视频、CUDA/BF16、MuJoCo或任何动作机制/收益成立。
最新记录见[只读采集验证](../audit/corp_ffn_capture_local_validation.json)；此前静态审计JSON中的文档hash属于实现前快照。

## 下一步真实样本只读验证方案（尚未执行）

入口现已准备：`python -m dacsc.readonly_validation`；[操作说明与验收包](READONLY_SERVER_VALIDATION.md)。
这是按下述R0/R1规格补齐的薄接线，使用原reader/视觉forward/logger及Torchvision。
当前入口仅验视觉分支，完整策略/动作项保持`not_run`，没有干预/排名/训练；真实数据与GPU执行仍待后续验收。

本节是待执行的本地方案，不是一次新实验的结果。当前只允许读真实视频、无梯度前向、记录和恒等校验；
不加门控、替换/清零激活、词表语义命名、SAE、动作归因、训练或服务器连接。
先完成R0采集验收，再记录R1外观响应；后续CUDA/BF16和MuJoCo仍按当前协议另行验证。

### 样本、模型和配对先冻结

- 数据来自本地真实RoboCasa365官方reader；校准只取原生styles11–30、episode `%10 != 9`，不使用source holdout、val、test或reserve来找排名。
- 小样本上限：3个task × 每task3个独立episode × 每episode头/中/尾3个不同帧，共最多27个三视角观测（81个view-frame）。
  不是81个独立episode。按固定seed123在有效source元数据集合中一次性抽取并保存清单，不根据激活或模型表现挑样本；不足则报告实际support。
- 每个episode的帧位置先固定为`0, length//2, length-1`并去重；同一episode的多个帧不当作独立统计重复。
- 身份键至少为`(dataset_relpath, task, episode_index, frame_index, view_name)`；style/layout从原始extras核对，不能从目录名猜或制造coarse标签。
  `CorpRoboCasaDataset`返回值当前没有layout字段，记录时须按同一个episode的原始metadata回填，不能把缺失当作默认layout。
- 使用同一已校验官方CoRP权重中的ViT-S/14做视觉采集；不换Registers的ViT-L模型。记录官方源checksum、适配初始化来源及新接口层随机初始化事实。
  只有另有可追溯、当前协议有效的RoboCasa checkpoint与train-only normalizer时，才做真实输入的完整策略/动作恒等校验。
  缺少它们时，仍可只验证视觉encoder，但完整策略项记`not_run`；不拿临时拟合、identity normalizer或零state替代真实输入契约。

### R0：真实reader与观察器的非侵入性

1. 核对官方源hash、实际reader/import路径、解码后端、视频/元数据身份。使用官方`LerobotDataset.__getitem__`，不改解码器，不用contact sheet或合成图冒充真实帧。
2. 每个观测只解码一次三个相机的原始RGB，缓存该批原始像素及hash。先用两观测batch验证三视角顺序，随后保持相同样本和batch划分；尾批可以为1，不补伪样本凑batch。
3. 调用现有`prepare_image`，走Pillow resize224、固定条件、ImageNet normalize；相机保持左/腕/右的官方view-major拼接。
   不做crop/翻转，不给RoboCasa加LIBERO旋转，不实例化训练`ColorJitter`或token-style模块。
4. 同一份预处理tensor分别做无hook与只读hook前向。固定checkpoint、模式、batch、精度和Torch RNG起点，
   在CPU FP32、`eval()`与`inference_mode()`下比较视觉输出；完整策略项满足前述normalizer条件时再比较representation/动作。
5. 输出必须有限且`torch.equal`（或`rtol=0, atol=0`）成立。校验原始257位置与输出256个patch的身份，
   逐样本/相机验证`flat_row=view_index*B+sample_index`，不只检查shape或全体均值。
6. 校验采集后state_dict digest、模型模式、Torch RNG和外部原有hook数量不变；自己的hook移除，失败帧不能沿用旧记录。

R0有读取、标签、相机、数值或恒等失败，就停止R1，不生成“已完成”汇总。
本地若缺视频/依赖，仅记录具体缺项，不联网取新模型或通过修改上游/兼容shim绕过。

### R1：同一帧的正常/外观响应，只观察不干预

- 条件先固定为`none`与现有`image_contract`中的确定性`color_jitter`（brightness1.6→contrast1.6→saturation2.2）；
  原始RGB共享，转换在resize后/normalize前，不对已normalize的图像再次施加扰动。
  这是固定评测stress，不是随机训练增广，不用不同style的另一episode充当“同状态反事实”。
- 每个条件分别完成R0的无hook/有hook恒等检查，再配对记录FFN与raw block输出。两条件的输出本来可以不同，不能要求它们相同才算通过。
- 保留layer、sample、view与token位置，不先对token平均或混合不同相机。
  raw block范数用与Registers对应的标准Torch向量范数；CLS与patch的分布/计数单列，不将最终LayerNorm后数据当作raw outlier检测输入。
- 这一阶段不要求产生outlier或神经元排名。可记录标准mean/min/max/分位数、有限性及原始响应；不新增自定义robustness总分。
  若后续使用官方detector，阈值、检测层、校准集和含CLS的原始定义需先记录，不能调到“必须找到神经元”。
- 当前`rank_register_images`只实现clean单相机图片预处理，没有shift参数。不能通过另存变换后的量化PNG或改预处理来绕过，
  然后声称已完成严格配对的跨条件官方排名；这类排名接线仍是待定义项，不在本轮实施。
- 检查并保存可读样图：若强度抹掉了把手/目标等任务线索，记录信息丢失，不把所有退化都解释为shortcut。

“外观响应变化”“outlier”和“动作致错单元”是不同结论；本节只交付前两项的描述性观察（若出现）。
RoboCasa新接口未学会任务时，不计算并解释动作重要性，更不把视觉观察结果称为门控收益。

### 待执行产物与验收

| 产物 | 必须内容 |
| --- | --- |
| `sample_manifest.jsonl` | 原生身份、source fold、task/style/layout、指令hash、episode长度、帧索引、相机、原视频/解码像素身份及实际support |
| `capture_manifest.json` | 项目/上游revision与hash、checkpoint来源、random-init边界、预处理契约、设备/精度、视图顺序、batch划分、采集位置及条件 |
| `identity_checks.json` | 每批每条件的有限性、输出/dtype/shape、恒等结果、权重/RNG/hook检查；视觉与完整策略项分开标PASS/FAIL/not_run及原因 |
| 只读summary与少量快照 | 按layer/view/CLS/patch保留身份；样图仅供检查，不能回作模型输入；固定少量观测保存完整tensor，其余保存标准汇总和manifest，避免全量逐帧大文件 |
| 执行日志 | 实际命令/输入、环境、失败项和未执行项；不以退出码0替代上述验收 |

上述文件名是方案要求，本轮没有伪造这些产物或真实视频PASS记录。新具身代码审计也不替代实际数据/硬件检查。

[R-repo]: https://github.com/nickjiang2378/test-time-registers/tree/860df43515c8d8e9e90952af25a46c26e4469570
[R-alg]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/shared/algorithms.py#L6-L83
[R-hook]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/dinov2/dinov2_hook_manager.py#L15-L22
[R-manager]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/shared/hook_manager.py#L65-L82
[R-fn]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/shared/hook_fn.py#L13-L46
[R-state]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/dinov2/dinov2_state.py#L39-L63
[R-config]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/configs/dinov2_large.yaml
[R-hub]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/dinov2/dinov2/hub/backbones.py#L86-L98
[R-vit]: https://github.com/nickjiang2378/test-time-registers/blob/860df43515c8d8e9e90952af25a46c26e4469570/dinov2/dinov2/models/vision_transformer.py#L213-L282
[N-repo]: https://github.com/foundation-model-research/NeuronPath/tree/5f93d0b9f2bcd2061640a3b90117f0087f068734
[N-model]: https://github.com/foundation-model-research/NeuronPath/blob/5f93d0b9f2bcd2061640a3b90117f0087f068734/models/modeling.py#L117-L133
[N-target]: https://github.com/foundation-model-research/NeuronPath/blob/5f93d0b9f2bcd2061640a3b90117f0087f068734/models/modeling.py#L304-L324
[N-joint]: https://github.com/foundation-model-research/NeuronPath/blob/5f93d0b9f2bcd2061640a3b90117f0087f068734/path/joint_attr.py#L212-L286
[N-pattern]: https://github.com/foundation-model-research/NeuronPath/blob/5f93d0b9f2bcd2061640a3b90117f0087f068734/path/influence_pattern.py#L170-L205
[D-repo]: https://github.com/dingmyu/davit/tree/fc79470d99babab23e45e0087790fa23cfd10a02
[D-attn]: https://github.com/dingmyu/davit/blob/fc79470d99babab23e45e0087790fa23cfd10a02/timm/models/davit.py#L174-L248
[D-model]: https://github.com/dingmyu/davit/blob/fc79470d99babab23e45e0087790fa23cfd10a02/timm/models/davit.py#L425-L486
[C-dino]: https://github.com/corp-policy/CoRP/blob/52445740aa9517ae396cc8fb14bae8fe5922f118/corp/_vendor/dinov2.py#L40-L116
[C-extract]: https://github.com/corp-policy/CoRP/blob/52445740aa9517ae396cc8fb14bae8fe5922f118/corp/representation/extractor.py#L39-L63
[C-resample]: https://github.com/corp-policy/CoRP/blob/52445740aa9517ae396cc8fb14bae8fe5922f118/corp/representation/cross_attention_resampler.py#L23-L57
[S-repo]: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas/tree/559c0f25a3cc5a20fc8b804774a88415404e0d22
[S-extract]: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas/blob/559c0f25a3cc5a20fc8b804774a88415404e0d22/src/ffn_value_vectors/extract.py#L65-L88
[S-project]: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas/blob/559c0f25a3cc5a20fc8b804774a88415404e0d22/src/ffn_value_vectors/extract.py#L91-L112
[S-hook]: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas/blob/559c0f25a3cc5a20fc8b804774a88415404e0d22/src/libero_experiments/hooks.py#L8-L40
[S-pi0]: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas/blob/559c0f25a3cc5a20fc8b804774a88415404e0d22/src/openpi_ffn_value_vectors/pi0_extract.py#L78-L138
[A-repo]: https://github.com/CWRU-AISM/action-atlas/tree/6025c47658d4f63891daf4be66757489db0e0949
[A-adapters]: https://github.com/CWRU-AISM/action-atlas/blob/6025c47658d4f63891daf4be66757489db0e0949/experiments/model_adapters.py#L283-L861
[A-hooks]: https://github.com/CWRU-AISM/action-atlas/blob/6025c47658d4f63891daf4be66757489db0e0949/experiments/hooks.py#L57-L171
[A-inject]: https://github.com/CWRU-AISM/action-atlas/blob/6025c47658d4f63891daf4be66757489db0e0949/experiments/cross_task_injection_v2.py#L162-L217
[A-sites]: https://github.com/CWRU-AISM/action-atlas/blob/6025c47658d4f63891daf4be66757489db0e0949/experiments/hooks.py#L174-L230
[A-recon]: https://github.com/CWRU-AISM/action-atlas/blob/6025c47658d4f63891daf4be66757489db0e0949/experiments/reconstruction_hooks.py#L144-L254
