# 研究方案与开源数据集选型总览

更新时间：2026-11-06

状态：研究方案与数据选型记录，不是论文结果，也不表示方法已经有效。

当前 V10 执行顺序和冻结配置以 [V10_PROTOCOL.md](V10_PROTOCOL.md) 为准；
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) 说明总体研究边界。本文用于把研究问题、模型选择、
数据划分、开源数据集泛化能力和历史证据放在同一张图上，避免把不同 benchmark、
不同训练预算和不同泛化类型混成一个结论。

---

## 0. 一句话结论

> 当前研究是：在 **CoRP-RoboCasa** 上先建立可靠的闭环基线，再研究外观变化如何
> 经由视觉 FFN 通道和 resampler 影响动作，最后才判断是否需要选择性的通道或
> token-channel 干预。

当前最重要的不是继续加一个门控模块，而是依次证明：

1. baseline 真的会做任务；
2. 控制外观变化后，同一个 baseline 真的会退化；
3. 候选单元的变化真的传递到动作；
4. 选择性干预优于同预算随机干预和标准像素增广；
5. clean 能力没有明显被破坏。

这五件事没有完成前，不能宣称“通道门控有效”，也不能把某个 benchmark 上的
官方成绩直接移植成我们的结论。

---

## 1. 研究问题

### 1.1 问题定义

研究对象限定为：

> 当任务目标、语言指令、几何状态、机器人状态和正确动作保持不变，只改变颜色、
> 材质、纹理、光照等视觉外观时，机器人策略为什么退化，以及能否选择性地削弱
> 这种影响。

这里有两个不同的问题：

| 类型 | 例子 | 主要风险 |
| --- | --- | --- |
| 外观变化 | 厨房风格、墙纸、台面材质、颜色、光照 | 与物体身份、任务线索可能相关 |
| 信息退化 | 强模糊、严重噪声、过曝、目标被完全遮挡 | 不是“外观无关”，而是关键信息被删除 |

因此，不能把 `color_jitter`、自然 `style` 和 Gaussian blur 混成同一个
“泛化”指标。它们可以都作为压力测试，但解释必须分开。

实验条件也要分三层：

| 条件 | 怎么构造 | 能支持什么结论 |
| --- | --- | --- |
| 受控视觉扰动 | 同一物理状态、同一几何，只改颜色/亮度，或检查过的渲染属性 | 视觉输入变化与动作之间的关系 |
| 自然 style 泛化 | train/val 使用不同 `style_id` | 对未见场景风格及其关联变化的泛化 |
| 官方 target 迁移 | 官方 target scenes/objects | 多因素外部迁移，不是纯外观因果 |

固定 layout、只换 `style_id` 也不能自动保证物理交互条件不变。RoboCasa 的 style
可能同时改变柜门、把手、设备类型等属性；如果这些属性改变了抓取几何，结论就不再
是“只改外观”。因此，自然 style 实验需要记录关键物体/把手是否改变，受控
color/blur 实验则要单列。官方说明见
[RoboCasa Scenes](https://robocasa.ai/docs/build/html/assets/scenes.html)。

### 1.2 三个研究问题

| 问题 | 要回答什么 | 需要的证据 |
| --- | --- | --- |
| Q1：外观变化在哪里被编码？ | 哪些层、视角、token 位置的 FFN 响应发生变化？ | 同状态配对、逐层/逐视角激活记录、episode/frame 身份 |
| Q2：哪些变化真正传到动作？ | 响应变化中哪些会影响动作，哪些被 resampler 或后续计算消化？ | clean→shift 指定单元替换、反向替换、同规模随机对照、动作和闭环结果 |
| Q3：能否选择性干预？ | 共享通道、token 条件通道和随机干预的差异是否稳定？ | 同预算比较、clean 代价、shift 收益、标准像素增广对照 |

当前只读 FFN 采集器支持 Q1 的一部分。Q2 的主动替换和 Q3 的可部署方法还没有实现，
不能把激活差异、梯度大小或遮挡结果当成动作因果结论。

### 1.3 贡献边界

可以争取的贡献是：

1. 区分“外观敏感”与“动作影响”，而不是只画一张敏感热图；
2. 研究 CoRP 的视觉表示经过 resampler 压缩后，哪些信息被过滤、哪些继续影响动作；
3. 用机制证据决定干预粒度，而不是预先假定 token-channel mask 更先进。

以下内容本身不构成新意：

- “FFN 通道 + 机器人 + mask”这个组合名称；
- 把域分数和梯度分数相加后乘 mask；
- 把一维共享 mask 改成二维 token-channel mask；
- 只比无增广 baseline，却不比标准像素增广或随机同预算对照。

---

## 2. 为什么选 CoRP

### 2.1 模型结构

当前使用的 CoRP 是官方实现上的 RoboCasa 适配，不是官方 RoboCasa 训练复现。

| 组件 | 当前角色 |
| --- | --- |
| 视觉编码器 | DINOv2 ViT-S/14，224 输入，256 个 patch tokens/view |
| 语言编码器 | T5-Efficient-Tiny |
| 特征压缩 | 语言条件 FiLM + learned-query resampler，48 tokens/view |
| 动作生成 | 1-layer DiT，flow matching |
| 参数量 | 官方 CoRP 约 48.9M |
| 机器人接口 | 当前适配为 3 视角、9D state、未来 10 步 × 12D action |

选择它的原因不是“它是最新 benchmark 冠军”，而是它把视觉表示、压缩接口和动作
生成拆得相对清楚，便于做机制分析。

### 2.2 官方论文报告的能力

以下是 CoRP 论文作者报告的数字，不是本项目复现：

| 数据集 | 设置 | CoRP | 对照 |
| --- | --- | ---: | --- |
| LIBERO | 4 suites 平均 | 97.0% | DP 72.4%，π0.5 96.9%，OpenVLA-OFT 97.1% |
| RoboTwin 2.0 | Clean | 75.78% | DP 28.0%，ACT 29.7%，π0 46.4% |
| RoboTwin 2.0 | Randomized | 73.36% | DP 0.6%，ACT 1.7%，π0 16.3% |

论文中值得注意的消融：

| 消融 | 结果 | 解释边界 |
| --- | ---: | --- |
| DINOv2 预训练 + 微调 | 97.0% | 官方作者设置 |
| 冻结 DINOv2 | 77.2% | 冻结会损失动作相关信息 |
| 随机初始化 ViT-S/14 | 78.1% | 预训练本身重要，但不是充分条件 |
| 全部 patch tokens | 83.2% | 信息和容量不是越多越好 |
| 48-token resampler | 97.0% | 指令感知压缩在作者设置中有效 |

结论：

- CoRP 适合作为“视觉表示如何影响动作”的紧凑试验台；
- 在 CoRP 作者的实验设置中，预训练编码器、任务适配和压缩都产生了明显贡献；
- 不能把冻结视觉塔当成默认前提；
- CoRP 论文没有研究自然外观域泛化，其 RoboTwin Randomized 也不是零样本
  style 迁移，因此我们的研究问题仍然不同。

### 2.3 从上游复用什么

优先复用顺序固定为：

> 已确认来源的官方入口 → PyTorch、Torchvision、SciPy 等基础库 → 必要的项目适配。

| 环节 | 上游实现 |
| --- | --- |
| CoRP 模型/加载 | 官方 `RepresentationExtractor`、`DiTActionGenerator`、`load_policy` |
| RoboCasa 数据读取 | 官方 DP `LerobotDataset` |
| 环境 reset/回放 | 官方 `playback_dataset.reset_to` |
| 任务 horizon | 官方 `get_task_horizon` |
| 成功判定 | 官方 `info["success"]` |
| target 场景 | 官方 `RoboCasaGymEnv(split="target")` |
| 图像 resize/augmentation | Pillow + Torchvision functional |
| EMA | PyTorch `AveragedModel` |

官方 CoRP 仓库主要提供 LIBERO 推理/评测入口，RoboTwin 部分仍是 TODO；训练循环和
RoboCasa 适配是我们的项目代码。因此必须称为：

> **CoRP adapted to RoboCasa / our training adaptation**

不能称为“官方 CoRP RoboCasa 复现”。

### 2.4 相邻工作与新颖性边界

下面这些工作已经把“机器人内部激活/特征干预”从概念验证推进到了实际实验，
因此不能再把“机器人 + hook + 消融”本身当作新意。

| 工作 | 实际贡献 | 对本项目的约束 |
| --- | --- | --- |
| Not All Features Are Created Equal / Action Atlas（arXiv 2603.19233） | 对多种 VLA 做层/路径消融、激活注入、SAE 和闭环验证 | 必须把我们的对象、干预位置和证据说清楚，不能只换一个模型名 |
| Registers（arXiv 2506.08010） | 定位高范数异常 token 相关的少量 FFN 神经元，并转移激活 | 高范数/外观响应不等于有害依赖，也不等于应当删除 |
| ToaSt（ICML 2026） | 以 FFN channel selection 和结构化剪枝改善视觉模型精度/效率 | 目标主要是效率与分类/检测，不能直接推出机器人动作鲁棒性 |
| Failure Attribution（arXiv 2610.27277） | 区分“检测到 corruption”和“移除该 corruption 对当前 episode 是否有收益” | 发现外观相关特征，不等于压制它会改善当前动作 |
| Pathways of Visual Information Flow in VLMs（arXiv 2607.03358） | 显示视觉信息可能通过直接/文本中介路径流动，消融后模型会启用替代路径 | 消融结果只能说明模型“能怎样”，不能自动说明正常计算“实际怎样” |

这些结果共同支持一个更窄但更可信的研究目标：

> 不研究“域敏感特征就应该删掉”，而研究“哪些视觉变化真正造成当前动作损害，
> 以及在没有 clean 参考时，哪种干预能够修复它”。

---

## 3. 当前 V10 方案

### 3.1 冻结的任务面板

V10 pilot 使用 5 个 RoboCasa target Atomic-Seen 任务：

```text
OpenCabinet
OpenDrawer
TurnOffStove
TurnOnElectricKettle
TurnOnSinkFaucet
```

这不是官方 50-task leaderboard 复现，而是先验证训练、评测和机制链路的 pilot。

### 3.2 数据划分

| 数据部分 | 范围 | 用途 |
| --- | --- | --- |
| train | pretrain styles 11–50，episode residue `0–8` | 参数学习、normalizer |
| source holdout | pretrain styles 11–50，residue `9` | 训练期离线诊断 |
| source closed-loop pilot | pretrain styles 11–30，20 episodes/task | 当前方案筛查的固定子面板，不等于完整 11–50 |
| internal val | pretrain styles 51–60 | 预算、checkpoint、方法选择 |
| official target | 官方 `split="target"`，scene pairs `(1,1)...(10,10)` | 最终外部测试，封存 |

关键纪律：

1. target 不参与训练；
2. target 不参与超参数、checkpoint 和方法选择；
3. target 结果出现后，不改方法再把它称作 validation；
4. 不同 split 各自冻结 manifest；同一 split 在不同方法/seed 间共享；clean/shift
   配对实验共享初始场景和策略噪声，但不把 train、source、internal-val 混成同一批；
5. 不同 seed、不同完成步数、不同 split 的 checkpoint 不能混表。

### 3.3 训练预算

V10 pilot 的冻结配置是：

```text
steps: 30,000
batch: 96
lr: 5e-5
EMA: 0.999
grad clip: 1.0
precision: BF16
task aux: 0.01
seed: 42
```

这是项目 pilot 预算，不是官方预算：

- CoRP 论文报告 LIBERO 80k、RoboTwin 120k，batch 392；
- RoboCasa 官方 DP 基线为 250k steps、batch 192；
- π0/π0.5 在 RoboCasa leaderboard 上报告 75k steps；
- GR00T N1.5 报告 120k steps。

30k 是否足够只能由 source/internal-val 学习曲线和闭环成功率决定，不能用
“loss 下降了”代替。

### 3.4 评估指标

主指标：

```text
闭环 success rate
```

辅助指标：

```text
macro task MSE
micro MSE
per-task MSE
5 组动作误差
12 维动作误差
```

MSE 只能解释动作数值误差，不能代替任务完成率。

### 3.5 Go/No-Go 门槛

| Gate | 必须满足 | 不满足时 |
| --- | --- | --- |
| 0. 训练身份 | 数据、顺序、batch、lr、EMA、steps、seed 一致 | 不能归因于方法 |
| 1. baseline 能力 | clean internal-val 非零且不是偶然单次成功 | 先修训练/任务/评测 |
| 2. 条件区分度 | 同 manifest、同初始状态、同噪声下 clean→shift 有可解释退化 | 换扰动、任务或 split |
| 3. 主指标 | 用闭环 SR，不以 MSE 代替 | 补 SR |
| 4. 随机对照 | cross/selected 方法必须和同预算 random control 比较 | 不能归因于选择机制 |
| 5. clean 保持 | 方法不能在提高 shift SR 时大幅破坏 clean | 视为不可接受 trade-off |

`4/20`、`3/20` 一类数字只能做工程筛查，不应被写成统计显著性阈值。最终需要多
episode、多 seed 和配对区间。

---

## 4. 方法路线：机制发现 + 最小方法

### 4.1 先做只读机制审计

当前采集对象：

- 12 个 DINOv2 block 的原始输出；
- 每层 `mlp.act` 的 1536 维 FFN 中间激活；
- CLS 与 patch token 分开；
- view-major 与 sample-major 显式转换；
- 三视角、episode、frame 身份可追溯。

不应当直接做的事：

- 用单个 batch 的激活大小给通道排永久名次；
- 把 CLS 排名直接套到 patch；
- 把 ViT-L 的阈值/层号复制到 ViT-S/14；
- 用梯度大推断“删除后一定提升”；
- 把 clean 全量视觉输出恢复当作 FFN 因果证明。

如果保存全部 12 层、三视角、FFN 和 raw block 的 FP32 激活，单个观测时刻单条件
约为 71 MB，clean/shift 一对约 142 MB；1000 对就是约 142 GB。采集应优先流式
统计、只保留少量完整配对样本，不应默认把所有激活长期落盘。

选层、选单元和验证单元必须尽量分开：

```text
source 校准：选层、选候选单元
独立 episode：做指定干预
internal-val：检查能否迁移
target：继续封存到最终评估
```

如果只在同一批帧上挑“最特别的通道”再展示它们“最有效”，很容易得到偶然结果。

### 4.2 再做指定单元干预

真正需要的是：

```text
clean 运行
→ shifted 运行
→ 只把少量指定层的指定 FFN/token 单元替换回 clean 值
→ 检查动作是否恢复
→ 反向替换
→ 对相同单元做衰减/置零
→ 同规模随机替换
```

所有运行必须固定：

- task/instruction；
- simulator state；
- 图像预处理；
- policy seed；
- flow sampling noise；
- 动作 mask 和 horizon。

报告：

- 动作偏移；
- 动作组误差；
- 任务成功率；
- 表示是否出现异常；
- 替换的 token×channel 数量；
- clean、shift、恢复三种条件的结果。

需要特别区分两种结论：

1. **clean 激活替换成功**：说明这些单元承载了可恢复的信息；
2. **相同单元衰减/置零也成功**：才支持“压制这条路径有部署收益”。

如果只有 clean 替换有效、衰减无效，合理结论是“需要恢复信息”，不是继续设计
mask。clean 参考图像属于诊断条件，最终方法必须另外通过：

```text
推理时只允许当前图像、机器人状态和任务指令
不读取 clean 图像
不读取 GT action
不读取 test domain 标签
```

闭环实验还要区分“同一时刻”和“同一状态”。clean 与 shift 运行几步后可能已经
处于不同阶段；直接把 clean 第 50 步的激活补给 shift 第 50 步，可能是在注入另一条
轨迹或另一个场景的信息。可靠的最小设计是：

```text
当前 rollout 的当前状态
→ 生成 clean/shift 两个观测
→ 做单步或短窗口替换
→ 检查动作与任务结果
```

完整 clean 轨迹重放可以作为行为覆盖或恢复性正对照，但不能单独解释为外观因果。
Action Atlas 的跨任务注入还报告了更强的混杂：注入任务 A 的激活后，π0.5 可能
直接执行任务 A 的运动轨迹，并绑定到源场景的绝对 workspace 坐标。因此，任何
clean/shift 激活替换都必须同时核对场景、对象位置、轨迹阶段和策略噪声。

### 4.3 最后再讨论方法

只有机制证据表明共享整通道过粗，才有理由测试 token-conditioned 通道选择。
候选比较至少包括：

```text
无干预
标准 Torchvision 像素增广
共享通道选择
token-conditioned 通道选择
各自的同预算随机干预
无动作保护版本
动作保护版本
```

随机对照要尽量匹配层、视角、CLS/patch 类别、token 位置分布、编辑数量和幅度；
无法同时匹配的因素必须单列。消融后模型也可能启用替代路径，因此要用动作和闭环
结果验证，而不能只看“某个激活被删掉了”。

旧 statistical token mixing 已因负结果和 donor 不匹配退役，代码不再保留，
也不作为 V10 的备选路径。

在最小因果实验成立前，不扩大训练预算，也不把“更大的模型/更长训练”当作修复
机制问题的手段。

---

## 5. 开源数据集与 benchmark 选型

### 5.1 一张表看清楚

| 数据集/benchmark | 主要能力 | 外观/场景泛化条件 | 是否有闭环 | 对当前项目的角色 |
| --- | --- | --- | --- | --- |
| **RoboCasa365** | 365 厨房任务、2500 场景、大量示范 | 原生 style/layout 标签；官方 pretrain/target 场景分离 | 是 | **唯一主训练和主评估数据源** |
| **RoboCasa-Shift** | RoboCasa365 上的鲁棒性诊断 | 光照、相机、base、Gaussian blur，matched pairs | 是 | **最接近我们的评估协议，优先参考** |
| **LIBERO-Plus** | LIBERO 上 7 类扰动、10,030 tasks | layout、camera、robot、language、light、background、noise | 是 | 外部鲁棒性诊断，不作为 RoboCasa 主 split |
| **LIBERO** | 4 个标准 suite、单臂桌面 | 任务 suite，不是原生 style holdout | 是 | CoRP 上游能力验证和公开对照 |
| **RoboTwin 2.0** | 50 个双臂任务、强域随机化、10 万+ 轨迹 | Clean/Randomized，背景/光照/杂乱等 | 是 | 未来双臂扩展；不是当前外观因果实验 |
| **RoboDojo** | 42 仿真 + 18 真机任务 | 五个能力维度含 Generalization | 是 | 更后期综合评估，Isaac Sim 成本高 |
| **CALVIN** | 长时序语言条件操作 | 多环境但外观轴不细 | 是 | 不作为当前首选 |
| **SimplerEnv** | real-to-sim 策略评估 | real-sim 对齐 | 是 | 未来真机/仿真迁移参考 |
| **DROID / Open X-Embodiment** | 大规模真机/跨实体数据 | 无统一外观 holdout 和闭环 target pair | 多数否 | 预训练数据参考，不直接替代当前 benchmark |

### 5.2 为什么 RoboCasa365 是主数据源

RoboCasa365 的核心优势不是“任务最多”，而是它同时提供：

1. `style_id` 和 `layout_id`；
2. 同一任务下的多场景采样；
3. pretrain styles/layouts 11–60 与 target 1–10 的官方分离；
4. 可指定 scene pair 的 reset；
5. 官方 `info["success"]` 和任务 horizon；
6. LeRobot 格式和官方 DP reader。

官方 README 说明：

- 365 tasks；
- 2500+ kitchen scenes；
- 600+ 小时人类示范；
- 1600+ 小时自动生成示范；
- 代码 MIT，assets/datasets CC BY 4.0。

官方论文的场景多样性消融也支持这种选择：

| 预训练场景数 | zero-shot target SR |
| --- | ---: |
| 5 scenes（5 layouts × 1 style） | 29.6% |
| 25 scenes（5 layouts × 5 styles） | 39.6% |
| 2500 scenes（50 layouts × 50 styles） | 44.7% |

这说明场景多样性本身会显著影响下游泛化。也说明我们只用 20 个 style 的早期
pilot 不能代表官方 RoboCasa365 训练规模。

### 5.3 RoboCasa365 不能单独证明“外观因果”

官方 target split 同时改变了：

- scene layout；
- style；
- object instances。

所以：

> target 结果适合作为外部迁移和最终压力测试，不适合单独解释“外观变化导致了
> 多少动作退化”。

任务划分与场景划分必须分开写：

- V10 的 5 个 Atomic target 任务仍是同一组 `OpenCabinet/OpenDrawer/...` 任务；
- 变化主要来自 target scenes、objects 和 scene/style 分布；
- 如果以后使用 Composite-Unseen，才涉及任务组合或任务级零样本问题；
- 不能因为使用 `split="target"` 就自动声称“任务组合发生了变化”。

主实验必须在 pretrain 内部构造：

```text
same task / same layout / same simulator state / same object poses
only verified visual attributes changed
physical geometry and state remain unchanged
```

如果使用自然 style，必须先核验柜门、把手、设备类型和物体几何没有改变，否则只能
表述为“自然 style 及其关联变化”，不能称作纯外观干预。或者使用已有 matched-pair
协议。source holdout 与 internal val 是组级比较，不是逐 episode 配对；
clean 与 color_jitter 才是同场景配对。

### 5.4 RoboCasa-Shift

官方论文：[arXiv 2610.27861](https://arxiv.org/abs/2610.27861)

官方代码：[HaochengWang23/RoboCasa-Shift](https://github.com/HaochengWang23/RoboCasa-Shift)

它基于 RoboCasa365，特点是：

- 4 个通过 clean gate 的 long-horizon tasks；
- 4 个单因素 shift：illumination、camera orientation、robot base、Gaussian blur；
- 每因素 3 个严重度；
- clean 与 shifted 共享 task、episode、simulator state 和 policy seed；
- 4 × 20 × 13 = 1,040 个正式 rollout；
- 同时报告 SR、paired outcome transition 和 stage-aware progress。

GR00T N1.5 的结果说明了“聚合 SR 不够”：

| Shift | L1 ΔSR | L2 ΔSR | L3 ΔSR |
| --- | ---: | ---: | ---: |
| Illumination | 0.012 | -0.025 | 0.025 |
| Camera | 0.000 | 0.188 | 0.138 |
| Robot base | 0.075 | 0.150 | 0.263 |
| Gaussian blur | 0.188 | 0.513 | 0.738 |

这里的 `ΔSR = clean SR - shifted SR`，正值表示退化。几项重要边界：

- 它只评估 GR00T N1.5，不是普遍排行榜；
- illumination 并不单调，强 shift 不一定更强退化；
- Camera L2 与 Blur L1 的聚合退化相同，但失败发生阶段不同；
- 它目前不是我们的训练数据或正式 v10 split；
- 它非常适合作为我们的评估设计参考。

### 5.5 LIBERO 与 LIBERO-Plus

官方 LIBERO 论文：[arXiv 2306.03310](https://arxiv.org/abs/2306.03310)

LIBERO-Plus 论文：[arXiv 2510.13626](https://arxiv.org/abs/2510.13626)

官方代码：[sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)

LIBERO 的价值：

- CoRP 官方权重和官方评测入口现成；
- 适合验证模型加载、动作接口和标准能力；
- CoRP 论文报告 97.0%，但这个数字是其训练设置下的结果。

LIBERO-Plus 的价值：

- 7 个扰动维度；
- 10,030 tasks；
- 包含 background texture、light、sensor noise、camera、robot、layout、language；
- 明确显示 VLA 的 benchmark 高分不能代表真实鲁棒性；
- 官方 README 报告了不同模型的分类别成绩。

官方 README 中的代表性成绩（%）：

| 模型 | Camera | Robot | Background | Noise | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA | 0.8 | 3.5 | 34.8 | 15.2 | 15.6 |
| π0 | 13.8 | 6.0 | 81.4 | 79.0 | 53.6 |
| OpenVLA-OFT+ | 92.8 | 30.3 | 93.9 | 89.3 | 79.6 |

这张表最重要的信息是“不同扰动轴的难度完全不同”，而不是模型总排名：
同一模型可能对背景较稳，却对相机或机器人初始状态极其敏感。不同列是不同
task family，不能把某一列的绝对值直接和另一列比较。

但 LIBERO-Plus 不是当前 RoboCasa 主实验：

- 动作空间、环境、相机和资产不同；
- 其“original/clean”并不等同于标准 LIBERO clean；
- 七类扰动是不同 task family，不能把类别间 SR 直接横向比较；
- 它适合外部鲁棒性验证，不适合直接作为我们 FFN 通道的配套数据。

我们的历史 smoke 只作为动机：

- 官方 CoRP（LIBERO 权重）在 LIBERO-Plus 部分原生条件上曾得到 14/14；
- 强 color_jitter 曾降到 3/14；
- Gaussian blur σ=2 曾降到 137/560，σ=4 降到 0/560；
- 样本小、clean 不是标准 LIBERO clean，不能和论文 97% 直接比较。

### 5.6 RoboTwin 2.0

官方代码：[RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

官方论文：[arXiv 2506.18088](https://arxiv.org/abs/2506.18088)

特点：

- 50 个双臂任务；
- over 100,000 条预采集轨迹；
- Clean 与 Randomized；
- 强随机化包括背景、光照、杂物、桌面高度等；
- MIT license。

它适合：

- 双臂操作和强视觉随机化；
- 未来验证方法是否能跨 embodiment；
- 复用官方 XPolicyLab 评测栈。

它不适合当前阶段直接替代 RoboCasa：

- 双臂动作和当前 12D 单臂接口不同；
- Randomized 是训练分布内的视觉变化，不自动等于零样本真实外观迁移；
- 预采集数据不提供像 RoboCasa `style_id` 那样完整的外观域标签；
- 需要重做数据读取、动作映射和闭环评测。

CoRP 论文的 RoboTwin 结果应这样理解：

```text
训练：2,500 clean + 25,000 randomized demonstrations
→ Clean/Randomized 是同一训练分布中的两组评测
→ 不能把 Randomized 的 73.36% 写成零样本外观泛化
```

### 5.7 RoboDojo

官方代码：[RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)

官方论文：[arXiv 2607.04434](https://arxiv.org/abs/2607.04434)

特点：

- 42 个仿真任务；
- 18 个真机任务；
- 5 个能力维度：Generalization、Precision、Long-Horizon、Memory、Open；
- 与 XPolicyLab 共享评测服务；
- 基于 Isaac Sim，环境成本明显高于 RoboCasa。

它适合作为“论文后期证明方法不是只在一个 benchmark 上有效”的外部评估，
不适合现在作为第一轮训练数据。

---

## 6. 官方报告成绩怎么读

### 6.1 不同 benchmark 不能直接排名

| 来源 | 指标 | 重要边界 |
| --- | --- | --- |
| CoRP 论文 LIBERO | 4 suites 平均 SR | 官方作者训练设置 |
| CoRP 论文 RoboTwin | Clean/Randomized SR | Randomized 训练分布内 |
| RoboCasa leaderboard | Atomic/Composite Seen/Unseen SR | 官方 target benchmark，训练和评测规模更大 |
| RoboCasa 论文场景消融 | zero-shot target SR | 5/25/2500 场景，受控比较 |
| LIBERO-Plus | 7 类扰动下的 SR | 单臂 LIBERO，扰动 task family 不可直接横比 |
| RoboCasa-Shift | paired ΔSR 和 stage failure | 4 tasks、GR00T N1.5、受控单因素 |

这些数字不能拼成一个统一排行榜：

- 数据集不同；
- 任务难度不同；
- action/embodiment 不同；
- 训练数据量和预算不同；
- clean 的定义不同；
- 有的是 zero-shot，有的是 in-distribution；
- 有的是单因素配对，有的是不同 task family。

### 6.2 RoboCasa365 leaderboard 的代表性快照

以下是本地固定 leaderboard 仓库中截至 2026-11-06 的部分记录。不同提交使用
RoboCasa 1.0.0/1.0.1，不能做严格跨版本比较：

| 模型 | Atomic-Seen | Composite-Seen | Composite-Unseen | 备注 |
| --- | ---: | ---: | ---: | --- |
| Diffusion Policy | 15.7 | 0.2 | 1.3 | 250k steps，batch 192 |
| π0 | 34.6 | 6.1 | 1.1 | 75k，batch 64 |
| π0.5 | 39.6 | 7.1 | 1.2 | 75k，batch 64 |
| GR00T N1.5 | 50.7 | 14.8 | 2.7 | 120k，batch 128，v1.0.1 |
| Awomo-0.5I | 86.5 | 58.8 | 45.2 | 当前快照中最高之一，v1.0.1 |

最重要的观察不是“谁的模型更大”，而是：

> 从 Atomic-Seen 到 Composite-Seen、Composite-Unseen，成功率通常明显下降。
> 这证明目标任务组合和未见任务的泛化仍然困难，但它不等于外观-only 泛化。

### 6.3 RoboCasa 官方 robustness 的边界

RoboCasa365 官方论文的 robustness 附录测试了：

- novel language；
- novel joint angles；
- novel base pose；
- novel camera pose。

作者报告模型通常对语言变化较稳，但相机、初始关节和 base pose 会造成退化。

这没有覆盖：

- style/texture 的自然外观泛化；
- color jitter；
- 材质替换；
- 同场景 only-style 反事实。

因此我们的工作不是重复官方 robustness 表，而是补：

> 外观变化是否通过视觉 FFN/resampler 影响动作，以及能否选择性干预。

---

## 7. 我们已有的实验证据

### 7.1 当前本地工程验证

已验证：

- CoRP 官方 release 权重可以严格加载；
- 三视角、CLS/patch 顺序和原始视觉输出一致性有回归；
- DINOv2/T5/resampler/DiT 的适配边界有测试；
- FFN 只读 hook 不改变原始输出；
- 数据 reader、动作接口、图像处理、EMA、断点恢复和输出锁有回归；
- 当前 V10 协议、target schedule、checkpoint training origin 和 SR 聚合有限制；
- 最新完整回归为 139/139 通过。

这些是工程和协议证据，不是真实 GPU 闭环成功证据。

### 7.2 V9 CloseDrawer pilot

旧 v9 的单任务、单 seed、20 episodes/split 结果：

| 方法 | clean source | clean val | jitter source | jitter val |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6/20 | 6/20 | 2/20 | 2/20 |
| random_stats | 8/20 | 10/20 | 2/20 | 2/20 |
| cross_style_stats | 9/20 | 10/20 | 2/20 | 0/20 |

能得到的结论：

- baseline 不是完全 0，至少在 CloseDrawer 的 source 条件下有少量成功；
- 强 color_jitter 下出现明显下降；
- cross_style 没有超过 random control；
- 旧 statistical mixture 方法叙事应停止；
- 这组数字不能支持最终方法结论。

不能得到的结论：

- 不能说 cross_style 有效；
- 不能说方法在 RoboCasa 上泛化；
- 不能把 6/20 当作稳定 baseline；
- 不能把单任务、单 seed、20 episodes 当作最终显著性证据。

### 7.3 旧 LIBERO-Plus smoke

旧实验使用官方 CoRP LIBERO checkpoint，不是当前 RoboCasa 权重：

| 条件 | 记录结果 | 用途 |
| --- | ---: | --- |
| 原生 7 类扰动的小样本 | 14/14 | 说明某些条件未观察到退化 |
| strong color_jitter | 3/14 | 外观压力可能有害 |
| Gaussian blur σ=2 | 137/560 | 低层信息退化明显有害 |
| Gaussian blur σ=4 | 0/560 | 严重信息丢失 |

这些结果只能作为研究动机和方法风险提示：

- clean 不是标准 LIBERO clean；
- episode 和 task 采样很少；
- 与当前 CoRP-RoboCasa 不同模型、不同环境；
- 不能把 blur 结果直接写成当前 V10 结论。

### 7.4 当前总结

```text
工程链路：基本可用
官方 CoRP 能力：有公开证据
当前 RoboCasa V10：尚未形成最终方法结论
FFN 机制：目前是研究目标，不是已证实根因
通道方法：没有足够证据进入论文正向结论
```

---

## 8. 建议的下一步

### 阶段 A：baseline gate

1. 用 V10 frozen manifests 训练 30k baseline；
2. 先看 clean source/internal-val 的闭环 SR；
3. 如果 baseline 接近 0，先修训练预算、任务难度、动作接口和 horizon；
4. 不启动任何未通过独立机制 gate 的方法比较。

### 阶段 B：条件区分度

1. 同一 checkpoint、manifest、初始状态和 policy noise；
2. 只改图像外观条件；
3. 检查 clean→color_jitter、clean→blur 的配对结果；
4. 记录 success→failure 和 failure→success 的 episode 转移，不只看平均值；
5. 若 clean 与 shift 没有差异，先换扰动或任务，不继续堆方法。

### 阶段 C：机制定位

1. 采集同一状态的 clean/shift FFN 激活；
2. 按层、视角、CLS/patch、token 位置记录；
3. 选择少量候选；
4. 做 clean→shift 替换、反向替换和同规模随机对照；
5. 对相同单元做衰减/置零，区分“恢复信息”和“压制路径”；
6. 同时观察 resampler 前后和动作输出；
7. 选层/选单元与验证 episode 分离，target 继续封存。

### 阶段 D：最小方法

只有 B/C 有稳定证据后，才比较：

```text
standard pixel augmentation
random channel intervention
shared channel selection
token-conditioned channel selection
```

### 阶段 E：扩规模

方向成立后再扩展：

```text
5 tasks → 多个 Atomic/Composite task
1 seed → 3 seeds
20 episodes → 50 episodes/task/split
internal val → official target
```

target 仍然只在方法、预算和 checkpoint 全部冻结后才碰。

---

## 9. 主要风险

| 风险 | 后果 | 处理 |
| --- | --- | --- |
| baseline clean 接近 0 | 所有方法比较都无意义 | 先修 baseline |
| style 变化没有造成退化 | 通道选择缺少问题空间 | 改条件或转为负结果/机制研究 |
| color_jitter 删除任务线索 | 不能解释成语义外观泛化 | 与 blur/信息退化分开 |
| target 同时换 style/layout/object | 无法归因外观 | 只用 target 做最终外部测试 |
| 只有 MSE 改善 | 不能证明任务能力 | 主结论必须用 SR |
| 没有 random control | 无法归因于选择机制 | 同预算随机对照 |
| 梯度/attention 重要性只是启发式 | 不能称为因果通道 | 用指定干预和闭环结果验证 |
| clean 替换有效但压制无效 | 不能证明 mask/门控有部署收益 | 转为信息恢复或继续定位 |
| 闭环不同轨迹被当作同状态 | 注入的是状态/轨迹信息 | matched state 或单步反事实 |
| 单任务、单 seed、20 episodes | 结果不稳定 | 分阶段扩 task/seed/episode |
| 训练预算远低于官方 | 不能和 leaderboard 直接比较 | 标明 pilot，不混表 |

---

## 10. 结论模板

当前可以写：

> 我们搭建了一个基于 CoRP 和 RoboCasa365 的可追溯实验框架，用于研究外观变化
> 如何通过视觉 FFN 和 resampler 影响机器人动作，并通过 source/internal-val
> 先筛选方法、最后封存 official target。

当前不能写：

> 我们提出了有效的通道门控，在 RoboCasa 上显著提升了外观泛化。

只有以下条件同时满足，才能写正向方法结论：

```text
baseline clean 可用
外观条件有配对退化
指定干预影响动作
选择方法超过 matched random
clean 没有明显崩溃
多 task/seed/episode 可重复
最后在封存 target 上报告外部结果
```

---

## 11. 官方来源

### 模型与训练

- [CoRP 官方代码](https://github.com/corp-policy/CoRP/tree/52445740aa9517ae396cc8fb14bae8fe5922f118)
- [CoRP 论文：Compact Robot Policies Need Fine-Grained Visual Representations](https://arxiv.org/abs/2610.08183)

### RoboCasa

- [RoboCasa 官方代码](https://github.com/robocasa/robocasa)
- [RoboCasa365 论文](https://arxiv.org/abs/2603.04356)
- [RoboCasa365 Leaderboard](https://robocasa.ai/leaderboard.html)
- [RoboCasa365 leaderboard 代码](https://github.com/robocasa-benchmark/leaderboard)
- [RoboCasa Scenes 文档](https://robocasa.ai/docs/build/html/assets/scenes.html)
- [RoboCasa-Shift 论文](https://arxiv.org/abs/2610.27861)
- [RoboCasa-Shift 代码](https://github.com/HaochengWang23/RoboCasa-Shift)

### LIBERO

- [LIBERO 论文](https://arxiv.org/abs/2306.03310)
- [LIBERO-Plus 论文](https://arxiv.org/abs/2510.13626)
- [LIBERO-Plus 官方代码](https://github.com/sylvestf/LIBERO-plus)

### 其他 benchmark

- [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
- [RoboTwin 2.0 论文](https://arxiv.org/abs/2506.18088)
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
- [RoboDojo 论文](https://arxiv.org/abs/2607.04434)
- [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)

### 机制分析与相邻工作

- [Action Atlas / Not All Features Are Created Equal（arXiv 2603.19233）](https://arxiv.org/abs/2603.19233)
- [Action Atlas 代码](https://github.com/CWRU-AISM/action-atlas)
- [Failure Attribution（arXiv 2610.27277）](https://arxiv.org/abs/2610.27277)
- [Pathways of Visual Information Flow in VLMs（arXiv 2607.03358）](https://arxiv.org/abs/2607.03358)
- [Registers（arXiv 2506.08010）](https://arxiv.org/abs/2506.08010)
- [ToaSt（ICML 2026）](https://proceedings.mlr.press/v306/moon26b.html)

### 项目内部文档

- [当前实验协议](EXPERIMENT_PROTOCOL.md)
- [V10 协议](V10_PROTOCOL.md)
- [通道干预上游映射](CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md)
- [只读服务器验收](READONLY_SERVER_VALIDATION.md)
