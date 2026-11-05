# 85 - CoRP模型可视化

日期：2026-11-05

## 1. 工具

真实 PyTorch 结构：

```text
torchview 0.2.7
torchinfo 1.8.0
graphviz Python 0.21
Graphviz dot 16.1.0
Homebrew /opt/homebrew
```

2.5D 架构图复用 PlotNeuralNet 官方 PyTikZ 入口：

```text
https://github.com/HarisIqbal88/PlotNeuralNet
commit 613be1f5d769718d315dabae823b3d376b26689c
Tectonic 0.17.0
```

Homebrew 使用 USTC 镜像安装。生成图使用原始官方 `corp_libero.pt`，
并通过 `load_corp_robocasa` 加载当前 `corp_robocasa_v6` 随机12维动作接口。

## 2. 绘图入口

脚本：

```text
scripts/visualize_corp.py
scripts/visualize_corp_25d.py
```

命令：

```bash
python scripts/visualize_corp.py \
  --depth 2 \
  --out audit/corp_model_visualization_v4

PATH="/opt/homebrew/bin:$PATH" .venv/bin/python \
  scripts/visualize_corp_25d.py \
  --compile \
  --out audit/corp_model_visualization_25d
```

Torchview 对 `Representation` dataclass 和 `PrefixCache` dataclass 的追踪不稳定。
脚本使用 tensor-only visualization wrapper，直接调用官方子模块并返回 tensor。
wrapper 只用于绘图，不改变模型代码、权重或前向语义。

2.5D 图只定义 CoRP 的架构块、维度和连接，渲染由 PlotNeuralNet 官方
`pycore.tikzeng`、`Box`、`Connection` 宏完成，没有自建图形引擎。

## 3. 输出

总览：

```text
audit/corp_model_visualization_overview/
```

细节：

```text
audit/corp_model_visualization_v4/
```

2.5D 架构图：

```text
audit/corp_model_visualization_25d/
```

每套包含：

```text
corp_full.svg / pdf / png
corp_representation.svg / pdf / png
corp_action_generator.svg / pdf / png
metadata.json
torchinfo.txt
Graphviz .gv
```

2.5D 目录包含：

```text
corp_25d.tex
corp_25d.pdf
corp_25d.png
```

生成物通过 `.gitignore` 保留在本地，不提交 Git。

## 4. 核对到的关键 shape

三视角输入：

```text
每个视角 [1,3,224,224]
拼接后    [3,3,224,224]
```

视觉编码：

```text
DINOv2 输出 [3,256,384]
每个视角    [1,256,384]
```

Resampler：

```text
每个视角 256 patch tokens
→ 48 learned queries
三视角共 144 tokens × 768
```

语言：

```text
[1,32,256] → [1,32,768]
```

state：

```text
[1,9] → [1,768]
```

拼接后的视觉语言状态前缀：

```text
144 visual + 32 language + 1 state = 177 tokens
[1,177,768]
```

动作生成：

```text
noisy action [1,10,12]
→ action tokens [1,10,768]
→ 单层 DiT prefix/action attention
→ velocity [1,10,12]
```

图中动作生成器展示一次 velocity forward，不展开训练时的完整 flow 采样循环。

## 5. 参数量

```text
visual encoder:        22,056,192
language encoder:      11,570,816
resampler:              2,921,536
state encoder:              8,448
action generator:      12,178,252
task aux head:            214,077
total:                 48,949,321
```

任务分类头只在训练期使用，不在推理动作路径中。

## 6. 当前初始化

加载报告：

```text
initialization_policy:
  libero_shared_random_12d_action_interface_v1

copied: 254
random: 10
```

随机项：

```text
state_encoder.projection.weight
action_generator.action_projection.weight
action_generator.velocity_projection.weight
action_generator.velocity_projection.bias
task_aux_head.norm.*
task_aux_head.mlp.*
```

RoboCasa action identity 与 LIBERO action min-max 的数值语义不一致，
因此 12 维动作接口整体随机初始化，不做前 7 维硬拷贝。

## 7. 使用边界

- Torchview 图用于核对真实模块、连接与 shape。
- 图很大，论文中的“官方 CoRP vs RoboCasa 适配”仍应使用 draw.io/Mermaid 重画。
- 自动图不能替代模型数值一致性测试。
- 本次只做本地可视化，未启动训练或连接服务器。

## 8. 2.5D方案

交互式3D页面在192个leaf module下可读性差，不适合当前目的。
该临时页面、脚本、依赖和截图均不保留。

当前采用 [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
官方 TikZ 宏与示例生成2.5D层叠图：

```text
三视角图像 → DINOv2 → Resampler
语言编码 → 视觉语言拼接 → State
→ DiT prefix/action → 12D velocity
```

2.5D图只服务于架构阅读和论文排版，不声称是真实计算图；真实模块、连接与shape仍以
Torchview图为准。

使用 PlotNeuralNet upstream commit
`613be1f5d769718d315dabae823b3d376b26689c` 生成：

```text
scripts/visualize_corp_25d.py
audit/corp_model_visualization_25d/corp_25d.tex
audit/corp_model_visualization_25d/corp_25d.pdf
audit/corp_model_visualization_25d/corp_25d.png
```

图中包含：

```text
3 views -> DINOv2 -> Resampler -> Visual prefix
T5-tiny ----------------------┐
State token ------------------┼-> Fusion prefix 177 x 768
                              └-> 1-layer DiT -> Velocity 10 x 12
DiT prefix -> Task head 61-way (train only, dashed)
```

渲染由 PlotNeuralNet 官方 `pycore.tikzeng`、`Box`、`Connection` 宏完成；
项目脚本只定义CoRP架构块、维度和连接。左侧“How to read”说明面板使用标准
TikZ `node` 绘制；PlotNeuralNet 官方 README 把通用 legend 列为 TODO，因此
这里没有另造图形引擎，只补一个与架构图同源的单页注释层。编译使用 Tectonic 0.17.0。

输出图使用显式白底黑字，兼容深色主题查看器；PNG 输出尺寸为 `2320 x 1405`。

## 9. 2.5D图怎么读：由浅入深

### 9.1 一句话

CoRP-RoboCasa 接收三路相机画面、语言指令和机器人状态，先把它们压缩成
一组固定长度的 token，再用 DiT 的 flow matching 生成一段 10 步、
每步 12 维的动作。

### 9.2 三个层次

先把它看成三段，而不是一堆层：

```text
观测输入
  -> 表征：画面/语言/状态变成 token
  -> 决策：token 条件化动作生成
  -> 输出：未来 10 步动作
```

对应到图中：

```text
视觉链：3 views -> DINOv2 -> Resampler -> visual prefix
辅助输入：T5 language + robot state
融合：visual + language + state = fusion prefix
动作：fusion prefix -> 1-layer DiT -> velocity -> action chunk
训练辅助：DiT prefix -> task aux（不参与推理动作）
```

### 9.3 模块、输入、输出和作用

下表中的 `B` 是 batch size，`H` 是动作时域，`D` 是隐藏维度。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 3 路相机 | 每路 `[B,3,224,224]` | 拼接后 `[3B,3,224,224]` | 同时提供左右环境视角和腕部视角 |
| DINOv2 ViT-S/14 | `[3B,3,224,224]` | `[3B,256,384]` | 将每张图切成 256 个 patch token；它编码局部视觉特征，不直接输出动作 |
| Resampler | 每视角 256 个 patch、48 个 learned query、语言 FiLM 条件 | 每视角 `[B,48,768]`，三视角拼接为 `[B,144,768]` | 把大量 patch 压缩成固定数量的视觉 token，并把视觉宽度统一到 768 |
| T5 language encoder | 指令文本，最大 32 token | token `[B,32,768]`，池化表示 `[B,768]` | 提供任务语义；池化表示还用于 Resampler FiLM 和 DiT 的 adaLN 条件 |
| State encoder | 9 维机器人状态 `[B,9]` | `[B,1,768]` | 注入末端执行器相对位置、相对旋转和夹爪状态 |
| Fusion prefix | 视觉 144 + 语言 32 + 状态 1 | `[B,177,768]` | 这是感知与动作生成之间的统一接口，177 不是随意数字 |
| 1-layer DiT | noisy action `[B,10,12]`、时间 `tau`、177-token prefix | velocity `[B,10,12]` | 学习从噪声到动作段的向量场；推理时用 Euler 积分生成动作 |
| Velocity head | DiT 的 768 维动作 token | `[B,10,12]` | 将每个动作 token 投回 12 维动作空间 |
| Task aux head | 选定的 DiT prefix hidden states | 61 类任务 logits | 只在训练时提供辅助分类约束；虚线表示它不进入推理动作路径 |

### 9.4 高维度理解

这张图真正表达的是三个分离：

1. **感知与决策分离**：DINOv2 和 Resampler 负责“看见并在固定预算内表达”，
   DiT 负责“基于这些表达生成动作”。视觉特征变多，不代表动作路径会直接变长。
2. **共享接口与任务专用头分离**：`177 x 768` 是共享 prefix；动作头输出连续动作，
   task aux 头输出离散任务类别。后者是训练辅助，不是部署时必须存在的分支。
3. **表示学习与动作生成分离**：task aux 让视觉 prefix 更容易保留任务相关信息，
   但最终控制信号仍由 DiT velocity 和 12 维动作接口产生。

另外，`10 x 12` 不是“10 个动作类别”：

```text
10 = action horizon，未来动作段长度
12 = 单个动作的维度
```

RoboCasa 的 12 维动作接口包含 base motion、control mode、末端位置、
末端旋转和夹爪控制。动作段通过 flow matching 生成，推理时从噪声开始，
做固定步数的 Euler 更新，而不是一次前向直接回归一个动作。

### 9.5 看图顺序

建议按下面的顺序读：

```text
1. 从左到右看视觉链：图像 -> patch -> 压缩后的视觉 token
2. 看 T5 和 State 如何汇入 Fusion
3. 看 Fusion 的 177 个 token 如何条件化 DiT
4. 看 DiT 输出 velocity，并经过动作积分得到 10 x 12 动作
5. 最后看虚线 task aux：只用于训练，不用于推理
```

这张图是架构阅读图，不声称是完整计算图。真实模块连接、参数和运行时 shape
以 Torchview 输出为准。
