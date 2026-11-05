# 85 - CoRP模型Torchview可视化

日期：2026-11-05

## 1. 工具

真实 PyTorch 结构：

```text
torchview 0.2.7
torchinfo 1.8.0
graphviz Python 0.21
Graphviz dot 16.1.0
modelviz-ai 0.1.0 (Three.js 交互式3D)
Homebrew /opt/homebrew
```

Homebrew 使用 USTC 镜像安装。生成图使用原始官方 `corp_libero.pt`，
并通过 `load_corp_robocasa` 加载当前 v5 初始化策略。

## 2. 绘图入口

脚本：

```text
scripts/visualize_corp.py
scripts/visualize_corp_3d.py
```

命令：

```bash
python scripts/visualize_corp.py \
  --depth 2 \
  --out audit/corp_model_visualization_v4

python scripts/visualize_corp_3d.py \
  --out audit/corp_model_visualization_3d
```

Torchview 对 `Representation` dataclass 和 `PrefixCache` dataclass 的追踪不稳定。
脚本使用 tensor-only visualization wrapper，直接调用官方子模块并返回 tensor。
wrapper 只用于绘图，不改变模型代码、权重或前向语义。

## 3. 输出

总览：

```text
audit/corp_model_visualization_overview/
```

细节：

```text
audit/corp_model_visualization_v4/
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

这些生成物体积约 15 MB，通过 `.gitignore` 保留在本地，不提交 Git。

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
  libero_shared_plus_aligned_7d_action_head_v1

copied: 257
random: 7
```

随机项：

```text
state_encoder.projection.weight
task_aux_head.norm.*
task_aux_head.mlp.*
```

动作头前 7 维复用官方 LIBERO 权重，新增 5 维为零。

## 7. 使用边界

- Torchview 图用于核对真实模块、连接与 shape。
- 图很大，论文中的“官方 CoRP vs RoboCasa 适配”仍应使用 draw.io/Mermaid 重画。
- 自动图不能替代模型数值一致性测试。
- 本次只做本地可视化，未启动训练或连接服务器。

## 8. 交互式3D方案已弃用

验证过 `modelviz-ai` 的 Three.js 交互式3D页面，但192个leaf module在三维空间中
可读性差，不适合当前目的。该临时页面、脚本和截图不保留。

下一步改为2.5D层叠图：优先复用 [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
的官方TikZ宏与示例，用层块表达：

```text
三视角图像 → DINOv2 → Resampler
语言编码 → 视觉语言拼接 → State
→ DiT prefix/action → 12D velocity
```

2.5D图只服务于架构阅读和论文排版，不声称是真实计算图；真实模块、连接与shape仍以
Torchview图为准。

已使用 PlotNeuralNet upstream commit
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
项目脚本只定义CoRP架构块、维度和连接。编译使用 Tectonic 0.17.0。
