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

## 8. Three.js 交互式3D

当前可用的开源自动3D工具中，`modelviz-ai` 可以直接从 PyTorch `nn.Module`
生成 Three.js 交互式 HTML。其PyTorch parser基于forward hook，只记录leaf module
和tensor shape；对 `Representation` / `PrefixCache` dataclass仍需tensor-only wrapper。

本次生成：

```text
audit/corp_model_visualization_3d/corp_3d.html
audit/corp_model_visualization_3d/corp_3d_screenshot.png
```

页面实际渲染验证通过，支持：

```text
拖拽旋转
滚轮缩放
Shift+拖拽平移
图层 hover/tooltip
```

抓到192个leaf module，覆盖DINO block、T5 block、resampler、state encoder
和DiT动作生成器。由于层数很多，需要缩放和平移查看完整路径。

该HTML通过unpkg加载Three.js 0.160；首次打开需要网络。若后续要离线分享，
应把Three.js和addons文件下载到本地并修改import map。

其他核对过的工具：

```text
3dNN_Visualiser (MIT): PyTorch exporter + React/Three.js + FastAPI/Docker；
  支持.nn3d交互浏览，但部署成本更高。

TensorSpace.js: 成熟3D WebGL前端，但主要面向Keras/TensorFlow.js，
  不能直接读取当前PyTorch模块树。

PlotNeuralNet (25k stars): TikZ 3D层叠图，适合论文最终排版；
  不是自动PyTorch计算图。
```
