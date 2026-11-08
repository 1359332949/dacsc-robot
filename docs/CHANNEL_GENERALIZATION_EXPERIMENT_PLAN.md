# CoRP 视觉通道与泛化实验计划

更新时间：2026-11-08

状态：研究计划与实验设计，不是方法结果，不代表通道干预已经有效。

本文件把当前会话中关于泛化、数据集、初始化、base/camera 和通道干预的结论收敛成一个
可执行计划。当前执行身份以 [V10_PROTOCOL.md](V10_PROTOCOL.md)、
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) 和当前 checkpoint/result 为准。

---

## 0. 最新证据与当前结论

### 已经完成的基础验证

| 项目 | 结果 | 说明 |
| --- | --- | --- |
| 原始 `OpenDrawer/source/clean` | `0/20` | 使用 `layout 11`，该 layout 不在 OpenDrawer 训练 split |
| seen smoke：`layout31/style12/reset` | `0/5`，全部 750 步 | 不是程序异常 |
| 训练 episode 初始 state | `4/5` 成功 | 模型具备可重复的 OpenDrawer 能力 |
| 官方 demonstration action replay | 第 155 步成功 | action、state、env、success 链通过 |
| reset 失败 trace | 750 步失败，gripper 从未闭合 | 失败发生在接近/抓取之前 |
| 训练成功 trace | 第 120 步 gripper 闭合，约 164/176 步成功 | 成功轨迹会接近、抓取、拉开 |

### 目前最重要的归因

最早怀疑的 base-only 解释已经被部分排除：

- 训练 `ep24` 使用 `style20 + stack_2_right`，成功；
- reset episode0 使用相近的 world base/camera 和同一个目标抽屉，但 `style12`，失败；
- 当前检查没有发现单靠 base/camera 位置就能解释该受控失败；
- 10 个 reset 样本中，style-target-drawer 组合均未在 train split 出现过；
- 因此当前更准确的候选是：**外观 style 与目标抽屉/fixture 的组合泛化失败**，
  而不是单纯 state normalizer 或 base pose 错误。

这不等于已经证明“通道机制”是根因。它只说明：

```text
同一任务、相近场景/机器人起点
+ 没见过的 style-target-fixture 组合
→ 策略没有进入正确的接近/抓取行为
```

### 当前不能直接做的事

- 不能用 `0/20` 或 `0/5` 宣称 CoRP 方法失败；
- 不能直接把训练 episode 初始 state 的解释当作官方 reset benchmark；
- 不能把 style 变化当成“纯颜色”变化；
- 不能在没有 clean 能力、没有可解释 clean→shift 退化前做通道干预；
- 不能把激活差异、梯度大小或 outlier 排名直接当作动作因果。

---

## 1. 具身泛化研究的抽象地图

### 1.1 泛化的定义

泛化研究可以统一写成：

```text
训练分布 P_train
    ↓ 训练策略 π
测试分布 P_test
    ↓ 闭环执行
success rate / stage failure
```

当 `P_test` 与 `P_train` 不同，才是 OOD 泛化问题。

### 1.2 研究角度

| 角度 | 改变什么 | 常见数据/协议 |
| --- | --- | --- |
| 外观 | 纹理、颜色、光照、材质 | 固定 state/layout 的颜色扰动、style split |
| 场景布局 | layout、家具位置 | train/target scene split |
| 物体 | 物体类别、实例、摆放 | object holdout |
| 初始状态 | robot base、关节、eef、物体初始状态 | reset / initial-state perturbation |
| 相机 | 相机位置、朝向、FOV | camera pose/FOV shift |
| 语言 | 指令改写、组合、歧义 | language rewrite |
| 物理 | 质量、摩擦、控制延迟 | dynamics randomization |
| 任务组合 | 已学子任务的组合 | compositional task split |
| 跨 embodiment | 不同机械臂、底盘 | embodiment holdout |
| sim-to-real | 仿真到真机 | sim-to-real benchmark |

当前项目主要研究外观，但实际评测还混入了：

```text
style-target-fixture 组合
reset 初始状态
base/camera 几何
```

因此实验必须把这些因素拆开。

---

## 2. 本项目真正要研究的通道问题

### 2.1 研究对象

以 CoRP 原始结构为蓝本：

```text
DINOv2 ViT-S/14
→ 每个视角 256 个 patch tokens
→ 每层 FFN 的 GELU 中间激活 [tokens, 1536]
→ resampler 压缩为 48 tokens/view
→ FiLM / DiT flow-matching action generator
```

通道干预的主要对象：

```text
某一个 DINO block 的 FFN 中间通道
作用于 patch token；CLS 暂时不作为动作因果对象
```

初期优先级：

1. FFN 中间通道；
2. 三视角共享的固定通道 mask；
3. 单 block、确定性、离线消融；
4. 暂不做动态 token-channel 网络；
5. 暂不把 DaViT、Register 或完整 SAE 移植到 CoRP。

### 2.2 研究问题

Q1：外观变化在哪些层、视角、token 位置的 FFN 通道上产生了稳定响应差异？

Q2：这些响应差异中，哪些会经过 resampler 影响动作，而不只是视觉特征变化？

Q3：在不读取 clean 图像、GT action 或测试 domain 标签的前提下，固定通道衰减/置零
能否改善 shift 条件，同时保持 clean 能力？

Q4：如果共享通道不足，token-conditioned 通道是否比同预算随机干预更好？

### 2.3 两个必须分开的结论

```text
clean 激活替换成功
```

说明：

> 这些单元承载了可恢复的信息，属于机制诊断。

```text
同单元衰减/置零在 shift 下也成功
```

才可能支持：

> 存在可部署的通道压制方法。

只有前者、没有后者，只能报告“恢复信息有效”，不能设计 gate 并宣称方法有效。

---

## 3. 先解决实验混淆，再做通道

### 3.1 当前可用的对照

当前 OpenDrawer 已经有两个重要锚点：

```text
train ep24: style20 + stack_2_right_group_1_2 → success
reset ep0:  style12 + stack_2_right_group_1_2 → fail
```

它们使用同一个目标抽屉，且 world base/camera 很接近。  
下一步应该把这个比较正式化为 manifest：

```text
task: OpenDrawer
layout: 31
target drawer ref: stack_2_right_group_1_2
style A: 20（seen style-target combination）
style B: 12（unseen style-target combination）
robot base/camera: 固定或记录差异
```

注意：style 可能改变几何和外观，因此这不是“纯颜色通道因果”实验。  
它用于回答：

> 未见 style-target 组合是否足以让策略在抓取前失败？

### 3.2 纯外观通道实验

通道机制实验应优先使用更干净的受控对：

```text
同一 task
同一 episode/frame 或同一 simulator state
同一 layout/fixture
同一 base/camera
同一 policy seed
只改变 color_jitter / blur / 已核验的渲染属性
```

这是：

```text
受控视觉扰动
```

自然 style-target 组合实验则单独作为：

```text
自然外观 + 任务目标组合泛化
```

不能把二者合并成同一个“style 泛化”指标。

### 3.3 数据支持检查

在启动闭环前必须检查：

- 该 task 是否有请求的 layout；
- 每个 style 是否有 train-split episode；
- 请求的 `fixture_refs` 是否在对应 style/layout 的 train split 中出现；
- 同一 style 的 fixture refs 是否改变目标物体身份；
- train/source/val 是否有 episode/style 泄漏。

如果只检查 style 范围，不检查 fixture/target，仍会把“新任务目标组合”误当成
“新外观”。

---

## 4. 建议的实验顺序

### E0：能力 gate，不做通道

目标：确认在训练分布内存在稳定非零任务能力。

建议：

```text
OpenDrawer
layout 31
训练支持的 fixture/target
训练或 source-holdout 初始 state
clean
先用 5–10 episodes
```

当前已有 4/5 训练初始 state 成功，但还不足以证明多种初始状态和 source-holdout
episode 的稳定性。只有 E0 通过，才进入 E1。

### E1：受控外观退化

目标：确认同一模型、同一状态和同一相机下，外观变化是否造成可解释的动作变化。

操作：

```text
同一个 observation
clean vs color_jitter
（可选）clean vs blur
固定 flow noise、policy seed、action mask
```

判定：

- 如果 clean 和 shift 动作几乎一致，不能进入通道因果实验；
- 如果 shift 动作在早期就改变方向，记录第一处分叉；
- MSE 只作为辅助，不替代闭环 SR。

### E2：只读 FFN 采集

目标：找到候选层/通道，而不是直接加 gate。

建议上限：

```text
3 tasks × 3 independent episodes × 3 frames = 最多 27 对 observation
```

只保留：

- CLS/patch 区分；
- 三视角 view-major 身份；
- 每层的 mean/std/quantile；
- 少量完整 FP32 activation snapshot。

不默认保存全部 12 层、三视角、clean/shift 全量激活。  
按当前文档估算，单个观测时刻一对 clean/shift 的完整激活约 142 MB；1000 对约 142 GB，
必须流式统计，不能无计划落盘。

#### E2 最新结果（2026-11-08）

四个训练支持初始 state 的 clean/color_jitter 配对已完成：

```text
clean        4/4 success
color_jitter 4/4 fail@750
```

只读 FFN 采集使用 ep24/28 选候选、ep64/73 独立验证。Patch 相对激活差异最大在
layer11/left、layer10/right 和 layer10/left；CLS 也在 layer11 最大。候选验证中，
layer0 最稳定（mean |validation z|=3.35、sign match=1、consistency=1），layer11
虽有最大差异，但一致性低于 layer0。这些结果属于 Q1 描述性证据，不是动作因果。

### E2b：最小 Q2 动作干预

已在 ep64/73 的固定候选上做初始 action chunk 干预，并在 ep64 做 300 步小闭环：

```text
clean                         success@179
color_jitter                  fail@300
layer11 clean replacement     fail@300
layer11 selected zero         fail@300
matched random zero           fail@300
```

layer11 clean replacement 能把初始 jitter action chunk 明显拉回 clean（ep64
MSE 0.106→0.021、cos 0.690→0.945；ep73 方向一致但幅度较小），但闭环中误差
重新累积，未恢复 success。selected zero 没有超过 matched random。

因此当前“单 block、20 channels、固定零衰减”不足以成为可部署 gate。E4 暂不启动；
下一步若要继续，应先把错误发生时刻、层间补偿和多层/更大 token-channel 支持集分开验证。

进一步做了 layer11 **全部 1536 个 patch FFN 通道**的 clean activation replacement
正对照（CLS 与 residual 保持原输入，clean reference 为特权诊断）。ep64/73 均运行
300 步：

```text
ep64: fail@300, 抽屉≈0, chunk cosine-to-clean mean=0.980
ep73: fail@300, 抽屉≈0, chunk cosine-to-clean mean=0.972
```

动作 chunk 已非常接近 clean oracle，但只替换 FFN 分支、保留 jitter residual，没有
恢复完整 clean block output；闭环仍然逐步漂移。当前结论是：layer11 FFN 分支容量不是
唯一瓶颈，需进一步验证整 block 输出、多层替换、resampler 或 DiT 路径，而不是继续做
单层 gate。

随后完成 layer11 **完整 block output** replacement：CLS/patch 全部替换，包含 attention、
post-attention residual、FFN、LayerScale。结果：

```text
ep64 layer11 block replace: success@177, drawer=0.962
ep73 layer11 block replace: success@172, drawer=0.980
12-block replacement: 与 layer11-only 轨迹完全相同
逐 chunk action MSE-to-clean=0, cosine≈1
```

这证明瓶颈位于 layer11 完整 block 内部，而不是 FFN 分支容量不足；全视觉路径和
resampler/DiT 在输入完整 clean block output 时均可恢复。下一步应分解 attention
branch、post-attention residual 和 complete FFN branch，寻找最小必要替换集合。

### E3：候选单元选择

候选选择必须在 source 校准集完成，不能在同一批 episode 上选择并证明。

候选可以来自：

- 外观响应差异；
- 对动作的敏感性；
- 官方 detector 的已有定义；
- outlier/response 统计。

任何选择都要记录：

- layer；
- view；
- CLS/patch；
- neuron/channel ID；
- 选择样本；
- 阈值；
- 是否使用 clean 参考；
- 是否有同预算随机集合。

### E4：最小通道干预

在单个 block 上做最小干预：

```text
FFN GELU 后的 patch 中间通道
× 共享开度 g_h
g_h = 1 为恒等
```

先做：

1. clean activation replacement（诊断，不可部署）；
2. shifted activation decay/zero（可部署候选）；
3. matched random channel decay；
4. embedding-channel control；
5. 无干预；
6. 标准像素增广对照。

必须固定：

- task/instruction；
- simulator state；
- image preprocessing；
- policy seed；
- flow sampling noise；
- action mask 和 horizon；
- 编辑通道数量、层、view 和编辑幅度。

### E5：自然 style-target 外部检验

只有 E4 在受控外观实验中成立，才做：

```text
style20 + stack_2_right（seen）
vs
style12 + stack_2_right（unseen）
```

这是组合泛化外部检验，不是纯外观因果。  
报告时必须同时写：

- style 是否改变几何；
- fixture/target 是否保持；
- base/camera 是否匹配；
- style-target 组合是否在 train split 出现。

---

## 5. 数据集与场景选择

### 主数据集

```text
/hy-tmp/datasets/robocasa365_oss
v1.0/pretrain/atomic/...
```

主任务：

```text
OpenDrawer
OpenCabinet
TurnOffStove
TurnOnElectricKettle
TurnOnSinkFaucet
```

首个机制实验建议只做：

```text
OpenDrawer
layout 31
```

原因：

- 已有训练初始 state 成功证据；
- 已有 style20/stack_2 成功和 style12/stack_2 失败的对照；
- task horizon、动作语义和 failure stage 容易观察；
- 20 episodes 的仿真成本比五任务矩阵低。

后续扩展顺序：

```text
OpenDrawer
→ 1–2 个额外 atomic task
→ 5-task panel
→ 3 seed
```

不要一开始直接做 61 task 或最终 target 1–10。

### 数据支持原则

每个请求必须同时检查：

```text
task
layout
style
fixture_refs / target identity
episode split
```

仅满足 `style ∈ 11–50` 不等于该 task/style/layout/fixture 组合有训练支持。

---

## 6. 硬件与预算

当前服务器观测：

```text
GPU: RTX 4090，约 48 GB 显存
host RAM: 约 755 GB
cgroup memory limit: 约 115 GB
```

当前 30k baseline 已完成，不再自动重训。

建议按阶段消耗：

1. E0/E1：OpenDrawer 5–10 episodes，先看方向和失败阶段；
2. E2：27 对 observation，只做一次或少量 batch 前向；
3. E3：离线统计和画图，不跑 MuJoCo；
4. E4：先 1 task × 少量通道 × 5 episodes；
5. 只有 shift 改善且 clean 保持，才扩到 3–5 task。

不建议：

- 在 E0 未通过时启动 E4；
- 一开始跑四方法 × 五任务 × 20 episodes；
- 为通道实验直接重训 50k/100k；
- 为保存激活而占满磁盘。

---

## 7. 指标与 go/no-go

### 主指标

```text
闭环 success rate
```

辅助：

```text
paired clean→shift success transition
first action divergence step
gripper close timing
drawer/eef trajectory
masked action MSE
stage failure
```

### 初期门槛

| Gate | 通过条件 | 不通过时 |
| --- | --- | --- |
| 能力 | 训练/源初始 state 有可重复非零成功 | 先修训练或评测 |
| 退化 | 同 state/camera 的 clean→shift 有可解释动作变化 | 换扰动或任务 |
| 传递 | 受控单元变化影响动作或闭环阶段 | 不设计 gate |
| 方法 | shift 改善优于 matched random | 停止或改假设 |
| 安全 | clean 不明显下降 | 不视为可接受方法 |

单任务、单 seed、20 episodes 只能做筛选，不能做最终统计结论。

---

## 8. 官方工作对照与本项目边界

### CoRP

- 官方论文报告 LIBERO 和 RoboTwin 2.0 结果；
- RoboTwin 训练包含 2,500 clean + 25,000 randomized demonstrations；
- Randomized 评测是训练分布内变化，不是零样本任意 reset；
- 官方仓库主要提供推理/LIBERO 评测，RoboTwin 部分仍不完整；
- 本项目是 `CoRP adapted to RoboCasa / our training adaptation`。

### RoboCasa365 / Diffusion Policy

- 官方 benchmark 有 pretrain/target scene split；
- 官方评估使用 `RoboCasaGymEnv` 的 reset；
- reset 可以恢复 `ep_meta` 中的 `init_robot_base_pos/ori`，否则会采样 base；
- 训练和评估的 reset 分布必须分别记录，不能混成“同一 seen 条件”。

### RoboCasa-Shift

- 研究 illumination、camera、robot base、blur；
- 使用 matched clean/perturbed rollouts；
- 重点报告阶段失败，而不是只看总 SR；
- 本项目的 base/camera 诊断与其协议一致。

### LIBERO-Plus

- 报告 camera viewpoint 和 robot initial state 是高风险扰动；
- 构造 20k+ 泛化轨迹做 mixed fine-tuning；
- 这说明“更多同分布训练步数”不等于“覆盖新的初始状态/相机分布”。

### RoboTwin 2.0

- 使用 clutter、lighting、background、tabletop height、language 和相机 pose
  randomization；
- 训练数据覆盖变化后再评测 Randomized；
- 可作为未来数据增广和 camera/base 覆盖设计的参考。

---

## 9. 最终建议

当前最有价值的研究问题不是：

```text
给 CoRP 加一个通道 gate，看看能不能涨点
```

而是：

```text
在受控同状态外观变化下，CoRP 的哪些 FFN 通道真正影响动作；
这些通道的干预能否超过同预算随机对照，并保持 clean 能力。
```

顺序必须固定为：

```text
能力 gate
→ 受控外观退化
→ 通道响应
→ 动作传递
→ 最小通道干预
→ 自然 style-target 外部检验
→ 多任务/多 seed
→ 最后 target
```

在此之前不要重训、换 backbone、启动完整四方法矩阵或接触 target 1–10。
