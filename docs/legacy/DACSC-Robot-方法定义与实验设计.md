# DACSC-Robot 方法定义与实验设计

> **历史设计，已退出当前实验协议。** 本文保留早期 DP/channel-gate 的设计来源，
> 不代表 CoRP 方法或验证结果。当前唯一执行协议为 `docs/EXPERIMENT_PROTOCOL.md`。

> 版本：v0.2（数据与基线审计版）
> 状态：方法定义与实验方案，尚未声称实验结论
> 主策略：Diffusion Policy（RoboCasa 官方 fork，已锁定）
> 主研究问题：机器人视觉策略在未见外观域下的动作泛化
> 最近核查：2026-10-15（核查来源见 16 节）

## 0. 版本说明

v0.2 相对 v0.1 的修订集中在"先固定实验对象、数据划分与梯度路径"，不增加新公式：

1. 废弃"style 1/2/3 训练、4 验证、5 测试"的划分 —— id 1–10 属于 target 场景（见 3.2、3.3）；
2. 基线锁定 RoboCasa 官方 Diffusion Policy fork，动作窗口改为官方 \(H=10,\ n_{\text{obs}}=2,\ n_{\text{action}}=8\)（见 4.1）；
3. 修正 DomainDrop 的机制描述与来源归属（见 4.2）；
4. 修复域分类器的 stop-gradient 与域梯度敏感性互相矛盾的硬错误（见 6.2、6.3）；
5. 通道门控插入位置改写为按真实网络结构（8 个 conv block）确定（见 5.2–5.4）；
6. "状态条件任务特征矫正"从候选创新降级为可选消融（见 5.5、6.9）；
7. 实验顺序改为 0–8 阶段，消融分为两轮（见 8、9、14）。

**执行前置条件**：14 节的阶段 0（数据审计）与阶段 1（基线复现）完成前，不进入门控模块实现。

## 1. 文档定位

本文档用于统一 DACSC-Robot 的问题定义、相关方法、公式来源、实现边界和实验计划。文档不把候选设计直接写成已经验证的结论，并对内容采用以下标记：

| 标记 | 含义 |
|---|---|
| 已有方法 | 原论文已经提出并验证的机制或公式 |
| 借鉴方法 | 原论文中的方法被改造后用于本问题 |
| 候选设计 | 我们计划实现并通过实验验证的组合 |
| 待验证假设 | 目前不能直接当作结论的判断 |

本文档的主要用途有三个：

1. 为代码实现提供明确的模块和损失定义；
2. 为论文或专利撰写保留方法来源和创新边界；
3. 为后续对比实验和消融实验提供统一的控制变量。

## 2. 研究问题

机器人视觉动作策略通常根据摄像头图像和机器人状态预测未来动作。在训练环境中，策略可能错误地利用桌面纹理、背景颜色、光照或任务无关物体等外观信息。部署到新的视觉环境后，虽然物体和任务关系没有改变，策略仍可能输出错误动作。

研究目标是：

> 在不损害任务相关视觉信息的前提下，识别并抑制对外观域变化敏感、但对动作预测贡献较小的视觉特征，同时保护对动作生成重要的通道，并对抑制后的任务特征进行矫正。

### 2.1 研究假设

在任务、机器人状态和物体几何关系基本保持不变时，以下因素可以发生变化而不应改变正确动作：

- 背景纹理；
- 桌面材质；
- 光照和阴影；
- 任务无关的干扰物；
- 场景风格或外观配置。

因此，理想的视觉表示应满足：

1. 对物体位置、目标位置和机器人状态保持动作敏感；
2. 对背景和外观域变化保持相对稳定；
3. 在外观变化后仍能生成一致的连续动作。

### 2.2 第一阶段边界

第一阶段固定使用 Diffusion Policy 作为下游动作策略，先研究视觉编码器中的通道处理机制。ACT、Octo、π₀ 和 GR00T 放在后续扩展实验中，不能与第一阶段混合为主要变量。

第一阶段暂不把以下内容作为核心贡献：

- 大型视觉语言模型预训练；
- 语言理解能力；
- 机器人本体跨形态迁移；
- 强化学习在线探索；
- 复杂世界模型预测。

## 3. 基本符号和数据定义

### 3.1 轨迹符号

一条机器人示范轨迹表示为：

\[
\tau_i=\{(o_t,s_t,a_t,d_i,l_i,q_i)\}_{t=1}^{T_i}
\]

其中：

- \(o_t\)：第 \(t\) 时刻的视觉观测；
- \(s_t\)：机器人本体状态，例如关节位置、末端位姿和夹爪状态；
- \(a_t\)：机器人动作；
- \(d_i\)：外观域标签（本项目中即 style_id）；
- \(l_i\)：场景布局标签（layout_id）；
- \(q_i\)：任务标签；
- \(T_i\)：第 \(i\) 条轨迹的长度。

策略接收连续动作窗口：

\[
\pi_\theta(o_t,s_t)\rightarrow a_{t:t+H-1}
\]

RoboCasa365 官方 Diffusion Policy 基线采用 \(H=10\)、\(n_{\text{obs}}=2\)、\(n_{\text{action}}=8\)，详见 4.1 节。

### 3.2 RoboCasa365 场景定义（2026-10-15 核实）

场景定义是 **scene = layout × style**：layout 决定户型平面，style 决定家具、电器与纹理的组合。

| 用途 | layout_id | style_id | 组合数 |
|---|---|---|---|
| pretrain | 50 个户型，id **11–60** | 50 个风格，id **11–60** | 50×50 = **2500 scenes** |
| target | 10 个户型，id **1–10** | 10 个风格，id **1–10**，一一配对 | **10 scenes** |

- target 的 style 与 pretrain 的 style 在夹具、电器、纹理上**完全不重叠**；
- 源码中的伪 id：`LAYOUT_GROUPS_TO_IDS = {-1: 1..10 \(test\), -2: 11..60 \(train\), -3: 1..60}`，`STYLE_GROUPS_TO_IDS` 同构；
- 环境参数支持 `layout_ids`、`style_ids`、`layout_and_style_ids`，另有 `"5x5"`、`"5x1"` 预设；
- 任务规模：365 tasks = 65 atomic + 300 composite，其中 220 个需要移动底盘；
- 官方 50 任务评测集 = Atomic-Seen 18 + Composite-Seen 16 + Composite-Unseen 16。

**关键约束：id 1–10 属于 target 场景，不能用于 pretrain 内部划分。**

### 3.3 数据划分约定

`pretrain → target` 这一步同时改变了 layout、style 和物体，三者耦合，**无法分离"外观域"这一个变量**。因此第一版采用 pretrain 内部划分来单变量研究外观域，target 作为二次验证：

| 用途 | layout | style |
|---|---|---|
| 训练 | 固定或有限 layout 子集 | 11–40 |
| 验证 | 相同 layout | 41–50 |
| 测试 | 相同 layout | 51–60 |
| 扩展测试 | 未见 layout 或 layout-style 组合 | 单独报告 |

该划分的有效性必须先由 10.2 节的数据审计确认；审计通过前不训练域分类器。

### 3.4 以 style_id 作为域标签的三个风险

1. pretrain 每个 task 只有 100 条 demo，场景对在候选集合中随机抽取，单 task 内 \(layout, style\) 覆盖极为稀疏；
2. style 共 60 类，若每类样本过少，域分类器准确率会接近随机水平，域梯度分数不可信；
3. style_id 的边界不等于外观域语义边界；必要时先用 `robocasa/scripts/build_fixture_style_groups.py` 把 style 归并为若干粗粒度外观域。

具体检查项见 10.2 节。

## 4. 相关方法及其作用

### 4.1 Diffusion Policy：官方基线（2026-10-15 核实并锁定）

第一阶段基线**锁定 RoboCasa 官方 fork**：`robocasa-benchmark/diffusion_policy`（最后更新 2026-05-25），其视觉核心来自 `robocasa-benchmark/robomimic` 的 `robocasa-dp` 分支。

官方训练与评测入口：

```bash
python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/<dataset-soup>
python eval_robocasa.py --checkpoint <ckpt> --task_set <task-set> --split <split>
```

基线配置（取自 `diffusion_policy/config/train_diffusion_transformer_bs192.yaml` 与 `config/task/robocasa/*.yaml`）：

```yaml
camera_views:
  - robot0_agentview_left
  - robot0_agentview_right
  - robot0_eye_in_hand
image_size: [256, 256]
crop_size: [224, 224]
visual_encoder: ResNet18ConvFiLM          # robomimic fork 自定义
language_conditioning: FiLM per conv block, CLIP 768-d, episode level
spatial_softmax_keypoints: 32
visual_embedding_dim: 64                   # 每路相机
policy_backbone: DiT                       # n_layer=12, n_cond_layers=4, n_head=8, n_emb=512
horizon: 10
n_obs_steps: 2
n_action_steps: 8
batch_size: 192
action_dim: 12                             # base_motion 4 + control_mode 1 + ee_pos 3 + ee_rot 3 + gripper 1
```

要点：

- **不再使用 \(H=16\) 作为默认动作窗口**，官方配置为 \(H=10\)、观测 2 帧、每次执行 8 步后重规划；
- `abs_action` 与 `rotation_6d` 在该分支不受支持（源码中 `assert not self.abs_action`），动作为 delta 表示；
- 训练显存建议 ≥24 GB，48 GB+ 更佳；推理 ≥8 GB；评测建议 16 核 / 64 GB 内存实例（官方文档）；
- episode horizon 不在配置文件中，评测脚本用 `get_task_horizon\(task\)` 覆盖（`eval_robocasa.py`）；registry 中 v1.0.1 之后为 300–7200 步；
- 训练超参：1000–2000 epochs × 500 steps、EMA、BatchNorm→GroupNorm、lr 1e-4、1000 步 warmup + cosine。

**时效性提示（2026-10-15）**：在官方 leaderboard 上，Vanilla Diffusion Policy 的 50 任务成绩为 Atomic-Seen 15.7 / Composite-Seen 0.2 / Composite-Unseen 1.3，是所有基线中最低的；同期 world-action model 类方法（Awomo-0.5I，2026-10-09）已达 86.5 / 58.8 / 45.2。因此本工作的定位是**表示层机制研究**：机制须保持 backbone-agnostic，并在后续至少在一种 VLA/WAM 上做迁移验证，否则会被认为是落后骨架上的增量。

### 4.2 DomainDrop：域敏感通道抑制（机制已核对原文）

DomainDrop（arXiv 2308.10285）在中间层插入域判别器，用**通道对域预测的贡献**作为域敏感性度量，并按概率丢弃敏感通道。

其通道分数**不是梯度乘积形式**，而是判别器权重与该通道激活的乘积：

\[
s_j = W^{y^d}_j \cdot \operatorname{GAP}(F_l(x))_j
\]

其中 \(W^{y^d}\in\mathbb R^C\) 是域判别器对真实域 \(y^d\) 的 FC 权重。原文其余设置：

- 判别器前接梯度反转层（GRL），权重 0.25，用于阻断域损失对主网络的直接更新；
- 丢弃比例 \(P_{\text{drop}}=0.33\)；
- 候选层为 4 个 residual block，每次迭代随机选一层；
- 用加权随机采样（WRS）生成二值掩码，复杂度 \(O(C)\)；
- 配合 dual consistency loss（权重 1.5，温度 5）。

本文借鉴其"用域判别信号定位不稳定通道 + 随机丢弃"的思想，但**不采用**"通道贡献 = 判别器权重 × 激活"的度量，改用 6.3 节的域梯度分数，并叠加动作重要性保护。引用时必须区分两者：**不得把 Taylor 梯度乘积形式归属为 DomainDrop 的原始方法**。

### 4.3 PLACE：渐进层级和通道级丢弃

PLACE 在不同层和不同通道执行随机丢弃，并逐步增加丢弃比例，避免模型长期依赖少量源域特征。

本文借鉴其渐进式训练策略，用于逐步提高 DACSC-Robot 的抑制预算。

### 4.4 DgCD：跨域分布差异通道丢弃

DgCD 直接估计不同域之间的特征分布差异，并根据 KL 散度等分布差异信号对通道进行丢弃。相比只依赖分类器权重，分布差异可以直接反映通道激活在不同域之间的变化。

本文将 DgCD 的分布差异作为域敏感性估计的一部分，而不是直接照搬其分类任务设置。

### 4.5 Taylor 一阶重要性：动作通道贡献估计

神经网络结构剪枝研究常用激活与损失梯度的乘积估计删除某个通道对损失的影响。本文将动作去噪损失替换为结构重要性计算中的任务损失，用于估计视觉通道对机器人动作的贡献。

### 4.6 Deep CORAL：跨域二阶统计对齐

Deep CORAL 通过对齐不同域特征的协方差，减少中间表示的二阶统计差异。本文把它作为可选的跨域特征正则项，用于抑制门控后特征残留的域差异。

### 4.7 DANN：域对抗表示学习

DANN 使用梯度反转层，使视觉编码器学习到难以被域分类器区分的特征。本文不在第一版中强制加入 DANN，而将其作为后续增强和消融选项，以避免域对抗、通道门控和动作蒸馏同时更新造成训练不稳定。

### 4.8 机器人视觉域增强方法

- DRAIL：区分任务相关区域和任务无关区域，对两类区域使用不同增广策略，并使用 Diffusion Policy 进行验证；
- RoboSaGA：利用策略显著性调节图像叠加增广强度，验证了 BC-MLP、BC-RNN 和 Diffusion Policy；
- RoboVIP：使用多视角视频扩散生成增强数据，在仿真中验证 Octo 和 π₀，在真实机器人中验证 Diffusion Policy；
- RoboAug：使用任务相关区域提取、场景增广和区域对比学习，主要实验使用 ACT，也扩展验证了 π₀。

这些方法主要属于数据增广或区域表示学习路线，可以作为 DACSC-Robot 的外部对比方法和组合方法。

## 5. DACSC-Robot 总体结构

### 5.1 处理链

```text
视觉观测 o_t（3 路相机 256×256 → crop 224）
    ↓
ResNet18ConvFiLM：_base_block + 8 个 conv block
    ↓
    [逐 block] FiLM（语言条件，基线已有）
    + [逐 block] 通道门控 g_{l,c}
    ↓
8 层特征 F_l（通道 64/64/128/128/256/256/512/512）
    ↓
域敏感性 u_{l,c}  +  动作重要性 v_{l,c}
    ↓
SpatialSoftmax(32) → Linear → 每路相机 64 维
    ↓
DiT 去噪网络
    ↓
动作序列 a_{t:t+H-1}（H=10）
```

### 5.2 门控插入位置（按真实网络结构确定）

ResNet18ConvFiLM 由 `_base_block`（conv1 + bn + relu + maxpool）和 8 个 conv block 组成，通道依次为：

```text
64 → 64 → 128 → 128 → 256 → 256 → 512 → 512
```

门控插入这些 conv block 的**输出**，且位于 SpatialSoftmax **之前**。

- SpatialSoftmax 用 1×1 卷积把 512 通道压成 32 个 keypoint map，此后的"通道"已不具备视觉通道语义，在其之后做通道门控会使"通道域敏感性"的解释失效；
- 输入 224×224 时，8 个 block 的特征图尺寸依次为 56/56/28/28/14/14/7/7，最终为 512×7×7（由源码 `output_shape\(\)` 的 `ceil\(H/32\)` 确定）。

### 5.3 门控与 FiLM 的顺序契约

基线本身已在**每个 conv block 之后**做通道级 FiLM 调制（\(x=(1+\gamma)\odot x+\beta\)，随后 ReLU）。因此必须显式固定门控与 FiLM 的相对位置，并作为消融项：

```text
方案 A：Conv Block → FiLM → Channel Gate
方案 B：Conv Block → Channel Gate → FiLM
```

门控置于 FiLM 之前更贴近"对原始域敏感特征操作"的语义，但两种顺序都必须实测。无论采用哪种，都必须在论文中写明：基线已含 FiLM，顺序会直接影响"域敏感性"的解释。

### 5.4 门控层集合（第一版）

```text
主实验：第 4、6、8 个 block 插入门控
消融 A：只在第 8 个 block 插入
消融 B：第 6、8 个 block 插入
消融 C：8 个 block 全部插入
```

### 5.5 候选贡献（修订）

1. 使用域梯度信号与跨域分布差异联合识别域敏感通道；
2. 使用动作去噪损失的 Taylor 重要性保护动作关键通道；
3. 在 Diffusion Policy 的视觉编码器内部做通道级软门控并施加抑制预算；
4. 在动作噪声预测空间施加外观变化一致性约束。

原第 3 条"使用机器人状态条件的任务特征矫正"**不再作为候选创新**，降级为可选消融项，理由见 6.9 节。

## 6. 公式定义

### 6.1 Diffusion Policy 动作去噪损失

对动作序列 \(a\) 加入扩散噪声：

\[
x_k=\sqrt{\bar{\alpha}_k}a+
\sqrt{1-\bar{\alpha}_k}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
\]

策略预测噪声：

\[
\hat\epsilon=\epsilon_\theta(x_k,o_t,s_t,k)
\]

动作策略损失：

\[
\mathcal L_{\mathrm{DP}}
=
\mathbb E
\left[
\|\epsilon-\hat\epsilon\|_2^2
\right]
\]

该损失继承 Diffusion Policy 的动作扩散训练形式。

### 6.2 域分类器（修正版：两阶段，梯度不回流）

域分类器用于**测量**域敏感性，其损失不得直接更新视觉 backbone。

原草案同时写了：

\[
\mathcal L_{\mathrm{dom}}=\operatorname{CE}\left(D_\psi(\operatorname{sg}(\bar f_l)),d\right)
\]

以及

\[
u^{\mathrm{grad}}_{l,c}=\left|\bar f_{l,c}\frac{\partial \mathcal L_{\mathrm{dom}}}{\partial \bar f_{l,c}}\right|
\]

二者互相矛盾：**一旦 \(\operatorname{sg}(\cdot)\) 生效，\(\partial \mathcal L_{\mathrm{dom}}/\partial \bar f_{l,c}\equiv 0\)，域梯度分支完全失效。** 修正为两阶段流程。

第 \(l\) 层视觉特征 \(F_l\in\mathbb R^{C_l\times H_l\times W_l}\)，通道全局平均池化：

\[
\bar f_{l,c}=\frac{1}{H_lW_l}\sum_{h=1}^{H_l}\sum_{w=1}^{W_l}F_{l,c,h,w}
\]

**步骤 1 — 用 detach 特征训练域分类器**

\[
\mathcal L_{\mathrm{dom}}=\operatorname{CE}\left(D_\psi(\operatorname{detach}(\bar f_l)),d\right)
\]

只更新 \(D_\psi\) 的参数，不更新 backbone。

**步骤 2 — 冻结 \(D_\psi\)，用未 detach 的特征求梯度**

```python
# ---- 每一轮训练都按此顺序执行 ----

# 步骤 1：训练域分类器（特征 detach）
for p in domain_head.parameters():
    p.requires_grad_(True)                     # 必须恢复，否则 D_psi 只会被训练一次
domain_logits = domain_head(features.detach())
domain_loss = cross_entropy(domain_logits, style_id)
domain_optimizer.zero_grad()
domain_loss.backward()
domain_optimizer.step()

# 步骤 2：冻结域分类器，用未 detach 的 features 求对特征的梯度
for p in domain_head.parameters():
    p.requires_grad_(False)                    # 只冻结"参数"，不改 features 的图
domain_logits = domain_head(features)          # features 必须保留计算图
domain_loss = cross_entropy(domain_logits, style_id)
grad = torch.autograd.grad(
    domain_loss, features,
    retain_graph=True, create_graph=False,
)[0]

# 步骤 3：仅读取梯度形成域敏感性分数（不执行 backward）
domain_score = (features * grad).abs().mean(dim=(2, 3))    # 逐通道
domain_score = ema_update(domain_score)                    # EMA 平滑

# 步骤 4：恢复 requires_grad，避免影响后续轮次与 checkpoint 语义
for p in domain_head.parameters():
    p.requires_grad_(True)
```

**步骤 4** — \(\mathcal L_{\mathrm{dom}}\) 不进入总损失，不更新视觉 backbone；域敏感性只能通过门控正则项（6.8 节）间接影响编码器。

实现注意（两个容易踩的坑）：

1. **`requires_grad_(False)` 必须在每轮步骤 1 之前恢复为 `True`。** 否则步骤 1 的 `backward()` 不会产生任何梯度，`domain_optimizer.step()` 空转，域分类器在第一轮之后永久冻结 —— 这会表现为"域分类准确率从第二轮起不再变化"。
2. **步骤 2 需要特征保留计算图。** 若与动作损失共享同一轮前向，用 `retain_graph=True`；若单独前向，必须使用带梯度的 `features`（不是 `detach()` 过的副本），否则 `autograd.grad` 返回空或报错。
3. 步骤 4 之后 `domain_score` 只作为打分写入 EMA，**不进入 `L_total`**；`L_dom` 也不进入 `L_total`。

### 6.3 基于域分类梯度的敏感性

在 6.2 节步骤 2/3 得到 \(\partial \mathcal L_{\mathrm{dom}}/\partial \bar f_{l,c}\) 之后，借鉴 Taylor 一阶结构重要性的形式定义域敏感性：

\[
u^{\mathrm{grad}}_{l,c}
=
\operatorname{EMA}
\left(
\left|
\bar f_{l,c}
\frac{\partial \mathcal L_{\mathrm{dom}}}
{\partial \bar f_{l,c}}
\right|
\right)
\]

分数越大，说明该通道对域判别贡献越大、越可能携带域特异信息。

**使用该分支的前置条件（任一不满足则退回 6.4 节的 KL 分支）**：

1. 域分类准确率显著高于随机水平（60 类时随机为 1.67%）；
2. 准确率在不同随机种子间稳定；
3. 每个风格的样本量足够（由 10.2 节审计给出）；
4. 若 style 类别过细导致 1–3 不满足，先把 style 归并为若干粗粒度外观域后再使用。

域分类器不是第一版的必要依赖：KL 分支（6.4 节）必须保留为无标签 fallback。

### 6.4 基于跨域分布差异的敏感性（按官方实现核对后修订）

> **核查状态（2026-10-16 已核实）**：DgCD 为 Kim et al., CIKM 2024，DOI 10.1145/3627673.3679539。
> **官方实现已找到**：`gyeomo/DgCD` 的 `domainbed/networks/DgCD.py`（**MIT 许可，99 行**）。
> 本节的描述已按**官方代码**重写；早期版本写的"逐通道高斯 KL"与官方不符，已废弃。

**DgCD 实际的打分流程**（逐步对应官方代码）：

1. 对特征做全局平均池化，得到逐通道向量 \(V \in \mathbb{R}^{B	imes C}\)
2. **域内标准化 + 对抗式仿射对齐**：按域（官方按"环境"，每 32 样本一块）与全体分别做
   均值/方差标准化，再乘上可学习的逐 (样本, 通道) 仿射 \(\text{Scaling}(\beta,\gamma)\)：

\[
\tilde{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad
\text{Scaling}(\tilde{x}) = \tilde{x}\odot\gamma + \beta
\]

3. **对通道维做 log-softmax**，把"通道向量"当成一个概率分布；
   用 KL 比较"全体分布"与"各域分布"：

\[
\mathcal L_{\mathrm{KL}} = \mathrm{KL}\left(\log\text{-softmax}(Z_{\text{all}}) \;\|\; \log\text{-softmax}(Z_{\text{env}})\right)
\]

4. **对抗步**：把上式的梯度**反向传播到仿射参数上并做梯度上升**，即主动寻找"让域间差异最大"的对齐方式：

\[
\beta \leftarrow \beta + \rho\,\frac{\nabla_\beta \mathcal L_{\mathrm{KL}}}{\sqrt{\|\nabla_\beta \mathcal L_{\mathrm{KL}}\|+\epsilon}}
\]

（对所有仿射参数做同样更新；这一步**不更新主干网络**。）

5. 用更新后的仿射参数**重算对齐**，并用 Gram 矩阵对角线归一化；
6. **打分用平方差而非 KL**：

\[
s_{b,c} = \left(Z^{\text{all}}_{b,c} - Z^{\text{env}}_{b,c}\right)^2
\]

7. **掩码**：min-max 归一化后做**加权随机采样**（WRS）丢弃比例 \(\text{ratio}\) 的通道，
   并**重归一化保持激活量级**：

\[
m \leftarrow m \cdot \frac{|m|}{\sum m}, \qquad m_c = 0 \text{ 的概率} \propto s_c
\]

**与"逐通道高斯 KL"的区别（早期文档写法，已废弃）**：

| | 早期文档（错） | DgCD 官方（对） |
|---|---|---|
| 分布形式 | 逐通道高斯 \(\mathcal N(\mu,\sigma^2)\) | **通道维 softmax** 分布 |
| 打分量 | KL 散度 | **平方差**（KL 只用于驱动对抗对齐） |
| 对抗性 | 无 | **有**（对仿射参数梯度上升） |
| 掩码 | 无 | **WRS + 重归一化** |

**移植说明**：官方代码硬编码 `.cuda()`、按每 32 样本切"环境"。我们的移植版
（`dacsc/dg_methods.py` 的 `DgCDScorer`）做了两处必要改动：
① 设备无关（本机用 Apple MPS）；② **按域标签切分环境**，而不是按固定块大小。
自研的"逐通道高斯 KL"版本保留为**消融对照**（配置项 `domain_score_mode: kl_gauss`）。

### 6.5 域敏感性融合

两个分数先进行层内标准化：

\[
\tilde u_{l,c}^{\mathrm{grad}}
=
\frac{u_{l,c}^{\mathrm{grad}}-mu_l^{\mathrm{grad}}}
{\sigma_l^{\mathrm{grad}}+\epsilon}
\]

\[
\tilde u_{l,c}^{\mathrm{disc}}
=
\frac{u_{l,c}^{\mathrm{disc}}-mu_l^{\mathrm{disc}}}
{\sigma_l^{\mathrm{disc}}+\epsilon}
\]

融合域敏感性：

\[
u_{l,c}
=
\alpha \tilde u_{l,c}^{\mathrm{grad}}
+
(1-\alpha)\tilde u_{l,c}^{\mathrm{disc}}
\]

### 6.6 动作 Taylor 重要性

使用 Diffusion Policy 去噪损失估计动作重要性：

\[
v_{l,c}^{\mathrm{Taylor}}
=
\operatorname{EMA}
\left(
\left|
\bar f_{l,c}
\frac{\partial \mathcal L_{\mathrm{DP}}}
{\partial \bar f_{l,c}}
\right|
\right)
\]

可以周期性进行实际通道遮挡校验：

\[
v_{l,c}^{\mathrm{mask}}
=
\mathcal L_{\mathrm{DP}}(F_l^{(-c)})
-
\mathcal L_{\mathrm{DP}}(F_l)
\]

其中 \(F_l^{(-c)}\) 表示删除第 \(c\) 个通道后的特征。最终分数可以写为：

\[
v_{l,c}
=
\beta\operatorname{Norm}
\left(v_{l,c}^{\mathrm{Taylor}}\right)
+
(1-\beta)\operatorname{Norm}
\left(v_{l,c}^{\mathrm{mask}}\right)
\]

第一版实现中，Taylor 分数用于训练，遮挡分数用于周期性校准，避免每个 batch 都进行昂贵的通道遮挡。

### 6.7 软通道门控

令 \(g_{l,c}\in[0,1]\) 表示通道保留程度：

\[
g_{l,c}=\sigma(\eta_{l,c})
\]

门控作用于 5.2 节确定的 conv block 输出，并按 5.3 节固定与 FiLM 的先后顺序。以"门控在校正模块之前"的形式写出：

\[
F_l^{\mathrm{gate}}
=
g_l\odot F_l
+
(1-g_l)\odot R_l(F_l,s_t)
\]

这里的 \(R_l\) 用于恢复被抑制通道中可能存在的任务相关信息。**注意：基线官方编码器已经在每个 conv block 之后做语言条件 FiLM，因此 \(R_l\) 与基线 FiLM 在机制上高度同构（差别仅在条件来源为 \(s_t\) 而非语言）。** 这直接决定 6.9 节的定位。

第一版中 \(R_l\) 取恒等映射（即 \(F_l^{\mathrm{gate}}=F_l\)，门控退化为普通软掩码），仅在消融中启用非平凡 \(R_l\)；否则无法区分收益来自通道门控还是额外网络容量。

### 6.8 门控目标

定义门控正则项：

\[
\begin{aligned}
\mathcal L_{\mathrm{gate}}
=
&\frac{1}{LC}\sum_{l,c}
\left[
\lambda_d u_{l,c}g_{l,c}
+
\lambda_a v_{l,c}(1-g_{l,c})^2
\right]\\
&+
\lambda_\rho
\left(
\frac{1}{LC}\sum_{l,c}(1-g_{l,c})-\rho
\right)^2
\end{aligned}
\]

其中：

- 第一项鼓励抑制域敏感通道；
- 第二项保护动作重要通道；
- 第三项控制整体抑制预算；
- \(\rho\) 为目标平均抑制比例。

该公式是 DACSC-Robot 的候选组合设计，不能描述为 DomainDrop、DgCD 或 Taylor 原文中的原始公式。
### 6.9 状态条件任务特征矫正（已降级为消融项）

> **定位修订（v0.2）**：本节不再作为候选创新。
>
> 理由：RoboCasa 官方视觉编码器 `ResNet18ConvFiLM` 已经在**每个 conv block 之后**施加通道级 FiLM 调制 \(x\mapsto(1+\gamma)\odot x+\beta\)，其中 \(\gamma,\beta\) 由 episode 级 CLIP 768 维语言嵌入经线性层生成（见 4.1 节源码核对）。因此下述模块与基线已有机制同构，仅把条件从语言换成机器人状态 \(s_t\)。若把它写成创新点，消融"去除状态条件矫正"很可能只退化为基线已有的东西，说服力不足。

保留形式（仅用于消融）：

\[
R_l(F_l,s_t)
=
F_l+
W_{2,l}\phi
\left(
\operatorname{LN}
\left(
W_{1,l}F_l+\operatorname{FiLM}_l(s_t)
\right)
\right)
\]

使用约束：

- 第一版 \(R_l=\mathrm{Id}\)，只在消融 C 中启用非平凡 \(R_l\)；
- 启用时必须同时报告与"基线 FiLM 顺序变体"（5.3 节方案 B）的对比，否则无法归因；
- 不使用大型 Transformer 矫正器，以便区分收益来自通道门控还是网络容量。


### 6.10 动作一致性

对同一动作轨迹构造原始观测 \(o_t\) 和外观变化观测 \(o_t^+\)，使用相同的带噪动作 \(x_k\)：

\[
\mathcal L_{\mathrm{cons}}
=
\mathbb E
\left[
\left\|
\epsilon_\theta(x_k,o_t,s_t,k)
-
\epsilon_\theta(x_k,o_t^+,s_t,k)
\right\|_2^2
\right]
\]

该设计将一致性约束放到动作扩散的噪声预测空间，而不是简单比较最终动作。

### 6.11 动作保护蒸馏

完整特征分支作为教师，门控矫正分支作为学生：

\[
\mathcal L_{\mathrm{pres}}
=
\mathbb E
\left[
\left\|
\operatorname{sg}
\left(
\epsilon_\theta(x_k,F_l,s_t,k)
\right)
-
\epsilon_\theta
\left(x_k,F_l^{\mathrm{gate}},s_t,k\right)
\right\|_2^2
\right]
\]

### 6.12 跨域协方差对齐

对门控后的域特征 \(Z_d\) 计算协方差矩阵 \(C_d\)：

\[
\mathcal L_{\mathrm{CORAL}}
=
\frac{1}{|\mathcal P|}
\sum_{(d_i,d_j)\in\mathcal P}
\frac{1}{4C^2}
\left\|
C_{d_i}-C_{d_j}
\right\|_F^2
\]

该项借鉴 Deep CORAL 的二阶统计对齐思想。

### 6.13 总损失

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{DP}}
+
\lambda_g\mathcal L_{\mathrm{gate}}
+
\lambda_c\mathcal L_{\mathrm{cons}}
+
\lambda_p\mathcal L_{\mathrm{pres}}
+
\lambda_{\mathrm{coral}}
\mathcal L_{\mathrm{CORAL}}
\]

其中各项的权重必须通过验证集和消融实验选择，不能直接假设固定值有效。

## 7. 训练流程

### 阶段一：策略预热

只优化：

\[
\mathcal L=\mathcal L_{\mathrm{DP}}
\]

目标是让策略先掌握基本动作。

### 阶段二：敏感性统计

冻结或低学习率更新主策略，统计：

- 域分类梯度分数；
- 跨域 KL 分布差异；
- 动作 Taylor 分数；
- 周期性通道遮挡损失。

所有分数使用 EMA，避免门控随 batch 剧烈变化。

### 阶段三：逐步启用门控

目标抑制预算逐步增加：

\[
\rho(t)
=
\rho_{\max}
\left(
\frac{t}{T_{\mathrm{warmup}}}
\right)^p
\]

该训练策略借鉴 PLACE 的渐进式丢弃思想。

### 阶段四：联合微调

同时优化动作去噪、门控、动作一致性、动作保护和跨域统计对齐。

### 推理阶段

推理阶段使用确定性软门控：

\[
F_l^{\mathrm{test}}
=
g_l\odot F_l+
(1-g_l)\odot R_l(F_l,s_t)
\]

第一版不在推理阶段使用随机通道丢弃。
## 8. 对比方法（两轮制）

所有方法使用同一 Diffusion Policy 官方 fork、同一数据划分、同一训练步数与同一随机种子设置。

### 8.1 第一轮：短任务集合筛选

任务集合固定 `atomic_seen`（18 个短任务），每组 3 个随机种子，短训练 + 少量 rollout。

| 编号 | 方法 | 目的 |
|---|---|---|
| A | Vanilla DP（官方配置原样） | 基线 |
| B | Random Dropout | 一般随机正则化 |
| C | KL-only Gate | 只用跨域分布差异 |
| D | DomainDrop-style Gate | 域判别信号指导抑制 |
| E | Action-aware Gate | 加入动作重要性保护 |
| F | DACSC Full | 完整方法 |

### 8.2 第二轮：扩展到全量

只把第一轮最优的 3–5 个配置扩展到：

```text
Atomic-Seen
Composite-Seen
Composite-Unseen
pretrain style split（11–40 / 41–50 / 51–60）
target split（layout 1–10 × style 1–10）
```

### 8.3 扩展对比（第二轮之后）

| 编号 | 方法 | 目的 |
|---|---|---|
| G | Deep CORAL | 仅统计对齐 |
| H | DANN | 仅域对抗表示学习 |
| I | RoboEngine / RoboSaGA / RoboAug / RoboVIP | 数据增广对比 |
| J | 数据增广 + DACSC | 两类方法是否互补 |
| K | DACSC + VLA/WAM backbone | 机制可迁移性验证 |

关键比较：

```text
A vs C：KL 分支是否有效
C vs D：域判别信号是否带来增量
D vs E：动作保护是否有效
E vs F：完整组合是否有效
I vs J：特征处理与数据增广的差异
F vs K：机制是否 backbone-agnostic
```

**注意**：官方 leaderboard 规定只能使用 `pretrain_human300` 且不得改动数据；若计划提交 leaderboard，则第 I/J 类含增广的方法不能作为榜单提交，只能作为离线对比实验。

## 9. 消融实验（两轮制）

### 9.1 第一轮（在 `atomic_seen` + style 划分上）

1. 去除域梯度分数（仅 KL）；
2. 去除 KL（仅域梯度）；
3. 去除动作保护；
4. 去除通道遮挡校准；
5. 门控层集合：只 block 8 / block 6,8 / block 4,6,8 / 全部 8 个（对应 5.4 节）；
6. 门控与 FiLM 顺序：FiLM→Gate 与 Gate→FiLM（对应 5.3 节）；
7. 不使用渐进抑制预算，直接固定抑制比例；
8. 不同抑制比例 \(\rho\) 的扫描。

### 9.2 第二轮（仅对保留的组件）

9. 去除动作一致性；
10. 去除动作保护蒸馏；
11. 去除 CORAL；
12. 固定门控 vs 动态门控；
13. 随机推理 vs 确定性推理；
14. 启用非平凡 \(R_l\)（6.9 节）并与 5.3 节方案 B 对比 —— 用于验证"矫正模块"相对基线 FiLM 是否真有增量。

**规模约束**：第一轮上限 8 组 × 3 种子 = 24 次短训练；第二轮上限 6 组 × 3 种子。不得一开始就训练 28 组以上模型并全部跑完整评测。


## 10. 数据集和划分

### 10.1 RoboCasa365 主实验

RoboCasa365 作为第一阶段主要机器人数据来源，提供：

- 图像；
- 机器人状态；
- 动作序列；
- 任务信息；
- 布局信息；
- 风格信息。

### 10.2 数据审计

正式训练前统计：

```text
task_name × layout_id × style_id × episode_count
```

检查：

- 同一任务是否覆盖多个 style_id；
- 同一 layout_id 是否覆盖多个 style_id；
- 某个 style_id 是否只出现于某个任务；
- 是否存在任务和风格完全绑定；
- 训练、验证和测试轨迹是否发生泄漏。
### 10.3 推荐划分（v0.2 修订）

**废弃**原"训练 style 1/2/3、验证 4、测试 5"的方案：id 1–10 属于 target 场景，不能用于 pretrain 内部划分。

第一轮（单变量研究外观域）：

| 用途 | layout | style |
|---|---|---|
| 训练 | 固定或有限 layout 子集（如 11–20） | 11–40 |
| 验证 | 与训练相同 | 41–50 |
| 测试 | 与训练相同 | 51–60 |
| 扩展测试 | 未见 layout（如 21–60 的子集） | 单独报告 |

第二轮（联合变化）：

```text
训练：多个 layout × 多个 style
测试一：未见 style（同 layout）
测试二：未见 layout（同 style）
测试三：未见 layout 与未见 style 的组合
```

第三轮（官方协议对齐）：`split=pretrain` 与 `split=target` 的全量评测，用于与官方 leaderboard 数值对照。

注意：训练/验证/测试的 style 子集必须通过 `env_kwargs.style_ids` 显式指定，并在 rollout 配置中确认生效；划分正确性由 10.2 节审计脚本的泄漏检查项背书。


## 11. 评测指标

### 11.1 任务成功率

- 已见风格成功率；
- 未见风格成功率；
- 未见布局成功率；
- 组合变化成功率；
- 抓取成功率；
- 放置成功率；
- 完整任务完成率。

### 11.2 泛化差距

\[
\Delta_{\mathrm{gen}}
=
S_{\mathrm{seen}}-S_{\mathrm{unseen}}
\]

### 11.3 风格稳定性

\[
\operatorname{Var}_{d}(S_d)
\]

### 11.4 动作质量

- Diffusion 去噪损失；
- 动作 L1/L2 误差；
- 末端位置误差；
- 末端姿态误差；
- 夹爪状态准确率；
- 轨迹平滑度；
- 碰撞次数；
- 动作抖动次数。

### 11.5 特征和门控分析

- 域分类准确率；
- 通道门控值分布；
- 每层实际抑制比例；
- 域敏感性和动作重要性的相关性；
- 遮挡前后动作损失变化；
- CORAL 距离；
- 不同风格下的特征协方差差异；
- Grad-CAM 或注意力可视化。

### 11.6 计算开销

- 参数量变化；
- 显存占用；
- 单步推理时间；
- 训练时间；
- 矫正模块计算量。

正式结果至少使用多个随机种子，并报告均值、标准差和置信区间。

## 12. 预期结果和判断标准

理想结果应同时满足：

```text
未见风格成功率提高
已见风格成功率基本保持
泛化差距缩小
不同风格之间的性能方差降低
动作预测误差没有明显恶化
通道抑制比例符合预设预算
```

如果未见域成功率提升但已见域大幅下降，说明动作保护不足或抑制比例过高。

如果域分类准确率下降但任务成功率没有提升，说明域信息被删除了，但模型没有学到更好的任务相关表示。

如果动作一致性损失下降而闭环成功率不变，说明离线动作预测一致性不能代表真实闭环泛化，需要增加仿真或真实机器人 rollout。
## 13. 当前创新边界（v0.2 修订）

已有研究基础：

- DomainDrop 的域判别信号指导通道丢弃（机制为判别器权重 × 通道激活，见 4.2）；
- PLACE 的渐进层与通道丢弃；
- DgCD 的分布差异引导通道丢弃（原文待核对）；
- Taylor 一阶结构重要性；
- Deep CORAL 的协方差对齐、DANN 的域对抗训练；
- Diffusion Policy 的动作扩散；
- RoboCasa365 已有的场景多样性研究（附录 H.1：5 scenes 29.6 → 25 scenes 39.6 → 2500 scenes 44.7，zero-shot）。

DACSC-Robot 的候选创新（修订后）：

1. 域梯度信号与跨域分布差异的联合通道决策；
2. 用 Diffusion Policy 去噪损失定义动作保护信号；
3. 在视觉编码器内部做通道级软门控并施加抑制预算；
4. 在动作噪声预测空间施加外观变化一致性约束。

**不再列入候选创新**：机器人状态条件的任务特征矫正（理由见 6.9）。

**时效性风险（2026-10-15 核查）**：

- **PAIR-VLA**（arXiv 2605.13105, 2026-05）用 paired visual variants 做动作不变性/敏感性双目标 RL 微调，与第 4 条在思想上高度重叠；必须取得该文并在论文中明确区分（RL vs 模仿学习、VLA vs DP、全局动作分布 vs 噪声预测空间）。
- 数据侧增广已高度拥挤：DRAIL、RoboSaGA、RoboAug、RoboVIP、NICE、RoboEngine。
- 但检索 2026 年 arXiv 未见"域敏感通道门控用于视觉运动策略"的工作（`abs:"domain-sensitive channels"` 仅命中联邦学习与测试时自适应；`abs:"channel-wise" AND abs:"visuomotor"` 零命中）——第 1、2、3 条仍是干净空位。
- RoboCasa365 官方 robustness 评测（附录 H.2）覆盖语言、相机位姿、关节噪声、底盘位姿扰动，**未覆盖外观/纹理/style 扰动**，该方向仍然成立。

所有结论必须通过实验验证后才能作为论文或专利的技术效果依据。

## 14. 实现路线（v0.2：先审计，后建模块）

```text
阶段 0：数据审计
        扫描 extras/*/ep_meta.json
        → task × layout × style 交叉表
        → 覆盖度、耦合度、泄漏检查、每域样本量

阶段 1：官方基线复现
        锁定 robocasa-benchmark/diffusion_policy
        → 确认配置键名、输入 shape、显存占用
        → 记录 ResNet18ConvFiLM 的 8 个 block 输出位置
        → 记录每个任务的 horizon（registry）
        → Atomic-Seen 上跑通闭环评测

阶段 2：KL 通道评分
        跨 style 特征分布统计
        → 检查不同 style 是否形成稳定分布差异
        → 得到 u^disc

阶段 3：域分类器与域梯度
        detach 训练 D_ψ → 冻结 → 未 detach 特征求梯度
        → 先报告域分类准确率，达标才启用 u^grad

阶段 4：动作 Taylor 重要性
        用 L_DP 估计 v^{Taylor}，周期性通道遮挡校准 v^{mask}

阶段 5：通道门控
        软门控 + 抑制预算 + 门控层集合 + FiLM 顺序

阶段 6：仿真闭环消融（两轮制，见 8、9 节）

阶段 7：RoboCasa target 场景验证（split=pretrain / split=target）

阶段 8：小规模 sim-to-real（可选，参考官方 DROID Panda 协议）
```

**阶段 0 与阶段 1 完成前，不实现门控模块。**

## 15. 参考文献

> 2026-10-15 逐条核验：以下 1–21 条均已确认存在（arXiv ID / DOI 可解析）。第 9 条标题已补全。

### 已有方法

1. Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).
2. Guo et al. [DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization](https://arxiv.org/abs/2308.10285).
3. Guo et al. [PLACE Dropout: A Progressive Layer-wise and Channel-wise Dropout for Domain Generalization](https://arxiv.org/abs/2112.03676).
4. Kim, Park, Lee, Chae. [Discrepancy-guided Channel Dropout for Domain Generalization](https://dl.acm.org/doi/10.1145/3627673.3679539). CIKM 2024.（无 arXiv 版本，原文需另行获取）
5. Molchanov et al. [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf).
6. Sun and Saenko. [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719).
7. Ganin et al. [Domain-Adversarial Training of Neural Networks](https://jmlr.csail.mit.edu/papers/volume17/15-239.html).
8. Bardes et al. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906).

### 机器人视觉域增强与鲁棒性

9. Hattori et al. [Task-Relevant and Irrelevant Region-Aware Augmentation for Generalizable Vision-Based Imitation Learning in Agricultural Manipulation](https://arxiv.org/abs/2603.04845).（标题已补全，原文含副标题）
10. Zhuang et al. [RoboSaGA: Enhancing Visual Domain Robustness in Behaviour Cloning via Saliency-Guided Augmentation](https://arxiv.org/abs/2608.11870).
11. Wang et al. [RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation](https://arxiv.org/abs/2601.05241).
12. Wang et al. [RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation](https://arxiv.org/abs/2602.14032).
13. [Improving Robotic Manipulation Robustness via NICE Scene Surgery](https://arxiv.org/abs/2511.22777).
14. Yuan et al. [RoboEngine: Plug-and-Play Robot Data Augmentation with Semantic Robot Segmentation and Background Generation](https://arxiv.org/abs/2503.18738). IROS 2025.
15. [What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models](https://arxiv.org/abs/2605.13105).（与本文第 4 条候选贡献存在重叠，须在论文中明确区分）

### 基准、数据与官方实现

16. Nasiriany et al. [RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](https://arxiv.org/abs/2603.04356). ICLR 2026.
17. Nasiriany et al. [RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://arxiv.org/abs/2406.02523). RSS 2024.
18. RoboCasa 官方文档（数据集结构 / 基准协议 / 策略学习算法）：https://robocasa.ai/docs/introduction/overview.html
19. 官方 Diffusion Policy fork：https://github.com/robocasa-benchmark/diffusion_policy
20. 官方 robomimic fork（`ResNet18ConvFiLM` 定义所在）：https://github.com/robocasa-benchmark/robomimic ，分支 `robocasa-dp`
21. 官方多任务排行榜与提交记录：https://github.com/robocasa-benchmark/leaderboard

### 引用注意事项

- 第 4 条 DgCD 无 arXiv 版本，6.4 节公式标注为"待按原文核对"；
- DomainDrop 的通道分数是判别器权重 × 通道激活（见 4.2），**不是** Taylor 梯度乘积；引用时须区分；
- 本项目的 H=10 / n_obs=2 / n_action=8 来自第 19 条仓库配置，不是第 1 条 DP 原文的默认值（DP 原文常用 horizon 16）。

## 16. 待确认事项（2026-10-15 核查结果）

### 16.1 已确认（可直接进入实现）

| # | 原问题 | 结论 | 来源 |
|---|---|---|---|
| 1 | RoboCasa365 中任务、布局和 style_id 的覆盖关系 | scene = layout × style；pretrain = 50 layouts(11–60) × 50 styles(11–60) = 2500；target = layout 1–10 × style 1–10 一一配对 = 10 scenes；365 tasks = 65 atomic + 300 composite；官方 50 任务 = 18+16+16 | 论文 2603.04356 §3.2；`scene_registry.py` |
| 2 | Diffusion Policy 使用的具体视觉编码器 | `ResNet18ConvFiLM`（robomimic fork）：3 相机 256×256 → crop 224 → 8 个 conv block（64/64/128/128/256/256/512/512，逐 block 语言 FiLM，CLIP 768-d）→ SpatialSoftmax(32) → Linear → 每相机 64 维；去噪网络为 DiT(12 层/512) | 4.1 节；`base_nets.py`、`obs_core.py`、`diffusion_transformer_hybrid_image_policy.py` |
| 3 | 要在哪些层插入通道门控 | 8 个 conv block 的输出，SpatialSoftmax 之前；第一版用 block 4/6/8 | 5.2、5.4 节 |
| 4 | 动作窗口 H 和观测窗口长度 | H=10、n_obs_steps=2、n_action_steps=8；episode horizon 由 registry 给出，v1.0.1 后为 300–7200 步 | `train_diffusion_transformer_bs192.yaml`；`eval_robocasa.py`；`dataset_registry.py` |
| 5 | 是否能得到可靠的 style_id 域标签 | 可以：`extras/episode_XXXXXX/ep_meta.json` 含 `layout_id`、`style_id`、`lang`；但有三个风险（稀疏、60 类过细、语义边界不齐），须先审计 | `lerobot_utils.py` `save_extra_demo_info`；`kitchen.py` `get_ep_meta` |
| 8 | 服务器显存和训练时间 | 训练 ≥24 GB（48 GB+ 更佳）、推理 ≥8 GB、评测 16 核/64 GB 内存；真正瓶颈是评测（50 tasks × 50 rollouts × 最长 7200 步 × 3 相机） | 官方文档；`env_runner` 注释 |

### 16.2 已决策（本版写入方案）

| # | 原问题 | 决策 |
|---|---|---|
| 6 | 先实现 KL 还是域分类梯度 | **先 KL**（无标签 fallback，不受样本量影响）；再用域分类器，且必须先报告准确率是否高于随机（60 类 = 1.67%）；不达标则合并 style 为粗粒度外观域。域分类器不是第一版必要依赖 |
| 7 | 第一阶段只仿真还是加真机 | **只做仿真闭环**（官方 50 rollouts/task 协议）。真机作为阶段 8 可选：官方用 DROID Panda + 3 相机 + 4 任务 + 140 条示范，Real Only 61.8% → Sim-and-Real 79.8% |

### 16.3 仍待执行 / 待确认

1. 运行 10.2 节数据审计脚本，确认 task × layout × style 交叉覆盖、耦合度与泄漏（阶段 0）；
2. 在 GPU 机器上完成官方 DP 基线 dry run：确认实际配置键名、显存占用、输入 shape、8 个 block 的输出位置、每任务 horizon（阶段 1）；
3. 取得 DgCD 原文（CIKM 2024，无 arXiv 版），核对 6.4 节公式；
4. 取得 PAIR-VLA（2605.13105）全文，明确与 6.10 节动作一致性约束的差异写法；
5. 确认是否计划提交官方 leaderboard（决定能否使用数据增广类方法）；
6. 根据审计结果决定 style 是否归并为粗粒度外观域，以及域分类器的类别数。

### 16.4 本地环境现状（2026-10-15）

- 工作机为 MacBook Air M4 / 16 GB，无 NVIDIA GPU、未安装 PyTorch；
- RoboCasa 数据集与 DP fork 均不在本地；
- 因此阶段 0 需在数据所在机器执行，阶段 1 需在训练服务器执行。
