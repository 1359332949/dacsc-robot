# DACSC-Robot 方法定义与实验设计

> 版本：研究设计初稿  
> 状态：方法定义与实验方案，尚未声称实验结论  
> 主策略：Diffusion Policy  
> 主研究问题：机器人视觉策略在未见外观域下的动作泛化

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

一条机器人示范轨迹表示为：

\[
\tau_i=\{(o_t,s_t,a_t,d_i,l_i,q_i)\}_{t=1}^{T_i}
\]

其中：

- (o_t)：第 (t) 时刻的视觉观测；
- (s_t)：机器人本体状态，例如关节位置、末端位姿和夹爪状态；
- (a_t)：机器人动作；
- (d_i)：外观域标签，例如风格、光照或背景配置；
- (l_i)：场景布局标签；
- (q_i)：任务标签；
- (T_i)：第 (i) 条轨迹的长度。

策略通常接收连续 (H) 步动作窗口：

\[
\pi_\theta(o_t,s_t)\rightarrow a_{t:t+H-1}
\]

数据划分时，(d_i) 不能自动等同于纯外观因素。RoboCasa365 中的风格、布局、家具和场景配置可能存在关联，因此正式实验前必须统计：

```text
task_name, layout_id, style_id, episode_count
```

第一轮实验应尽量固定任务和布局，仅改变 style_id，降低任务、几何和外观之间的混杂。

## 4. 相关方法及其作用

### 4.1 Diffusion Policy：动作策略基础

Diffusion Policy 将机器人动作序列表示为条件扩散过程，训练模型从带噪动作中恢复专家动作。它适用于高维连续动作，并能表示多个合理动作模式。

本文将 Diffusion Policy 作为动作生成器，而 DACSC-Robot 处理其前面的视觉特征。

### 4.2 DomainDrop：域敏感通道抑制

DomainDrop 使用域分类器识别中间视觉特征中对域变化敏感的通道，并在训练过程中对这些通道进行抑制，同时配合层级训练和一致性约束提升域泛化能力。

本文借鉴其“利用域判别信号定位不稳定通道”的思想，但不把动作任务中的通道全部视为可删除特征。

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

DACSC-Robot 的处理链为：

```text
视觉观测 o_t
    ↓
视觉编码器 E_φ
    ↓
多层特征 F_l
    ↓
域敏感性估计 u_{l,c}
    +
动作重要性估计 v_{l,c}
    ↓
软通道门控 g_{l,c}
    ↓
状态条件任务特征矫正 R_l(F_l,s_t)
    ↓
Diffusion Policy
    ↓
动作序列 a_{t:t+H-1}
```

DACSC-Robot 的候选贡献由四部分构成：

1. 使用域分类梯度和跨域分布差异联合识别域敏感通道；
2. 使用动作去噪损失的 Taylor 重要性保护动作关键通道；
3. 使用机器人状态条件的残差模块矫正被抑制的视觉特征；
4. 在 Diffusion Policy 的噪声预测空间施加动作一致性和动作保护约束。

## 6. 公式定义

### 6.1 Diffusion Policy 动作去噪损失

对动作序列 (a) 加入扩散噪声：

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

### 6.2 域分类器

第 (l) 层视觉特征为：

\[
F_l\in\mathbb R^{C_l\times H_l\times W_l}
\]

对通道进行全局平均池化：

\[
\bar f_{l,c}
=
\frac{1}{H_lW_l}
\sum_{h=1}^{H_l}
\sum_{w=1}^{W_l}F_{l,c,h,w}
\]

域分类器为：

\[
\hat d=D_\psi(\bar f_l)
\]

域分类损失为：

\[
\mathcal L_{\mathrm{dom}}
=
\operatorname{CE}(\hat d,d)
\]

第一版建议对域分类器输入使用停止梯度：

\[
\mathcal L_{\mathrm{dom}}
=
\operatorname{CE}
\left(
D_\psi(\operatorname{sg}(\bar f_l)),d
\right)
\]

这样域分类器主要负责测量域敏感性，不会在早期直接扰乱动作策略。

### 6.3 基于域分类梯度的敏感性

借鉴 Taylor 一阶结构重要性形式，定义：

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

该分数越大，说明通道对域分类结果越敏感。

### 6.4 基于跨域分布差异的敏感性

对于域 (d)，估计通道激活的高斯近似：

\[
q_{l,c}^{(d)}
=
\mathcal N
\left(
\mu_{l,c}^{(d)},
(\sigma_{l,c}^{(d)})^2
\right)
\]

所有域的总体分布为 (q_{l,c}^{(*)})。定义：

\[
u^{\mathrm{disc}}_{l,c}
=
\frac{1}{|\mathcal D|}
\sum_{d\in\mathcal D}
D_{\mathrm{KL}}
\left(
q_{l,c}^{(d)}
\middle\|
q_{l,c}^{(*)}
\right)
\]

对于高斯分布：

\[
D_{\mathrm{KL}}
\left(
\mathcal N(\mu_d,\sigma_d^2)
\middle\|
\mathcal N(\mu_*,\sigma_*^2)
\right)
=
\log\frac{\sigma_*}{\sigma_d}
+
\frac{\sigma_d^2+(\mu_d-\mu_*)^2}
{2\sigma_*^2}
-\frac12
\]

该项属于 DgCD 思路在机器人视觉动作特征上的候选改造，正式实现前需要根据 DgCD 原文代码确认统计窗口和通道梯度的具体计算方式。

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

其中 (F_l^{(-c)}) 表示删除第 (c) 个通道后的特征。最终分数可以写为：

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

令 (g_{l,c}\in[0,1]) 表示通道保留程度：

\[
g_{l,c}=\sigma(\eta_{l,c})
\]

被门控的特征为：

\[
F_l^{\mathrm{gate}}
=
g_l\odot F_l
+
(1-g_l)\odot R_l(F_l,s_t)
\]

这里的 (R_l) 用于恢复被抑制通道中可能存在的任务相关信息。

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
- (ho) 为目标平均抑制比例。

该公式是 DACSC-Robot 的候选组合设计，不能描述为 DomainDrop、DgCD 或 Taylor 原文中的原始公式。

### 6.9 状态条件任务特征矫正

矫正模块可以采用轻量残差结构：

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

第一版不使用大型 Transformer 矫正器，以便区分收益来自通道门控还是复杂网络容量。

### 6.10 动作一致性

对同一动作轨迹构造原始观测 (o_t) 和外观变化观测 (o_t^+)，使用相同的带噪动作 (x_k)：

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

对门控后的域特征 (Z_d) 计算协方差矩阵 (C_d)：

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

## 8. 对比方法

所有方法应尽量使用同一个 Diffusion Policy、同一数据划分、同一训练步数和同一随机种子设置。

| 编号 | 方法 | 目的 |
|---|---|---|
| A | Vanilla Diffusion Policy | 基础性能 |
| B | 普通随机 Dropout | 一般随机正则化 |
| C | PLACE-style Dropout | 渐进层和通道随机丢弃 |
| D | DomainDrop-style | 域分类器指导抑制 |
| E | DgCD-style | 跨域分布差异指导抑制 |
| F | DomainDrop + 动作保护 | 检验动作保护作用 |
| G | DgCD + 动作保护 | 检验分布差异和动作保护组合 |
| H | DACSC-gate | 仅使用联合门控 |
| I | DACSC-gate + correction | 加入特征矫正 |
| J | DACSC-Robot | 完整方法 |
| K | Deep CORAL | 仅统计对齐 |
| L | DANN | 仅域对抗表示学习 |
| M | DRAIL、RoboSaGA 或 RoboVIP | 数据增广对比 |
| N | 数据增广 + DACSC-Robot | 验证互补性 |

关键比较：

```text
C vs D vs E：不同域敏感性处理方式
D vs F：动作保护是否有效
E vs G：动作重要性是否改善 DgCD
H vs I：矫正模块是否有效
I vs J：一致性和统计对齐是否有效
A vs J：完整方法是否提升域泛化
M vs J：特征处理和数据增广的差异
M vs N：两类方法是否互补
```

## 9. 消融实验

至少包含以下消融：

1. 去除域分类梯度分数；
2. 去除跨域 KL 分布差异；
3. 只使用域敏感性，不使用动作保护；
4. 去除 Taylor 动作重要性；
5. 去除通道遮挡校准；
6. 去除状态条件矫正；
7. 去除动作一致性；
8. 去除动作保护蒸馏；
9. 去除 CORAL；
10. 不使用渐进抑制预算；
11. 比较不同抑制比例；
12. 比较不同插入层；
13. 比较固定门控和动态门控；
14. 比较随机推理和确定性推理。

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

### 10.3 推荐划分

第一轮：

```text
训练：固定任务和布局，style 1、2、3
验证：相同任务和布局，style 4
测试：相同任务和布局，style 5
```

第二轮：

```text
训练：多个布局和多个风格
测试：未见布局与未见风格组合
```

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

## 13. 当前创新边界

以下属于已有研究基础：

- DomainDrop 的域敏感通道抑制；
- PLACE 的渐进层和通道丢弃；
- DgCD 的分布差异引导通道丢弃；
- Taylor 一阶结构重要性；
- Deep CORAL 的协方差对齐；
- DANN 的域对抗训练；
- Diffusion Policy 的动作扩散。

DACSC-Robot 的候选创新集中在：

1. 将域敏感性和动作重要性用于机器人视觉通道联合决策；
2. 使用 Diffusion Policy 去噪损失定义动作保护信号；
3. 对被抑制特征使用机器人状态条件的任务特征矫正；
4. 在动作噪声预测空间中进行外观变化一致性约束；
5. 通过抑制预算、动作保护和分布对齐共同避免过度删除。

这些内容必须通过对比实验和消融实验验证后，才能进一步作为论文或专利中的技术效果依据。

## 14. 实现路线

### 第一阶段：基础复现

```text
RoboCasa365 数据审计
→ Vanilla Diffusion Policy
→ 闭环评测脚本
→ 风格和布局划分
```

### 第二阶段：通道分析

```text
提取多层特征
→ 域分类器
→ 域梯度分数
→ 跨域分布差异
→ 动作 Taylor 分数
```

### 第三阶段：门控实现

```text
软门控
→ 抑制预算
→ PLACE 式渐进训练
→ DomainDrop 和 DgCD 对比
```

### 第四阶段：DACSC 完整实现

```text
动作保护
→ 状态条件矫正
→ Diffusion 噪声一致性
→ CORAL 对齐
```

### 第五阶段：扩展实验

```text
DRAIL / RoboSaGA / RoboVIP 数据增广
→ ACT
→ Octo
→ π₀
```

## 15. 参考文献

1. Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).
2. Guo et al. [DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization](https://arxiv.org/abs/2308.10285).
3. Guo et al. [PLACE Dropout: A Progressive Layer-wise and Channel-wise Dropout for Domain Generalization](https://arxiv.org/abs/2112.03676).
4. Kim et al. [Discrepancy-guided Channel Dropout for Domain Generalization](https://dl.acm.org/doi/10.1145/3627673.3679539).
5. Molchanov et al. [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf).
6. Sun and Saenko. [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719).
7. Ganin et al. [Domain-Adversarial Training of Neural Networks](https://jmlr.csail.mit.edu/papers/volume17/15-239.html).
8. Bardes et al. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906).
9. Hattori et al. [Task-Relevant and Irrelevant Region-Aware Augmentation for Generalizable Vision-Based Imitation Learning](https://arxiv.org/abs/2603.04845).
10. Zhuang et al. [RoboSaGA: Enhancing Visual Domain Robustness in Behaviour Cloning via Saliency-Guided Augmentation](https://arxiv.org/abs/2608.11870).
11. Wang et al. [RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation](https://arxiv.org/abs/2601.05241).
12. Wang et al. [RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation](https://arxiv.org/abs/2602.14032).

## 16. 待确认事项

在开始正式代码实现前，需要确认：

1. RoboCasa365 中任务、布局和 style_id 的覆盖关系；
2. Diffusion Policy 使用的具体视觉编码器；
3. 要在哪些层插入通道门控；
4. 动作窗口 (H) 和观测窗口长度；
5. 是否能得到可靠的 style_id 域标签；
6. 是否先实现 KL 分布差异，还是先实现域分类梯度；
7. 第一阶段是否只使用仿真闭环，还是同时加入真实机器人实验；
8. 服务器显存和训练时间能否支持多组消融实验。
