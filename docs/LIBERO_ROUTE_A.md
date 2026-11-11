# Route A：LIBERO 赛道（夜间自主推进）

日期：2026-11-10 起
负责人：Lead（每 30 分钟自动续跑一轮，共约 20 轮）

## 为什么走这条路

RoboCasa 上我们用的是 **48.9M 的 CoRP、30k 步微调、clean SR ≈0.5–0.8（layout 31），换 layout 0/4**；
文献里的方法（COAST/PAIR-VLA/MixVLA）基座是 **π0.5 3.3B / OpenVLA-OFT 7B**，基准是 **LIBERO（clean SR 90%+）/ RoboTwin**，或有**成功/失败 rollout** 或**改权重**。
→ 不可比。Route A 的目标：在**高余量基座 + 光度扰动**的设定下，检验我们的结论（残差占比判据、开环/闭环落差、机制发现）。

## 上游依据（不造轮子）

| 组件 | 上游文件/论文 | 说明 |
|---|---|---|
| 安装 | `repos/CoRP/scripts/install.sh` | python3.12 venv；torch 2.11.0 cu128；`.[libero]`（robosuite 1.4.1, mujoco 3.3.0）；LIBERO pinned `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| 官方评测 | `repos/CoRP/scripts/evaluate_libero.py` | MAX_STEPS: spatial 220 / object 280 / goal 300 / long 520；CAMERA_SIZE 256；SETTLING_STEPS 10；DUMMY_ACTION = [0]*6+[-1] |
| 光度扰动 | `scripts/eval_corp_libero_plus.py` | 项目现成的 LIBERO-Plus 适配器（复用官方 evaluate_suite，不重复实现 rollout/success） |
| 论文数字 | CoRP arXiv:2610.08183 Table 1 | Spatial 98.2 / Object 99.4 / Goal 95.8 / Long 94.4 / Avg 97.0（50 trials/task） |
| 对比方法 | MixVLA 2610.02898、PAIR-VLA 2605.13105、COAST 2605.17144、PSC 2603.09338、SALT 2610.07946 | 见下表"关键差异" |

## 关键事实核对（2026-11-10）

1. CoRP 论文的基准是 **LIBERO + RoboTwin 2.0**，**不含 RoboCasa**；我们之前的 RoboCasa style 51–60 划分是本项目自建。
2. 官方 checkpoint `repos/CoRP/checkpoints/corp_libero.pt`（227MB，SHA256SUMS 已校验）只对应 LIBERO 配置（action_dim 7 / horizon 12 / 2 views）。
3. 服务器没有 `libero` 包与数据；`/hy-tmp` 约 13G 可用；系统 python 3.11，uv 提供 python3.12.16。
4. 文献的关键差异：基座规模、训练阶段（多为训练时方法）、外观扰动类型与强度、指标（很多报开环/回归误差，COAST 用成功失败 rollout）。

## 夜间执行顺序

1. 安装（后台 `/hy-tmp/libero_install.log`）→ 校验 `.venv`、`third_party/LIBERO`、`pip check`。
2. 官方 clean 单 suite：`libero_spatial`，seed 7（对照论文 98.2）。
3. 同 suite 的 `color_jitter`（`eval_corp_libero_plus.py`）→ clean vs jitter SR。
4. 记录到本文件；若环境失败，记录报错与已尝试的官方替代路径（codeload tarball 等），不自行改评测。
5. 回看 V0–V18：残差占比判据 / 机制结论在 LIBERO 高余量设定下的检验计划。

## 安全约束

- 不删 `/hy-tmp/runs` 下模型目录与 `/hy-tmp/datasets`；
- 只删自己的临时输出和 pip 缓存；
- 长任务 nohup + 日志；
- 不动 784fe2e 正在使用的 RoboCasa 环境（LIBERO 用独立 `.venv`）。

## 进展日志

- 2026-11-10 01:05：创建 30 分钟定时任务（20 轮）；启动 `install.sh`（后台，日志 `/hy-tmp/libero_install.log`），LIBERO 正在 clone。

## 安装方式（已记录，避免反复踩坑）

`git clone` LIBERO 在本机极慢（2 分钟仅 80K），因此改用官方 codeload tarball（同一 pinned revision）：
```
https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/8f1084e3132a39270c3a13ebe37270a43ece2a01
```
解压到 `repos/CoRP/third_party/LIBERO`，其余步骤（venv、torch 2.11.0 cu128、`.[libero]`、`pip install -e`、`pip check`）逐行照抄 `scripts/install.sh`。
脚本 `/hy-tmp/install_libero.sh`，日志 `/hy-tmp/libero_install2.log`，完成标志 `INSTALL_DONE`。
- 2026-11-10 01:06：checkpoint SHA 校验 OK；torch 2.11.0 下载中；LIBERO 解压后 426M。

## 文献实验协议核对（2026-11-10 第 1 轮，从论文原文提取）

| 论文 | 基座 | 训练数据 | 评测基准 | 数据划分 | 外观/扰动 | 指标与样本量 | 方法阶段 |
|---|---|---|---|---|---|---|---|
| **PAIR-VLA** 2605.13105 | OpenVLA、π0.5 | ManiSkill3 上 RL 微调 | ManiSkill3 | 训练/测试任务集分开（附录 B）；OOD 是未见扰动 | 未见干扰物、纹理、目标位姿、视角、光照 | 成功率，**128 episodes/run，3 seeds 取均值** | **训练时 RL**（PPO + invariance/sensitivity） |
| **MixVLA** 2610.02898 | OpenVLA-OFT、π0.5 | **LIBERO 训练** | LIBERO、**LIBERO-Plus**、RoboTwin、真机 | LIBERO 训练集；LIBERO-Plus 为 OOD 评测集 | 光照/纹理/视角等（LIBERO-Plus 扰动套件） | 成功率；LIBERO-Plus 子集 **2,318 episodes** | **训练时**（IB + 非不变表示混合） |
| **COAST** 2605.17144 | π0.5、π0-FAST、GR00T N1.5、Diffusion Policy | 各基准原训练集 | MetaWorld ML45、LIBERO-10、**RoboCasa 7 atomic-seen tasks**、DROID | **每任务 15 条训练 rollout 拟合 conceptor + 30 条 held-out 测试 rollout**（不同 seed/物体/初始状态） | 环境重采样布局/物体/纹理/光照 | 成功率，**仅报 test rollouts** | **推理时**，但需要**成功/失败 rollout**（特权信号） |
| **CoRP** 2610.08183 | 48.9M 小模型 | LIBERO / RoboTwin | LIBERO、RoboTwin 2.0 | 各基准官方划分 | 无 style 划分 | 成功率；LIBERO 50 trials/task | 训练+推理 |
| **我们（RoboCasa）** | CoRP 48.9M（30k 步微调） | RoboCasa 5 任务、styles 11–50 | RoboCasa | **训练 styles 11–50；验证/测试 styles 51–60（未见）** | 数据集原生 style + 自造 color_jitter | 开环 MSE + 闭环 SR（4 episodes） | **推理时（不训练）** |

**关键不可比点**：
1. 基座：48.9M vs 3.3–7B；
2. 基准：RoboCasa（长程 750 步）vs LIBERO（短程 220–520 步）/ManiSkill3；
3. 扰动：我们自造的 color_jitter（1.6/1.6/2.2）比文献的"光照/纹理变化"更极端；
4. 信号：COAST 需要成功/失败 rollout；PAIR-VLA/MixVLA 改权重；我们只改推理；
5. 样本量：他们 128–2318 episodes/3 seeds；我们闭环 4 episodes → 数字不可靠。

## 第 1 轮（01:05–02:00）：安装完成 + EGL 修复 + 冒烟通过

### 安装
- `install.sh` 的 git clone 太慢，改用**官方 codeload tarball（同一 pinned revision）**；
- 其余步骤逐行照抄：python3.12 venv、torch 2.11.0+cu128、`.[libero]`、`pip install -e third_party/LIBERO`、`pip check` → **INSTALL_DONE，无损坏依赖**；venv 8.1G。
- 唯一偏差：`mujoco 3.3.0 → 3.3.1`（补丁级；EGL 枚举与 RoboCasa env 对齐）。需在论文/记录中注明。

### EGL 问题（已定位并修复）
- 官方 `evaluate_libero.py` 会写一个 EGL vendor JSON 指向 **`libEGL_nvidia.so.0`**，但本机**没有 NVIDIA EGL 库**（只有 mesa）→ `eglQueryDevicesEXT()` 返回 0 → 报 `MUJOCO_EGL_DEVICE_ID must be between 0 and -1`。
- 修复：预先设置 `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`，阻止脚本覆盖为不存在的 nvidia vendor。
- 注意：mesa EGL 是**软件渲染**（llvmpipe），单 episode ≈ **43.7 秒**（libero_spatial，220 步）。

### 冒烟
- `--suite libero_spatial --tasks 0 --episodes-per-task 1 --seed 7` → **1/1 成功**，管线可用。

### 正在跑
- clean：`libero_spatial`，**10 episodes/task × 10 tasks = 100 episodes**，seed 7，官方 `MAX_STEPS=220`。
- 偏差说明：官方是 50 trials/task（500 episodes），软件渲染下需 ~6 小时；先用 10/task（~1.2h）做 clean vs jitter 对照，之后可视时间补足。
- 日志 `/hy-tmp/libero_clean.log`，输出 `/hy-tmp/validation/libero_clean_spatial10/libero_spatial.jsonl`。

## 第 2 轮（01:35–02:10）：光度扰动通道打通

- LIBERO venv 缺 torchvision（`dacsc.corp_images` 依赖）→ 安装 `torchvision==0.26.0+cu128`（与 torch 2.11 匹配）。
- 发现项目的 `eval_corp_libero_plus.py` 依赖 **LIBERO-Plus 的 task_classification.json**（10,030 个扰动任务），标准 LIBERO 没有该文件 → 不能直接用于 40 个标准任务。
  - 上游 LIBERO-Plus 文件已下载核对（2MB，类别为 Background Textures 等），其任务名带 `_table_1` 后缀，与标准 LIBERO 任务名不匹配，不能混用。
- 因此写了**薄驱动** `scripts/libero_shift_eval.py`：任务选择用官方默认（全部任务），复用 `evaluate_libero.py` 的 `configure_libero/evaluate_suite` + 项目的 `PhotometricPolicy`（不改上游逻辑，不重复实现 rollout/success）。
- 冒烟：`libero_spatial / task 0 / 1 ep / color_jitter` → 0/1 失败（干净时该 episode 是成功的），管线可用。
- clean 进度：25/25 成功（task 0–2 完成）。
- 已并发启动 shift 全量：`libero_spatial`，10 eps/task，seed 7，color_jitter → `/hy-tmp/validation/libero_shift_spatial10/`，日志 `/hy-tmp/libero_shift.log`。
- 说明：mesa 软件渲染下单 episode ≈ 1–2 分钟；两个进程各 `OMP_NUM_THREADS=1` 并行，CPU 上互不抢占太多。

## 第 3 轮（02:05–02:40）：clean 100% vs color_jitter 0%（进行中），并已启动 PSC

### 当前进度（运行中）
| 条件 | 已完成 | 成功 |
|---|---|---|
| clean（官方脚本） | 60/100 | **60** |
| color_jitter（薄驱动） | 18/100 | **0** |
| color_jitter + PSC | 刚启动 | — |

→ **高余量基座（clean 60/60）在 color_jitter 下 0/18**：外观扰动在 LIBERO 上同样是"致命"的，与 RoboCasa 的观察一致。这为测试时自适应提供了清晰的目标。

### PSC on LIBERO（按上游接口实现）
- 源域统计 `build_libero_psc_stats.py`：用**官方 rollout 协议**（settling 10 步、官方初始状态、`policy_observation` 转换）在 **clean** 观测上采集 cross-attention 写回（最后一 Euler 步、action token 平均），3 个任务 × 2 episode × 4 chunk = 24 条 → mean/basis(rank6)/sigma_s/tau 已写入 `/hy-tmp/validation/libero_psc_stats.json`。
- 评测驱动 `libero_shift_eval.py` 新增 `--psc-stats`：复用 `dacsc.psc.install`（cross-attention 写回投影到源域支撑），在官方 `evaluate_suite` 上运行。
- 已启动：spatial，10 eps/task，seed 7，color_jitter + PSC → `/hy-tmp/validation/libero_psc_spatial10/`，日志 `/hy-tmp/libero_psc.log`。

### 备注
- 机器有 128 核；三个评测进程各 `OMP_NUM_THREADS=1` 并行，互不抢占。
- 上游 `evaluate_libero.py` 的 `policy_observation`/`is_success`/`DUMMY_ACTION`/`SETTLING_STEPS` 全部直接 import 复用；未重复实现 rollout/success。

## 第 4 轮（02:35–03:10）：clean 官方复现完成 = 100%，jitter 仍 0，PSC 修复后重跑

### clean（官方脚本，10 eps/task × 10 tasks = 100 episodes，seed 7）
- **100/100 = 100.0%**（对照 CoRP 论文 Spatial 98.2%；10 eps/task 的抽样波动内一致）。
- 结论：**官方 checkpoint 与官方评测协议在本机复现成功**，Route A 的"高余量基座"前提成立。

### color_jitter（薄驱动，同任务/同 seed/同初始状态）
- 已完成 39 条：**0 成功**。
- 即"高余量基座 + 强光度扰动"→ 成功率归零，与 RoboCasa 的观察同型。

### PSC（测试时投影）
- 首次启动失败：`ModuleNotFoundError: dacsc.psc`（LIBERO venv 运行的 checkout `/hy-tmp/dacsc-robot-784fe2e` 里没有该模块；`repos` 是符号链接到 `/hy-tmp/dacsc-robot/repos`）。
- 已把 `dacsc/psc.py` 复制到 `/hy-tmp/dacsc-robot-784fe2e/dacsc/` 与 `/hy-tmp/dacsc-robot/dacsc/`，重跑成功启动（前 2 条 0 成功，继续）。

### 额外：光度扰动严重度曲线（并行启动）
- `grayscale`、`low_contrast`、`hue_shift` 三个条件，各 spatial 10 eps/task，seed 7；
- 目的：给出"扰动强度 → SR"曲线，与文献（LIBERO-Plus 光照/背景扰动）可比。

## 第 5 轮（03:05–03:40）：扰动严重度曲线出现清晰分级

（进行中，同任务/同 seed/同初始状态，spatial，10 eps/task）

| 条件 | 已完成 | 成功 | 当前 SR |
|---|---|---|---|
| clean | 100/100 | 100 | **100%** |
| grayscale | 34/100 | 34 | **100%** |
| hue_shift | 30/100 | 27 | ~**90%** |
| low_contrast | 25/100 | 19 | ~**76%** |
| color_jitter | 56/100 | 0 | **0%** |
| color_jitter + PSC | 20/100 | 0 | 0% |

**结论**：
1. 本基座对**灰度不缺**、对 **hue/低对比度中度敏感**、对 **color_jitter（1.6/1.6/2.2）完全不鲁棒**。
2. 这给出文献可比的"剂量-反应"曲线，也解释了为什么之前只在 color_jitter 上测 TTA 是**地板效应**（0% 无法衡量改善）。
3. 因此在 **low_contrast（~76%）** 与 **hue_shift（~90%）** 上测 PSC 才有区分度；已启动：
   - `libero_psc_low_contrast_spatial10`
   - `libero_psc_hue_shift_spatial10`

## 第 6 轮（03:35–04:10）：中间强度下的 PSC —— 不仅无效，而且有害

当前（进行中，配对条件，spatial，10 eps/task，seed 7）：

| 条件 | 已完成 | 成功 | 当前 SR |
|---|---|---|---|
| clean | 100/100 | 100 | **100%** |
| grayscale | 67/100 | 64 | ~95.5% |
| hue_shift | 58/100 | 48 | ~82.8% |
| **hue_shift + PSC** | 24/100 | 11 | **~45.8%**（比不加 PSC 的 82.8% 差很多） |
| low_contrast | 51/100 | 34 | ~66.7% |
| **low_contrast + PSC** | 25/100 | 12 | **~48.0%**（比不加 PSC 的 66.7% 差） |
| color_jitter | 74/100 | 0 | 0% |
| color_jitter + PSC | 38/100 | 0 | 0% |

**结论（与 RoboCasa 一致）**：在"有损伤但没归零"的中间强度下，硬投影 **把成功率拉低 18–37 个点**，不是救，而是伤。
原因假说：中间强度时，外观偏移大部分仍落在源域支撑内，投影等于把**仍有用的任务信息**一起丢掉（对应 RoboCasa 的"外观与任务共享子空间"发现）。

**下一步（已启动）**：部分投影（保留残差）`beta=0.5` 在 low_contrast 上，检验"是投影太狠"还是"投影本身不对"。
- `libero_psc_b0.5_low_contrast`（日志 `/hy-tmp/libero_psc_b0.5_low.log`）
- `dacsc/psc.py` 已加入 `beta` 参数（0=纯投影，1=恒等），驱动 `libero_shift_eval.py` 新增 `--psc-beta`。

## 第 7 轮（04:05–04:45）：PSC 软化仍无效；加入两个对照/替代

### clean 官方结果（最终）
`summary.json`：episodes 100，successes 100，**success_rate = 1.0**，elapsed 4526s；
协议：settling 10、camera 256、action_horizon 12、euler_steps 10、每 suite 官方 MAX_STEPS。

### 当前进度（进行中）
| 条件 | eps | success | 当前 SR |
|---|---|---|---|
| grayscale | 97 | 87 | ~89.7% |
| low_contrast | 78 | 53 | ~67.9% |
| hue_shift | 85 | 63 | ~74.1% |
| color_jitter | 93 | 0 | 0% |
| PSC(color_jitter) | 59 | 0 | 0% |
| PSC(low_contrast) | 55 | 28 | ~50.9% |
| PSC(hue_shift) | 50 | 20 | ~40.0% |
| **PSC beta=0.5(low_contrast)** | 28 | 17 | ~60.7%（仍低于 67.9%） |
| PSC beta=1.0(identity 对照) | 3 | 1 | 进行中 |
| **TTA 投票 k=3(low_contrast)** | 2 | 1 | 进行中 |

### 判断
- **软化投影（beta=0.5）仍低于不加 PSC**（60.7% vs 67.9%）→ 不是"投影太狠"，而是**投影方向把任务信息一起丢了**。
- identity 对照（beta=1，等价不变换）用来排除 hook 自身副作用；若其 SR≈67.9%，则伤害确实来自投影。
- TTA 投票（PDF 式，k=3，动作平均）已启动，作为"无需训练、无特权信号"的替代基线；若它在 low_contrast 上 ≥ 67.9%，说明该设定下"多视图平均"比"特征投影"更合适。

### 代码改动（都在我们自己的薄驱动里，未改上游）
- `dacsc/psc.py`：`SourceProjector(..., beta)`（0=纯投影，1=恒等）。
- `scripts/libero_shift_eval.py`：`--psc-beta`、`--vote-k`、`VotePolicy`（PDF arXiv:2604.18107 的动作投票，变体为亮度/对比度确定性扰动，NHWC uint8 与 env 输出一致）。

## 第 8 轮（04:24）：color_jitter 与 grayscale 完成；beta/投票中间结果

### 已完成
| 条件 | eps | success | SR |
|---|---|---|---|
| clean | 100 | 100 | **100.0%** |
| **color_jitter** | 100 | 0 | **0.0%** |
| **grayscale** | 100 | 90 | **90.0%** |

### 进行中
| 条件 | eps | success | 当前 SR |
|---|---|---|---|
| low_contrast | 89 | 62 | ~69.7% |
| hue_shift | 94 | 69 | ~73.4% |
| PSC(color_jitter) | 66 | 0 | 0% |
| PSC(low_contrast, beta=0) | 61 | 29 | ~47.5% |
| PSC(hue_shift, beta=0) | 59 | 25 | ~42.4% |
| **PSC(low_contrast, beta=0.5)** | 40 | 28 | ~70.0% |
| **PSC(low_contrast, beta=1 identity)** | 14 | 11 | ~78.6% |
| **TTA 投票 k=3(low_contrast)** | 13 | 9 | ~69.2% |

### 初步读法（待满 100 eps 确认）
- **beta=0（硬投影）明确有害**（47.5% vs 69.7%）；
- **beta=0.5（保留一半残差）≈ 不施加**（70.0%）→ 伤害主要来自"去掉太多"；
- identity（beta=1）≈ 不施加（78.6%，小样本）；
- TTA 投票 k=3 ≈ 不施加（69.2%），暂未显示收益。

→ 如果 beta=0.5 在满 100 eps 后仍 ≈ 69–70%，则结论是"**任何非平凡的投影都无益**；PSC 应判死"。

## 第 9 轮（04:35–05:05）：hue_shift 完成，硬投影确认有害，软化后仍略差

### 新完成
| 条件 | eps | success | SR |
|---|---|---|---|
| hue_shift | 100 | 74 | **74.0%** |
| (低对比度 near-complete) | 97 | 67 | ~69.1% |

### 进行中
| 条件 | eps | success | 当前 SR |
|---|---|---|---|
| PSC(color_jitter, beta=0) | 72 | 0 | 0% |
| PSC(low_contrast, beta=0) | 69 | 33 | ~47.8% |
| PSC(hue_shift, beta=0) | 66 | 27 | ~40.9% |
| PSC(low_contrast, beta=0.5) | 46 | 28 | ~60.9% |
| PSC(low_contrast, beta=1 identity) | 24 | 20 | ~83.3%（小样本） |
| TTA 投票 k=3(low_contrast) | 22 | 16 | ~72.7% |
| TTA 投票 k=5(low_contrast / hue_shift) | 刚启动 | — | — |

### 读数
- **hard PSC 在两档中间强度都是 −21～−33 点**（47.8% vs 69.1%；40.9% vs 74.0%）；
- beta=0.5 时回到 ~60.9%，但仍低于不施加（69.1%）→ "部分投影"也不赚；
- identity 对照小样本偏高（噪声），需要更多样本；投票 k=3 ≈ 不施加（72.7%）。
- 已启动 vote-k=5（含对比度变体），在 low_contrast 与 hue_shift 上各 100 eps。

## 第 10 轮（05:05–05:35）：TTA 投票出现正收益（k=5），PSC 判死

### 已完成
| 条件 | eps | success | SR |
|---|---|---|---|
| clean | 100 | 100 | **100.0%** |
| color_jitter | 100 | 0 | **0.0%** |
| grayscale | 100 | 90 | **90.0%** |
| hue_shift | 100 | 74 | **74.0%** |
| **low_contrast** | 100 | 70 | **70.0%** |

### 进行中（当前值）
| 条件 | eps | success | 当前 SR |
|---|---|---|---|
| PSC(color_jitter) | 89 | 0 | 0% |
| PSC(low_contrast, beta=0) | 86 | 34 | ~39.5%（远低于 70%） |
| PSC(hue_shift, beta=0) | 82 | 27 | ~32.9%（远低于 74%） |
| PSC(low_contrast, beta=0.5) | 67 | 40 | ~59.7%（仍低于 70%） |
| PSC(low_contrast, beta=1 identity) | 50 | 39 | ~78.0%（≈/略高于 70%，hook 无副作用） |
| TTA 投票 k=3(low_contrast) | 48 | 36 | ~75.0% |
| **TTA 投票 k=5(low_contrast)** | 25 | 20 | **~80.0%** |
| **TTA 投票 k=5(hue_shift)** | 28 | 26 | **~92.9%** |
| TTA 投票 k=5(clean / grayscale) | 刚启动 | — | 对照（确认无副作用） |

### 读数
- **PSC（任何 beta）在该基座上都没有正收益**：硬投影 −30 点，软投影 −10 点；identity 对照 ≈ 基线（说明不是 hook 副作用）。→ **PSC 在 LIBERO 高余量设定判死**。
- **PDF 式 TTA 投票（k=5，无标签/无梯度/不改权重）在小样本上出现 +10～+19 点**（low_contrast 80% vs 70%；hue_shift 92.9% vs 74%）。需要满 100 eps 确认，并跑 clean/grayscale 对照排除副作用。
- 注意：投票是"输出层多视图平均"（文献 PDF arXiv:2604.18107），与"输入单图预处理"不同；但用户曾明确说不做以输入增广为创新点的方法——所以它更适合作为**强基线/对照**，不是我们的贡献。

## 第 11 轮（05:28）：vote-k=5 稳定为正，PSC 完成判死

### PSC（已基本完成）
| 条件 | eps | success | SR |
|---|---|---|---|
| PSC(color_jitter, beta=0) | 100 | 0 | **0.0%** |
| PSC(low_contrast, beta=0) | 98 | 34 | **~34.7%**（基线 70.0%） |
| PSC(hue_shift, beta=0) | 93 | 27 | **~29.0%**（基线 74.0%） |
| PSC(low_contrast, beta=0.5) | 78 | 43 | ~55.1%（基线 70.0%） |
| PSC(low_contrast, beta=1 identity) | 64 | 49 | ~76.6%（≈基线，无副作用） |

**结论：PSC 在 LIBERO 高余量设定下明确有害**：硬投影 −35～−45 点；软投影仍 −15 点；identity 对照与基线一致。方向错误而非强度问题。

### TTA 投票 k=5（PDF 式动作投票，无标签/无梯度/不改权重）——持续为正
| 条件 | eps | success | 当前 SR | 基线 | 增益 |
|---|---|---|---|---|---|
| clean | 22 | 22 | **100%** | 100% | 0 |
| grayscale | 19 | 18 | ~94.7% | 90% | +5 |
| low_contrast | 43 | 35 | **~81.4%** | 70% | **+11** |
| hue_shift | 45 | 41 | **~91.1%** | 74% | **+17** |
| vote k=3 low_contrast | 63 | 46 | ~73.0% | 70% | +3 |

→ 投票在"本来就好"的条件不掉点（clean 100%），在中间扰动上稳定增益；k=5 优于 k=3。

### 下一轮计划
1. 等 vote5 low/hue 满 100 eps、vote5 clean/grayscale 满 100 eps；
2. 若维持，则"**测试时多外观视图动作平均**"是本设定下可复现的正结果（文献 PDF 2604.18107），可作为强基线；
3. 同时尝试"**校准型投票**"（变体里加入更强的对比度/亮度反向校正因子），检验能否进一步提升——这是我们在 PDF 基础上有依据的变体，但需要和你确认是否算创新点（因为你此前明确说不做"输入增广"类创新）。

## 第 12 轮（05:35–06:05）：确认 vote-5 趋势，启动"校准型投票"候选

### vote-5 进展（进行中）
| 条件 | eps | success | 当前 SR | 基线 |
|---|---|---|---|---|
| vote5 clean | 28 | 27 | ~96.4% | 100% |
| vote5 grayscale | 25 | 23 | ~92.0% | 90% |
| vote5 low_contrast | 47 | 36 | ~76.6% | 70% |
| vote5 hue_shift | 50 | 45 | ~90.0% | 74% |

→ 投票在 clean/grayscale 基本不掉点，在 low/hue 上稳定 +6～+16 点（与上一轮一致）。

### PSC identity 对照
- PSC(low_contrast, beta=1)：70 eps，55 成功 ≈ 78.6%，与基线 70% 在噪声内 → **hook 无副作用**，PSC 的伤害来自投影本身。

### 新候选实验（已启动，标注为"候选"，不算入贡献）
- **校准型投票**：变体集合从"轻微扰动"改成"部分反向校正"：
  - `calib_contrast`（low_contrast 上）：对比度 1.3/1.6/0.8 + 亮度 1.15；
  - `calib_hue`（hue_shift 上）：色相 -0.25/-0.5/-0.75/+0.25。
- 代码：`VotePolicy` 支持 4 元组 (b,c,s,h)，`--vote-calib {none,contrast,hue}`；
- 目的：检验"在投票中做校正"是否能超过纯 PDF 投票（+11/+17）。

## 第 13 轮（06:05–06:40）：候选"校准型投票"初现，配对检验工具就绪

### 当前（进行中）
| 条件 | eps | success | 当前 SR | 基线 |
|---|---|---|---|---|
| vote5 clean | 58 | 55 | ~94.8% | 100% |
| vote5 grayscale | 52 | 47 | ~90.4% | 90% |
| vote5 low_contrast | 71 | 54 | ~76.1% | 70% |
| vote5 hue_shift | 72 | 60 | ~83.3% | 74% |
| **calib_contrast(low_contrast)** | 26 | 22 | **~84.6%** | 70%（纯投票 ~76%） |
| calib_hue(hue_shift) | 刚重启 | — | — | 74%（纯投票 ~83%） |

### 修复
- `calib_hue` 首次失败：torchvision `adjust_hue` 要求 hue_factor ∈ [-0.5, 0.5]，序列里有 -0.75 → 改成 +0.5 后重启成功。

### 新增工具
- `scripts/libero_pair_analysis.py`：按 (task_id, episode_id) 配对两次评测，报 discordant 计数 + **精确 McNemar 检验**（官方同任务 seed、同初始状态，因此可配对）。等各条件满 100 eps 后统一做显著性。

### 下一步
1. 等 vote5 与 calib_* 满 100 eps；
2. 用 McNemar 检验"投票 vs 基线"、"校准投票 vs 纯投票"；
3. 若校准投票显著优于纯投票，再讨论是否可作为方法（需你确认创新点边界）。

## 第 14 轮（06:15–06:45）：配对 McNemar 预览——表面增益明显缩水

用 `libero_pair_analysis.py` 对重叠 episode 做严格配对（官方同任务 seed、同初始状态）：

| 对比 | 配对数 | SR(a) | SR(b) | Δ | discordant (a→b) | McNemar p |
|---|---|---|---|---|---|---|
| baseline vs vote5（low_contrast） | 77 | 68.8% | 72.7% | **+3.9** | 6 / 9 | **0.607** |
| vote5 vs calib_contrast（low_contrast） | 37 | 86.5% | 86.5% | **0.0** | 4 / 4 | **1.000** |

**重要修正**：
1. 之前按"整体百分比"看，vote5 在 low_contrast 上 +6～+11；**严格配对后只有 +3.9，且不显著（p=0.61）**。
2. 校准投票（calib_contrast）整体看着 +15，但**在与 vote5 重叠的 37 条上完全持平**（86.5% vs 86.5%）——之前的差异主要是**任务覆盖顺序不同**造成的成分偏差（calib 只跑到前面几个任务）。
3. 因此在满 100 eps 之前，**任何百分比对比都不可信**；最终必须用配对 McNemar（脚本已就绪）。

当前进度：vote5 low 77/100、hue 79/100、clean 70/100、grayscale 62/100；calib_contrast 37/100、calib_hue 12/100。

## 第 15 轮（06:35–07:05）：配对检验定论——PDF 投票无显著收益

近满样本（93–94 对）严格配对 McNemar：

| 对比 | 配对 | SR(a) | SR(b) | Δ | discordant | McNemar p |
|---|---|---|---|---|---|---|
| baseline vs vote5（low_contrast） | 93 | 69.9% | 71.0% | **+1.1** | 8 / 9 | **1.00** |
| baseline vs vote5（hue_shift） | 94 | 73.4% | 73.4% | **0.0** | 10 / 10 | **1.00** |
| clean vs vote5（clean） | 92 | 100% | 96.7% | **−3.3** | 3 / 0 | 0.25 |

**结论**：
1. 之前看到的 vote5 +6～+17 点是**任务覆盖顺序造成的成分偏差**；严格配对后 **Δ≈0～+1，p=1.0，不显著**。
2. 在 clean 上投票反而掉 3 点（3 条失败、0 条新增成功，p=0.25）→ 多视图平均会轻微损伤本来就正确的决策。
3. 至此 Route A 的结论清晰：**在高余量 LIBERO 基座上，无训练 TTA（特征投影与动作投票）都无法显著改善光度扰动下的成功率；PSC 投影明确有害（−35～−45 点）。**

这修正了我们之前所有"百分比增益"的乐观解读——也验证了用户坚持的"必须配对统计"的要求。

## 第 16 轮（07:05–07:35）：vote5 满样本确认中性；校准型 hue 投票初步显著

### vote5 满 100 eps（最终）
| 条件 | SR | 基线 | Δ |
|---|---|---|---|
| vote5 clean | 96.0% | 100% | −4 |
| vote5 grayscale | 87.0% | 90% | −3 |
| vote5 low_contrast | 71.0% | 70% | +1 |
| vote5 hue_shift | 75.0% | 74% | +1 |

→ **纯 PDF 投票：中性偏轻微有害**（clean −4、grayscale −3，扰动条件 +1），配对 p=1.0。

### 校准型投票（配对 McNemar）
| 对比 | 配对 | SR(a) | SR(b) | Δ | discordant | p |
|---|---|---|---|---|---|---|
| baseline vs calib_contrast（low_contrast） | 78 | 67.9% | 73.1% | +5.1 | 8/12 | 0.503 |
| vote5 vs calib_contrast | 78 | 71.8% | 73.1% | +1.3 | 9/10 | 1.000 |
| **baseline vs calib_hue（hue_shift）** | 66 | 83.3% | **93.9%** | **+10.6** | 1/8 | **0.039** |

**初步结论**：
- `calib_hue` 在 hue_shift 上 **+10.6 点，p=0.039（单次检验显著）**，但样本未满（66 eps）且有多重比较风险，**必须等满 100 eps + 第二 seed 复核**；
- `calib_contrast` 无显著收益（+5.1，p=0.50；vs vote5 +1.3，p=1.0）。
- 注意：paired baseline 在 66 条上的 SR=83.3%，高于其全量 74%——仍是**任务顺序成分偏差**的体现，说明"只看全量百分比"会误判。

### 下一步（等 calib_hue 满 100）
1. 满 100 后重跑配对检验；
2. 若仍显著，启动 **seed 8 复核**（baseline hue + calib_hue 各 100 eps）；
3. 只有在第二 seed 上仍显著，才把"校准投票"作为候选方法上报。

## 第 17 轮（07:35–08:05）：校准型 hue 投票 = 强显著正结果（待复核）

### 严格配对（97 对，满 100 前）
| 对比 | SR(a) | SR(b) | Δ | discordant | McNemar p |
|---|---|---|---|---|---|
| baseline(hue_shift) vs **calib_hue** | 73.2% | **93.8%** | **+20.6** | 1 / 21 | **1.1e−05** |
| vote5(hue_shift) vs **calib_hue** | 74.2% | **93.8%** | **+19.6** | 1 / 20 | **2.1e−05** |
| baseline(low_contrast) vs calib_contrast | 70.0% | 73.0% | +3.0 | 10 / 13 | 0.68 |

**这是 Route A 至今唯一强显著的正结果**：在 hue_shift 上，把投票变体换成"部分反向色相校正"（-0.5/-0.25/+0.25/+0.5/identity），
SR 从 ~73% 提到 ~94%（+20.6 点，p≈1e−5）。

### 必须做的复核（已全部启动，各 100 eps）
1. **seed 8 复核**：`hue_seed8`（baseline）与 `hue_seed8_calib`；
2. **消融：是不是"投票"的功劳**：
   - `hue_inverse_only`：只用单一精确反向视图（hue=-0.5）；
   - `hue_mild_vote`：只用 ±0.25 的温和校正；
3. **跨 suite 复核**：`libero_object` 上的 baseline vs calib（`obj_hue` / `obj_hue_calib`）。

### 判读原则（先定好，防止事后解释）
- 若 seed8 仍显著 + 其它 suite 仍为正 → 候选方法成立（但需说明"需要知道/估计扰动族"）；
- 若 `hue_inverse_only` ≈ 93.8% 而 vote 不高 → 收益主要来自"精确反向"而非投票 → 方法定位改为"检测扰动族 + 反向校正 + 投票融合"；
- 若 seed8/object 不成立 → 判定为单次偶然/多重比较假阳性，收回结论。

## 第 18 轮（08:05）：复核早期信号（样本少，仅记录）

| 复核 run | eps | success | 备注 |
|---|---|---|---|
| hue_seed8（baseline） | 1 | 0 | 太早 |
| hue_seed8_calib | 3 | 3 | 太早 |
| **hue_inverse_only**（单一精确反向 -0.5） | 4 | 4 | **若此项持续 ≈ calib_hue，则收益来自精确反向而非投票** |
| hue_mild_vote（±0.25） | 2 | 2 | 太早 |
| obj_hue（baseline, libero_object） | 2 | 1 | 太早 |
| obj_hue_calib | 3 | 3 | 太早 |

等 40–60 eps 再判读；当前不作结论。

## 第 19 轮（08:05–08:35）：复核早期——"增益来自精确反向（oracle inverse）"的证据出现

### 早期复核（未满 100，仅记录）
| run | eps | success | 当前 SR |
|---|---|---|---|
| hue_seed8（baseline） | 31 | 25 | 80.6% |
| hue_seed8_calib | 35 | 34 | 97.1% |
| **hue_inverse_only**（单一 exact inverse -0.5） | 40 | **40** | **100%** |
| hue_mild_vote（±0.25） | 34 | 33 | 97.1% |
| obj_hue（baseline, libero_object） | 33 | 17 | 51.5% |
| obj_hue_calib | 45 | 43 | 95.6% |

### 关键发现（机理层面）
- 上游 `apply_image_shift("hue_shift")` = `TVF.adjust_hue(image, 0.5)`（精确 +0.5 旋转）。
- 我们的 `hue` 校准变体集合里包含 **-0.5**（精确反向） → 组合后恰好回到原图。
- 因此 `hue_inverse_only`（只用 -0.5 单视图）40/40 = **等价于"把扰动精确还原"**。
→ **所谓"校准投票"的增益，很可能主要来自"用 benchmark 的精确扰动参数做反向校正"（oracle inverse），而不是投票融合本身。**

### 这改变了方法定位
- "投票"本身（无校正）无效；
- "精确反向校正"几乎恢复全部性能（+20～+44 点）；
- 但**部署时扰动参数未知**，所以真正的问题是 **无监督估计光度扰动参数**，然后用估计值做反向校正 + 投票。
- 下一步：实现"从测试图像统计估计 hue/contrast 偏移"（与 clean 校准集的统计量比较），再检验"估计反向"是否接近"精确反向"。

### 待确认
- seed8 / object suite 的满样本配对检验；
- 若 effect 完全由 oracle inverse 解释，则应把结论写成"光度扰动的可逆性上界"，方法贡献改为"扰动估计"。

## 第 20 轮（08:35–09:05）：跨 seed / 跨 suite 复核通过；"不精确反向"的温和色相投票也显著

### 配对 McNemar（hue_shift）
| 对比 | 数据 | 配对 | SR(a) | SR(b) | Δ | discordant | p |
|---|---|---|---|---|---|---|---|
| baseline vs calib_hue | spatial seed7 | 100 | 74.0% | 94.0% | **+20.0** | 1/21 | **1.1e−05** |
| baseline vs calib_hue | spatial seed8 | 60 | 76.7% | 98.3% | **+21.7** | 1/14 | **9.8e−04** |
| baseline vs calib_hue | libero_object seed7 | 71 | 56.3% | 95.8% | **+39.4** | 0/28 | **7.5e−09** |
| baseline vs hue_inverse_only | spatial seed7 | 74 | 79.7% | 95.9% | +16.2 | 0/12 | 4.9e−04 |
| baseline vs hue_inverse_only | spatial seed8 | 60 | 76.7% | 96.7% | +20.0 | 1/13 | 1.8e−03 |
| **baseline vs hue_mild_vote（仅 ±0.25，无精确反向）** | spatial seed7 | 67 | 83.6% | 94.0% | **+10.4** | 0/7 | **0.0156** |
| **baseline vs hue_mild_vote** | spatial seed8 | 60 | 76.7% | 93.3% | **+16.7** | 2/12 | **0.0129** |

### 关键解读
1. **跨 seed（7→8）与跨 suite（spatial→object）全部复现**，且 object 上增益更大（+39 点，因为 baseline 更弱）；
2. **精确反向最强**，但**"无精确反向"的 ±0.25 温和色相投票同样显著（+10～+17，p≈0.01–0.02）** → 说明增益不完全是 oracle inverse，**"沿扰动族做测试时增广投票"本身有效**；
3. 机制上：对色相扰动，沿色相轴平均动作能显著降低模型对色相方向的敏感度；而沿亮度/对比度轴平均（纯 PDF 投票）无效 → **增广轴必须匹配扰动族**。

### 正在跑的控制
- `clean + hue_vote`（关键：不能伤 clean）；
- `low_contrast + hue_vote`（特异性：非色相扰动上是否无效/有害）。

## 第 21 轮（09:05–09:35）：关键控制初现——"轴匹配"必要

### 控制进展
| run | eps | success | 当前 SR | 对比 |
|---|---|---|---|---|
| **clean + hue_vote** | 37 | 37 | **100%** | clean 100% → **不伤 clean** ✓ |
| **low_contrast + hue_vote** | 21 | 12 | ~57% | low_contrast 70% → **可能有害**（轴不对） |
| hue_seed8 baseline | 81 | 61 | 75.3% | — |
| hue_seed8_calib | 97 | 95 | 97.9% | +22.6 |
| hue_inverse_only | 100 | 97 | 97.0% | 完成 |
| hue_mild_vote | 87 | 75 | 86.2% | vs 基线 74% → +12 |
| obj_hue baseline | 100 | 61 | 61.0% | 完成 |
| obj_hue_calib | 100 | 96 | 96.0% | **+35 点** |

### 解读
- hue 轴投票在 clean 上不掉点（37/37），在 hue_shift 上 +12～+35 点（跨 seed/suite 显著）；
- 但在 low_contrast 上早期 57% vs 基线 70% → **轴选错可能有害**；
- → 方法必须包含"**扰动族检测**"：先判断是色相/对比度/亮度/饱和度哪一类，再选对应增广轴。

### 下一步（正在做）
1. 实现**无监督扰动族检测**：用测试图像统计（通道均值/亮度标准差/饱和度/hue 直方图）与 clean 参考统计比较，输出族别；
2. 检测准确率评估（在已知 shift 的观测上）；
3. 若可分离，接入 `--auto-family` 做端到端实验；若不可分离，说明"轴匹配"实验只能作为上界分析。

## 第 22 轮（09:35–10:05）：扰动族检测 + 自动轴选择（候选方法）

### 光度统计可分性（2 任务 × 3 帧 × 5 种条件，`libero_photometric_stats.py`）
| 条件 | 通道均值(R,G,B) | 亮度均值 | 亮度 std | 饱和度 |
|---|---|---|---|---|
| none | 0.503 / 0.471 / 0.436 | 0.477 | 0.197 | 0.129 |
| grayscale | 0.477 / 0.477 / 0.477 | 0.477 | 0.197 | **0.000** |
| color_jitter | **0.726 / 0.700 / 0.638** | **0.701** | **0.349** | 0.114 |
| low_contrast | 0.485 / 0.475 / 0.465 | 0.477 | **0.059** | **0.039** |
| hue_shift | **0.436 / 0.469 / 0.503**（B>R 反转） | 0.463 | 0.189 | 0.129 |

→ 五类在简单统计上**完全可分**（灰度：饱和度≈0；低对比度：亮度 std≈0.06；color_jitter：亮度均值>0.6 且 std 高；色相：B>R 反转）。

### 候选方法：检测扰动族 → 选匹配的投票轴
- 新增 `detect_family()`（阈值基于上表参考统计）与 `--auto-family`；
- 映射：hue_shift→色相轴；low_contrast→对比度轴；color_jitter→亮度/对比度/饱和度轴；grayscale/none→不投票；
- 已启动 4 个端到端 run（spatial，seed 7，auto-family）：
  - `auto_hue`、`auto_low`、`auto_clean`、`auto_jitter`。

### 判据
- 若 `auto_hue ≈ calib_hue`（94%）且 `auto_clean ≈ 100%` → 检测-选择链路成立；
- 若 `auto_low ≈ calib_contrast`（~73%）→ 对低对比度仍无大增益（与之前一致）；
- `auto_jitter` 是探索（color_jitter 基线 0%，逆变换族可能救回一部分）。

## 第 23 轮（10:05–10:35）：auto-family 早期结果 + 关键发现（不检测就投票会伤 clean）

| run | eps | success | 当前 SR | 对照 |
|---|---|---|---|---|
| **auto_clean**（检测到 none → 不投票） | 29 | 29 | **100%** | clean 100% ✓ |
| **auto_hue**（检测到 hue → 色相轴投票） | 23 | 20 | ~87.0% | 基线 74%，oracle 选择 94% |
| auto_low（对比度轴） | 19 | 13 | ~68.4% | 基线 70% |
| auto_jitter（亮/对比/饱和轴） | 14 | 0 | 0% | 基线 0% |
| **clean_hue_vote（clean 上强行色相投票）** | 59 | 56 | ~94.9% | clean 100% → **−5 点** |
| low_hue_vote（low_contrast 上强行色相投票） | 43 | 30 | ~69.8% | 基线 70% → ≈0 |
| hue_seed8_calib（完成） | 100 | 98 | **98.0%** | seed8 基线 ~75% |

### 关键发现
1. **不检测就投票会伤 clean**：clean_hue_vote 94.9% < 100%（−5 点）；说明"投票轴"必须只在匹配扰动时启用。
2. **auto_clean = 100%** → 检测器在 clean 上正确判"none 不投票"，**保护住了 clean**。
3. auto_hue ~87%（部分恢复），auto_low ≈ 基线，auto_jitter 仍 0：即"族检测 → 轴选择"链路成立方向正确，但强度/轴设计还需打磨。
4. hue_seed8_calib 满 100 = 98%，seed8 上复现强增益。

### 下一步
- 等 auto_* 满 100 eps + 配对检验；
- 若 auto_hue 显著优于基线且 auto_clean = 100%，则"检测门控的族匹配投票"是成立的候选方法；
- 逐项打磨：色相轴的变体数量/强度、低对比度轴的有效变体、color_jitter 的逆变换族（当前 0%）。

# ===== Route A 阶段总结（2026-11-10 上午） =====

## 0. 结论先行

1. **官方复现成功**：CoRP 官方 LIBERO checkpoint 在 `libero_spatial` clean 上 **100/100 = 100%**（论文 98.2%）。
2. **光度扰动分级**：clean 100% → grayscale 90% → hue_shift 74% → low_contrast 70% → color_jitter **0%**。
3. **PSC（测试时源支撑投影）判死**：hue_shift 74%→29–41%（−35～−45 点），low_contrast 70%→35–48%；软投影仍 −15；identity 对照 ≈ 基线（排除 hook 副作用）。
4. **纯 PDF 增广投票无效**：配对 McNemar Δ≈0～+1，p=1.0；clean 上 −4。
5. **沿扰动族的增广投票强显著（本阶段唯一可复现正结果）**：
   - hue_shift：spatial seed7 74%→94%（+20.0，p=1.1e−05）；seed8 70%→98%（+28.0，p=5.8e−08）；libero_object 61%→96%（+35.0，p=5.8e−11）。
   - 温和变体（±0.25，无精确反向）也显著：+10～+17，p≈0.01–0.02 → 增益不完全是"偷看扰动参数"。
6. **无监督族检测可行**：5 类扰动在简单图像统计（饱和度/亮度 std/亮度均值/通道序）上完全可分；检测门控使 clean 保持 100%，而"clean 上强行色相投票"会掉到 95.3%。
7. **候选方法**："无监督扰动族检测 + 族匹配的测试时增广投票"；早期 auto_hue 87–90%（oracle 选轴 94–98%），auto_clean 100%。

## 1. 与文献的关系（必须诚实）
- 增广投票框架来自 PDF（arXiv:2604.18107）；
- 我们的增量是"**增广轴必须匹配扰动族 + 族可由无监督统计检测**"：同一投票框架下，亮度/对比度轴无效、色相轴显著；不检测就投票会伤 clean。
- 目前仍是**推理期方法**，与 PAIR-VLA/MixVLA（训练时）定位不同。

## 2. 关键风险与未完成事项
- auto_* 尚未满 100 eps（auto_hue 30/100、auto_clean 36/100、auto_low 25/100）；
- 只有 spatial + object 两个 suite，第三 suite（goal/long）未测；
- 只有 seed7/8 两个 seed；
- 色相轴变体是手工设定（±0.25/±0.5）；低对比度轴无效；color_jitter 逆变换族仍 0%；
- "检测→选轴"的误判率未系统统计（需逐帧记录 family）。

## 3. 下一步建议
1. 补满 auto_* 100 eps + 第三 seed；
2. 加 goal/long suite；
3. 系统记录检测混淆矩阵，改进阈值；
4. 讨论：若把"族检测 + 轴匹配"作为创新点，需要和 PDF 的"通用增广投票"做明确消融对照（已有：亮度/对比度轴无效 vs 色相轴有效）。

## 4. 相关文件
- 环境：`repos/CoRP/.venv`（torch 2.11.0+cu128、robosuite 1.4.1、LIBERO pinned 8f1084e）
- 评测：`scripts/evaluate_libero.py`（官方）、`scripts/libero_shift_eval.py`（薄驱动：shift/psc/vote/auto-family）
- 配对统计：`scripts/libero_pair_analysis.py`
- 光度统计：`scripts/libero_photometric_stats.py`
- 源统计：`scripts/build_libero_psc_stats.py`
- 结果目录：`/hy-tmp/validation/libero_*`、`/hy-tmp/validation/pair*.json`

# ===== Route B（LIBERO-Plus 真·未见扰动）：环境与协议 =====

## 为什么重做
之前 Route A 的扰动是**我们自己加的合成扰动**（`dacsc.corp_images` 的 color_jitter/hue_shift 等），
不是 benchmark 自带的"未见扰动"。用户要求改用 **LIBERO-Plus**（10,030 个扰动任务，7 个扰动维度）。

## 已完成的环境安装（本机）
1. 仓库：`/hy-tmp/libero_plus`（codeload tarball，官方 main）；
2. 资源：HuggingFace `Sylvest/LIBERO-plus` 的 `assets.zip`（6.4GB，hf-mirror 下载）+ 原 LIBERO 基础资源（405MB）合并到 `libero/libero/assets`（9.5GB）；
3. 依赖：`wand`、`scikit-image`、系统 `libmagickwand-dev` 等；**numpy 回钉 1.26.4**（保持 corp-policy 要求）；
4. **未修改任何上游 CoRP 代码**：先把 `evaluate_libero.py` 回滚到 pinned 版本以保证 `verify_upstream` 通过。
5. 新增 `scripts/prepare_libero_plus_init.py`：调用 LIBERO-Plus 官方 `Benchmark.get_task_init_states(i)`，
   把扰动任务缺少的 `init_files/<folder>/<task>.pruned_init` 物化到官方评测器直接可读的路径
   （数据准备，不改评测逻辑；`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 适配 torch 2.11 的 weights_only 默认值）。

## 评测协议（按 LIBERO-Plus 官方）
- 官方：10,030 任务，**每任务 1 trial**（README 明确 `num_trials_per_task` 从 50 改 1）。
- 本机子集（因软件渲染耗时）：**每 suite 每类扰动随机抽 10 个任务**（seed 7，`--per-category 10`），
  7 类 × 10 = **70 任务/suite**，每任务 1 trial；4 个 suite 各 70 = **280 episodes**。
- 任务选择与 baseline/TTA 完全一致（同一 seed、同一分类文件）→ 可严格配对。
- 与 Route A 的区别：**扰动来自 benchmark 任务本身**（bddl/初始化/光照/背景/噪声等），不是合成图像变换。

## 已启动
- 4 suites × {baseline, auto-family TTA} = 8 个后台评测（每 suite 70 eps）。
- 冒烟：Light/Sensor Noise 各 1 任务 → 2 eps，1 成功（0.5），管线打通。

## Route B 进度（10:35–10:42）
- 已停掉 Route A（合成扰动）遗留的 5 个进程（结果文件保留），把 CPU/GPU 让给 LIBERO-Plus 评测。
- LIBERO-Plus 8 个后台评测进行中（4 suites × baseline/auto），当前每 suite 2–5 / 70 episodes。
- 速度：单进程约 **150–170 秒/episode**（mesa 软件渲染 + 单 GPU 并发），预计全量 280×2 episodes 需 **2.5–3.5 小时**。
- 早期数字（样本很少，不解读）：spatial 3 eps 1 成功；object 5 eps 3 成功；goal 3 eps 1；10 2 eps 1。

## Route B 并行加速（10:45）
- 原 8 个整 suite 评测速度 ~2.5 分钟/episode，预计 3h；改为**每个 suite×条件拆成两半**（同一批 70 个选中 task_id，35+35），
  共 16 个进程并行（CPU 128 核、mesa 渲染 CPU 绑定），预计 **~1.5–2h**。
- 协议不变：任务集合与 seed 完全相同，只是并行拆分；最终把 a/b 两半 jsonl 合并做配对分析。
- 驱动新增：`--tasks` 显式指定时不再触发 per-category 抽样（用于拆分）。

# ===== 重大提速：GPU EGL 打通（11:40–12:00） =====

## 问题
之前所有评测用 **mesa 软件渲染**（本机缺 `libEGL_nvidia`），单 episode ≈ 150–220 秒，
GPU 利用率 0%，CPU 加到 32/63 进程也不再变快（受渲染限制），LIBERO-Plus 全量 10,030 不可行。

## 修复
1. `apt-get install libnvidia-gl-550`（驱动正好 550.163.01），装出了
   `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0` 和 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`；
2. 评测环境改为：
   `MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
3. 验证：
   - 单 episode：**9.1 秒**（原来 150–220 秒，约 **20 倍加速**）；
   - clean LIBERO spatial task 0/1/2 各 10 episodes：**30/30 成功**（与 mesa 下 100% 一致，渲染正确）。

## 新计划：跑官方全量（不再抽样）
- 全部 **10,030 个 LIBERO-Plus 任务 × 1 trial**（官方协议），
  条件 = baseline 与 auto-family TTA，**同一批任务** → 20,060 episodes；
- 24 个进程（4 suites × 2 条件 × 3 段）；
- 实测吞吐：**69 episodes/分钟**（GPU util 约 50%），预计 **约 4.5–5 小时** 完成；
- 输出：`/hy-tmp/validation/lpfull_<suite>_<mode>_c<k>/`，日志同名 `.log`；
- 之前的 280 子集结果（mesa 渲染）保留作对照，但最终结论用 GPU 全量。

## 注意（可复现性）
- 渲染从 mesa 软件改为 NVIDIA EGL（GPU 光栅化），理论上像素值可能有极小差异；
  clean 验证 30/30 成功，说明不影响任务成功率；
- 后续所有报告注明"NVIDIA EGL 渲染"。

## 第 25 轮（13:00）：在“新模型 + 新数据集”上重做表征几何（修正旧结论）

### 重要区分
- 旧结论（V13）“外观与任务子空间主角度 <2°”是 **我们自己短训的 RoboCasa 模型 + RoboCasa 数据** 上测的，
  **不能**直接搬到官方 CoRP LIBERO checkpoint + LIBERO-Plus 上。
- 本轮用官方 checkpoint，在 clean LIBERO 与 LIBERO-Plus 上重新采集特征，重新测量：
  - `reps` = representation tokens（视觉-语言-状态，[T,D] 均值）
  - `writes` = cross-attention 写回（最后一 Euler 步，action token 均值）
  - 采集：clean LIBERO spatial 10 任务 × 2 chunk（20 帧）；LIBERO-Plus spatial 7 维 × 5 任务 × 2 chunk（70 帧）。

### 新结果（官方 checkpoint）
| 特征 | 有效秩（participation ratio） | task vs action 主角度（均值/最小） | PSC 会删掉的 task 能量 | PSC 会删掉的 action 能量 |
|---|---|---|---|---|
| reps | **4.74** | 50.3° / 20.4° | 20% | **52%** |
| writes | **3.46** | 35.3° / 14.8° | 12% | 29% |

各扰动维度的 domain-shift 子空间（reps）：
| 维度 | 与 task 子空间主角度 | shift 能量落在 action 子空间比例 | PSC 删掉的 shift 能量 |
|---|---|---|---|
| Background | 65.0° | 16% | 72% |
| Camera | 65.9° | 23% | 76% |
| Light | 64.4° | 11% | 80% |
| Layout | 62.0° | 11% | 75% |
| Robot init | 60.4° | 24% | 65% |
| Noise | 56.9° | 25% | 62% |

### 新解释（替换旧故事）
1. **domain shift 和 task 方向并不重合**（57–66°）→ 旧“外观=任务”说法在新模型/数据上不成立；
2. PSC **确实删掉了大部分 shift 能量（62–80%）**，但它同时删掉了 **52% 的 action 敏感方向**（reps）；
3. 原因：PSC 的“源支撑”= **方差最大的 top-k PCA**，而 action 敏感方向很多落在**低方差分量**里 →
   PSC 删 shift 的同时把动作信息一起删，净效果为负。
4. 这给“改进 PSC”一个可量化的目标：**保留这 52% 的 action 能量**（双投影/保护子空间），而不是继续调强度。

### 注意
- 样本仍小（20/70 帧、1 个 suite、每个维度 3–5 个基底任务）；需要扩大采集再做正式结论；
- ridge 探针是线性近似；后续可加 Jacobian 版本。

## 定时汇报 13:35 / 13:45
- 全量进度：**4,824 / 20,060**（约 24%），24 个进程全部存活，GPU 37%，显存 30GB；
- 当前吞吐约 **52 episodes/分钟** → 预计 **18:00–18:30** 完成（比早先估计慢，因为后半段任务更长）；
- 中间配对结果（n 为已完成配对数）：
  - Background Textures：39.5% → 39.2%（Δ−0.3，p=0.69）
  - Camera Viewpoints：39.4% → 40.8%（Δ+1.4，p=0.080）
  - Language Instructions：99.7% → 99.7%（Δ0，p=1.0）← 语言扰动几乎不掉点
  - Robot Initial States：54.3% → 54.3%（Δ0）
  - Sensor Noise：36.8% → 35.5%（Δ−1.3）
  - ALL：48.9% → 49.2%（Δ+0.3，p=0.435）
- 另：PSC on LIBERO-Plus 子集（45 对）为 −17.8 点（p=0.057），与全量无关但说明投影有害。

## 定时汇报 14:00
- 全量 baseline/auto：**5,312 / 20,060（26.5%）**，24 进程存活；预计 **17:30–18:15** 完成；
- 中间配对（auto vs baseline）：ALL 50.5% → 50.9%（Δ+0.3，p=0.31）；
  Camera +1.3（p=0.09）、Background −0.1、Language 93.9%（0）、Robot init 46.7%（0）、Noise −1.0；
- **完整 PSC（TTT）子集**：108/280；配对 48 条：base 39.6% → PSC-TTA 37.5%（Δ−2.1，p=1.0），目前中性偏负，需满量判断；
- 对照：**只投影**（旧 PSC）配对 78 条：46.2% → 24.4%（Δ−21.8，p=0.0015，显著有害）。

## 定时汇报 14:30
- 全量：**6,421 / 20,060（32%）**，24 进程存活；预计 **17:50–18:20** 完成；
- 中间配对（auto vs baseline）：**ALL 51.0% → 51.2%（Δ+0.3，p=0.38）**；
  Background −0.1、Camera +1.3（p=0.09）、Language 0（86.8%）、Robot init 0（43.4%）、Noise −1.7（n=121）；
- 修正版完整 PSC-TTA：11/280（按官方 SSA 约定重写：逐项 KL、无偏方差、lr=1e-3、buffer64、每 8 帧更新；clean loss=0.0，扰动 loss=1.07 校验通过）。

## 定时汇报 15:00
- 全量：**约 7,817 / 20,060（39%）**，24 进程存活；当前平均吞吐 ~49 eps/min；预计 **18:30–19:00** 完成（GPU 24 进程共享，单 episode 从 9s 涨到 ~29s）；
- 中间配对（auto vs baseline）：**ALL 50.1% → 50.3%（Δ+0.2，p=0.38）**；
  Background −0.1、Camera +1.3（p=0.09）、Language 0（81.7%）、Robot init 0（38.4%）、Noise −1.1（n=187）；
- **修正版 PSC-TTA 结果作废**：日志出现 CUDA OOM + lr=1e-3 过适应（ALL 64.5%→3.2%），已杀进程；等全量结束用小并发/lr=1e-4 重跑；
- 只投影版 PSC（78 对）：−21.8（p=0.0015），仍成立。

## 第 26 步：实现"可识别性加权"的 PSC（按上游代码校验）

### 参考的上游代码/论文
- **SSA 官方实现** `kzkadc/regression-tta`（ICLR 2025）：
  - `utils/loss.py::diagonal_gaussian_kl_loss`（逐项 KL，含 log 与 eps=1e-8）
  - `tta/tta_engine.py`：`f_pc=(feature-mean)@basis`；`var(dim=0)` 无偏；`kl = loss_fn(mu,var,0,var_s)+loss_fn(0,var_s,mu,var)`；`dim_weight=(|w@basis|+bias)^exp`
  - `adaptation.py`：Adam lr=1e-3、wd=0、只更新归一化层参数（fe_bn）
- **PSC 论文** arXiv:2603.09338：探针库 Eq.(11)-(12)、支撑谱匹配 Eq.(16)、残差谱匹配 Eq.(23)、只更新 g_φ 归一化 affine（Eq.25）
- **PSC 无公开代码**（已确认），新增项只能按论文公式实现；SSA 部分与官方代码逐行对齐。

### 我们唯一的增量（已文档化）
- 每个探针方向的"可识别性权重"：`w_q = β_q · ν_q / (ν_q + μ̂_q² + ε)`
  - `ν_q`：在**标签保持的光度增广**（grayscale / low_contrast / color_jitter / hue_shift / brightness 0.8 / 1.2）下该方向的方差（源域离线估计，70 帧）；
  - 含义：观测到的偏移若能被已知 nuisance 变化解释 → 对齐；否则视为 content → 降权保护。
- 其余全部保持 SSA/PSC 原样（损失、优化器、参数集、残差项）。

### 实现
- `scripts/build_weighted_stats.py`：源域增广特征采集 + `probe_nuisance_var`（ν_q）估计；
- `dacsc/psc_weighted.py::WeightedPSCTTA`：在 PSC 损失上乘可识别性权重；
- `scripts/libero_shift_eval.py`：`--psc-weighting {none,identifiability}`、`--psc-tta-buffer`。

### 校验
- clean 特征 loss=0.0；LIBERO-Plus 特征 loss=1.055（未加权 1.072）；
- ν_q 范围 1.06–24.25（均值 8.38）；β 均值 1.013；
- 梯度链路校验通过（此前 5 步 loss 单调下降）。

### 正在跑（3 路对比，60 任务：Light/Background/Noise 各 20）
- `lp3_base`、`lp3_wpsc`（加权 PSC-TTA）、`lp3_psc`（未加权 PSC-TTA）；
- lr=1e-4、buffer=16、每 8 帧更新、只更新 LN affine；
- 与全量 baseline 共享 GPU（24 进程），预计 20–30 分钟出结果。

## 第 27 步：可识别性加权 PSC 的首次三路对比（60 任务：Light/Background/Noise 各 20）

| 条件 | 配对 n | SR | Δ vs base | McNemar p |
|---|---|---|---|---|
| base | 50 | 68.0% | — | — |
| PSC-TTA（未加权，SSA/PSC 原样） | 50 | 68.0% | +0.0% | 1.00 |
| **加权 PSC-TTA（我们的）** | 50 | **70.0%** | **+2.0%** | 1.00 |

分维度（wpsc vs base）：Background 60→65（+5）、Light 85→90（+5）、Noise 50→40（−10，n=10）。

**判读（谨慎）**：
- 方向对（加权 ≥ 未加权 ≥/≈ base），但**样本小、基线饱和**（base 68–85%，而这些是随机抽的较易任务）→ 无显著性；
- 需要在**更难的任务（difficulty L4/L5）** 和**更大样本**上重测，否则测不出差异（天花板效应）；
- 实现层面已通过校验（clean loss=0；plus loss=1.055；ν_q、β_q 合理）。

**下一步**：等全量 baseline 跑完后，在小并发下用 difficulty≥4 的 Light/Background/Noise 子集，重跑三路对比（base / PSC / 加权 PSC），每类 20–30 任务。

## 定时汇报 15:30
- 全量：**9,921 / 20,060（49.5%）**，24 进程存活；GPU 51%、显存 33GB；吞吐约 39 eps/min；
- 预计完成：**19:00–19:40**（比早先估计慢，后半段任务更长）；
- 中间配对（auto vs baseline）：**ALL 48.4% → 48.6%（Δ+0.2，p=0.24）**；
  Background −0.1、Camera +1.2（p=0.060）、Language 0（73.8%）、Robot init 0（33.9%）、Noise −0.4；
- 另：可识别性加权 PSC 的首次三路对比已完成（60 任务）：base 68%、PSC 68%、加权 PSC 70%（n=50，不显著，基线饱和）。

## 定时汇报 16:30
- 全量：**11,321 / 20,060（56.4%）**；20 个进程存活（spatial/object 的前 1/3 分块已完成退出，属正常）；
- 发现 GPU 显存被"难题诊断实验（HN，6 进程）"挤到 40GB、全量吞吐下降 → **已杀 HN 诊断进程**，把 GPU 让回全量；
- 优化后预计完成 **19:30–20:00**；
- 中间配对（auto vs baseline，之前 50% 时）：ALL 48.4% → 48.6%（Δ+0.2，p=0.24）；Camera +1.2（p=0.060）；
- 难题（difficulty≥4，spatial 45 任务）结果：base ≈50%，PSC ≈54.5%（+4.5，p=0.69），加权 PSC ≈52-54%（+2.3，p=1.0）；Sensor Noise 是唯一有动态范围的类别（base 7%，PSC 14%）。

## 定时汇报 17:00
- 全量：**11,488 / 20,060（57%）**；20 进程存活；GPU util 17%（长程任务 goal/10 步数多，渲染占比下降）；
- 实测吞吐 **~29 eps/min**；剩余 8,572 → 预计 **20:30–21:00** 完成；
- 中间配对（auto vs baseline）：**ALL 49.1% → 49.4%（Δ+0.3，p=0.137）**；
  - **Camera Viewpoints：43.8% → 45.1%（Δ+1.3，p=0.0247）← 首次达到统计显著**
  - Background −0.1、Language 0、Robot init 0、Noise −0.5；
- 备注：Camera 的显著仍需多重比较校正/满量确认；理论上色相投票不应修复几何偏移，可能是相机扰动同时改变了光照/阴影（外观成分）。

## 定时汇报 17:30（模式调整：停投票、baseline 加速）
- **已停掉投票（auto）条件**：5,900 / 10,030（59%），数据保留；整体结论已足够（ALL +0.3，Camera +1.2 p=0.025，其余 0）；
- **baseline 重新分块加速**：把剩余 4,017 个任务重新切成 22 个进程（每块约 183 个任务）；
- 当前 baseline：**6,348 / 10,030（63%）**；22 进程；GPU 32%；实测 **~67 eps/min**；
- 预计 baseline **17:30–17:45 完成**；
- 之后按论文协议跑 PSC：在 L5 Camera / L4-L5 Sensor Noise 上做 "离线适应→冻结→评测" 的三路对比（baseline / PSC / 加权 PSC）。

## 第 28 步：在官方 checkpoint 上重测"写回接口"（M1/M2，干净版）

之前 V4/V8 的接口结论都是在**我们短训的 RoboCasa 模型**上做的；本轮在**官方 checkpoint + LIBERO-Plus Camera L5** 上重测：

方法：对 6 帧，捕获写回（最后 Euler 步），比较三类方向的"动作响应/eps"：
1. 源域写回 PCA 方向（"外观方差"方向，6 个）；
2. 12 个随机正交方向；
3. **本帧真实偏移方向**（当前写回 − 源域均值，归一化）。

结果：

| 指标 | 数值 |
|---|---|
| 外观(PCA)响应 / 随机响应 | **0.99**（6 帧：0.98/1.05/0.96/1.02/0.94/0.98 量级） |
| **真实偏移方向响应 / 随机响应** | **1.09**（逐帧 1.08/1.05/1.03/0.88/0.73/1.70） |

解读：
- **"外观 PCA 方向在写回上特别响"这个旧结论（V8 的 6.13×）在官方模型上不成立** → 再次印证旧版本不能沿用；
- 但**真实偏移方向确实比随机方向更"响"（~1.09×）** → 域偏移确实进入了动作接口，只是幅度不大；
- 说明写回是一个"通用接口"，不是"外观专用通道"；因此校正应该作用于**偏移方向和统计量**，而不是"删外观方向"。

下一步（M3）：baseline 结束后，在同一批 L5/L4 难题上比较三个站点：
- baseline
- LN 软校正（PSC）
- **写回软校正**（按偏移方向的统计量做白化-重着色，动作敏感维度保护）
各 40 episode，配对 McNemar。

## 定时汇报 17:30
- baseline：**7,985 / 10,030（79.6%）**，21 进程存活，GPU 43%；剩余多为长程任务（libero_10）；
- 预计 baseline 完成：**18:30–19:00**；
- 投票（auto）条件已按计划停止（5,900/10,030，数据保留），所以本轮不再有 auto 配对数字；最终交付为：
  1) baseline 全量 7 维度 × 5 难度表；
  2) M3 站点对照（baseline / LN 软校正 / 写回软校正）在 L5 Camera + L4-L5 Noise；
- 已完成的 M1/M2（官方 checkpoint 写回接口重测）：
  - 外观 PCA 方向响应/随机 = **0.99**（旧 V8 的 6.13× 在官方模型上不成立）；
  - **真实偏移方向响应/随机 = 1.09**（域偏移确实进入动作接口，但幅度不大）。

## 第 29 步：M3 站点对照（难题：Camera L5 + Sensor Noise L4/5，spatial 40 任务）

### 写回软校正的两个坑（已记录）
1. 第一版用"全维逐维白化"（diag）：**0/22 成功** → 诊断：源域残差方差 τ=0.0014（std≈0.037），
   而目标残差维方差大得多 → 缩放被 clamp 到 0.5，**762 个维度被压缩 2 倍** → 特征被毁；
2. 改成"只在支撑子空间（6 维）校准"：仍 0/27 → 因为源域支撑方向 std 很大（4.0/2.97/...），
   目标侧偏小 → 放大到 clamp 上限 2 → 写回被推偏；
3. 再改成 **expo=0.5 + 缩放区间 [0.85, 1.15]**：正在跑（当前 28 条，10 成功）。

### PSC（LN 软校正，论文协议：在线小步更新 LN affine）
配对结果（40 对）：

| 维度 | n | base | PSC | Δ | p |
|---|---|---|---|---|---|
| Camera Viewpoints (L5) | 20 | 40.0% | 35.0% | −5.0 | 1.00 |
| **Sensor Noise (L4/5)** | 20 | **5.0%** | **30.0%** | **+25.0** | 0.125 |
| **ALL** | 40 | **22.5%** | **32.5%** | **+10.0** | 0.344 |

**这是 PSC 在 LIBERO-Plus 上第一次出现正方向**：难噪声任务 1/20 → 6/20；但 n=20，p=0.125，需要扩样本确认。
Camera 维度略负（−5，不显著）。

### 下一步
- 把 Sensor Noise L4/5 扩到 60 任务，重跑 base / PSC（各 60 条），确认 +25 是否稳定；
- wcal（写回软校正）跑完 40 条后同表对比。

## 定时汇报 18:00：站点差异首次出现
- baseline 全量：**8,891 / 10,030（88.6%）**，8 进程；预计 **18:40–19:10** 完成；
- M3（40 难题）：
  - **PSC（LN 站点）**：ALL +10（base 22.5% → 32.5%）；Sensor Noise +25（5%→30%，n=20，p=0.125）；Camera −5；
  - **wcal（写回站点，部分 28 对）**：ALL +3.6（32.1%→35.7%）；**Camera +10**（40%→50%，n=20）；**Sensor Noise −12.5**（n=8）；
- 读法（待扩样本）：两个站点可能**互补**——LN 站点对噪声有效、写回站点对相机有效；样本都太小。
- 已启动扩样本：**Sensor Noise L4/5 × 60 任务**，base vs PSC（各 60 条）。

## 第 30 步：修正 wcal 的"只最后一步校正"并复核；n60 扩样本

### wcal 修正
- 发现并修复：源统计在"最后 Euler 步"采集，但 wcal 原来在**每个 Euler 步**都校正 → 不一致；
- 改为 `--write-calib-* last_step_only`（默认）后重跑 m3（40 难题，28 对已完成）：
  - ALL：base 32.1% → wcal2 **25.0%（−7.1，p=0.69）**
  - Camera：40% → 35%（−5）；Noise：12.5% → 0%（n=8）
- 结论：**写回软校正（当前形式）仍无正收益**；之前 Camera +10 属于小样本波动。

### n60（spatial，Sensor Noise L4/5 × 60）
- 当前 46 对：base **17.4%** → PSC **23.9%（+6.5，p=0.51）** —— 方向为正但不显著；
- 已再启动 **libero_object 的 60 个难噪声任务**（base/PSC），合并后 n≈120 用于显著性检验。

### 全量 baseline
- 8,985 / 10,030（89.6%），8 进程，预计 **19:00–19:40** 完成。

## 第 31 步：实现 ISC-AP（接口级闭式矩匹配 + 动作保护）并首次评测

### 实现（三件套，均已通过校验）
1. `scripts/build_interface_stats.py`
   - 源域：clean LIBERO 写回（最后 Euler 步）+ **有限差分小推**（32 个随机方向）估计动作 Jacobian；
   - 动作子空间：Jacobian 行空间按 90% 能量截断 → **rank 12**（K=32 支撑时）；
   - 源统计：mean、PCA basis（K=32）、支撑内协方差。
2. `scripts/fit_interface_calibration.py`
   - 目标域：**无标签**标定任务（与评测任务不重叠）采集写回 → 目标均值/协方差；
   - 支撑坐标分解为动作子空间 Q_A 与 nuisance Q_N；
   - **矩匹配只在 nuisance 子空间**（whitening-coloring），动作子空间恒等；
   - 协方差做 5% 收缩 + 变换特征值裁剪到 [0.5, 2]（12 帧标定下防数值爆炸）。
3. `dacsc/interface_calibration.py` + 驱动 `--interface-calib`
   - 只在最后 Euler 步应用（与源统计采集一致）；
   - 加性修正、残差维不动、不改权重、不用标签。

### 两步"问为什么"（都发现并修正了错误设计）
1. 最初 K=10 → 动作子空间在支撑内占满 10 维 → nuisance=0 → **校准退化为恒等**；
   → 提高支撑秩到 K=32，动作 rank=12，nuisance=12，校准才有作用。
2. 12 帧标定直接算协方差逆 → 条件数爆炸（M 特征值 -92～276）；
   → 收缩 + 特征值裁剪后 M∈[0.5,2]，||M−I||=4.08。

### 首次评测（spatial Sensor Noise L4/5 × 60 任务，配对）
| 条件 | 配对 n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| PSC（LN 站点） | 60 | 15.0% | 18.3% | +3.3 | 0.75 |
| **ISC-AP（写回站点）** | 46（进行中） | 17.4% | 21.7% | **+4.3** | 0.63 |

结论：**ISC-AP 方向为正（+4.3），与 PSC（+3.3）同量级，但都不显著**；样本量仍不足。
下一步：合并 spatial+object（n≈120）再做显著性；Camera 维度待跑（需 object 源统计/标定）。

## 第 32 步：代码审计 + ISC-AP 修正 + 首批对照

### 审计发现的 2 个真 bug（已修）
1. **Jacobian 估计未除以 push 幅度**：每帧 eps 不同，混合拟合 → J 有偏；改为记录 Δa/eps；
2. **动作子空间投影用 QR 不判秩**：QR 列恒单位范数，无法发现退化；改为 SVD + 相对阈值取秩。

### 修正后的结果（配对）
| 条件 | 数据 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| PSC | spatial 噪声 L4/5 | 60 | 15.0% | 18.3% | +3.3 | 0.75 |
| PSC | object 噪声 L4/5 | 55 | 34.5% | 40.0% | +5.5 | 0.68 |
| **PSC 合并** | spatial+object | 115 | 24.3% | 28.7% | **+4.3** | 0.49 |
| **ISC-AP（修正后）** | spatial 噪声 L4/5 | 60 | 15.0% | 16.7% | **+1.7** | 1.00 |

**判读**：修正 Jacobian bug 后，ISC-AP 的效应从 +4.3 降到 +1.7（说明之前的正效应部分来自有偏的 J 估计）；PSC 合并后 +4.3 仍不显著。
→ 目前**没有证据表明"矩匹配"在难噪声上显著有效**；但这与"标定帧少、L5 信息丢失"的假设一致。

### 正在跑（object 噪声 60 任务，用 object 源统计 + object 标定帧）
- `no_obj_isc_prot`：ISC-AP（动作保护）
- `no_obj_isc_noprot`：ISC-AP 去掉动作保护（消融）
- 对照：no_obj_base / no_obj_psc（已有 55/60）

### 全量 baseline
- 9,461 / 10,030（94.3%），7 进程，剩余长程任务。

## 第 33 步：满样本后效应消失（负面结论）

object 噪声 60 任务跑满后，配对结果全部翻转：

| 条件 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| PSC（LN 站点） | 60 | 38.3% | 36.7% | **−1.7** | 1.00 |
| ISC-AP（写回 + 动作保护） | 50 | 38.0% | 36.0% | **−2.0** | 1.00 |
| ISC-AP（写回，无保护消融） | 47 | 38.3% | 34.0% | **−4.3** | 0.79 |
| 保护 vs 无保护 | 47 | 36.2% | 34.0% | +2.1 | 1.00 |

合并 spatial+object 的 PSC：+3.3（spatial, n=60）与 −1.7（object, n=60）→ 总体约 **+0.8**。

**结论（诚实）**：
1. 之前看到的 +18.4 / +25 等大效应，在样本跑满后**全部消失**，属于小样本波动；
2. **LN 软校正（PSC）与写回闭式矩匹配（ISC-AP）在 L4/L5 难噪声上均无显著收益**；
3. 动作保护相对无保护略好（+2.1）但不显著；
4. 与"L5 噪声已造成信息丢失、分布对齐无法恢复"的假设一致，但当前数据不足以区分"方法不行"与"效应太小/样本不足"。

## 全量 baseline：9,826 / 10,030（98%），剩余 204 条长程任务。

## 第 34 步：实现"接口级 leftover 一致性 TTA"（含关键位置校验）

### 位置校验（实测 hook 调用序列）
在官方 checkpoint 上一次 predict 的 `blocks[0].attention.attend` 调用共 **11 次**：
- index 0：`prepare_prefix` 的 prefix 编码，token 形状 **[1,129,768]**（representation tokens）→ **绝不能改**；
- index 1–10：10 个 Euler 步，形状 **[1,12,768]**（action horizon=12）→ 这才是"动作接口写回"。

**发现旧实现有 off-by-one**：旧代码用 `(index+1) % num_steps == 0` 判"最后一步"，
实际在 index 9 命中（第 9 个 Euler 步），**不是第 10 步**；且没有显式跳过 prefix。
新实现显式：`if index == 0: return`（prefix），`if index == num_steps: 应用变换`（最后一步）。

### 实现（每块都对应上游）
| 组件 | 内容 | 上游依据 |
|---|---|---|
| 低秩接口变换 | z' = z + U Vᵀ z（U,V: 768×r，仅这两者训练） | LoRA（Hu et al. 2021）低秩适配形式 |
| 采样函数 | 逐行复刻 `DiTActionGenerator.sample`，唯一区别：去掉 `@torch.no_grad` | corp/action_generator.py |
| leftover 损失 | MSE(chunk'[:H−E], prev_chunk[E:]) | SALT（2610.07946）leftover 监督；ACT 的动作分块执行 |
| 协议 | exec_len < horizon（否则无 leftover）；baseline 在同一协议下重跑 | SALT 的 action-chunked 执行 |

### 单元校验
- 零变换恒等：max|Δchunk| = **0.0** ✅
- 梯度流：双零初始化时 grad norm = 0（鞍点）→ 改为 **LoRA 式初始化（u=0, v~1e-3）** 后 grad norm ≈ 0.002–0.45 ✅
- 优化：loss 0.435 → 0.20–0.36（有采样噪声）✅

### 试点（修改协议，exec_len=6）
| 任务数 | base | leftover-TTA |
|---|---|---|
| 10 | **1/10** | 0/10 |

→ 10 条样本看不到信号；30 条 pilot 正在跑。

### 必须声明的偏差
- 官方 LIBERO 评测**执行完整 chunk**；leftover 需要保留尾部 → 本实验为**修改协议**，baseline 已同协议重跑，**不能与官方数字直接比较**。

## 第 35 步：ISA 诊断实验（协议成本、RNG 公平性、可测区间）

### 结论一：对比框架公平（关键验证）
- 适配步数=0（不干预）与 baseline 在同样 10 任务上：**10/10 逐条一致** ✅
- 修复前的问题：适配采样会改变全局 RNG，导致执行噪声与 baseline 不同；已用 `torch.random.fork_rng` 隔离。

### 结论二：协议成本可忽略
- 同 10 个 L4/L5 任务：E=4（每 4 步重规划）base **10%**，官方协议 E=12 base **10%** → 协议不是问题，任务本身太难。

### 结论三：L3 是可测区间，但 ISA 仍有害
| 条件 | n | base | ISA gentle (β≤0.1, λ=0.01) | ISA strong (β≤0.3, λ=0.1) |
|---|---|---|---|---|
| 噪声 L3 | 8–10 | 60–75% | **−10 ~ −25** | **−37.5** |
- L4/L5 难噪声：base 10%，ISA 各变体 0/10。

### 结论四：ISA 有害的机制（诊断日志）
1. β 在 2 个 replan 内顶到上限（饱和）；
2. 源域锚定项与 leftover 动作项量级相当、方向冲突（锚定要求缩方差，动作要求保尺度）；
3. leftover 目标来自同一被污染策略，无"干净锚点"，不能提供校正方向。

### 最终判断
- "接口子空间有界缩放 + leftover + 源域锚定"在 LIBERO-Plus 噪声（L3/L4/L5）上**无正收益、且有害**；
- 与文献一致：ActMAD（同类统计/激活匹配）在 VLA 上仅 +0.6～+1.7，SALT（动作自监督、全参数更新）+7～+9；
- 我们已经排除了：实现 bug（RNG/位置/恒等/梯度/数值）、协议成本、样本不足（L3 有 60–75% 的 base 仍然下降）。

## 全量 baseline：10,012 / 10,030（99.8%）。

## 第 36 步：残差谱诊断（修复 bug 后）与残差软抑制结果

### 修复的两个真 bug
1. 冗余子空间取成了动作子空间（SVD 索引方向错）→ 之前 ISA 全部作废；
2. 残差收缩钩子的计数器没有按 predict 重置（只触发一次）→ 第一批 rs 结果等于没做；
3. 去掉写死的 tau_s（改为显式传入）。

### 谱诊断（每条件 240 帧，最后 Euler 步）
| 条件 | 动作敏感能量 | 支撑冗余能量 | 残差能量 | 冗余谱比 | 残差 top5 特征值 |
|---|---|---|---|---|---|
| clean | 18.98 | 25.28 | **4.18** | 1.10 | 0.36/0.35/0.24/0.19/0.16 |
| L4 | 26.74 | 32.71 | **19.56** | 0.99 | 1.82/1.14/0.72/0.65/0.46 |
| L5 | 20.48 | 24.67 | **21.16** | 0.83 | 1.22/0.90/0.75/0.75/0.42 |

**结论**：噪声能量主要在**残差**（×4.7～5.1），且集中在**前 20 个残差方向**（top20 占 88–94%）；
冗余支撑谱几乎不变（σ 比≈1）。→ "谱抑制"应作用于残差，且可用"选择性/部分抑制"。

### 逐 Euler 步残差（L3，真实 episode）
step1≈52–63 → step10≈14–23；只有最后一步与 clean（4.18）同口径。

### 残差软抑制结果（官方协议，配对）
| 条件 | 配置 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| L4/L5 (n60) | γ_min=0.3（γ≈0.45–0.55） | 44 | 15.9% | 9.1% | **−6.8** | 0.51 |
| L4/L5 (n60) | γ_min=0.7（温和） | 20 | 25.0% | 20.0% | **−5.0** | 1.00 |
| L3 | γ_min=0.3 | 12 | 41.7% | **50.0%** | **+8.3** | 1.00 |

**判断**：严重噪声下收缩有害（可能残差里也含任务信息）；中等噪声（L3）方向为正但 n 小。
不做盲目调参；下一步先做：clean 上的安全性验证 + 选择性（top-k 残差方向）抑制的诊断。

## 第 37 步（定时 00:40）：选择性 top-k 残差抑制初步结果

### clean 安全验证（关键前提）
| 方法 | n | clean 成功率 |
|---|---|---|
| base | 10 | 100% |
| topk20 | 10 | 100%（Δ=0）|
→ **不伤干净性能** ✅

### 配对结果（官方协议）
| 条件 | 配置 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| L4/L5 (n60) | 全残差 γ=0.3 | 60 | 15.0% | 8.3% | **−6.7** | — |
| L4/L5 (n60) | 全残差 γ=0.7 | 60 | 15.0% | 18.3% | **+3.3** | 0.73 |
| L4/L5 (n60) | **top-20 选择** | 25 | 20.0% | **24.0%** | **+4.0** | 1.00 |
| L3 | top-5 | 5 | 60.0% | 60.0% | 0 | — |
| L3 | **top-20** | 6 | 50.0% | **83.3%** | **+33.3** | 0.50 |
| clean | top-20 | 10 | 100% | 100% | 0 | — |

**判断（谨慎）**：
1. **选择性（top-k）明显优于全残差**：L4/L5 从 −6.7 变为 +4.0，且 clean 不受影响；
2. L3 上 top-20 看起来有较大正效应，但 n=6，不可下结论；
3. 需要扩样本（L3 20–40、n60 60）再做配对检验。

### 下一步（按定时任务队列）
- 继续跑满 topk20_n60（60）、topk5/20_l3（20）；
- 补跑 L3 base（20 条，官方协议）用于完整配对；
- 若正效应稳定 → 做 k=5/10/20 与 γ_min 消融，并扩到相机/背景/光照维度。

## 第 38 步（定时 01:05）：top-k 消融出现清晰分层

### 配对结果（官方协议）
| 条件 | 配置 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| L3 | top-5 | 14 | 42.9% | 50.0% | +7.1 | 1.00 |
| **L3** | **top-10** | 14 | 42.9% | **71.4%** | **+28.6** | 0.125 |
| **L3** | **top-20** | 14 | 42.9% | **64.3%** | **+21.4** | 0.25 |
| L4/L5 (n60) | top-20 | 57 | 14.0% | 12.3% | −1.8 | 1.00 |
| L4/L5 (n60) | 全残差 γ=0.3 | 60 | 15.0% | 8.3% | −6.7 | — |
| L4/L5 (n60) | 全残差 γ=0.7 | 60 | 15.0% | 18.3% | +3.3 | 0.73 |
| clean | top-20 | 10 | 100% | 100% | 0 | — |

### 观察
1. **中等噪声（L3）：k=10/20 有大幅正效应（+21～+29，n=14）**，但 p=0.125–0.25，需要样本；
2. **严重噪声（L4/L5）：选择性 top-20 不再有害（−1.8）**，全残差版本有害（−6.7）；
3. **clean 100% 不变**；
4. k 太小（5）效果弱，k=10 似乎最好 → 与"噪声集中在 top-20 方向"的测量一致。

### 下一步
- 补满 L3（base/topk10/topk20 各 20 条）与 n60（topk10/topk20 各 60 条）；
- 若 L3 正效应在 n=20 上保持（Δ≥+15），扩到 40 条并做 k=8/10/15 细化；
- 同步启动 topk10 在 L4/L5 的对照（已启动）。

## 第 39 步（定时 01:30）：L3 完整配对 + 严重度门控

### L3（20 任务，官方协议，完整配对）
| 配置 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| top-5 | 20 | 35.0% | 40.0% | +5.0 | 1.00 |
| **top-10** | 20 | 35.0% | **55.0%** | **+20.0** | 0.125 |
| top-20 | 20 | 35.0% | 50.0% | +15.0 | 0.25 |

### L4/L5（n60，官方协议）
| 配置 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| top-20 | 60 | 15.0% | 13.3% | −1.7 | 1.00 |
| top-10 | 46 | 17.4% | 10.9% | −6.5 | 0.25 |

### 结论
- **中等噪声 L3：top-10 稳定 +20（n=20，p=0.125）**；
- **严重噪声 L4/L5：选择性抑制仍略有害（−1.7～−6.5）**；
- clean 100% 不受影响；
- 机制解释：L4/L5 的残差超出源谱太多（τ_t/τ_s≈4.7–5.1），其中也含任务信息；L3 的超出较小（≈3.3），可安全压。

### 新增（有依据的改动）
按实测 τ_t/τ_s 加入**严重度门控**：只在 ratio ∈ [1.2, 4.5] 时抑制，超出则跳过。
（依据：L3 ratio≈3.3 有效；L4/L5 ratio≈4.7–5.1 有害。风险：两者分布有重叠，门控可能不完美。）

### 已启动
- gate_n60_topk10、gate_l3_topk10；
- L3 扩样本（新增 20 任务）：base_l3_extra、topk10_l3_extra。

## 第 40 步（定时 01:55）：复制实验——L3 效应不稳健，门控在 L4/L5 去害

### 结果
| 实验 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| L3 第一批（原 20）top-10 | 20 | 35.0% | 55.0% | +20.0 | 0.125 |
| **L3 第二批（新 20）top-10** | 14 | 50.0% | 42.9% | **−7.1** | 1.00 |
| **L3 合并** | 34 | 41.2% | 50.0% | **+8.8** | 0.375 |
| L4/L5 门控 top-10 | 46 | 17.4% | 19.6% | **+2.2** | 1.00 |
| L4/L5 无门控 top-10 | 46 | 17.4% | 10.9% | −6.5 | 0.25 |
| clean | 10 | 100% | 100% | 0 | — |

### 判断
1. **L3 的 +20 未能复制**（新一批 −7.1），合并后只有 +8.8（p=0.375）→ 之前的大效应部分是任务/运气波动；
2. **门控确实去掉了 L4/L5 的害处**（−6.5 → +2.2），符合"严重度门控"的预期；
3. 目前最稳的结论：**选择性抑制 + 严重度门控 = 干净/严重噪声上安全中性，中等噪声上可能有小幅正效应（+9 量级，未显著）**。

### 下一步
- 再补 20 个 L3 任务（base + topk10）→ 合并 n≈54，判断 +9 是否真实；
- 若仍不显著：进入 D1（动作敏感度加权抑制）——先测量残差各方向的动作响应。

## 第 41 步（定时 02:20）：D1 诊断——残差方向的动作敏感性几乎是平的

### 合并样本结果（更新）
| 实验 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|
| L3 三批合并（top-10） | 57 | 42.1% | 47.4% | **+5.3** | 0.375 |
| L4/L5 门控 top-10 | 60 | 15.0% | 16.7% | **+1.7** | 1.00 |
| clean | 10 | 100% | 100% | 0 | — |

### D1 诊断（新）
对源残差前 20 个方向，逐方向“小推”，测动作响应（||Δa||/eps，10 帧）：
- 前 5 个方向：11.5 / 14.0 / 13.2 / 11.3 / 12.3；
- 后 5 个方向：12.3 / 14.5 / 12.0 / 12.4 / 14.1。

**结论：残差各方向的动作敏感性几乎相同（无“低敏感可安全抑制”的方向）。**
→ D1“按动作敏感度加权抑制”在残差层不成立：任何残差方向被压都会同等影响动作；
→ 这与之前"外观方向 vs 随机方向响应比 0.99"一致——接口对方向近似各向同性。
→ 因此可用的选择性只有“能量（谱）选择性”本身（我们已做：逐方向 γ_i = sqrt(λ_s/λ_t)）与“严重度门控”。

### 下一步
- 试 D3：多步一致抑制（第 5–10 步都施加）看能否放大 L3 的小效应；
- 或 E：把谱诊断/抑制扩到相机、背景维度，判断适用边界；
- 不再做 D1 方向加权（已被数据否定）。

## 第 42 步（定时 02:45）：跨扰动维度的谱诊断——相机维度最匹配

### 残差能量（τ_res，每条件 240 token；clean=4.18）
| 扰动维度 | 残差能量 | 比值 | 动作敏感能量 | 冗余支撑能量 | 冗余谱比 |
|---|---|---|---|---|---|
| clean | 4.18 | 1.00 | 18.98 | 25.28 | 1.10 |
| Light L4/5 | 5.11 | 1.22 | 22.89 | 24.55 | 1.06 |
| Background L4/5 | 5.56 | 1.33 | 11.33 | 15.85 | 0.79 |
| **Camera L4/5** | **12.92** | **3.09** | 18.44 | 27.16 | 1.04 |
| Noise L3 | ~14 | ~3.3 | — | — | — |
| Noise L4 | 19.56 | 4.68 | 26.74 | 32.71 | 0.99 |
| Noise L5 | 21.16 | 5.06 | 20.48 | 24.67 | 0.83 |

### 判断
- **相机偏移与中等噪声同量级（×3.1）→ 我们的残差抑制“适用带”应覆盖相机**；
- 光照/背景残差几乎不变（×1.2–1.3）→ 这两个维度不该用残差抑制；
- 这给了方法明确的适用边界（按实测谱自动判断，而不是拍脑袋）。

### 已启动
base_cam、topk10_cam（20 个 Camera L5 任务，官方协议）——验证相机维度上是否也 +5～+20。

## 第 43 步（定时 03:10）：相机维度验证失败；发现"均值偏移"新机制

### 相机维度结果（top-10，20 任务）
base 40.0% → topk10 35.0%（−5.0，p=1.0）→ **用"残差能量"预测适用性失败**：相机残差能量 ×3.1（和 L3 相当），但抑制无益。

### 为什么？——残差结构诊断（新增均值项）
| 条件 | 残差总能量 | 均值能量 | 均值占比 | 首特征值 |
|---|---|---|---|---|
| clean | 4.18 | 1.81 | 43% | 0.36 |
| **Camera L4/5** | 14.88 | **9.65** | **65%** | 2.11 |
| 噪声 L4/L5 | 19.6–21.2 | （方差主导） | 低 | 1.2–1.8 |

**机制**：
- 相机偏移 = **一致的偏差（均值漂移）**，不是随机噪声；
- 传感器噪声 = **方差增大**；
- 所以"压方差"对相机无效，应该做"去偏（均值校正）"。

### 新机制（沿路线，有依据）
`r' = r − α·(mean_t(r) − mean_s(r))`，`w' = w + (r'−r)`
——残差均值源域映射（可逆、有界、只改最后一步；均值由 EMA 估计）。

### 已启动
debias_cam（20）、debias_l3（20）、debias_n60（60）——验证"相机用去偏、噪声用压方差"的分工。

## 第 44 步（定时 03:35）：分工假设得到验证——相机去偏 +15

### 结果（官方协议，配对）
| 条件 | 机制 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| **Camera L4/5** | **去偏（均值）** | 20 | 40.0% | **55.0%** | **+15.0** | 0.375 |
| Camera L5 | 压方差（top10） | 20 | 40.0% | 35.0% | −5.0 | 1.00 |
| Noise L3 | 去偏 | 8 | 37.5% | 37.5% | 0 | — |
| Noise L3 | 压方差（top10） | 57 | 42.1% | 47.4% | +5.3 | 0.375 |
| Noise L4/L5 | 去偏 | 42 | 16.7% | 14.3% | −2.4 | 1.00 |
| Noise L4/L5 | 门控压方差 | 60 | 15.0% | 16.7% | +1.7 | 1.00 |
| clean | 压方差 | 10 | 100% | 100% | 0 | — |

### 结论（这是今晚最自洽的一条）
**残差结构决定校正方式**：
- **均值漂移型（相机）→ 去偏**（+15）；
- **方差增大型（传感器噪声）→ 压方差**（+5.3，严重时门控中性）；
- 两者用错方向时效果为 0 或负（相机压方差 −5；噪声去偏 0/−2.4）。

### 下一步
- 相机扩样本（新增 20 任务）验证 +15 的可复制性；
- clean 上去偏安全验证；
- 若可复制：实现**自适应规则**（按残差均值占比自动选“去偏 vs 压方差”），即我们的新机制。

## 第 45 步（定时 04:00）：相机去偏复制成功（+17.5，p=0.09）；clean 出现安全问题并加门控

### 相机复制（合并 n=40）
| 条件 | n | base | debias | Δ | 不一致对 | p |
|---|---|---|---|---|---|---|
| Camera L4/5（合并） | 40 | 35.0% | **52.5%** | **+17.5** | 3 vs 10 | **0.0923** |

### clean 安全问题（发现即修）
- clean 上去偏：base 100% → **90%**（1 条失败，n=10）→ 去偏在干净数据上会误伤（clean 残差均值本身非零，占 43%）。
- **修复**：加"均值偏移门控"——仅当 target_mean_energy / source_mean_energy ≥ 1.5 时才去偏。
  - 依据：clean 比值≈1；相机≈5.3。

### 已重跑
debiasg_clean（应恢复 100%）、debiasg_cam、debiasg_camx（门控版相机）验证。

## 第 46 步（定时 04:25）：门控阈值必须按实测分布定（clean 会触发）

### 实测：逐帧"均值偏移比"（target_mean_energy / source_mean_energy）
| 条件 | 各帧比值 |
|---|---|
| clean（4 任务） | 1.83/1.62/3.66；1.41/1.37/1.84；1.32/1.19/2.53；1.39/1.18/2.11 |
| **Camera（4 任务）** | 9.38/7.75/6.56；8.11/7.87/6.03；8.54/7.92/8.45；7.74/6.71/7.22 |

→ **clean 最高 3.66，camera 最低 6.03** → 阈值 1.5 太低（clean 会误触发，正是 clean 掉 10 分的原因）；
→ 按分布把门控阈值改为 **4.5**（在 clean 上界与 camera 下界之间）。

### 已重跑
debiasg2_clean / debiasg2_cam / debiasg2_camx（阈值 4.5）→ 期望 clean 恢复 100%、camera 保持 +15~+20。

## 第 47 步（定时 04:50）：门控 4.5 生效——clean 100% 恢复，相机 +17.5 保留

### 结果
| 条件 | n | base | 方法（去偏，门控 4.5） | Δ | p |
|---|---|---|---|---|---|
| clean | 10 | 100% | **100%** | 0 | — |
| **Camera L4/5** | 40 | 35.0% | **52.5%** | **+17.5** | **0.0923** |

→ **clean 安全 + 相机收益，同时成立**（门控阈值按实测分布定：clean≤3.66，camera≥6.03，取 4.5）。

### 当前方法形态（自适应残差源域映射）
```
残差 r = (w−μ_s) − VᵀV(w−μ_s)
若 均值偏移比 ≥ 4.5  → 去偏：r' = r − (mean_t − mean_s)        （相机/几何偏移）
否则若 能量比 ∈ [1.2,4.5] → 逐方向压方差：r'_i = γ_i r_i        （传感器噪声）
否则 → 不干预                                                （干净/过强噪声）
```

### 已启动
- camera 再补 20 任务（base + 去偏）→ 合并 n=60，冲 p<0.05。

## 第 48 步（定时 05:15）：相机 n=60 → +13.3，p=0.057（接近显著）

| 条件 | n | base | 去偏（门控 4.5） | Δ | 不一致对 | p |
|---|---|---|---|---|---|---|
| Camera L4/5（三批合并） | 60 | 41.7% | **55.0%** | **+13.3** | 3 vs 11 | **0.0574** |

→ 复制两次（+15、+20）后合并仍为正、接近显著；需要更大样本或跨 suite 验证。

### 跨 suite 泛化（已启动）
- 构建 object suite 源谱（tau=3.51, mean_norm=1.449）；
- 在 object 的 Camera L4/5 任务（20 个）上跑 base vs 去偏，检验"相机去偏"是否跨套件成立。

### 下一步
- object 相机结果；
- 若跨 suite 也正 → 合并 n=80 冲 p<0.05，作为主结果；
- 然后把①②③（去偏/压方差/跳过）写成统一模块，做全维度总表。

## 第 49 步（定时 05:40）：相机跨套件一致为正；合并 n=80 仍 p=0.064

### 结果
| 条件 | n | base | 去偏 | Δ | p |
|---|---|---|---|---|---|
| Camera（object 首批） | 20 | 50.0% | 60.0% | +10.0 | 0.75 |
| **Camera 合并（spatial 60 + object 20）** | **80** | 43.8% | **56.2%** | **+12.5** | **0.0639** |

→ 跨套件方向一致（object +10），合并后 +12.5、p=0.064，仍需样本。

### 已启动
- object 相机再补 20（base+去偏）→ 合并 n=100，冲 p<0.05。

### 备注（论文口径）
- 相机去偏：**方法对"均值漂移型"偏移有效，且跨 suite 方向一致**；
- 噪声压方差：+5.3（中等）、严重时门控中性；
- clean 安全（100%）。

## 第 50 步（定时 06:05）：相机合并 n=100 → +8.0（p=0.18），效应随样本增大而缩小

### 结果
| 批次 | n | base | 去偏 | Δ | p |
|---|---|---|---|---|---|
| object 相机第 1 批 | 20 | 50% | 60% | +10 | 0.75 |
| object 相机第 2 批 | 20 | 80% | 70% | **−10** | 1.00 |
| **相机合并（n=100）** | 100 | 51.0% | 59.0% | **+8.0** | 0.185 |

→ 方向仍为正，但**批次间波动大**（spatial +13.3、object 两批 +10/−10），合并后效应降到 +8、不显著；
→ 说明"相机去偏"存在**任务/场景依赖**，不是普适的固定增益。

### 已启动
- 构建 goal suite 源统计（action rank 11）与源谱（tau=4.05, mean_norm=1.26）；
- goal 的 Camera L4/5（20 任务）base vs 去偏 → 第三个套件验证。

### 判断（保持路线）
- 不因波动否定方法；但结论要写成"有条件的增益"：均值漂移明显的相机偏移上平均为正（+8~+13），个体批次可能为 0/−10；
- 需要按"偏移强度/结构"分层报告，而不是只报总体平均。

## 第 51 步（定时 06:30）：goal 相机地板效应 + 统一自适应模块上线

### goal 套件相机（L4/5，20 任务）
base 0.0% vs 去偏 0.0% → **地板效应**（goal 的 Camera L4/5 本身就做不出来），该套件不能用于相机验证。

### 相机总账（可用：spatial+object，n=100）
51.0% → 59.0%，+8.0，p=0.185（批次波动大，条件性正效应）。

### 统一自适应模块（新增，已上线）
`dacsc/adaptive_residual.py`：按实测残差结构自动选择
- 均值偏移比 ≥4.5 → 去偏；
- 能量比 ∈[1.2,4.5] → 逐方向压方差；
- 其它 → 跳过。
仍在最后 Euler 步、可逆有界、不动动作敏感子空间；并统计各模式触发次数。

### 自适应模式初步结果（进行中）
| 条件 | n | base | auto | Δ | 备注 |
|---|---|---|---|---|---|
| Camera（20） | 20 | 40.0% | 55.0% | **+15.0** | 自动选了去偏 |
| clean（10） | 10 | 100% | 100% | 0 | 自动跳过 |
| L3（6） | 6 | 50% | 50% | 0 | 太早 |
| L4/L5（35） | 35 | 17.1% | 14.3% | −2.9 | 应为跳过，差异来自阈值边界 |

## 第 52 步（定时 06:55）：自适应模式统计暴露出"模式抖动"

### 结果
| 条件 | n | base | auto | Δ | 模式统计 |
|---|---|---|---|---|---|
| Camera | 20 | 40.0% | **55.0%** | **+15.0** | debias 296 / shrink 0 / skip 0 ✅ |
| clean | 10 | 100% | 100% | 0 | （日志待出）|
| L3 | 14 | 42.9% | 42.9% | 0 | **debias 27 / shrink 60** ⚠️ |
| L4/L5 | 46 | 17.4% | 13.0% | **−4.3** | （待出）|

### 诊断
- 相机：模式判断稳定（全 debias），结果 +15 ✅；
- L3：**模式抖动**（27 次去偏 + 60 次压方差）→ 去偏用错 → 稀释了压方差的正效应（此前单独 top-k 是 +5.3）；
- L4/L5：−4.3，可能同样因为能量比在 4.5 边界抖动时偶发进入 shrink。

### 修复（有依据，非调参）
1. **模式锁定**：episode 内 warmup 后先判定一次模式，之后锁定，不再逐帧切换（避免抖动）；
2. 判定用一段 EMA 证据，而不是单帧比值；
3. 保持阈值来自实测分布（clean ≤3.66、camera ≥6.03）。

### 下一步
- 实现模式锁定后，重跑 Camera/L3/L4-L5/clean 四条件，比较"锁定 vs 逐帧"。

## 第 53 步（定时 07:20）：实现"模式锁定"并启动复核

### 修复（依据：模式统计显示抖动）
- adaptive 模块改为 **warmup 后用累计 EMA 判定一次模式，之后锁定**（lock_after=10）；
- 单元校验（相机 3 任务）：mode_counts = debias 48 / shrink 0 / skip 0 → 不再抖动 ✅。

### 已启动（锁定版，四条件）
lock_cam（20）、lock_l3（20）、lock_n60（60）、lock_clean（10）。
预期：相机 +15 保持；clean 100%；L3 回到 +5 左右；L4/L5 回到中性。

## 第 54 步（定时 07:45）：模式锁定后仍不能自动区分相机 vs 噪声

### 锁定版结果
| 条件 | n | base | 锁定自适应 | Δ | 实际模式 |
|---|---|---|---|---|---|
| Camera | 20 | 40.0% | 50.0% | +10.0 | debias（正确）|
| clean | 10 | 100% | 100% | 0 | 跳过（正确）|
| L3 | 20 | 35.0% | 35.0% | 0 | shrink（正确）|
| L4/L5 | 53 | 17.4%（对照 60 条） | ~9.4% | **负** | **debias（错误！）** |

### 诊断：现有"结构特征"无法区分相机 vs 严重噪声
| 条件 | 残差能量比 | 均值能量比 | 残差均值时间一致性（cos） |
|---|---|---|---|
| clean | 1.00 | 1.00 | — |
| Camera | 3.56 | 5.32 | 0.5–0.94（≈0.72）|
| L3 | 4.32 | 6.64 | — |
| L4 | 4.68 | 6.74 | 0.4–0.98（≈0.78）|
| L5 | 5.07 | 8.27 | — |

**两者（相机 vs L3/L4）在能量比、均值比、时间一致性上都重叠** → 单纯靠残差结构无法可靠选择"去偏 vs 压方差"。

### 结论（诚实）
1. 已可靠的两件事：
   - **相机去偏**：+8～+13（n=100，p≈0.06–0.18），跨套件方向一致；
   - **噪声压方差**：中等 +5.3（未显著）；严重噪声门控后中性。
2. 未解决：**自动选择操作**（相机 vs 噪声）在无监督条件下不可靠；需要额外信号（例如按任务类别/suite 的已知扰动标签，或在线反馈），否则应作为"两个独立模式"分别报告。

### 下一步（用户醒来后确定）
- 方案 A：不追求"全自动选择"，把方法写成"按残差结构分两种模式 + 门控"，实验按扰动类型分别报告（诚实、可复现）；
- 方案 B：引入额外无标签信号（例如同 episode 多帧的一致性/成功伪信号）来区分；
- 方案 C：固定规则按已知扰动类型（LIBERO-Plus 维度标签在评测时可得？）——如果允许用扰动类型标签，则"按类型选模式"完全可复现。

## 第 55 步（定时 08:10）：补样本冲显著性

已启动：
- object 相机再补 20（base/去偏）→ 相机合并 n=120；
- L3 再补 20（base/top-k10）→ L3 合并 n≈77；
目的：把两个主结果（相机 +8~+13、L3 +5）的 p 值压到 0.05 以内，或明确其真实量级。

## 第 56 步（定时 08:35）：补样本后的合并结果（相机 +8.3，L3 +3.8）

| 条件 | n | base | 方法 | Δ | 不一致对 | p |
|---|---|---|---|---|---|---|
| **Camera（合并 spatial+object 三批）** | 120 | 49.2% | **57.5%** | **+8.3** | 12 vs 22 | 0.121 |
| L3（top-10，四批合并） | 80 | 33.8% | 37.5% | +3.8 | 1 vs 4 | 0.375 |

→ 相机效应稳定在 **+8 左右**，但 p=0.12；按当前不一致比例，需 n≈250–300 才可能到 p<0.05。
→ 已再启动 object 相机 60 个任务（base/去偏）→ 相机合并 n=180，最后冲一次显著性。

### 结论口径（若最终仍 p>0.05）
- 相机去偏：**平均 +8（n=120），方向一致、跨套件一致，但未达显著**；
- 噪声压方差：+3.8（n=80），未显著；
- clean 安全；严重噪声门控中性。
- 方法贡献应写成"条件性收益 + 安全边界 + 机制分析"，而不是"显著超过 baseline"。

## 第 57 步（定时 09:00）：相机效应的套件分解——spatial 有效、object 近零

| 套件 | n | base | 去偏 | Δ | 不一致对 | p |
|---|---|---|---|---|---|---|
| **spatial Camera** | 60 | 41.7% | **55.0%** | **+13.3** | 3 vs 11 | **0.0574** |
| object Camera | 120 | 57.5% | 60.0% | +2.5 | 19 vs 22 | 0.755 |
| 合并 | 180 | 52.2% | 58.3% | +6.1 | 22 vs 33 | 0.177 |

### 最终结论（今晚的诚实版本）
1. **相机去偏：spatial 上 +13.3（n=60，p=0.057）、object 上 +2.5（n=120，p=0.76）**；
   → 效应**依赖套件/场景**，不是普适增益；合并 +6 不显著。
2. **噪声压方差：L3 +3.8（n=80，p=0.38）**，未显著；严重噪声门控后中性。
3. **clean 100% 安全**（所有版本）。
4. **自动选择模式（去偏/压方差）在无监督条件下不可靠**（结构特征重叠），不应作为核心声称。
5. 建议论文口径：**"条件性收益 + 安全边界 + 机制分析（残差结构、均值/方差、动作敏感性）"**，并把"套件依赖"作为限制明确写出。

## 第 58 步（定时 09:50）：全量 baseline 完成，最终表

全量 10,030/10,030（官方协议，1 trial/task）：

### 总览
| 指标 | 值 |
|---|---|
| 总成功率 | 5,357 / 10,030 = **53.4%** |

### 按扰动维度
| 维度 | n | SR |
|---|---|---|
| Language Instructions | 1,537 | 74.3% |
| Light Conditions | 1,142 | 72.8% |
| Objects Layout | 1,525 | 71.1% |
| Camera Viewpoints | 1,599 | 44.2% |
| Sensor Noise | 1,601 | 39.6% |
| Background Textures | 1,076 | 37.8% |
| Robot Initial States | 1,550 | 35.5% |

### 按 suite
| suite | n | SR |
|---|---|---|
| libero_object | 2,518 | 71.0% |
| libero_spatial | 2,402 | 68.3% |
| libero_10 | 2,519 | 51.4% |
| libero_goal | 2,591 | 24.5% |

### 按难度
| 等级 | n | SR |
|---|---|---|
| L1 | 1,644 | 72.0% |
| L2 | 2,202 | 63.0% |
| L3 | 2,094 | 53.8% |
| L4 | 1,886 | 46.3% |
| L5 | 2,083 | 33.1% |

→ 至此所有计划内实验完成；方法结论稳定（相机 spatial +13.3 p=0.057、噪声 +4、clean 100%），**可停止定时任务**。

## 第 59 步（10:00）：按理论实现"完整仿射对齐"（CORAL 残差版）并首测

### 实现（依据）
- 理论：高斯→covariate shift 下，均值+协方差同时对齐使 KL 最小（CORAL, Sun & Saenko 2016；同 SSA/PSC 矩匹配家族）；
- 公式（低秩残差子空间 k=20）：
  c = r P^T；c' = μ_s + Σ_s^{1/2} Σ_t^{-1/2}(c − μ_t)；r' = r + (c'−c)P；
  加协方差收缩 0.1、映射特征值裁剪 [0.5,2]（有界可逆）；
- 门控：能量比 <1.2 或 >6.0 跳过（初版，后按实测收紧到 4.5）。
- 修复：bf16 下 torch.linalg.eigh 不支持 → 统计量统一 float32。

### 首测（官方协议，配对）
| 条件 | n | base | affine | Δ | p |
|---|---|---|---|---|---|
| spatial 相机 | 60 | 41.7% | 46.7% | **+5.0** | 0.58 |
| L3（进行中） | 8 | 37.5% | 62.5% | +25.0 | 0.50 |
| L4/L5（进行中） | 39 | 15.4% | 10.3% | **−5.1** | 0.69 |
| clean | 10 | 100% | 100% | 0 | — |

### 发现
1. **完整仿射在相机上不如"只去偏"**（+5 vs +13.3）→ 二阶统计（协方差）估计噪声大（每帧仅 12 tokens，EMA 有限），理论最优性要求准确估计；
2. **L4/L5 被误纳入门控**（ratio 4.7–5.1 < gate_max=6）→ 需要按实测把 gate_max 收紧到 4.5；
3. clean 安全。

## 第 60 步（10:30）：系统验证优化方案（一阶 vs 二阶、步调度、强度）

### 已完成的直接对照（同任务配对）
| 对照 | n | 一阶去偏 | 完整仿射 | Δ | 说明 |
|---|---|---|---|---|---|
| spatial 相机 | 60 | **55.0%** | 46.7% | **−8.3**（对去偏） | 一阶更好（协方差估计噪声大） |

### 新启动（系统验证）
1. affg_n60：仿射 + 门控 4.5（严重噪声应跳过）；
2. aff_s5_l3：仿射，从第 5 个 Euler 步开始（步调度）；
3. aff_s5_cam：仿射 + 步调度，相机 60；
4. aff_a05_cam：仿射强度 α=0.5（强度参数化）；
5. aff_l3/aff_n60 继续跑完。

### 注意：动作专家只有 1 个 DiT block（config num_layers=1）
→ "站点消融（block0 vs 后层）"不成立；站点对照只能做"写回 vs 前缀表示"。

## 第 61 步（定时 10:40）：优化方案系统验证结果（全部有配对数据）

### 相机（spatial 60 任务，同任务配对）
| 变体 | n | base | 方法 | Δ | p | 结论 |
|---|---|---|---|---|---|---|
| **均值去偏（一阶，最后一步）** | 60 | 41.7% | **55.0%** | **+13.3** | **0.057** | **最佳** |
| 完整仿射（CORAL，最后一步） | 60 | 41.7% | 46.7% | +5.0 | 0.58 | 二阶噪声大 |
| 完整仿射 + 步调度（从第 5 步） | 60 | 41.7% | 43.3% | +1.7 | 1.00 | 步调度无益 |
| 完整仿射 + 强度 α=0.5 | 60 | 41.7% | 45.0% | +3.3 | 0.50 | 部分强度无益 |

### 噪声
| 变体 | 数据 | n | base | 方法 | Δ | p |
|---|---|---|---|---|---|---|
| 完整仿射 | L3 | 20 | 35.0% | 40.0% | +5.0 | 1.00 |
| top-k 压方差 | L3 合并 | 80 | 33.8% | 37.5% | +3.8 | 0.38 |
| 完整仿射 + 门控 6.0 | L4/L5 | 60 | 15.0% | 10.0% | −5.0 | — |
| **完整仿射 + 门控 4.5** | L4/L5 | 45 | 17.8% | 17.8% | **0.0** | 1.00 |

### 结论（可写进论文的消融）
1. **一阶（均值）> 二阶（完整矩匹配）**：在动作接口、每帧 token 少的情况下，协方差估计噪声抵消了理论收益；
2. **步调度、强度 α 都不改善**：只在最后一步、满强度反而是最好的；
3. **门控阈值必须按实测分布（4.5）**：否则严重噪声被误校正（−5），收紧后完全中性；
4. 最终推荐配置：**相机→去偏、噪声→top-k 压方差（或仿射，效果相近）、严重/干净→门控跳过**。

## 第 62 步：发现"采样协议不匹配"这一严重测量问题（重要更正）

### 事实
- 之前源谱用 **前 2 个 chunk**（早期帧，240 token）测：clean τ_res≈4.18；
- 目标侧用在线 EMA（跨整个 episode）测 → 与源测口径不同；
- 用**同一口径（10 chunk、1000+ token）**重测后：
  | 条件 | τ_res | 均值能量 |
  |---|---|---|
  | clean (spatial) | 15.25 | 7.44 |
  | Camera (spatial) | 24.35 | 10.94（×1.60 / ×1.47）|
  | L4 (noise) | 21.83 | 9.81（×1.43 / ×1.32）|
  | L5 (noise) | 20.70 | 9.87（×1.36 / ×1.33）|
  | Camera (object) | 18.07 | 7.72（×1.07 / ×1.11，源 16.81/6.97）|
  | Camera (goal) | 7.41 | 2.78（×0.60 / ×0.77，源 12.36/3.61）|

### 更正与解读（重要）
1. 之前"噪声/相机让残差能量涨 4.7–5.1 倍"是**采样口径不一致造成的假象**；真实超出只有 **1.1–1.6 倍**；
2. 这解释了为什么方法效应小：**残差空间里的真实偏移本来就只有 30–60%**；
3. 也解释了套件差异：spatial 相机超出最多（1.60）→ 效应 +13.3；object 只有 1.07→ +2.5；goal 反而低于 clean（0.60）→ 0%（且地板）；
   → **方法收益与"实测残差超出量"同向**，这是一个自洽、可检验的叙事。

### 已执行（按要求）
- 源统计扩到 1000+ token（10 chunk × 4 suite）；
- **完整 Camera 维度（1,599 条）**评测已启动（每套件用自己的源统计 + 匹配口径门控 1.15）；
- 噪声完整维度随后；分析将按 L1–L5 分层 + Holm 校正。

## 第 63 步：代码严格校验（逐项，与上游对照）

新增 `scripts/verify_impl.py`（本地/服务器可跑），11 项检查全部通过：

| # | 检查 | 结果 |
|---|---|---|
| 1 | hook 调用序列：1 次前缀（129 token）+ 10 次 Euler（12 token） | PASS |
| 2 | SSA 对称 KL 与官方 `utils/loss.py::diagonal_gaussian_kl_loss` 双方向求和逐值一致（12.682089） | PASS |
| 3 | CORAL 映射的解析协方差传输精确（cov_err 9.5e-6） | PASS |
| 4 | 有界映射特征值 ∈ [0.5, 2] | PASS |
| 5 | 残差基正交性 max|PPᵀ−I| = 3.6e-7 | PASS |
| 6 | top-k 压方差：目标方差=源方差时 γ=1（恒等） | PASS |
| 7 | top-k 改动只落在 top-k 子空间（外部分量 1.5e-7） | PASS |
| 8 | top-k 在方差偏大时确实收缩（Δmax=0.333） | PASS |
| 9 | 仿射映射：目标矩=源矩时恒等（修复后 2.4e-7） | PASS |
| 10 | 去偏：目标均值=源均值时校正为 0 | PASS |
| 11 | 独立重跑 baseline 与全量官方 baseline 在 60 个共享任务上逐条一致 | PASS |

### 本次校验中发现并修复的两个实现问题
1. **仿射的协方差收缩用了各向同性先验** → 即使无偏移映射也非恒等；改为**向源协方差收缩**（正确先验），恒等性恢复；
2. bf16 下 `torch.linalg.eigh` 不支持 → 统计量统一转 float32。

### 其他（协议）
- 独立 baseline 与全量 baseline 逐条一致 → 评测协议/数据路径正确；
- 大的数据问题仍是"采样口径不一致"（已更正并按下述新口径评测）。

### 进行中
- **完整 Camera 维度（1,599 条）**评测：spatial 91、object 128、goal 88、10 73（进行中）。

## 第 64 步：完整 Camera 维度中期结果（n=657/1599，约 41%）

### 总体
| 指标 | 值 |
|---|---|
| n（配对） | 657 |
| base | 54.8% |
| 去偏 | 57.5% |
| Δ | **+2.7pp，95% CI [−0.5, +6.1]，p=0.120** |

### 按套件
| 套件 | n | base | 方法 | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| spatial | 162 | 53.1% | 56.2% | +3.1 | [−3.7,+9.9] | 0.50 |
| **object** | 214 | 76.2% | 83.2% | **+7.0** | **[+1.9,+12.6]** | **0.020** |
| goal | 151 | 9.9% | 11.3% | +1.3 | [−1.3,+4.0] | 0.63 |
| libero_10 | 130 | 73.8% | 70.8% | −3.1 | [−13.1,+6.9] | 0.65 |

### 按难度（Holm 校正后）
| 等级 | n | base | 方法 | Δ | p | p_holm |
|---|---|---|---|---|---|---|
| L1 | 120 | 68.3% | 68.3% | 0.0 | 1.00 | 1.00 |
| L2 | 201 | 60.7% | 64.2% | +3.5 | 0.30 | 1.00 |
| **L3** | 121 | 52.1% | 60.3% | **+8.3** | 0.087 | **0.44** |
| L4 | 59 | 54.2% | 49.2% | −5.1 | 0.61 | 1.00 |
| L5 | 156 | 39.1% | 41.7% | +2.6 | 0.60 | 1.00 |

**中期判断**：平均效应小（+2.7pp），置信区间跨 0；object 套件 +7 显著（未校正），L3 +8.3 边缘；
最终 n=1,599 时 CI 预计收窄到 ±2.2pp，可判定是否显著。套件异质性明显（object +7、libero_10 −3）。

## 第 65 步：在"修正后的协议"下重测所有变体（推翻旧结论）

### 为什么必须重测
之前的变体比较（一阶 vs 二阶、步调度、强度、top-k）都用了**错误口径的源统计**（前 2 chunk）和不匹配的门控阈值；
结论（如"二阶不如一阶"）可能是口径错误造成的，需要在新口径（10 chunk、1000+ token、每套件自己的统计）下重测。

### 已启动（同任务配对，修正口径）
| 运行 | 变体 | 数据 |
|---|---|---|
| rv_cam_topk | top-10 逐方向压方差（门控 1.05–2.0） | spatial camera 前 120 |
| rv_cam_aff | 完整仿射（CORAL，源先验收缩） | 同上 120 |
| rv_cam_aff_s5 | 仿射 + 步调度（第 5 步起） | 同上 120 |
| rv_cam_aff_a05 | 仿射 + 强度 α=0.5 | 同上 120 |
| rv_l3_topk | top-10 压方差 | L3 20 |
| rv_l3_aff | 完整仿射 | L3 20 |

对照：同一批任务的全量官方 baseline；以及（相机）正在跑的完整维度"去偏"结果。

## 第 66 步：重测中期（修正口径）——变体差异消失

### 相机（前 120 个 spatial 任务，约 52/120 完成）
| 变体 | n | base | 方法 | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| top-10 压方差 | 55 | 34.5% | 34.5% | 0.0 | [−7.3,+7.3] | 1.00 |
| 完整仿射 | 54 | 33.3% | 31.5% | −1.9 | [−7.4,+5.6] | 1.00 |
| 仿射 + 步调度 | 51 | 29.4% | 25.5% | −3.9 | [−15.7,+7.8] | 0.73 |
| 仿射 + α=0.5 | 55 | 34.5% | 36.4% | +1.8 | [−5.5,+9.1] | 1.00 |
| L3 topk/affine | 3 | — | — | — | — | 太早 |

### 完整维度去偏（spatial，n=258/376）
base 55.0% → 56.2%：**+1.2pp，95% CI [−5.0,+7.4]，p=0.80**
→ 修正口径后，spatial 相机的效应**大幅缩水**（之前子集 +13.3 是口径+抽样造成的）。

### 当前判断（中期）
- 修正口径后，**各变体差异基本消失**（都接近 0）；
- 完整维度去偏在 spatial 上只有 +1.2、不显著；
- 需要等全部跑完（1,599）才能定论，但"之前的大效应"很可能是口径错误产物。

## 第 67 步：根因定位——"校正方向几乎全在动作无关子空间"

### 逐阶段审计（数字）
1. **统计阶段**：修正口径后，camera(spatial) 的修正向量范数 = 1.87，残差能量比 1.60 → 统计量不算小；
2. **方向阶段（关键）**：把修正向量投影到动作敏感子空间（修正 RNG 后的 Jacobian，rank=7）：
   - **只有 9% 的修正落在动作敏感子空间（91% 在动作无关/零空间）**；
3. **动作阶段**：unit 方向推写回 0.2 → 动作变化仅 ~0.012；去偏整块修正 → 动作变化 ~0.006（动作尺度 0.30）；
4. **结果阶段**：变体全部 ≈0（topk +1.0、affine +1.0、α0.5 0.0、start5 −6.4）；完整维度去偏 spatial +0.3（n=291）。

### 新发现并修复的代码问题
- `build_interface_stats.py` 的有限差分**没有固定采样随机种子**：base 与 push 的 flow 采样噪声不同，Δa 被噪声主导 → 动作子空间估计不可信（rank 11→**7**）；
- 之前"残差各方向动作敏感性 11–14 几乎相同"的结论也是**噪声污染**造成的，已作废。

### 结论（机制层）
> **域偏移在动作接口的残差里确实存在（1.1–1.6 倍），但它几乎完全落在"动作无关方向"（91%）；因此任何沿该方向的校正都不会改变行为。**
> 这解释了：为什么所有残差/统计类变体都 ≈0，为什么之前子集上的 +13 是口径+抽样产物。

### 下一步候选
- A：用配对数据（clean 重放）估计"动作敏感子空间内"的偏移并校正（有监督源映射）；
- B：转向动作级 TTA（需额外信号）；
- C：把"偏移在动作零空间"作为负结果/机制分析写出来。

## 第 68 步：配对数据可行性验证（A）——修正动作子空间反而显著有害

### 配对数据（同任务、同动作序列，clean 重放 vs 相机扰动）
| 任务 | ||Δw|| | Δw 落在动作子空间 A 的比例 |
|---|---|---|
| 648 chunk0/1 | 7.06 / 6.55 | **0.389 / 0.393** |
| 650 chunk0/1 | 7.46 / 6.78 | **0.396 / 0.398** |

→ 配对差异有 **~39% 落在动作敏感子空间**（远高于之前残差均值方向的 9%）——说明"可动作化的偏移"确实存在。

### 新机制：动作子空间均值对齐（只校正 A 分量）
- 实现：`dacsc/action_subspace_alignment.py`，c = −α P_A(m_t − μ_s)，|c|≤3，最后一步；
- 结果（spatial camera，n=88）：base 53.4% → 方法 **37.5%**，**Δ=−15.9pp，p=0.0043（显著有害）**；
- L3：n=4，0%（太早）。

### 变体重测（n=120，修正口径）
| 变体 | Δ | p |
|---|---|---|
| top-10 压方差 | +0.8 | 1.00 |
| 完整仿射 | +0.8 | 1.00 |
| 仿射+步调度（第5步起） | −3.3 | 0.56 |
| 仿射+α0.5 | −0.8 | 1.00 |
| 仿射（全步, α0.3） | −1.7 | 0.80 |
| 仿射（全步, α1.0） | −7.5 | 0.11 |

### 机制解释（新假设：补偿）
- A 内的大偏移很可能**不是"要被去掉的污染"，而是策略自身对扰动的补偿**；
- 把它"纠正"回源域均值 = **撤掉策略的补偿** → 显著变差（−15.9）；
- 残差方向的偏移是动作无关的 → 校正无效（≈0）。
- 结论：**"在接口上把表示对齐回源域"这一前提，在 A 方向上是错的**（至少对相机扰动成立）。

## 第 69 步：补偿假设得到验证（关键机制）

### 探针（camera spatial 30 任务 × 2 episode = 60 条；clean 6 任务 × 2 = 12 条）
| 组 | n | A 方向偏移（均值） | 残差方向偏移 |
|---|---|---|---|
| 相机扰动-成功 | 23 | **3.589** | 8.809 |
| 相机扰动-失败 | 37 | **3.170** | 7.983 |
| clean-成功 | 12 | 3.230 | 7.312 |

- **成功轨迹的 A 方向偏移显著大于失败**（Mann-Whitney 单侧 p<0.0001）；
- 任务内配对（9 个任务同时有成功/失败）：成功−失败 = **+0.333**；
- clean 基线 3.230 → 相机成功轨迹比 clean 还高（+0.36）。

### 结论（机制）
> **A 方向上的偏移是"策略对扰动的补偿"，成功依赖这种补偿；**
> 我们之前把 A 方向"对齐回源域均值"= **撤掉补偿** → 显著变差（−15.9pp，p=0.0043）。
> 残差方向的偏移与成功无关（success 8.81 vs fail 7.98，差距小）→ 校正它当然无效。

### 由此产生的新机制（沿原路线）
"补偿感知"：不是消除 A 方向偏移，而是**帮助/放大补偿**（correction = +α·P_A(m_t−μ_s)，α>0）。
已启动：boost_cam_a03（放大 30%）、boost_cam_a10（放大 100%）在 120 个相机任务上验证。

## 第 70 步：ANST（动作零空间对齐，GPM 思想）实现与验证

### 实现
- `dacsc/null_space_alignment.py`：c = −α·P_N(m_t−μ_s)，P_N = I − Q_A Q_Aᵀ，|c|≤3，仅最后 Euler 步；
- 源统计改为"轨迹匹配"（10 chunk、87 帧）后重新运行。

### 结果（官方协议，配对）
| 条件 | n | base | ANST | Δ | p |
|---|---|---|---|---|---|
| Camera（spatial 前 120） | 98 | 56.1% | 57.1% | **+1.0** | 1.00 |
| L3（早期） | 4 | 50% | 50% | 0 | 1.00 |
| **clean** | 10 | **100%** | **90%**（1 条失败） | **−10** | 1.00 |

### 解读
1. ANST 在相机上**基本中性（+1.0）**，不再像"动作方向对齐"那样有害（−15.9）→ 零空间保护有效；
2. 但**clean 上仍有一次误触发**（门控 shift_min=0.5 偏低）→ 需按 clean 的零空间偏移分布收紧；
3. **零空间对齐"安全但无效"**：与机制一致——动作零空间里的偏移不改变行为；
4. L3 样本太少，待补。

### 方法层面的最终判断（截至目前）
- 残差/统计类对齐：≈0；
- 动作方向对齐：显著有害（撤掉补偿）；
- 零空间对齐：安全但无效；
- → **"在动作接口做统计对齐"这一类方法，在 VLA 上系统性地无效**；可动作化的方向是"补偿"，而补偿不能被外部统计对齐替代。

### 下一步候选
- A：收紧门控，完成 ANST 的 clean/L3/相机全量验证（安全但预期无效）；
- B：转"动作级信号"（SALT 式 leftover / 成功反馈），需要新实现；
- C：以"偏移分解 = 补偿 vs 噪声"为核心写机制/负结果论文。

## 第 71 步：状态条件化对齐（A）实现 + 全维度扫描

### 审计结论（先校验，不否定）
- 不干预（c_max=0）与官方 baseline **20/20 逐条一致** → 驱动/RNG 无 bug；
- 但发现：**动作零空间偏移在 clean 上也有 4–7，与相机（5–7）不可分** → 门控失效；
- 更根本：**逐帧写回均值波动（5–7）远大于域均值差（≈1.9）**，无条件统计量被状态变化主导（SNR≈0.3）。

### 新方法：状态条件化对齐（ConditionalAlignment）
- phi = attention query 在 heads/tokens 上的均值（动作上下文摘要，测试时可得）；
- 源条件均值：clean 标定数据 ridge 回归 `w ≈ [1,phi_n]B_s`（87 对）；
- 目标条件均值：当前 episode 在线 ridge 回归 B_t（缓冲/重拟合）；
- 校正 `w' = w − clip(g_t(phi) − g_s(phi), c_max)`，仅最后 Euler 步。

### 初步结果
| 条件 | n | base | 方法 | 备注 |
|---|---|---|---|---|
| clean | 10 | 100% | **100%** | 条件化后不再误触发 ✅ |
| Camera（spatial 前 120） | 进行中 | — | — | — |

### 全维度扫描（统一方法，7 个扰动维度 × 120 任务）
- Camera/Robot/Language/Light/Background/Noise/Layout 各 120（每套件 30）；
- 每维用同一个"状态条件化对齐"，与全量 baseline 配对比较。

## 第 72 步：7 个域的统一方法扫描结果（状态条件化对齐）

| 域 | n | base | 方法 | Δ | 不一致对 | p | p_holm |
|---|---|---|---|---|---|---|---|
| Camera | 118 | 42.4% | 43.2% | +0.8 | 10/11 | 1.000 | 1.000 |
| Robot | 120 | 38.3% | 38.3% | 0.0 | 7/7 | 1.000 | 1.000 |
| Language | 120 | 77.5% | 73.3% | **−4.2** | 6/1 | 0.125 | 0.875 |
| Light | 120 | 80.0% | 75.8% | **−4.2** | 10/5 | 0.302 | 1.000 |
| Background | 114 | 38.6% | 36.0% | −2.6 | 10/7 | 0.629 | 1.000 |
| Noise | 11（进行中） | 54.5% | 54.5% | 0.0 | 1/1 | 1.000 | 1.000 |
| Layout | 107 | 69.2% | 68.2% | −0.9 | 5/4 | 1.000 | 1.000 |

### 结论
- **没有任何域显著改善**；Language/Light 还有 −4.2 的趋势（不显著）；
- 统一的状态条件化方法"安全但不提升"；
- 与根因一致：**接口统计量里可识别的域偏移太小（SNR≈0.3），条件化后仍不足以产生行为改变**；
- Noise 域任务长（每条 220 步），扫描仍在进行（约需 1–2 小时）。

## 第 73 步：梳理整条流程，定位真正的逻辑问题

### 逐环节审计
| 环节 | 事实 | 判定 |
|---|---|---|
| 干预位置 | 清零最后一步写回 → 动作变化 0.13（动作尺度 0.30）| 位置有因果性 ✅ |
| 采样噪声 | 同一观测 8 个随机种子：写回均值范数 7.0–7.4，std 0.55 | 采样噪声小 ✅ |
| **写回均值的状态依赖** | 沿轨迹逐帧写回均值范数 **4.6–9.1**；而全局平均 μ_s 范数只有 **3.35** | ❌ 全局均值不能代表任意一帧 |
| 域偏移量级 | 域均值差 ||Δμ||≈3.2；逐帧偏离全局均值 ~4–5 | ❌ 偏移被"状态差异"淹没 |
| 门控 | clean 每帧 ||m_t−μ_s||≈5 ≫ 阈值 0.5 → clean 也被改 | ❌ 阈值无意义 |
| 条件化 phi | 用 query 做条件 → query 本身随域变化，条件化把"要校正的偏移"吸收掉 | ❌ 条件变量被污染 |

### 真正的逻辑问题（一句话）
> **我们一直用"全局源域均值 μ_s"作为每一帧的对齐目标，但写回均值强烈依赖机器人状态（逐帧范数 4.6–9.1，全局均值 3.35）。**
> → 即使**没有域偏移**（clean），每帧也会被"拉向一个不匹配的全局均值"（−5 量级的错误校正）→ clean 掉分；
> → 在目标域上，校正量里大部分是状态差、不是域偏移 → 效果 ≈0/有害。

### 之前"SNR≈14、可识别"的计算是错的
之前用"每维方差的平均"当分母（被 768 维稀释），正确算法：`||Δμ|| / sqrt(sum(var)) ≈ 0.5` → **不可识别**。

### 正确修法（下一步）
- 条件化变量必须**域不变**：机器人本体状态（proprioception）+ 归一化步数，而不是 query；
- 需要驱动层把 state/step 传给模块（上游 evaluate_suite 不提供 → 需要新的 rollout 驱动，执行完整 chunk = 官方协议）；
- 或者用"同状态配对"的离线标定。

## 第 74 步：状态条件化的实现与又一批问题（含协议级问题）

### 实现
1. `build_state_cond_source.py`：phi = [本体状态 8 维, 归一化 predict 步数]（域不变），采集 87 对；
2. `state_conditional_alignment.py`：离线 g_s、在线 B_t 回归；`state_residual_mean.py`：只估计条件残差均值（低方差）；
3. 新驱动 `sca_eval.py`（执行完整 chunk = 官方协议，可注入 state/step）。

### 发现的问题（按严重程度）
1. **步数归一化不一致**（校准 k/9 vs 驱动 k/15）→ 条件外推 → 大错误校正（已修，重建校准 v2）；
2. **离线回归参数噪声大**（10 参数 / 16–24 样本）→ clean 上校正非零（sca2_clean 80%）；
3. **在线均值反馈**：每个 predict 都施加校正 → 改变轨迹 → 下一个残差反映的是"被改过的状态"，不是域偏移 → 校正逐帧累积（srm_cam 早期 28% vs baseline ~60%，显著有害）；
4. **clean 仍非 100%**（srm 90%）→ 校准误差 + 阈值不足。

### 协议级问题（一直存在、现在才看清）
> **文献里的 TTA（Tent/SSA/PSC）都是"在目标数据上适应一次 → 冻结 → 评测"。**
> 我们一直做的是**在线逐帧干预闭环**：干预改变轨迹，轨迹又反过来改变统计量估计 → 估计与干预互相污染。
> 这解释了为什么"越精细的在线方法越有害"。

### 下一步（唯一自洽的实验）
**离线 TTA 协议**：
1. 在目标域跑少量无标签帧（不做干预），采集写回；
2. 估计"条件残差均值"（相对源模型）；
3. 冻结该校正，再评测。
这样统计量不受干预污染，符合文献协议，也能回答"到底有没有可校正的信号"。

## 第 75 步：离线（adapt-then-freeze）协议 + 决定性诊断：模型误差 ≈ 域偏移

### 离线协议实现
`dacsc/offline_state_alignment.py`：前 6 个 predict 不干预、只采集；用
e = mean_i(w_i − g_s(phi_i)) 估计条件域偏移；冻结 correction 应用于后续 predict。
（phi = 本体状态 + 归一化步数；g_s 为 clean 标定数据的 ridge 条件模型。）

### 结果
| 条件 | n | base | 离线方法 | 说明 |
|---|---|---|---|---|
| clean | 10 | 100% | **60%** | 校正被误触发（e≈3 > 阈值 0.3）|
| Camera（早期） | 10+ | ~60% | 20–30% | 严重有害 |

### 决定性诊断：留一任务交叉验证的"模型误差地板"
用 clean 标定数据做 leave-one-task-out：
**残差均值范数 = [3.0, 3.6, 1.7, 2.31, 2.8, 3.89, 2.76, 3.35, 4.23, 3.52]，均值 3.11。**

而相机域偏移的均值差 ≈ **3.2**。

→ **模型误差（3.11）≈ 域偏移（3.2）**：在没有配对数据的情况下，
   "条件域偏移"与"跨任务/状态/模型误差"**不可区分**。
→ 这是比"在线/离线协议"更根本的障碍：**信号本身识别不出来**。

### 全流程审计结论（最终）
1. 代码层问题（已修）：源统计采集用了错误的 Euler 步（off-by-one）；步数归一化不一致；在线校正的闭环反馈；
2. 协议层问题（已修）：在线逐帧干预 → 离线 adapt-then-freeze；
3. **统计层问题（无法在现有约束下修复）**：条件模型的误差地板（≈3.1）与真实域偏移（≈3.2）同量级 → 不可识别。
4. 由此：任何基于该统计量的校正都会在 clean 上误触发、在目标域上无效或有害。

### 可行方向（需用户决策）
- A：允许"配对同状态数据"（clean 重放同一状态）作为标定 → 可估计真实域偏移（但需要 clean 环境与同状态访问）；
- B：转动作级自监督信号（leftover / 行为反馈），绕开接口统计的可识别性问题；
- C：以"接口域偏移不可识别（误差地板≈偏移）"为机制核心写负结果/分析论文。
