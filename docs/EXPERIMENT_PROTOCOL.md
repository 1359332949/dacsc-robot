# 当前实验协议：CoRP-RoboCasa / appearance shift

2026-10-30。本文件是唯一当前执行协议，替代旧 notes/71、72。

## 研究问题和证据边界

目标：**紧凑机器人策略在未见外观变化下保持任务成功率，同时保持正常条件性能。**
基线固定为 CoRP-RoboCasa。先建立有效 clean 闭环能力，再在同一模型、任务、初始状态和
策略噪声下测 appearance shift，最后评估标准图像增广、特征风格增广以及动作重要性保护这一项增量。
channel/token 是待检验的干预选择；当前代码的 token statistics 仍是候选方法。

当前不能写成结论的内容：

- 旧 CoRP-RoboCasa 离线 MSE 无明显 source/val gap，不能推出闭环成功率无 gap。
- CloseFridge 的 0/10 只能描述该 checkpoint 在该协议下失败，不能自动归因为训练不足。
- 旧官方 CoRP/LIBERO-Plus 的 14 个任务 paired 结果是动机证据，不能替代 RoboCasa 模型的验证。
- CoRP 论文 RoboTwin Randomized 训练包含随机化示范，不能用两组 success 差证明零样本外观泛化。
- legacy DP 的 u/v 分析、DDPM gate 和 custom flow 结果不是 CoRP token 方法结果；梯度分数不等于因果证明。

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

| 环节 | 来源 / 接口 | 边界 |
| --- | --- | --- |
| 模型/加载 | [官方 CoRP](https://github.com/corp-policy/CoRP/tree/52445740aa9517ae396cc8fb14bae8fe5922f118) `RepresentationExtractor`、`DiTActionGenerator`、`load_policy` | 保留 DINOv2/T5/K48/1-layer DiT；仅适配3 views、9D state、12D action、H10 |
| 数据 | [官方 RoboCasa DP reader](https://github.com/robocasa-benchmark/diffusion_policy/blob/41212698b6f481ed92a55f0d5f1778ec56bea417/diffusion_policy/dataset/lerobot_dataset.py) | 不重写视频读取、动作键拼接和时间索引；在外层过滤原生标签/补 mask |
| 重放 | [官方 reset_to](https://github.com/robocasa/robocasa/blob/dc73a8ce18e4b8947e8df2ae2958245ca823f3b4/robocasa/scripts/dataset_scripts/playback_dataset.py) | 不调用 label_subtasks 中的修改副本，不修补 XML、不偷偷 retry |
| 布局/时限 | 官方 `LayoutType.TRAIN`、`unpack_layout_ids`、`get_task_horizon` | layout 11–60；时限按任务 registry，删除固定600步和旧 override |
| 图像 | Pillow bilinear + [Torchvision functional](https://docs.pytorch.org/vision/main/transforms.html) | 三条路径使用同一图像契约；保留 LIBERO 官方观测旋转，不给 RoboCasa添加该旋转 |
| EMA | [PyTorch AveragedModel / get_ema_multi_avg_fn](https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema-for-training) | 不重写 EMA 算术；适配参数名/共享权重和非持久 donor buffer |
| 互斥 | [filelock](https://py-filelock.readthedocs.io/en/latest/how-to.html) | 输出所有权和日志创建位于同一锁内 |

共享权重从**已训练的 LIBERO CoRP policy checkpoint**迁移，不只是 DINOv2 视觉预训练。
先用官方 loader 完整加载源模型；仅声明的 state/action 接口和新 camera query/embedding 允许初始化。
其他 missing/unexpected/shape mismatch 报错。报告中的单位是 state-dict tensor，需区分共享参数别名。

必须使用名称 **CoRP adapted to RoboCasa / our training adaptation**：官方公开仓库只有推理，
我们的训练循环并非官方训练代码；当前未实现论文 task-classification auxiliary loss（lambda=.01）。
保留这个差异并公开记录，不凭猜测重建一个 head 后称为忠实复现。
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

## donor 与动作保护契约

训练期 statistics mixing：eval 返回原视觉特征，不需要 style、task 或动作标签计算 donor。

- `random_stats`：`torch.randperm(B)`，允许 self-pair/同 style；沿用 TFS-ViT 样本置换思想。
- `cross_style_stats`：每个 recipient 在 batch 不同原生 style 的样本中均匀抽一个。
  直接用 `torch.multinomial`；recipient 之间可复用 donor。缺标签或只一个 style 立即报错，无 fallback。
- `cross_style_stats_action_aware`：donor 规则不变，动作重要性仅调整调制强度。
- 一对样本共用 donor，左对左、腕对腕、右对右，不跨相机；官方 view-major 输出先转换再逆转换。
- **不限制同 task/layout**，不能称为“同任务、只变外观”的受控训练干预。
- mean/std detach；独立 donor RNG 由 run seed 初始化并完整恢复，不消耗 flow RNG。
- action-aware 为先前训练梯度构造的 patch 位置 EMA/min-max 权重，跨样本/视角统计；不是对象追踪或因果重要性。

来源：[TFS-ViT FeatureStylization](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/blob/main/domainbed/pretrained_models/DeiT_models/models.py)。
当前固定 alpha、单个 pre-resampler 插入点、多视角及异 style 规则都属于适配，
不是完整 TFS-ViT 的 Beta mixing、多层/token 策略。仅凭该模块不能把完整 TFS-ViT 填入比较表。
`sample_pair_v2` donor 契约不匹配的 checkpoint 拒绝加载。
CoRP checkpoint只接受上述三种训练期mixing模式，拒绝已退役的normalize分支或未声明的模式。
SR命令中的style范围必须逐段升序、唯一且落在原生pretrain ID 11–60内，不能静默跳过反向范围。

## 统一预算与保存

当前统一 runner 使用 batch30、lr5e-5、pilot EMA .99、grad clip1、BF16、单帧/视角、Euler10。
原训练适配中的 AdamW betas=(.9,.95)、weight_decay=1e-4、eps=1e-8和前10%预算的线性warmup
保持数值不变，完整写入结果的 `optimization`。这些细节是项目配方，不是官方CoRP公布的训练配置。
`STEPS` 必须显式声明；默认只跑 none。正式预算、EMA、任务集、训练 seeds 必须在比较前冻结，
不能根据 test 结果反复调整。51/200步只用于工程检查，2k/10k也不自动构成学会任务的证据。

CPU/CUDA RNG、optimizer、scheduler、EMA、donor、importance、像素增广RNG、DataLoader位置完整保存；
checkpoint原子替换，保存独立于评测周期。精确续训目前要求 workers=0、同数据路径/metadata/预算/架构。
不能把一个2k预算改写成10k后称为同预算的完整 resume；新的预算需另立实验。
已完成预算的 checkpoint 不再走训练 resume；显式使用独立 evaluator，避免将保存点之前的 metrics 当作最终评测。
模型格式为 `corp_robocasa_v3`、训练状态版本3，保存 raw/EMA/实际完成步数、初始化权重 digest、图像/上游契约和预算。
旧格式不迁移。test 默认不执行。

## 评测与判定

主指标：闭环 success rate；辅指标：masked per-dimension/per-task MSE。
所有方法同一初始化、数据顺序、预算、训练 seeds、评测帧及仿真 scene manifests。
SR source 为训练 styles 下新生成的冻结场景，不是数据中的 `%10==9` episodes。
scene manifest 保存模型 XML、仿真状态、metadata 和 RNG；之后调用官方重放，再核对 state/style/layout。

聚合必须检查任务清单、episode数、manifest digest、checkpoint SHA256、条件、推理预算、source/val范围和完整性。
schema `corp_robocasa_sr_v5` 保存实际训练seed、预算、优化配方、像素/特征增广、初始化及训练数据digest，不接受旧JSON冒充完整配对记录。
写入前锁文件，日志禁止截断；中断/失败不能产生“全完成”的汇总。
旧进程不遵守新锁，以后部署前须确认旧 writer 退出，再使用新目录；本轮未远程部署。

实验顺序固定：

1. 官方 demo 和本地回归通过，再验证真实 RoboCasa/CUDA 数据与重放。
2. 训练 no-intervention baseline，在冻结的 source/val 场景验证 clean 闭环能力。
3. 同 checkpoint 的 clean/appearance 配对评测，确认问题及任务语义仍成立。
4. 比较标准图像增广、忠实的上游特征增广及动作保护增量；必须有随机/无保护对照。
5. 只有验证集证据支持才扩预算；所有选择完成后再显式最终 test。

报告训练seed和episode/task support，使用适当的配对差值/置信区间（可复用 SciPy），
不能把相邻帧当独立episode，也不能仅以5个百分点或一次51步结果下结论。
如果 baseline 未学会，或 ours 不能优于合理对照，记录失败范围和不确定性，停止扩大方法；不据此宣称普遍不可能。

## 官方 demo 与本地测试分工

模型：直接运行 CoRP `scripts/smoke_test.py`，不再维护重复模型 demo。
标准 LIBERO：直接运行其 `scripts/evaluate_libero.py`。
RoboCasa GT replay：直接使用其 `scripts/dataset_scripts/playback_dataset.py --dataset ... --use_actions`。
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

SR比较manifest示例（路径相对manifest目录；数据必须已按SR v4生成）：

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
