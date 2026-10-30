# 当前实验协议：CoRP-RoboCasa / appearance shift

2026-10-29。本文件是唯一当前执行协议，替代旧 notes/71、72。

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

CPU/CUDA RNG、optimizer、scheduler、EMA、donor、importance、DataLoader位置完整保存；
checkpoint原子替换，保存独立于评测周期。精确续训目前要求 workers=0、同数据路径/metadata/预算/架构。
不能把一个2k预算改写成10k后称为同预算的完整 resume；新的预算需另立实验。
已完成预算的 checkpoint 不再走训练 resume；显式使用独立 evaluator，避免将保存点之前的 metrics 当作最终评测。
模型格式为 `corp_robocasa_v2`，保存 raw/EMA/实际完成步数、初始化权重 digest、图像/上游契约和预算。
旧格式不迁移。test 默认不执行。

## 评测与判定

主指标：闭环 success rate；辅指标：masked per-dimension/per-task MSE。
所有方法同一初始化、数据顺序、预算、训练 seeds、评测帧及仿真 scene manifests。
SR source 为训练 styles 下新生成的冻结场景，不是数据中的 `%10==9` episodes。
scene manifest 保存模型 XML、仿真状态、metadata 和 RNG；之后调用官方重放，再核对 state/style/layout。

聚合必须检查任务清单、episode数、manifest digest、checkpoint SHA256、条件、推理预算、source/val范围和完整性。
schema `corp_robocasa_sr_v3` 不接受旧 JSON。写入前锁文件，日志禁止截断；中断/失败不能产生“全完成”的汇总。
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
