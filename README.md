# DACSC-Robot

研究紧凑机器人策略在未见外观条件下的闭环泛化。当前使用 **CoRP adapted to RoboCasa**，
先验证能完成任务的无干预基线，再检验视觉表示增广。方法效果尚未成立。

**当前 V10 执行顺序与冻结配置：[V10_PROTOCOL.md](docs/V10_PROTOCOL.md)。**
研究边界和总体比较原则见 [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)；
机器可读事实源是 [v10_atomic_target_pilot.json](configs/v10_atomic_target_pilot.json)。
研究问题、数据选型、官方泛化成绩和当前 pilot 证据的总览见
[RESEARCH_PLAN_AND_DATASETS.md](docs/RESEARCH_PLAN_AND_DATASETS.md)。
仓库只保留当前 CoRP-RoboCasa 执行链；历史实验文档和退役实现不再保留在工作树中。

## 先复用上游

| 环节 | 实际调用 |
| --- | --- |
| CoRP 模型与严格加载 | 官方 `corp` 模块、`corp.checkpoint.load_policy` |
| 官方模型 smoke / 标准 LIBERO rollout | `repos/CoRP/scripts/smoke_test.py` / `evaluate_libero.py` |
| RoboCasa 视频与动作读取 | 官方 DP `LerobotDataset`；可选缓存复用官方 RoboCasa `CachedLeRobotSingleDataset` |
| RoboCasa 重放与 task horizon | 官方 `playback_dataset.reset_to` / `get_task_horizon` |
| 图像处理 | Torchvision functional / Pillow bilinear resize |
| 像素训练增广 | Torchvision `ColorJitter(.4, .4, .4, hue=0)`，与固定评测扰动独立 |
| 参数 EMA | PyTorch `AveragedModel` / `get_ema_multi_avg_fn` |
| 输出互斥 | `filelock.FileLock` |

`upstream.lock.json` 固定源码 revision 和被调用入口文件的 SHA256；运行前检查，
不自动打补丁，不静默降级。仅覆盖清单中的文件，不代替整个仿真环境的版本验证。

默认训练缓存只覆盖配置中列出的任务；`CACHE_VIDEO=0` 改用官方 DP 流式读取。
缓存使用官方 `pyav` 全帧路径，在线读取使用官方 `opencv` 时间戳路径；
`readonly_validation video` 会用真实视频逐像素比较两者，不一致直接失败，不把
“更快”当成“输入等价”。

通道方向先做只读机制审计，见[上游映射表](docs/CHANNEL_INTERVENTION_UPSTREAM_MAPPING.md)。
`dacsc/corp_ffn_capture.py`只观察CoRP原block与GELU输出，不接入训练/推理主入口、不替换attention或增加门控。
其可选分析依赖单独固定于`analysis_upstream.lock.json`；原Registers logger和detector由已核验的外部源文件直接加载。

## 本地验证，不启动训练

GPU环境使用独立的CoRP/RoboCasa依赖锁和离线wheelhouse，避免每次现场解析冲突：
见[GPU环境复现说明](environment/README.md)。

在已有依赖环境中，从仓库根目录运行：

```bash
.venv/bin/python -m dacsc.upstream
.venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=repos/CoRP .venv/bin/python repos/CoRP/scripts/smoke_test.py --device cpu
```

第三条直接使用官方 demo，检查真实发布权重 checksum、严格加载、参数量和动作生成。
`tests/` 保留数据/模型接口、mask、共享权重 EMA、断点恢复、配对和拒绝错误输入的回归。
其中叶模块假环境仅用于本地契约验证，不能当作真实仿真成功率。

新增FFN回归要求预先准备`analysis_upstream.lock.json`中Registers固定revision的四个`shared/`源文件。
可用作者checkout并设置`DACSC_REGISTERS_REPO=/absolute/path/to/test-time-registers`；
本地默认查找`.cache/analysis_sources/registers/<revision>/`。缺失或hash不符直接失败，测试不联网、不静默跳过。
它还使用作者源文件的NumPy/Pillow/Matplotlib/tqdm依赖；这不是主训练新增的必需依赖，不修改官方模型或共享环境。
当前完整本地验证记录统一写入`audit/current_validation.json`；历史快照不再作为入口。

环境安装遵循 [CoRP 官方说明](https://github.com/corp-policy/CoRP) 和
[RoboCasa 官方说明](https://robocasa.ai/docs/build/html/introduction/installation.html)。
官方 CoRP LIBERO 环境和 RoboCasa 环境的 NumPy/MuJoCo 等约束不同，分别管理；
不得使用“一键混装”环境。

## 当前入口

| 工作 | 入口 |
| --- | --- |
| 真实视频/官方CUDA demo/只读FFN验收 | `python -m dacsc.readonly_validation`，见[只读验收操作说明](docs/READONLY_SERVER_VALIDATION.md) |
| 只准备官方MuJoCo重放命令 | 同上`replay-plan`；不执行仿真 |
| CoRP-RoboCasa 训练适配 | `scripts/train_corp_robocasa.py`；必须显式传 `--steps` |
| V10 frozen baseline | `scripts/run_v10_pilot.sh`；默认 5 tasks、30k、batch96、BF16 |
| held-out 离线 MSE | `scripts/eval_corp_robocasa.py` |
| RoboCasa 单任务 SR | `scripts/eval_corp_robocasa_sr.py` |
| 多任务 source/val SR 与严格聚合 | `scripts/run_corp_sr_split.sh` |
| official target SR | `scripts/run_v10_target_sr.sh` |
| 已完成评测的严格配对与跨训练seed统计 | `scripts/compare_corp_evaluations.py`，仅文件分析 |
| LIBERO-Plus 外观条件扩展 | `scripts/eval_corp_libero_plus.py`；直接委托官方 `evaluate_suite` |

RoboCasa 数据根目录使用 `--data-root` 或 `DACSC_DATA_ROOT`，其下应有
`v1.0/pretrain/atomic/<Task>/<date>/lerobot`。CoRP 源码通过 `CORP_REPO` 或 `repos/CoRP` 定位。
所有入口 `--help` 可在不启动训练的情况下查看参数。

当前 checkpoint 为 `corp_robocasa_v8`，训练状态版本4，SR schema为`corp_robocasa_sr_v11`，
action指标仍为`corp_action_mse_v1`。旧 checkpoint/SR 文件不转换、不兼容。
test 默认封存，只有最终评估显式启用。真实 RoboCasa/CUDA 的复核仍待服务器环境验证。

CoRP 的 DINOv2/T5/resampler/FiLM/DiT 共享层复用官方 LIBERO 权重；RoboCasa 的
9D state projection 和 12D action/velocity projection 重新初始化并训练。LIBERO motion
min-max 与 RoboCasa action identity 不同，不能复制前7维动作头。
`initialization_policy` 写入 checkpoint、result 和 SR training origin；旧初始化结果
不能与这一协议混表。

CoRP-RoboCasa 默认启用论文 Eq.7 的 task-classification auxiliary loss：
`--task-aux-lambda 0.01 --task-aux-depth 1`。任务头只读取 `V×K` 个视觉前缀 token，
逐 token 做 LayerNorm、取均值并经过跨深度共享 MLP；推理不读取 task label/logits。
官方 CoRP 发布库只有推理，因此该 trainer/head 是公式复现而非官方训练代码；
论文未公布 MLP 隐藏宽度，当前固定为 256 并写入 checkpoint、result 和 SR 训练身份。
`--task-aux-lambda 0` 是显式无辅助损失 ablation，必须与新默认分开报告。

指标计算直接用PyTorch、NumPy和SciPy，不创建加权总分。micro/task-macro MSE、五组/12维误差及
逐episode support分开报告；跨seed的SD使用ddof=1、区间调用SciPy Student-t/SEM。
单seed或零观察变异不报假精确CI；统计工具不启动模型或训练。定义与manifest例子见当前协议末节。
指标改进的本地证据见 `audit/current_validation.json`。

## 像素训练增广基线

`--train-augmentation color_jitter` 直接使用 Torchvision 随机
`ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=0)`。
强度参考 [timm 通用 ImageNet 训练配方](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms_factory.py)，
**不是 CoRP 官方训练增广的复现**，也不引入分类任务的 crop/flip。

只在 source train 的 resize 后、ImageNet normalize 前生效；每个观测独立抽样，三视角共用一次抽样。
使用独立 RNG 并随完整训练状态恢复，不消耗 flow 的全局 RNG。验证、测试、推理和 normalizer 拟合不使用随机增广。
`--eval-shift color_jitter` 仍是固定强度、固定顺序的评测 stress，两者不能混称。
像素增广默认关闭；V10 pilot 通过 `TRAIN_AUGMENTATION=color_jitter` 选择该对照。
输出目录仍由 `--out-dir` 独占，不能覆盖已有 run。

## 本地交付

提交并保持工作区干净后，`bash scripts/pack_for_server.sh /path/to/new/output` 使用标准
`git archive` 导出完整已提交代码（包括根目录manifest、测试和协议），拒绝覆盖已有包。
它不上传、不部署，不再混入本机未跟踪的第三方源码/权重；上游依赖按锁定revision准备。

只读验收交付使用`python scripts/pack_readonly_validation.py --registers-source SOURCE --out NEW_DIRECTORY`：
导出标准Git bundle（保留commit身份）、hash固定的作者分析源文件和验收说明。拒绝脏树和覆盖，不连服务器、不安装依赖、不运行模型。
本地验证结果见`audit/current_validation.json`；真实视频、CUDA/BF16和MuJoCo仍需在目标服务器执行。
