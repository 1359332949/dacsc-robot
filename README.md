# DACSC-Robot

研究紧凑机器人策略在未见外观条件下的闭环泛化。当前使用 **CoRP adapted to RoboCasa**，
先验证能完成任务的无干预基线，再检验视觉表示增广。方法效果尚未成立。

**唯一当前执行协议：[EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。**
`notes/` 是历史记录；`docs/legacy/`、legacy DP/channel-gate 脚本不能作为 CoRP 主实验入口。

## 先复用上游

| 环节 | 实际调用 |
| --- | --- |
| CoRP 模型与严格加载 | 官方 `corp` 模块、`corp.checkpoint.load_policy` |
| 官方模型 smoke / 标准 LIBERO rollout | `repos/CoRP/scripts/smoke_test.py` / `evaluate_libero.py` |
| RoboCasa 视频与动作读取 | 官方 DP `LerobotDataset`，已移除旧 CLIP cache 私有补丁 |
| RoboCasa 重放与 task horizon | 官方 `playback_dataset.reset_to` / `get_task_horizon` |
| 图像处理 | Torchvision functional / Pillow bilinear resize |
| 像素训练增广 | Torchvision `ColorJitter(.4, .4, .4, hue=0)`，与固定评测扰动独立 |
| 参数 EMA | PyTorch `AveragedModel` / `get_ema_multi_avg_fn` |
| 输出互斥 | `filelock.FileLock` |

`upstream.lock.json` 固定源码 revision 和被调用入口文件的 SHA256；运行前检查，
不自动打补丁，不静默降级。仅覆盖清单中的文件，不代替整个仿真环境的版本验证。
token statistics、不同 style donor 和 action-aware 是本工程适配，**不是完整 TFS-ViT 复现**。

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
`tests/` 保留数据/模型接口、mask、共享权重 EMA、donor、断点恢复、配对和拒绝错误输入的回归。
其中叶模块假环境仅用于本地契约验证，不能当作真实仿真成功率。

新增FFN回归要求预先准备`analysis_upstream.lock.json`中Registers固定revision的四个`shared/`源文件。
可用作者checkout并设置`DACSC_REGISTERS_REPO=/absolute/path/to/test-time-registers`；
本地默认查找`.cache/analysis_sources/registers/<revision>/`。缺失或hash不符直接失败，测试不联网、不静默跳过。
它还使用作者源文件的NumPy/Pillow/Matplotlib/tqdm依赖；这不是主训练新增的必需依赖，不修改官方模型或共享环境。
只读采集的本地证据见`audit/corp_ffn_capture_local_validation.json`；此前86项记录仍对应`7efc94a`快照。

环境安装遵循 [CoRP 官方说明](https://github.com/corp-policy/CoRP) 和
[RoboCasa 官方说明](https://robocasa.ai/docs/build/html/introduction/installation.html)。
官方 CoRP LIBERO 环境和 RoboCasa 环境的 NumPy/MuJoCo 等约束不同，分别管理；
不要使用已删除的“一键混装”脚本。`audit/corp_pipeline_local_validation.json` 保留a147242底座的历史验证，
指标/统计改动的历史验证见 `audit/standard_metrics_local_validation.json`；本轮训练增广验证及源码hash见
`audit/pixel_augmentation_local_validation.json`。
安装/下载官方 demo 的依赖、模型和仿真资产属于准备步骤；本轮未运行任何安装仿真环境或训练任务。

## 当前入口

| 工作 | 入口 |
| --- | --- |
| 真实视频/官方CUDA demo/只读FFN验收 | `python -m dacsc.readonly_validation`，见[只读验收操作说明](docs/READONLY_SERVER_VALIDATION.md) |
| 只准备官方MuJoCo重放命令 | 同上`replay-plan`；不执行仿真 |
| CoRP-RoboCasa 训练适配 | `scripts/train_corp_robocasa.py`；必须显式传 `--steps` |
| 顺序运行统一预算 | `scripts/run_corp_experiment.sh`；默认只有 `none`，必须指定 `STEPS`、新 `OUT` |
| held-out 离线 MSE | `scripts/eval_corp_robocasa.py` |
| RoboCasa 单任务 SR | `scripts/eval_corp_robocasa_sr.py` |
| 多任务 source/val SR 与严格聚合 | `scripts/run_corp_sr_split.sh` |
| 已完成评测的严格配对与跨训练seed统计 | `scripts/compare_corp_evaluations.py`，仅文件分析 |
| LIBERO-Plus 外观条件扩展 | `scripts/eval_corp_libero_plus.py`；直接委托官方 `evaluate_suite` |

RoboCasa 数据根目录使用 `--data-root` 或 `DACSC_DATA_ROOT`，其下应有
`v1.0/pretrain/atomic/<Task>/<date>/lerobot`。CoRP 源码通过 `CORP_REPO` 或 `repos/CoRP` 定位。
所有入口 `--help` 可在不启动训练的情况下查看参数。

当前 checkpoint 为 `corp_robocasa_v5`，SR schema 为 `corp_robocasa_sr_v7`，action指标仍为 `corp_action_mse_v1`。
旧 loss、图像变换、donor 或评测协议的文件不转换、不兼容、不混入新对比。
test 默认封存，只有最终评估显式启用。真实 RoboCasa/CUDA 的复核仍待服务器环境验证。

官方 7D 动作头的前 7 维与 RoboCasa 的 eef position/rotation/gripper 严格对应，
训练初始化会复用这部分权重，新增 base motion/control mode 5 维从零学习。
`initialization_policy` 写入 checkpoint、result 和 SR training origin；v4/v6 旧结果
不能与这一初始化协议混表。

CoRP-RoboCasa 默认启用论文 Eq.7 的 task-classification auxiliary loss：
`--task-aux-lambda 0.01 --task-aux-depth 1`。任务头只读取 `V×K` 个视觉前缀 token，
逐 token 做 LayerNorm、取均值并经过跨深度共享 MLP；推理不读取 task label/logits。
官方 CoRP 发布库只有推理，因此该 trainer/head 是公式复现而非官方训练代码；
论文未公布 MLP 隐藏宽度，当前固定为 256 并写入 checkpoint、result 和 SR 训练身份。
`--task-aux-lambda 0` 是显式无辅助损失 ablation，必须与新默认分开报告。

指标计算直接用PyTorch、NumPy和SciPy，不创建加权总分。micro/task-macro MSE、五组/12维误差及
逐episode support分开报告；跨seed的SD使用ddof=1、区间调用SciPy Student-t/SEM。
单seed或零观察变异不报假精确CI；统计工具不启动模型或训练。定义与manifest例子见当前协议末节。
指标改进的本地证据见 `audit/standard_metrics_local_validation.json`。

## 像素训练增广基线

`--train-augmentation color_jitter --token-style-mode none` 直接使用 Torchvision 随机
`ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=0)`。
强度参考 [timm 通用 ImageNet 训练配方](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms_factory.py)，
**不是 CoRP 官方训练增广的复现**，也不引入分类任务的 crop/flip。

只在 source train 的 resize 后、ImageNet normalize 前生效；每个观测独立抽样，三视角共用一次抽样。
使用独立 RNG 并随完整训练状态恢复，不消耗 flow 的全局 RNG。验证、测试、推理和 normalizer 拟合不使用随机增广。
`--eval-shift color_jitter` 仍是固定强度、固定顺序的评测 stress，两者不能混称。
当前像素基线与 token-style 互斥，默认都关闭；runner 通过 `TRAIN_AUGMENTATION=color_jitter MODES=none`
选择像素基线，输出路径为 `$OUT/$TRAIN_AUGMENTATION/$MODE/seed_$SEED`。本轮没有执行该训练入口。

## 本地交付

提交并保持工作区干净后，`bash scripts/pack_for_server.sh /path/to/new/output` 使用标准
`git archive` 导出完整已提交代码（包括根目录manifest、测试和协议），拒绝覆盖已有包。
它不上传、不部署，不再混入本机未跟踪的第三方源码/权重；上游依赖按锁定revision准备。

只读验收交付使用`python scripts/pack_readonly_validation.py --registers-source SOURCE --out NEW_DIRECTORY`：
导出标准Git bundle（保留commit身份）、hash固定的作者分析源文件和验收说明。拒绝脏树和覆盖，不连服务器、不安装依赖、不运行模型。
本地准备验证记录见`audit/readonly_validation_local_preparation.json`：108项回归通过；真实视频、CUDA/BF16和MuJoCo仍为未执行。
