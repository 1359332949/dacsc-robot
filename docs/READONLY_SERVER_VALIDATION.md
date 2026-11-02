# CoRP只读验收包：执行顺序与边界

这是**后续部署的操作说明**。准备/解包不等于服务器验证通过；本轮只在本地整理、测试、提交和打包。
不调用训练入口main、不做backward或optimizer更新、不加门控、不执行学习策略SR矩阵、不读取test/reserve作校准。

## 包里是什么

`scripts/pack_readonly_validation.py`在干净commit上生成：

- `code.bundle`：标准`git bundle`，保留commit和Git身份。包含已提交历史，不包含ignored的数据、环境、模型或仿真资产。
- `registers_analysis.tgz`：`analysis_upstream.lock.json`固定的四份作者原文件及来源清单，逐字节导出，不改算法。
- 本说明、`bundle_manifest.json`和`SHA256SUMS`。manifest中服务器、真实视频、CUDA、MuJoCo、训练均为`not_run`。

本地打包（不上传）：

```bash
.venv/bin/python scripts/pack_readonly_validation.py \
  --registers-source .cache/analysis_sources/registers/860df43515c8d8e9e90952af25a46c26e4469570 \
  --out .cache/readonly_delivery/NEW_COMMIT_DIRECTORY
```

拒绝脏工作区、源hash不符和重复输出。服务器以后收到包时，先核对已交付的`SHA256SUMS`文件hash，
再用`sha256sum -c SHA256SUMS`验证内容，在**新目录**`git clone code.bundle CODE_DIRECTORY`并核对HEAD，
展开Registers文件至独立分析源码目录。不能覆盖正在运行的checkout或旧run目录。

CoRP/DP/RoboCasa源码、官方checkpoint及相邻`config.json`、官方依赖环境和仿真资产需提前准备；包不带权重，入口不下载/安装、不改上游、不加兼容shim。
按两个lock中的revision准备源码；`python -m dacsc.upstream`检查清单文件。它不代替全部依赖或资产验证。
不要只解开旧`git archive`包后运行新验收入口：本入口要求真实、干净的Git checkout。

## 0. 冻结环境与输出

后续命令在新checkout根目录执行。`python`必须是已核对的环境解释器；视频与CUDA自动阶段使用同一解释器/库版本，
receipt还核对platform、Torch/Torchvision/CUDA版本，不能拿另一环境的PASS继续。MuJoCo可使用单独的官方仿真环境，需另存其版本/实际import路径。

示意变量先替换成真实绝对路径，不直接照抄占位符：

```bash
export CORP_REPO="$PWD/repos/CoRP"
export PYTHONPATH="$PWD:$CORP_REPO:$PWD/repos/robocasa:$PWD/repos/diffusion_policy:$PWD/repos/robomimic"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA=/ABSOLUTE/PATH/robocasa365
REGISTERS=/ABSOLUTE/PATH/registers
CHECKPOINT="$CORP_REPO/checkpoints/corp_libero.pt"
OUT=/ABSOLUTE/NEW/PATH/readonly_COMMIT_TIMESTAMP
python -m dacsc.upstream
```

环境准备沿用官方安装说明，**不靠本入口自动修环境**。锁定源码导入到其他路径会失败，不做静默fallback。
所有阶段输出互斥、拒绝复用非空目录；失败保留`report.json`/`error.txt`，修好后换新目录重跑。
每步返回成功并核对receipt后再手动执行下一步，不串接训练。

## 1. 真实视频：只读官方reader，不加载模型

```bash
python -m dacsc.readonly_validation video --data-root "$DATA" --out "$OUT/video"
```

复用固定DP的`LerobotDataset.__getitem__`及其官方`LeRobotSingleDataset`视频后端，外层仍是`CorpRoboCasaDataset`。
薄记录器只截取同一次真实reader返回，不重写解码器，不再为获取原图重复解码。

选择先冻结、再读图：seed123，最多3 tasks × 每task3独立episodes × 每episode头/中/尾去重帧，
最多27组三视角观测/81 view-frames。只选styles11–30且episode `%10 != 9`；不足记录实际support，不补val/test。

验收与产物：

- `selection.json`在读视频之前生成；`sample_manifest.jsonl`含dataset/task/episode/frame、原生style/layout、指令、元数据/视频/解码像素hash和实际后端。
- 原RGB为有限`[3,256,256]`、固定Pillow resize224/ImageNet normalize；三视角左/腕/右，标签与extras对齐。
- state9D、action`[10,12]`、尾部mask与repeat-last正确；样图用于人工核查，不回作输入。
- `observations.pt`缓存这次读出的像素和样本，后续条件共享它；不读其他episode做排名、不调用训练增广。
- `report.json.status=passed`只说明此阶段通过。真实视频、metadata、相机身份或动作mask任何失败，停止。

这不是“模型已会做任务”的证据。

## 2. CPU FP32：真实像素的只读非侵入性

```bash
python -m dacsc.readonly_validation capture \
  --video-artifacts "$OUT/video" --checkpoint "$CHECKPOINT" \
  --registers-source "$REGISTERS" --device cpu --precision fp32 --out "$OUT/cpu_fp32"
```

直接复用官方CoRP loader和视觉forward，沿用已测的三视角适配器，但**只执行其中视觉encoder**。
不向完整策略传零state或临时normalizer；新state/action/right-query接口的随机初始化明示且不执行。
完整策略/动作恒等性在这个入口始终是`not_run`，不能拿本结果替代未来有效RoboCasa checkpoint验证。

每批2观测（尾批可1），同一raw像素分别做`none`及固定评测`color_jitter`；每条件内比较无hook/有hook/独立原始hook。
采集原GELU和raw block输出，Registers logger原函数不变。

验收：同精度同输入时输出逐元素一致、数值有限、每层CLS/patch与`row=view_index*B+sample_index`对齐，
权重digest、CPU/CUDA RNG及原有hook数量不变，临时hook清理完整，无梯度。两条件之间**不要求输出相同**。

产物：`capture_manifest.json`、`identity_checks.json`、`activation_summary.json`、首批两条件原始tensor快照和样图。
summary只使用Torch范数/mean/min/max/quantile，分CLS/patch/相机/层保留身份；不产生新的“风格分数”、坏神经元标签或排名。
确认扰动是否抹掉目标等任务线索；描述性变化不能当作shortcut或动作因果结论。

## 3. CUDA/BF16：先官方demo，再真实像素

执行前另存`nvidia-smi`原始输出（GPU/UUID/driver），与receipt中的Torch/CUDA、解释器及GPU能力记录一起交付。
当前入口不安装或升级driver；缺失这份硬件记录时在人工汇总中列为缺项。

```bash
python -m dacsc.readonly_validation official-cuda \
  --video-artifacts "$OUT/video" --device cuda:0 --out "$OUT/official_cuda"
```

此子命令使用当前解释器原样委托官方`repos/CoRP/scripts/smoke_test.py`，依次FP32/BF16；不维护另一份模型demo。
先检查CUDA/BF16支持，无CUDA直接失败。验收官方checkpoint checksum、严格加载、48,689,159参数、有限动作`[1,12,7]`和`smoke_ok`。
原始stdout/stderr分别保留在`official_fp32.log`/`official_bf16.log`。**官方demo仍是合成LIBERO输入，不是真实RoboCasa输入验收。**

之后才运行真实输入的两个精度：

```bash
python -m dacsc.readonly_validation capture \
  --video-artifacts "$OUT/video" --official-cuda-artifacts "$OUT/official_cuda" \
  --checkpoint "$CHECKPOINT" --registers-source "$REGISTERS" \
  --device cuda:0 --precision fp32 --out "$OUT/cuda_fp32"
python -m dacsc.readonly_validation capture \
  --video-artifacts "$OUT/video" --official-cuda-artifacts "$OUT/official_cuda" \
  --checkpoint "$CHECKPOINT" --registers-source "$REGISTERS" \
  --device cuda:0 --precision bf16 --out "$OUT/cuda_bf16"
```

前置receipt必须同commit/lock/环境，所有列出的文件hash必须匹配，CUDA官方demo还绑定同一video receipt。
**分别验证同精度内的恒等性，不要求FP32与BF16逐元素一致**。BF16必须实际捕获到BF16激活，不能只是CLI标签。

训练增广边界在真实像素上单独检查：`TrainingPixelAugmentation`与同RNG的Torchvision原`ColorJitter`精确一致，
private RNG可恢复、不消耗全局CPU/CUDA RNG。它的输出**不进入任何模型forward**；模型只接收固定评测条件。
这是算子验收，绝不调用训练main、loss、backward或optimizer。

## 4. MuJoCo：官方状态重放，然后官方GT动作重放

前三阶段通过后选择一个官方首episode确为source的真实dataset。保持官方脚本不变，不能把另一episode改名、改XML或改style迁就入口。

```bash
python -m dacsc.readonly_validation replay-plan \
  --dataset /ABSOLUTE/SOURCE/TASK/DATE/lerobot \
  --sim-python /ABSOLUTE/OFFICIAL_SIM_ENV/bin/python \
  --replay-output "$OUT/mujoco" --out "$OUT/replay_plan"
```

此命令**只检查并给出命令，不运行MuJoCo**。核对episode_000000的原生source style、states长度/有限性、原XML可解析及文件hash；
官方脚本当前`--n 1`实际读取ordinal0，所以不能仅检查“选中的某个source episode”。若首episode非source，换dataset，不打补丁。

从`report.json.details.commands`读取两条argv，人工核对后在仿真解释器中执行。
先创建新的`$OUT/mujoco`目录，设置`MUJOCO_GL=egl`，把完整stdout/stderr写进各自的新日志；
argv会直接调用锁定的`repos/robocasa/robocasa/scripts/dataset_scripts/playback_dataset.py`：

1. state：`--n 1 --camera_height 224 --camera_width 224 --video_path .../state_replay.mp4 --verbose`
2. action：同一dataset/episode，加官方`--use-actions`，输出`action_replay.mp4`。

不使用`--first`截帧，不自己实现`reset_to`或动作重排，不启动学习策略rollout。
保存仿真环境版本/import路径、资产路径、两条实际命令/完整日志；两次都要完整走完episode，无资产/XML/NaN/EGL异常。
用现有视频工具完整解码两份非空视频并人工检查。原`playback diverged` warning必须保留、单列处理，不能隐藏后报确定性PASS。

**重放结束/视频可解码与任务成功是不同验收项**：当前官方脚本不输出success数，不能编造success=1。
若另查官方success判据，保留读取依据；不把GT重放成功当学习策略成功。
将检查结果另存`mujoco_acceptance.json`，字段包括`status`、`state_replay`、`action_replay`、`video_decode`、
`divergence_warnings`、`success_check`、原始日志/视频hash和环境；未检查项写`not_run`及原因。
`replay-plan`的PASS仅表示命令准备好，**不是MuJoCo验收PASS**。

## 结束条件

任一步失败停止，输出目录保留。人工汇总各阶段PASS/FAIL/not_run与缺项，不用一次退出码替代全部验收。
整轮验证不会自动接着训练、门控或参数扫描。只有真实链路通过后，再讨论研究对象、受控干预和有效任务基线。
