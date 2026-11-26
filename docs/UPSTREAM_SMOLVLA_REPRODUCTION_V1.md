# SmolVLA 原版入口复现体检 V1

**结果：** 下载并原样运行公开 roboeval，兼容环境十任务 Clean 为 **9/10**，通过起步检查。按公开记录主要版本另建独立环境，原版单例 **1/1（70步）**。这些是 Frozen 基线体检，不是 TTA 收益，不是100局正式成绩。

## 1. 执行的是真实开源入口

- 仓库：https://github.com/sim-too-real/roboeval （原 ActuallyIR/roboeval 重定向到此）。这是社区评测仓库，不能称为 SmolVLA 论文作者的官方仓库。
- checkout `6e9d9daf1526ff344b3742b935fde167faeffb13`，源码没有修改。安装副本19个Python文件逐字节匹配 checkout。
- 核心 src/roboeval 和 run_smolvla_libero_spatial.py 与公开79/100结果上传提交 `89d52225d82d63613f40e5f4e29c81c74d950516` 没有差异。历史结果manifest写的是较早的base commit ba7b...，不能仅按该字段checkout就认为重建了完整运行树。
- 10局直接调用原脚本：`scripts/run_smolvla_libero_spatial.py --episodes-per-task 1 --max-steps 220 --seed 0 --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。可复用本仓库 scripts/run_smolvla_upstream_native.sh；它只配置路径和环境，不重写Python推理。
- 真vanilla LIBERO，EGL，256像素，10步settling，原始HuggingFace配置，n_action_steps=1、Euler10、chunk50。CoRP的H=E=12没有强加到SmolVLA。

## 2. 所有有效原版评测分开汇报

| 批次 | 任务/起点 | 结果 | 用途 |
|---|---|---:|---|
| 兼容环境 CLI 首次单例 | task0/init0 | 0/1，220步 | 原样保留，不能凭单局判断总体 |
| 兼容环境原脚本固定10局 | 十任务各init0 | **9/10** | 起步体检主批次 |
| 公开主要版本独立环境 CLI 单例 | task0/init0 | **1/1，70步** | 新环境兼容核查，不代替10局 |

原版内部采样随机性保留；这些批次不能简单合并为一个方法成功率，也不能与旧自写8/10、未完成的20例做恢复/损害或增益比较。

| Native task ID | 结果 | 步数 |
|---:|---|---:|
| 0 | 成功 | 74 |
| 1 | 成功 | 111 |
| 2 | 失败 | 220 |
| 3 | 成功 | 94 |
| 4 | 成功 | 132 |
| 5 | 成功 | 89 |
| 6 | 成功 | 123 |
| 7 | 成功 | 113 |
| 8 | 成功 | 93 |
| 9 | 成功 | 122 |

十局总耗时约 13.9 分钟。task2仍失败，未删例或补到成功。

## 3. 确认的旧入口差异

1. **加载路径/精度不同。** 原始Hub配置 load_vlm_weights=True；旧离线配置将其设为False，并将基础模型名替换成本地路径。两个基础config JSON本身完全相同，但构建分支不同。原版实际视觉dtype为BF16，旧路径为FP32。788个权重按目标dtype匹配不能证明计算精度等价。此前“只省下载、不改变数值路径”的判断不成立。
2. **像素归一化位置不同。** 原版先在NumPy/CPU除255，我们在GPU除255。同图最大差5.96e-8，同一已加载模型/状态/Torch RNG的动作最大差0.00798187；两边分别重放均可重复。只把输入计算改为作者CPU路径后，输入及动作逐位一致。这是单观测输入差异的因果验证，不是整体闭环掉分归因。
3. **原版seed不是完整模型随机数控制。** 原版reset接收seed但忽略它；Flow内部仍随机采噪。同观测两次reset(seed77)动作L2差0.20743644，CUDA RNG前进；显式重置Torch RNG后逐位一致。当前原版体检未偷偷修该行为；后续严格配对实验需单独登记可审计的随机数控制。
4. **旧30局没有跑完。** 用户要求转原版时，自写扩展已完成20例、10成功、零程序错误。四个活动子进程与其driver按确切身份停止，结果没有删除；不把10/20当作预定30例最终胜率。

确认不等价，不等于已经证明每项差异各造成多少闭环损失；本轮没有做完整单因素闭环归因。

## 4. 镜像与版本证据

- GPU服务器无法直连官方Hub，使用 hf-mirror.com 下载；基础模型2,029,990,624字节，经官方LFS SHA-256核对通过。aria2c并行下载只改变传输，不改模型。
- 基础模型SHA：`b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`；SmolVLA权重SHA：`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。
- Python/PyTorch等依赖通过阿里云PyPI和cu128 wheel镜像安装到独立目录/环境；不覆盖CoRP环境。
- 9/10兼容环境：Python3.10、Torch2.10、Transformers4.57.6、NumPy1.26.4、MuJoCo3.2.7。它不是历史报告环境的完全复制。
- 独立环境单例：Python3.12.3、Torch2.11.0+cu128、TorchVision0.26.0+cu128、Transformers5.6.0、NumPy2.4.4、MuJoCo3.2.7、robosuite1.4.1、Hub1.11.0等主要版本按公开记录配置。硬件仍是RTX3090，不能声称与历史RTX5090完全相同。
- 重新下载LeRobot固定提交483be9aa，与旧源码存档176个Python文件全部相同；原始roboeval核心19文件与安装副本相同。
- 新环境前几次因jsonlines、serial、matplotlib缺失停在import阶段，日志保留；不是有效任务失败，也没有修改算法来绕过错误。实际模型导入、完整原版单例均已通过。

## 5. ActMAD对照修复的边界

修复了近期多帧双相机view-major错位；三项CPU测试与真实CoRP生命周期检查通过。适配确实改变参数和动作，重置后逐位恢复Frozen。它仍明确标为ActMAD-inspired，不能称原作者空间L1算法的忠实复现。没有新的ActMAD闭环胜率，也没有SmolVLA上V1适配收益。旧错误对照成绩不能因代码修复而自动有效。

## 6. 下一阶段

以原版Frozen入口和完整记录的兼容环境为参照；正式大样本与方法比较前，先把Flow随机数配对控制写成独立、可审查的协议差异，再固定同一名单、多种子、Clean与原成功Shift。不能用10局90%替代正式100局，更不能把恢复原版推理流程当作原创TTA增益。保护的Blind/Held-out清单未打开，本轮是已接触任务的开发体检。

## 证据

紧凑JSON：docs/data/upstream_smolvla_original_v1/summary.json。原始逐例JSON、stdout、环境、源SHA及旧中止批次在本地 remote_artifacts/upstream_smolvla_original_v1/。
证据包：upstream_repro_evidence_v1_licensed.tar.gz，SHA-256 `298ba145c537252f49aeea57eb3b69522cede2f0f5e8fc14314af3ea86e36db2`。模型权重及大型依赖不放入Git。
