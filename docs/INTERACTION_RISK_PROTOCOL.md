# 看—动—再看：最小工程规格与研究边界

日期：2026-11-21。当前阶段只做旧分析修正、数据采集和探针正确性验证。

## 当前实验的假设与停止条件

问题：同一初始状态、相同实际执行动作下，Lighting 是否只改变被观测的视觉条件，以及固定输入探针能否测量这种变化？

这不是风险模型实验。视觉特征变化、探针响应变化及 transition 差异均不能单独证明任务危险，也不能证明下一帧预测误差可预测失败。

- 使用冻结 CoRP；H=E=12，7 维控制动作，10 个 Euler step，FP32 / OSMesa。
- 从实际 frozen Lighting episode 日志按 state/seed 选择最小样本；结果只作 instrumentation，不能据此估计总体风险分类精度。
- Clean 原策略产生动作，裁剪到 [-1,1] 后，将**完全相同的命令**逐步执行在 Clean / Lighting 2292。
- 使用原生 step 返回的观测，不在边界额外 sim.forward、刷新或恢复状态。控制回放成功不等于 Lighting 策略自身成功。
- 每步比较 qpos/qvel/time/body pose/EEF/site pose/ctrl；边界比较 policy state8。超过容差即停止并保留失败原因。
- 建模前检查相同结构、几何和已列出的物理参数。该检查不构成对所有 MuJoCo 内部状态的完全等价证明。
- 禁止进入 target-test，禁止更新策略或 Adapter，禁止对本批数据拟合四种风险评分器。

## 连续数据契约

每条配对轨迹保存边界观测数组，长度 N+1；第 i 个动作 chunk 对应 observation[i] → observation[i+1]，不存在重新配对或跨 episode 拼接。

| 字段 | 含义 |
|---|---|
| boundary_step / boundary_sim_time | 相对执行步、真实模拟时间；相邻时间差必须等于执行长度 / control_freq |
| state8 / RGB | 每个边界的当前本体状态、两路预处理图像 |
| tokens_agentview / tokens_eye_in_hand | 分别保存 [N+1,48,768]，不跨视角平均 |
| actions_planned | 原策略输出，未裁剪的控制命令 |
| actions_executed / executed_mask / executed_steps | 实际裁剪命令、连续前缀 mask、执行长度；补齐位置为零 |
| full_chunk / fixed_horizon_eligible | 完整 12 步与部分 chunk 分开；部分 chunk 不能进入固定时间跨度模型 |
| success | simulator task checker 的事件，不由 reward/done 猜测 |
| terminated / truncated / done | 原始 API 语义与采集上限区分；采集上限只增加 truncated，保留原始 done |
| x_probe / probe_tau / probe_action_index | 同一张量、相同流时间、相同动作位置，文件含张量及 hash |
| probe_anchor_state8 / language | 本次配对轨迹初始本体状态固定、语言固定，视觉探针只替换图像 |
| probe_hidden / probe_velocity | 官方 action-generator forward 的 output_norm hook 和输出；不手工重写 block |

legacy 4-tuple API 不提供独立 terminated；显式 TimeLimit.truncated 优先，其余 done 记为 terminated，并记录这种映射。success 可以为 true 而 done 仍为 false；collector 会停止但不篡改原始 done。

探针的 x_probe 独立随机生成，tau=0.5、动作索引=0，跨状态/时间/域固定；初始 state8 在每个配对 episode 内固定。不同初始状态之间的探针比较仍包含 state anchor 差异，不能解释成纯视觉变化。探针既不参与控制，也不提供风险标签。策略自身轨迹探针留作后续补充，本次不实现。

## 本次验收

1. 对真实 Clean/Lighting 输入，官方 policy.predict 注入固定 z0，与显式 Euler 比较每一步 x、velocity、最终控制动作和裁剪动作。
2. 相同条件重复 probe 输出一致；同一个 x_probe 不被改写；固定状态和语言不变。
3. 只换一个视角图像时，另一视角 token 不变。
4. 两条轨迹每步动作相同、测得状态一致，连续时刻和特征形状正确。
5. 实际测试完整 chunk 和末尾截断 chunk；mask/时间/长度验证通过。
6. 审核元数据、源文件 hash、数据 hash、原始日志选样来源。

## 通过后才启动的观测器

单一候选预测量：

`f(z_agent_t, z_wrist_t, state8_t, executed_actions_t, mask_t) -> delta_z_(t,t+12)`。

先只使用完整 chunk。对照为 state8+动作、视觉输入且不带动作的预测器；输出目标、训练样本和评估口径保持一致。后者不是直接用下一帧信息预测下一帧。

Source Fit 拟合模型、降维、均值/方差和对角或收缩残差尺度；Source Validation 选定报警阈值；所有操作保留状态分组。高维特征不直接求未经正则化的完整协方差逆。

第一阶段先证明正常交互预测的 held-out 表现。第二阶段用相同动作的 Clean/Lighting 回放测量无害光照导致的残差增量。通过后才能检验 Lighting 自身轨迹内的风险排序。受控回放中共享动作，不能直接用其成败代替真实 Lighting 策略风险。

## 风险定义与未来评价边界

- 当前动作偏差标签：当前状态、相同采样条件下，相对于参照动作的 XYZ / SO(3) / gripper 分量超过预先冻结阈值。配对 Clean 参照仅能作为离线 privileged 标签，不能进入部署输入；它衡量行为分歧，不天然等于任务失败。
- 未来风险：分数在 chunk 结束、新观测到达后才可用；未来事件窗口从这个可用时间开始，预设窗口长度，禁止把当前 transition 的事后误差算作提前预测。
- 提前量：首次风险事件执行步减去首次有效报警执行步，报告未报警与提前/滞后。不得以 episode 结束代替事件时间。
- Episode success 只做轨迹级评价，不复制到所有 chunk。超时不能自动解释为物理失败或不可恢复。
- 风险阈值与任务事件判据须在风险实验前冻结；本次没有生成风险标签或宣称任何 AUROC。
- Lighting 内部单独报 AUROC/AUPRC、按状态分解的召回、Clean 描述性误报率和轨迹最大分数。不得用 Clean-vs-Lighting 的分类性能替代风险预测。
- 模型仅 Source Fit 训练，阈值仅 Source Validation 决定；target-dev 留出状态不得参与调参。如需要 target-dev 方法选择，另划内层开发与外层留出；报告 pilot 消耗。
- 以初始状态为不确定性估计单位，seed/flow-step/action-position/chunk 不能当独立状态。
- 十个验证状态的“5% FPR”仅为描述性目标，不作严格统计保证。报告每状态误报、每轨迹最大分数和状态级置信区间。
- 当前尚无可靠正常观测器、无害光照鲁棒性、Lighting 内部风险预测或 Clean-safe adaptation 的证据。

## 本轮查阅的原始来源

- TTA-VLA 作者项目页（2026-11-21 读取）：https://tta-vla2.github.io/
  - 使用真实环境时间的视觉/动作表征变化及语言语义锚点。它支持检查 temporal transition，但不能替我们证明风险可预测性；flow time 不等于环境时间。
- scikit-learn 官方 Common pitfalls：https://scikit-learn.org/stable/common_pitfalls.html
  - 模型、预处理和选择操作只使用训练集。本轮修复常数和 beta，并加入“改变测试标签，预测不得变化”的回归测试。
- Gymnasium 官方 Handling Time Limits：https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
  - 区分 termination 和 truncation。本轮保留 legacy done，采集时间上限单独标注。
- 实际运行的 CoRP `corp/policy.py` / `corp/action_generator.py` 和 LIBERO-Plus `env_wrapper.py` 已逐段核对；运行 metadata 保存 CoRP 源文件 hash。

文献中的成功率和历史会话提到但本轮未核验的论文结论不作为本次实验结果。上述工程验证也不构成专利新颖性结论。

## 可复现命令（当前服务器）

服务器目录为 `/hy-tmp/patch-diag`。首次运行使用原有 `multidomain_matrix.json` 和冻结 split；输出目录必须不存在，防止覆盖实验。

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/hy-tmp/libero_config
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/hy-tmp/libero_plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
python scripts/collect_interaction_replay.py   --matrix /hy-tmp/validation/multidomain_matrix.json   --checkpoint repos/CoRP/checkpoints/corp_libero.pt   --split docs/data/tta_split_libero_spatial_perturbation.json   --episode-log docs/data/perturb_dev_v2.json   --out runs/interaction_replay/NEW_RUN_DIRECTORY --chunks 3 --max-steps 36
python scripts/validate_interaction_replay.py   --root runs/interaction_replay/NEW_RUN_DIRECTORY   --out runs/interaction_replay/NEW_RUN_DIRECTORY/reopened_validation.json
```

`--states` 不传时，程序从指定 seed 的真实 frozen 日志中，分别取编号最小的失败和成功状态；本轮实际是 34 和 36。该结果分层采样只服务于代码验收，不能用于估计风险总体分布或模型性能。

旧实验重算：

```bash
OPENBLAS_NUM_THREADS=4 python scripts/b3_oracle10.py   --input runs/patch_diag/b3_flow_pairs_10.json   --legacy runs/patch_diag/b3_oracle10.json   --out runs/patch_diag/NEW_B3_CORRECTED_REPORT.json
python -m unittest tests.test_b3_fold_isolation tests.test_interaction_records -v
```
