# 64 - CoRP-RoboCasa 任务 0：per-dim / per-task MSE 与 success rate 阻塞

日期：2026-10-24
checkpoint：`/hy-tmp/runs/corp_robocasa_p2_61task_200_v2/model.pt`（200 步，61 task）
评测样本：每 task 32 帧（1952 样本/split）

## 1. 结论

```text
per-dim / per-task MSE：✅ 已完成
style shift 伤不伤 CoRP：未见系统性退化
success rate：❌ 被 RoboCasa 资产 + robosuite 版本阻塞
```

## 2. per-dim MSE（200 步 checkpoint）

| split | overall | eef_pos | eef_rot | gripper | base_motion | control_mode |
|---|---|---|---|---|---|---|
| source (train styles 11-30, 10% episode holdout) | 0.1618 | 0.3569 | 0.0254 | 0.6478 | 0.0135 | 0.0923 |
| val (styles 31-40) | 0.1548 | 0.3455 | 0.0281 | 0.5795 | 0.0143 | 0.0997 |
| test (styles 41-50) | 0.1574 | 0.3487 | 0.0258 | 0.5862 | 0.0142 | 0.1227 |

- 误差由 **eef_position + gripper_close** 主导（0.36 / 0.59–0.65）。
- 常量维 base_motion / control_mode **没有稀释结论**：它们在 held-out 上反而更低。
- eef_position 三个 split 几乎一致（0.345–0.357）。

## 3. per-task gap（val/test vs source）

```text
val : mean -0.0070  median -0.0036  min -0.0672  max +0.0882  27/61 退化
test: mean -0.0043  median -0.0016  min -0.0626  max +0.0621  27/61 退化
```

- gap 分布对称，**没有系统性退化**；略多于一半的 task 在 held-out 上更好。
- 32 帧/task 下仍有单 task 波动（±0.06–0.09），不能把单个 task 的 gap 当结论。

## 4. success rate 阻塞（重要）

计划用 `RoboCasaGymEnv` 直接跑 rollout（服务器没有 DP runner 需要的 `demo.hdf5`）。
实际创建 env 时遇到两个问题：

### 4.1 robosuite 版本不匹配

```text
robocasa/environments/kitchen/kitchen.py:522 传 load_model_on_init=False
robosuite 1.5.2 ManipulationEnv.__init__ 不接受该参数
→ TypeError
```

运行时 shim（吸收该参数、不改 vendored 代码）可以绕过。

### 4.2 RoboCasa 资产缺失

```text
/assets 只有 3.4M（官方约 10GB）
fixtures/windows 不存在
objects 目录为空
→ FileNotFoundError: .../fixtures/windows/Window032/model.xml
```

**没有资产就无法创建任何 RoboCasa kitchen env，也就无法测 success rate。**

## 5. 待决策

1. 是否下载 RoboCasa 资产（~10GB）以支持 live-env success rate？
2. 是否接受运行时 shim（robosuite 1.5.2 + vendored robocasa），还是安装匹配版本？
3. 在资产到位前，是否先做纯代码的 gate 语义修复（任务 1）？
