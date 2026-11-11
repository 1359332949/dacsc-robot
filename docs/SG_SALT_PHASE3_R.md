# 工单 3.0-R：评测链路修复报告（执行正确性）

> 触发：外部审查发现 `9812188` 的 `scripts/salt_eval.py` 尚未包含 SG-SALT / episodes / always-on 等入口改动，
> 而启动器已按新协议调用。**审查结论属实。**

## 1. 已确认的问题（P0）

| # | 问题 | 证据 |
|---|---|---|
| A | `9812188` 的 `salt_eval.py` 没有 `sg_salt` 模式、`--lambda-protect`、`--a-basis`、`--episodes-per-task` | 该 commit 的 `git status` 显示 `scripts/salt_eval.py` 仍是未提交修改；本地 `git status` 可见 `M scripts/salt_eval.py` |
| B | 5 个 formal seeds 未执行 | 旧入口只有单次 `run_episode`、固定 `states[0]` |
| C | always-on 未传入 | 启动器未传 `--gate-mode always`，SALT 正式批次实际使用 **feature gate** |
| D | **实际运行的 SALT 批次无效** | 检查 `p3_formal_salt`：120 episodes 中 **仅 3 条 updates>0**（feature gate 几乎不触发） |

## 2. 运行目录与代码来源（重要）

- 远程工作目录 `/hy-tmp/dacsc-robot-784fe2e` **不是 git 仓库**（`git rev-parse` 失败）；
- 实际运行文件通过 `scp` 从本地仓库部署，因此**“远程 SHA”必须记录为部署文件的 sha256**；
- 修复后部署的 sha256（记录于本次提交后）：

```
scripts/salt_eval.py     <见下文部署后哈希>
scripts/launch_phase3.py <见下文部署后哈希>
dacsc/sg_salt.py         <见下文部署后哈希>
```

- 本地仓库 commit 是唯一权威源；`9812188` 缺失的驱动改动已在本次提交补齐。

## 3. 修复内容

| 修复 | 实现 |
|---|---|
| SG-SALT 真正接入 optimizer | `sg_salt_loss` 返回 `(total, salt, penalty)`，`total` 用于 backward；`--mode sg_salt` |
| CLI 统一 | `--episodes-per-task`、`--lambda-protect`、`--a-basis`、`--gate-mode always`（启动器显式传入） |
| 5 个 formal seeds | `seed = args.seed + task_id*1000 + episode`，episode 0..4；初始状态 `states[episode % len]` |
| always-on | 启动器对非 Frozen 方法显式传 `--gate-mode always` |
| 配对记录 | 输出新增 `init_state_index`、`lambda_protect`、`a_basis_sha256`、`penalty_last`、`salt_loss_last` |
| A 缺失必须报错 | `sg_salt_loss(lam>0, a_basis=None)` → `ValueError`；driver 在 sg 模式要求 `--a-basis` |
| 保护项梯度独立验证 | 新增 `test_penalty_only_gradient_reaches_beta` |
| λ 说明 | 协议改为“真实观测 + 零 leftover 目标的代理梯度标定”，不可在结果后改 λ |

## 4. 三路 smoke（非正式 seed 303，单任务 705，always-on）

| 方法 | success | updates | gate_mode | init_state | penalty_last | λ | a_basis_sha |
|---|---|---|---|---|---|---|---|
| Frozen | ✅ | 0 | always | 0 | — | 0 | — |
| SALT | ✅ | 70 | always | 0 | — | 0 | — |
| SG-SALT | ❌ | 180 | always | 0 | **0.00755** | **5.0** | 已记录 |

- SG-SALT 的 `penalty_last` 非零、`lambda_protect=5.0`、`a_basis_sha256` 已写入记录；
- 三种方法使用同一 task / seed / init_state_index（0），满足配对要求；
- SG 单条失败不作结论（单 episode，且 β 全饱和）。

## 5. 对已跑批次的影响

| 批次 | 状态 | 处理 |
|---|---|---|
| Frozen 120 eps | 有效（frozen 不使用 gate） | 保留 |
| 旧 SALT 120 eps（feature gate） | **无效**（仅 3 条 adapted） | 保留为 `p3_formal_salt_gatefeature_invalid/`，不用于结论 |
| SALT（修复后） | 重新启动（always-on） | 进行中 |
| SG-SALT | 待 SALT 完成 | 待启动 |

## 6. 测试

- `pytest tests/test_salt.py tests/test_sg_salt.py` → **32 passed**（21 + 11）
- 新增：penalty-only 梯度、lam>0 缺 A 报错。

## 7. 结论

- 审查发现的 P0 属实，已修复并 smoke 验证；
- 当前正式 SALT/SG 批次将按冻结协议（always-on、λ=5.0、5 episodes、seeds 7–11）执行；
- 旧 SALT 批次不作为结果；
- 仍禁止根据正式结果调整 λ/lr/η/A/任务/seeds。

_commit SHA 见本次提交；部署 sha256 在运行后补记。_
