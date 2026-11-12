# B2 v3 受限 dry-run 报告（单任务/单分支/单候选）

> 批准范围：1 episode、1 branch、1 candidate（`prod_step_005`）；只验证执行链路，不产生 TTA 有效性结论。
> 命令：
> `python scripts/sg_b2_v3.py --dry-run --task-id 705 --episode 0 --branch 8 --candidate prod_step_005 --out /hy-tmp/validation/b2v3_dryrun`
> 输出：`/hy-tmp/validation/b2v3_dryrun/b2_v3.jsonl`（1 条完整记录）。

## 1. dry-run 前的 P1 修正

| 问题 | 处理 |
|---|---|
| 默认输出可能覆盖正式目录 | dry-run 强制显式 `--out`，且禁止等于 `/hy-tmp/validation/b2v3` |
| raw 候选误带 production 诊断 | raw 的 `step_scale/accepted/backtracks/local_a_change/cumulative_a_change` 记为 `null` |
| `make_env` 固定 `states[0]` | 改为 `states[episode % len(states)]` |
| success 被统一当 censor 丢弃 | censor 记录保留 `success_control/update`；分析脚本新增 control/update success、update-only、control-only、both、neither 计数 |

另修复了 dry-run 暴露的两个实现错误：
- 脚本缺少 `if __name__ == "__main__": main()` 入口（导致静默退出）；
- `sg.adapter.beta.zero_()` 在 `requires_grad` leaf 上原地操作报错 → 加 `torch.no_grad()`；
- 输出目录未创建 → 启动时 `mkdir(parents=True)`。

## 2. dry-run 结果（执行链路检查，不看 τ 方向）

| 检查 | 实际值 | 判定 |
|---|---|---|
| 任务 / episode / branch / candidate | 705 / 0 / 8 / `prod_step_005` | 符合批准范围 |
| 分支资格 | `contact=0.0549 ≤ 0.10`，分支前未 success | ✅ |
| 生产更新 | `raw_norm=1.02e-5`，`effective_norm=1.02e-5`，`beta_max=4.05e-6` | ✅ 非零 |
| β 边界 / 截断 | `clip_ratio=0.0`，`step_scale=1.0` | ✅ |
| A guard / backtrack | `accepted=true`，`backtracks=0`，`local_a_change=5.26e-8` | ✅ 接受更新 |
| control / update 动作变化 | 4 段分别 `1.74e-7, 1.30e-7, 2.42e-7, 2.70e-7`；均 < 0.05 | ✅ 非零且未触发 cap |
| 执行窗口 | `steps_control=24`，`steps_update=24`，无 success/termination | ✅ |
| τ 参考距离 | `d_ref=0.16697`（episode 初始），`d_branch≈0.16780`，`d_end_control=0.158081`，`d_end_update=0.158082` | ✅ 使用固定 d_ref 与各自环境 |
| τ | `tau_prox = -2.38e-6` | 记录成功（非统计结论） |
| 分析联调 | `analyse_b2_v3.py` 成功读取；1 complete、0 censor；**success_outcomes.neither=1**（修正后）；分类 unidentifiable | ✅ |

**说明**：生产候选的 β 量级很小（max 4e-6），动作变化 ~2e-7，因此 τ 落在不可辨识区间；
本次 dry-run 的意义是确认"候选 β 真正进入动作生成、4×6 配对、终止/censor、d_ref 与 schema 联通"，
不能据此判断更新有益/有害。

## 2.1 raw 三候选受限 dry-run（补充，路径验证）

同一 task 705 / ep0 / branch 8，分别隔离输出：

| candidate | proposal/applied norm | α | predicted ‖J_A·Δβ‖ | 动作变化范围 | 24 步 | τ_prox |
|---|---|---|---|---|---|---|
| raw_proj_100 | 0.02049 / 0.02049 | 1.0 | 1.02e-17 | 2.5e-4–4.0e-4 | ✅ | -0.01127 |
| raw_full_100 | 0.04264 / 0.04264 | 1.0 | 2.24e-2 | 5.4e-4–1.6e-3 | ✅ | -0.01222 |
| raw_sens_100 | 0.03739 / 0.03739 | 1.0 | 2.24e-2 | 5.9e-4–2.3e-3 | ✅ | -0.00227 |

三个 raw 路径均未触发统一缩放（α=1）与 action-cap censor；`raw_proj` 的 A 变化 ~1e-17 验证投影方向保持。
单 episode 不作为效应结论。

## 3. dry-run 暴露但尚未覆盖的检查

- 分支前两环境的 observation/body pose 一致性：dry-run 记录未单独写入该字段；
  但 zero-update 逐分支审计（branch 8/12，轨迹/步数差 0.0）与双环境确定性预检已覆盖该协议。
- 生产候选若被 A guard **拒绝/回滚**的分支：本次 accepted=true，未触发拒绝路径；
  正式实验必须单独统计拒绝率与拒绝原因（脚本已记录 accepted/backtracks/step_scale）。
- success 分支：本次双方均未 success，success 分类字段为 0；正式数据会统计四类结果。

## 4. 结论与请求

- 受限 dry-run **通过执行链路验收**：β 非零进入采样、两分支各 24 步、τ 使用固定 d_ref、分析脚本可解析。
- 未产生任何 TTA 有效性或统计结论；单 episode 不构成证据。
- 请求：批准**冻结当前 schema/代码**，并授权正式 24-episode B2（branch 8/12 + raw η=100 + production 候选，
  仍分开报告）。若需先审阅 dry-run JSON，可在授权前提出字段修改。
