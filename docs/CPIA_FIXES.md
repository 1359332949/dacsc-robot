# CPIA 评估框架修复记录（P0，2026-11-11）

来源：外部代码审计（reviewer），针对远端 `7ee163c` 的 `scripts/leftover_tta_eval.py` 与 `dacsc/leftover_tta.py`。

## 已修复问题

### 1. `--adapt-once` 不是"每个 episode 一次"（P0）
- 问题：`transform.adapted_once` 是**进程级**标志，第一次 episode 适配后，后续所有 episode 都不再适配；
  U/V/Adam 状态也跨 episode 累积。
- 修复：新增 `LowRankInterfaceTransform.reset_adaptation(lr, seed)`：
  - 每个 episode 开始调用：`u.zero_()`、`v ~ N(0,1e-3)`、**重建 Adam**、清空 `adapted_once/gated/loss_trace/update_a_frac`；
  - 明确区分两种设置：
    - **Episodic TTA**（每次 episode 重置后适配一次）——论文主实验采用；
    - **Continual TTA**（跨 episode 累积）——需单独报告，不能混用。
- 验证：2 tasks × 2 episodes 冒烟测试，两个 episode 均 `episode_adapted=true`、`uv_norm` 各自重置（≈0.0031），不再只适配第一条。

### 2. disagreement gate 没有真正阻止更新（P0）
- 问题：`if disagreement < threshold: transform.gated = True`，但随后**仍然无条件**执行 `adapt_steps` 的
  `loss.backward(); optimizer.step()`；gate 只影响目标选择。
- 修复：引入 `skip_update = (verifier is not None and disagreement < threshold)`，
  **`skip_update=True` 时完全跳过更新**；episode 记录 `gate_skipped`、`adapt_steps_done`。
- 验证：`--verifier leftover --disagree-threshold 1e9` 时 `gate_skipped=true`、`adapt_steps_done=0`、`uv_norm=0.0`。

### 3. post-update 采样前未 reset hook（P0，诊断值污染）
- 问题：更新后用 `sample_with_grad` 计算 `post_chunk`，但 `transform.call` 仍是上一次计数值，
  prefix attention 也可能被低秩变换处理 → `loss_trace` 的 post 值不可信（不影响正式执行）。
- 修复：post 采样前先 `transform.reset()`；pre/post 使用同一固定采样种子。

### 4. verifier 候选监督区间（P1）
- 问题：按重叠段（`c[:overlap]` vs leftover）选候选，但蒸馏 loss 用**整个 chunk**，
  把没有 leftover 监督依据的后半段也当目标。
- 修复：默认只在重叠区间计算 `MSE(chunk[:H-E], target_chunk[:H-E])`；
  如需全 chunk 蒸馏，使用显式 `--distill-full`（作为单独消融）。

## 其它（沿用）
- 任务列表支持 `0-9` 范围解析；
- 每 episode 记录字段：`episode_adapted / gate_skipped / disagreement / adapt_steps_done / uv_norm`；
- proto 保真：bf16 autocast 与上游一致；`fork_rng` 隔离测试时更新；
- CPIA 为修改协议（`exec_len=6`），与官方协议（完整 chunk）分别报告。

## 重跑矩阵（下一步）
1. Frozen（`mode=base`，同协议）
2. Candidate-only（`mode=cand`，无接口更新）
3. CPIA-episodic（`mode=tta --adapt-once`，修复后）
4. CPIA-continual（不加 `--adapt-once`）——单独报告
5. 统计：配对 McNemar + Tango/Newcombe 90% CI + 任务级 cluster bootstrap；clean 用非劣效检验。

> 在修复后的框架上重跑之前，不应对 CPIA 的效果下结论（旧 n=120 结果混合了"真正适配"与"未适配" episode）。
