# 76 - token_style 51 步四组 smoke（修复后）

日期：2026-10-24
前提：C1 已修（`979108b` / `d11f138`），hook 通过 `load_corp_robocasa` 注册
配置：CoRP-RoboCasa，5 task，51 步，1 seed（smoke，不是结论）
输出：`/hy-tmp/runs/token_smoke_51/{mode}/result.json`

## 结果

| mode | train loss | source MSE | val MSE |
|---|---|---|---|
| none | 0.4055 | **0.1433** | **0.1443** |
| random_stats | 0.3966 | 0.1727 | 0.1694 |
| cross_style_stats | 0.3728 | 0.1686 | 0.1539 |
| cross_style_stats_action_aware | 0.3706 | 0.1687 | 0.1539 |

## 解读

1. **四组 loss/MSE 不同** -> hook 已生效，不再是 no-gate。
2. 51 步 smoke 下，**none 的 source/val MSE 最低**；token_style 三组都更高。
3. `cross_style_stats` 与 `action_aware` 几乎相同（0.1539），
   说明在 51 步/5 task 下 action-aware 权重尚未产生可测差异。
4. 这只是 smoke：1 seed、51 步、5 task，**不能作为方法有效/无效的结论**。

## 下一步

- 按冻结协议：3 seeds × 51/200 步，per-dim/per-task MSE + success；
- 条件：clean / color_jitter（A 前提已成立：100% -> 21.4%，14 task 小样本）；
- 需要 random_stats 作为强制对照（本 smoke 已包含）。
