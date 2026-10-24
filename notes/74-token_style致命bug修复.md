# 74 - token_style 致命 bug（C1）与 H/M 系列修复

日期：2026-10-24

## C1（致命）：hook 从未注册

**原因**：`load_corp_robocasa()` 内部 `CorpRoboCasa(config)` 不传 token_style；
训练脚本随后 `model.token_style = TokenStyleModulator(...)` 直接改属性，**不会注册 hook**。

**影响**：所有 `--token-style-mode` 训练实际都是 no-gate；
此前的 `corp_token_smoke`（3 步 `cross_style_stats`）**作废**。

**修复**：
- `CorpRoboCasa.set_token_style(ts)`：先移除旧 hook，再注册新 hook；
- `load_corp_robocasa(..., token_style=ts)` 直接构造带 hook 的模型；
- 训练脚本先构造 token_style，再传入 load；并 `assert model._style_handle is not None`。

**回归测试**：`scripts/test_corp_token_style.py` 现在走 `load_corp_robocasa` 路径，
断言 `_style_handle is not None` 且 `token_style.last_stats_t` 非空。

## H 系列

| 项 | 修复 |
|---|---|
| H1 strict load | `load_adapted_checkpoint` 检查 unexpected / missing（仅允许 `token_style.*`）|
| H2 deterministic | `set_seed` 加 `cudnn.deterministic=True, benchmark=False` |
| H3 RNG 隔离 | donor 采样改用独立 `torch.Generator`；flow noise 在 extract 之前生成 |
| H4 fp32 loss | `F.mse_loss(pred.float(), target.float())` |
| H5 GPU sync | `update_action_importance()` 默认不返回 summary；仅 `return_summary=True` 时同步 |

## M 系列

| 项 | 修复 |
|---|---|
| C2 | `corrupt_observation` 缺 key 时 `raise KeyError` |
| M1 | LIBERO-Plus summary/records 记录 `shift`、`blur_sigma` |
| M2 | RoboCasa SR 记录 `actual_style` |
| M4 | `action_importance` 尺寸不匹配时 `warnings.warn` |
| M5 | `domain_ids=None` 时 `cross_style_stats` 回退 random 会 warn |
| M6 | docstring 改为 "token sequence (patch or resampler)" |
| M7 | `F.interpolate(antialias=True)` vs PIL 的差异已在 note 中记录 |

## 验证

```text
test_corp_robocasa      PASS
test_corp_token_style   PASS（含 load_corp_robocasa hook 回归）
test_stale_mask         PASS
test_gate_position      PASS
```

## 结论

- no-gate baseline 不受影响；
- **C1 修复前，任何 token-style 结论都无效**；
- 修复后必须先重跑 smoke，确认 `mod_abs_mean > 0`、`styled != base`、`importance` 非零，
  再谈 appearance-shift 方法实验。
