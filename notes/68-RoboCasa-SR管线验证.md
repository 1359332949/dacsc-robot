# 68 - RoboCasa success-rate 管线验证（GT action replay）

日期：2026-10-24

## 方法

用数据集原始 episode 的 ground-truth action 复现 rollout：

```text
extras/episode_000000/model.xml.gz   -> reset_to(env, model)
extras/episode_000000/states.npz     -> states[0] (184D sim state)
data/chunk-000/episode_000000.parquet-> action (modality order)
  -> reorder 为 convert_action 顺序
  -> env.step(convert_action(...))
```

关键：必须先 `reset_from_xml_string(model.xml)`，否则 sim state 维度是 178，与记录的 184 不匹配
（场景模型不同）。

## 结果

```text
episode=0  steps=294  states=(294,184)
after model reset: sim state dim = 184 (matches)
GT replay: success=1 @ step 282/294
```

## 结论

```text
RoboCasa success-rate 管线：✅ 正确
  - 观测 key 正确
  - action 顺序 / convert_action 正确
  - success 判定正确
  - episode 初始状态可精确复现

CoRP-RoboCasa 200 步模型 clean 0/2：✅ 模型太弱，不是管线 bug
```

## 含义

- 200 步 checkpoint 的 source MSE 0.105、eef_pos 0.357、gripper 0.648，
  做 CloseFridge 这种精确操作失败是合理的。
- 要用 RoboCasa success rate 做方法评测，必须先训练到能解任务的规模（2k–20k 步量级）。

## 框架修正（与 note 67 一起看）

```text
语义/外观偏移（style/background/light/layout） -> CoRP 鲁棒（无退化）
低层图像退化（Gaussian blur σ=2/4）           -> CoRP 崩（50% / 0%）
RoboCasa 适配模型 success rate                -> 0/2（模型太弱）
```

真正可复现、可研究的薄弱点是 **low-level corruption**，不是 style/domain。
