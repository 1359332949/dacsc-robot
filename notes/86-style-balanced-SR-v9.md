# 86 - Style-balanced SR v9：修复 RoboCasa 场景复用

日期：2026-11-06

## 1. 发现的问题

旧的 `v6_close_drawer_20ep_v1` manifest 虽然命令传入 `--styles 11-30`，
但 20 个 source episode 的 `actual_style` 全部是 20，val 的 20 个 episode
全部是 35。因此旧的 source 5/20 与 val 2/20 只是一个固定 style pair 的初步结果，
不能当作 20 个 source style 对 10 个 val style 的泛化证据。

原因不是 RoboCasa 不采样。`RoboCasaGymEnv` 构造时会先做一次 reset，
`Kitchen.reset` 随后会把 `_ep_meta` 中的 `layout_id/style_id` 当作固定场景；
后续再次 reset 时如果没有清掉 `_ep_meta`，就会继续复用第一次场景。

## 2. 修复

- `prepare_episode` 创建 manifest 前显式设置目标 `layout_id/style_id`；
  没有显式场景时清空 `_ep_meta`，保证后续 reset 重新采样。
- manifest 格式升级为 `robocasa_eval_episode_v2`，保存 requested/actual scene。
- SR schema 升级为 `corp_robocasa_sr_v9`，固定 `layout_id` 并记录完整 `style_order`。
- 每个 episode 的 style 按 `episode % len(styles)` 轮转：
  source 20 个 episode 覆盖 styles 11-30；val 20 个 episode 让 styles 31-40 各出现两次。
- 一个 run 内固定 layout 11，避免把 layout 变化误读成 style 变化。
- 聚合器校验 summary、record、requested scene、actual scene 与固定 layout 完全一致。

## 3. 回归验证

- 本地 130 个回归测试全部通过。
- 真实 RoboCasa smoke：requested `(layout=11, style=11/12)` 得到 actual
  `(layout=11, style=11/12)`，manifest format 为 `robocasa_eval_episode_v2`。
- 新 SR v9 单 episode 入口已实际跑通并写出 requested/actual scene。

## 4. GPU 队列

训练保持同一预算：

```yaml
steps: 10000
batch_size: 96
lr: 5e-5
ema_decay: 0.99
task_aux_lambda: 0.01
seed: 42
```

顺序：

1. `random_stats`
2. `cross_style_stats`
3. `cross_style_stats_action_aware`

训练结束后自动在冻结的 style-balanced manifests 上跑 baseline + 三个方法：

```text
CloseDrawer, 20 episodes/split
source: styles 11-30
val: styles 31-40
layout: 11
shifts: none, color_jitter
```

旧 v8 结果保留但不进入 v9 主对比。
