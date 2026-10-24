# 65 - CoRP 在 LIBERO-Plus 上的初步鲁棒性 smoke

日期：2026-10-24
checkpoint：**官方 CoRP（LIBERO 训练）`corp_libero.pt`**
suite：`libero_spatial`
采样：每扰动类别 2 个 task × 1 episode
环境：`/hy-tmp/models/corp/CoRP/.venv`（robosuite 1.4.1）+ LIBERO-Plus assets（9.5G）

## 结果

| 扰动类别 | 成功 |
|---|---|
| Camera Viewpoints | 2/2 |
| Robot Initial States | 2/2 |
| Language Instructions | 2/2 |
| Light Conditions | 2/2 |
| Background Textures | 2/2 |
| Sensor Noise | 2/2 |
| Objects Layout | 2/2 |
| **合计** | **14/14 = 100%** |

## 口径（必须写清楚）

```text
本文测的是：官方 CoRP（LIBERO 训练）在 LIBERO-Plus 上的零样本鲁棒性
不是：     我们的 CoRP-RoboCasa（atomic61 适配版）在 LIBERO-Plus 上
```

两者模型不同，不能混写。

## 含义（初步）

- CoRP 官方 checkpoint 在 LIBERO-Plus 的 7 类系统扰动上**没有出现退化**。
- 与 RoboCasa365 的结论一致（source/val/test 无 gap）。
- 这进一步削弱“style shift 伤害策略”这一前提。

## 边界

- 只测了 `libero_spatial`，每类 2 个 task、1 episode，样本很小；
- 选的是每类前 2 个 task，可能偏容易；
- 要下强结论需要每类 20–50 个 task，并覆盖 `libero_object/goal/10`。

## 工程修复（记录）

1. RoboCasa assets：官方 Box 源不可达，改用官方 HF 镜像
   `robocasa/robocasa-assets` + `IIFAN/robocasa365-lightwheel-*`，共 24G。
2. RoboCasa env 需要 robosuite master；LIBERO-Plus 需要 robosuite 1.4.1
   → 两者分环境：LIBERO-Plus 用 CoRP 自带 venv。
3. LIBERO-Plus 的 `np.fromstring` 在 NumPy 2.x 已移除 → 改为 `np.frombuffer`（vendored 一行修复）。
4. `torch.load` 读 init states 需 `weights_only=False`（本地可信文件）。

## 下一步

- 扩大 LIBERO-Plus 采样（每类 20–50 task，覆盖 4 个 suite）确认结论；
- 若仍无退化 → gate/token 方法都缺增益空间，转向“负结果 + 机制分析”。
