# 52 - 全量纯 DINOv2 style 审计

日期：2026-10-22  
数据：RoboCasa365 pretrain/atomic，61/61 tasks，6,935 episodes  
特征：**冻结 DINOv2 ViT-S/14**，无语言、无 FiLM、无低维状态、无策略训练  
划分：episode-level；逐 style 6 train / 2 val episodes  
规模：50 styles × 8 episodes = 400 episodes；3 frames/episode = 1,200 frames

结果：

- `audit/pure_dino_style_full/results.json`
- `audit/pure_dino_style_full/summary.csv`
- `audit/pure_dino_style_full/label_audit.json`
- `audit/pure_dino_style_full/selected_episodes.jsonl`
- `audit/pure_dino_style_full/contact_sheet.png`

## 1. 标签审计

- 每个 style 有 137–158 个独立 episode；
- 覆盖 53–56 个 task、42–50 个 layout（style 11–15 抽样）；
- 选中 400 episodes，每 style 6 train / 2 val；
- style50 每类 val support = 2 episodes；
- coarse3 的 train/val episodes：
  - 组 0：120 / 40
  - 组 1：120 / 40
  - 组 2：60 / 20（因为 51–60 只有 10 个 style）

标签来源是每个 episode 的 `extras/episode_XXXXXX/ep_meta.json`，直接读取原始 `style_id` / `layout_id`，没有经过策略的 FiLM 或 dataset wrapper。

## 2. 主结果（episode-level balanced accuracy）

随机水平：style50 = 0.02；coarse3 = 0.333；layout10 = 0.10；task10 = 0.10。

| label | 最佳特征 | 分类器 | episode bal acc | frame bal acc |
|---|---|---|---:|---:|
| style50 | right agentview patch mean | linear | **0.520** | 0.493 |
| coarse3 | right agentview CLS | linear | **0.683** | 0.617 |
| layout10 | left agentview patch mean | MLP | **0.500** | 0.498 |
| task10 | wrist patch mean | MLP | 0.575 | 0.494 |

## 3. 哪个视角有 style

Patch mean：

| view | style50 | coarse3 |
|---|---:|---:|
| left agentview | 0.490 | 0.567 |
| **right agentview** | **0.520** | **0.667** |
| wrist | 0.170 | 0.475 |
| 三视角 concat | 0.410 | 0.617 |

CLS：

| view | style50 | coarse3 |
|---|---:|---:|
| left agentview | 0.340 | 0.567 |
| right agentview | 0.340 | **0.683** |
| wrist | 0.140 | 0.408 |

结论：

1. **style 主要在 agentview**；右 agentview 最强；
2. **腕部相机几乎看不到 style**（style50 仅 0.17，接近随机）；
3. 三视角 concat **没有超过右 agentview**，说明 style 信号主要已经在右 agentview 中；
4. CLS 的 coarse3 很强，但 style50 弱于 patch mean。

## 4. 控制 task/layout

| 控制 | coarse3 episode bal acc |
|---|---:|
| 全局 | 0.667–0.683 |
| 同 task（NavigateKitchen） | 0.444 |
| 同 layout（layout 12） | 0.532 |

控制后明显下降，说明全局 probe 确实借用了一部分 task/layout 线索；  
但在同一 task、同一 layout 内仍然高于随机 0.333，说明 **style 本身仍有可识别信号**。

## 5. 与之前 ResNet18 DomainDrop 的关系

之前 note 49 的 ResNet18 stage-end GAP 域头在 3 粗域上接近 1/3。  
这次纯 DINOv2 审计说明：

```text
不是 style 标签不可学
而是 ResNet18 GAP + FiLM/策略特征不适合作为 style 域表示
```

具体来说：

- pure DINOv2 right agentview patch mean：coarse3 0.667，style50 0.520；
- ResNet18 GAP 在线 DomainDrop 域头：coarse3 ≈ 0.33；
- 纯视觉 probe 与在线域头的差距，来自特征空间和在线优化，而不是标签本身。

## 6. 视觉抽样

`contact_sheet.png` 是 **CloseMicrowave + layout 13 + styles 11/29/37** 的 3×3 图：

- 同一 task、同一 layout；
- 不同 style 的柜体、台面、墙面、地板、微波炉外观明显不同；
- 9 张图，3 个 style × 3 个 episode 帧。

## 7. 当前结论

```text
style 标签本身：✅ 有可学习视觉信号
粗域 3 组：✅ right agentview 可到 0.667–0.683
50 style：✅ right agentview 可到 0.520
腕部视角：❌ style 信号很弱
同 task/layout 控制：⚠️ 下降到 0.44–0.53，但仍高于随机
ResNet18 GAP 在线域头：❌ 不是合适的 style 表示
```

下一步应该：

1. 用 **right agentview DINOv2 patch mean** 作为 DomainDrop 的域特征/冻结域分支；
2. 用该特征 warm-start domain head，再做 layout/task sanity check；
3. 再跑 51 步 DomainDrop screen；
4. 不要继续用 ResNet18 GAP + 在线 51 步域头来判断 style 标签是否有效。
