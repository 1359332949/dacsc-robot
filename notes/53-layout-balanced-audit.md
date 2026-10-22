# 53 - Layout-balanced 纯 DINOv2 审计

日期：2026-10-22  
目的：修正 note 52 中 layout10 只有 2 类的问题。  
采样：按 layout10 分组，每组 20 episodes；共 200 episodes。  
特征：右侧 agentview，冻结 DINOv2 ViT-S/14，patch mean / CLS。  
划分：episode-level，每组 14 train / 6 val。

结果：

- `audit/pure_dino_layout_balanced/results.json`
- `audit/pure_dino_layout_balanced/summary.csv`
- `audit/pure_dino_layout_balanced/label_audit.json`
- `audit/pure_dino_layout_balanced/selected_episodes.jsonl`

## 1. layout10 support 现在是对的

10 个 layout10 类，每类：

```text
14 train episodes / 6 val episodes
42 train frames / 18 val frames
```

## 2. layout10 结果

| 特征 | 分类器 | episode bal acc | frame bal acc |
|---|---|---:|---:|
| right agentview patch mean | MLP | **0.283** | 0.233 |
| right agentview patch mean | linear | 约 0.20–0.25 | 约 0.20 |

随机水平：0.10。

结论：

> **layout10 只是弱可分（0.25–0.28），低于我们设定的 0.30 sanity 门槛。**
>
> 这不是 branch/head 接口问题，而是 layout10 这个标签在 pure DINOv2 agentview 特征里本身较弱。

## 3. layout-balanced sample 上的 style 结果不能看

这个 sample 为了 layout 平衡，style 分布严重偏斜：

```text
style 11: 127 episodes
style 12: 72 episodes
style 13: 1 episode
```

因此：

- style50 的 0.976 是 **selection artifact**，不是有效的 style 可分性；
- coarse3 只包含 class 0（styles 11–13 全在 11–30），coarse3 probe 无意义。

有效的 style 结论仍然来自 note 52 的 **style-balanced 400-episode** 审计：

```text
style50 0.520
coarse3 0.683
same-layout coarse3 0.532
```

## 4. task10

layout-balanced sample 的 task10 best 约 0.62（wrist CLS MLP），  
但 task10 是人为连续分组，不能替代 layout/style 的语义 sanity。

## 5. 当前结论

```text
style-balanced audit：✅ style50/coarse3 有效
layout-balanced audit：✅ support 正确
layout10：⚠️ 0.25–0.28，弱可分，低于 0.30
layout-balanced style50：❌ 只有 3 个 style，不能使用
same-layout coarse3：✅ 0.532，仍是可靠 style 控制
```

因此：

> **layout10 不适合作为 frozen branch 的唯一 sanity gate。**
>
> 更合理的下一步是直接用有效的 coarse3 head（离线 0.658）做 51 步 no-mask 在线验证；  
> layout 若要作为 sanity，需要重新定义成更可分的 layout 聚类/受控 layout 子集，而不是连续 ID 的 10 组。

## 6. warm-start 状态

- 右侧 agentview + frozen DINOv2 branch 已接入策略；
- 2-class layout head 已替换为 10-class layout-balanced head；
- coarse3 head 保留 style-balanced 版本（离线 episode bal 0.658）；
- `test_domain_head_branch.py` 已通过；
- layout10 head 离线只有 0.25–0.28，不建议拿它作为主 sanity。

下一步：

```text
coarse3 online no-mask 51 步
→ 在线 domain acc >0.50
→ 再加 mask / 3 seeds / 200 步
```
