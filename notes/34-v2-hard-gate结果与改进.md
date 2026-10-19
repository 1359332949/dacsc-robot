# v2 hard gate 结果与改进（2026-10-19）

## 已实现

- per-sample top-k hard mask（33%）
- straight-through
- mask 重归一化
- 训练时 dropout，推理默认关闭
- 单层 probe（layer 11）
- DomainDrop-style 10 类粗域分类器（消融）
- DgCD per-sample 分布差异打分（当前主线）
- domain-balanced batch sampler（每 batch 覆盖 10 个粗域）
- 动作重要性进入 mask 分数
- gated/ungated 特征一致性
- action importance EMA 修复：只更新当前层，未选层保留原值
- CLIP lang_emb 缓存
- checkpoint 保存 `domain_head/action_importance/gate_step`

## 200 步 classifier 版（balanced）

输出：`/hy-tmp/runs/hard200_balanced`

```text
mask suppression: 33.07%
domain_acc: 9.4% -> 17.2%
dp_loss: 1.196 -> 0.311
aux_loss: 0.236 -> 0.324
```

结论：mask 和 DP 正常；10 类域头偏弱。

## 200 步 DgCD 版（balanced）

输出：`/hy-tmp/runs/hard200_dgcd`

```text
mask suppression: 33.07%
dp_loss: 1.194 -> 0.311
aux_loss: 0.0 -> 0.036（仅一致性项）
consistency: 0.0 -> 0.359
```

DgCD 模式不再依赖弱分类器；但日志里的 `domain_acc` 不适用（没有分类头），
需要下一步用 mask 的跨域重叠率来评价域选择性。

## 判断

✅ 训练链路和 hard mask 已正常
✅ 33% 强扰动稳定
✅ DP loss 正常下降
✅ 动作重要性非零
⚠️ 尚未证明不同 style 的 mask 具有明显差异
⚠️ 固定 layer 11 只是 probe，不是最终随机层版本

## 下一步（不直接扩 2k/20k）

1. 用固定平衡 batch 计算不同 style 的 mask IoU / 选择差异；
2. 比较 `action_lambda=0` 与 `action_lambda=1`，证明动作保护有效；
3. 通过后把 `hard_layer_indices` 改成 `null`，启用随机单层；
4. 再做 2k pilot，最后才考虑 20k。

## 当前不建议

- 不要直接启动 20k。
- 不要把 classifier 版的 17% 域准确率当成方法失败或成功。
- 不要把 DgCD 版的 `domain_acc` 当指标（它没有分类头）。
