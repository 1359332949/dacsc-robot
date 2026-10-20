# Token-level 遮挡结果（2026-10-20）

## 设置

- flow K48 checkpoint
- 3 个固定 batch
- 每个观测步 145 tokens
- 只遮挡 144 个 visual tokens 中的 48 个（33%）
- low-dim/language token 不遮挡
- 同一 flow noise

## 结果

| 条件 | velocity MSE | 相对 random |
|---|---:|---:|
| full | 1.15084 | — |
| top-style token | 1.14845 | **-0.00034** |
| top-action token | 1.14893 | **+0.00014** |
| random token | 1.14879 | — |

## 解释

- 三种遮挡都只造成约 0.2% 的差异；
- top-style token 遮挡并不比 random 更伤动作；
- top-action token 也没有显著大于 random；
- 说明当前 K=48 token 有较强冗余，hard token dropout 不是有效干预。

## 决策

停止 hard token dropout / token hard gate。  
下一步转向 token statistics / TFS-ViT 风格 stylization，而不是继续做二值 token mask。

记录：

- `audit/token_occlusion_20261020.json`
- `audit/token_occlusion_20261020.csv`
