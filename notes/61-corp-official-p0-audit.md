# 61 - CoRP 官方发布版 P0 保真审计

日期：2026-10-22
官方代码：`https://github.com/corp-policy/CoRP`
官方 commit：`52445740aa9517ae396cc8fb14bae8fe5922f118`
官方 checkpoint：`corp_libero.pt`
结果目录：`audit/corp_official_p0/`

## 1. 结论

```text
官方 checkpoint 可严格加载                         ✅
fp32 / bf16 GPU smoke                             ✅
真实 LIBERO rollout / success 判定                 ✅
官方配置中的架构参数可核实                          ✅
官方训练循环、优化器、batch/lr 配置可核实            ❌ 发布仓库没有
联合微调与 task auxiliary loss 可由代码证明          ❌ 只有论文声明
可以声称“复现了官方训练协议”                         ❌
可以声称“官方 LIBERO checkpoint 推理链路可用”         ✅
```

因此当前可以建立的是：

> **Official CoRP inference baseline on LIBERO**

还不能建立的是：

> **Official CoRP training baseline on RoboCasa**

后者必须明确写成 **CoRP adapted to RoboCasa / our reproduction**，除非官方补充训练代码。

## 2. 官方发布物事实

### 2.1 仓库范围

发布的 `README.md` 只提供 LIBERO 推理说明，RoboTwin2.0 仍是 `TODO`。仓库中没有：

```text
train.py
optimizer configuration
training dataloader
flow-matching training loss
fine-tuning recipe
task-classification auxiliary loss implementation
```

`corp/` 中只有推理所需的 model、checkpoint loader、preprocessing 和 action sampler。

### 2.2 Checkpoint

```text
size: 227,747,291 bytes
sha256: f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6
```

checkpoint payload keys：

```text
config_sha256
format
model
normalizer
```

没有 optimizer state、scheduler state 或训练 step。

### 2.3 参数量

```text
total:                      48,689,159
visual_encoder:             22,056,192
language_encoder:           11,570,816
cross_attention_resampler:   2,883,904
state_encoder:                   7,680
action_generator:            12,170,567
```

## 3. 官方发布的架构

来源：`checkpoints/config.json` 与 `corp/` 实现。

```text
image_size: 224
action_dim: 7
action_horizon: 12
state_dim: 8
hidden_dim: 768
views: agentview + eye_in_hand
```

### 3.1 视觉

```text
DINOv2 ViT-S/14
P=256 patch tokens/view
discard CLS token in extraction
multi-view shared weights
```

### 3.2 语言

```text
T5-Efficient-Tiny
max_length: 32
LayerNorm + projection to 768
language tokens 保留并拼接到视觉 token
pooled language 用于 FiLM 调制 resampler queries
```

FiLM 位置：

```text
pooled_language
→ Linear(768,64) → SiLU → Linear(64,1536)
→ gamma,beta
→ queries = queries * (1 + gamma) + beta
→ cross-attention resampler
```

### 3.3 Resampler

```text
tokens_per_view: 48
num_heads: 8
film_rank: 64
每个 view 有独立 learned queries + view embedding
```

### 3.4 State

```text
Linear(8,768) + learned state token offset
```

### 3.5 Action generator

```text
1-layer DiT
hidden_dim: 768
num_heads: 12
mlp_ratio: 4.0
adaln_rank: 64
max_horizon: 128
flow sampling Euler steps: 10
```

## 4. 官方 LIBERO 推理协议

来源：`scripts/evaluate_libero.py`、`corp/preprocessing.py`。

```text
camera render size: 256
image transform: 180° rotation → bilinear resize to 224
no center crop
settling steps: 10
action chunk: 12
Euler steps: 10
```

状态：

```text
eef_pos[3] + eef_rotvec[3] + gripper_qpos[2]
→ 8D
→ checkpoint state mean/std z-score
```

动作归一化：

```text
前 6 维 motion：checkpoint action_min/max 线性反归一化
第 7 维 gripper：保持模型输出
```

## 5. 真实 LIBERO 验证

### 5.1 Smoke

```text
fp32: parameters=48,689,159 action_shape=(1,12,7) smoke_ok
bf16: parameters=48,689,159 action_shape=(1,12,7) smoke_ok
```

### 5.2 真实 episode

`libero_spatial`，全部 10 tasks，每 task 1 episode，seed 7，bf16：

```text
task 0 success=1 steps=78
task 1 success=1 steps=111
task 2 success=1 steps=93
task 3 success=1 steps=84
task 4 success=1 steps=128
task 5 success=1 steps=98
task 6 success=1 steps=107
task 7 success=1 steps=105
task 8 success=0 steps=220
task 9 success=1 steps=117

success rate: 9/10 = 0.90
elapsed: 63.69 s
```

这个数字不能和论文的 50 episodes/task 结果直接比较，只证明官方推理与评测链路真实可用。

证据：

- `audit/corp_official_p0/corp_p0_audit.json`
- `audit/corp_official_p0/smoke_fp32.log`
- `audit/corp_official_p0/smoke_bf16.log`
- `audit/corp_official_p0/spatial10x1_summary.json`
- `audit/corp_official_p0/spatial10x1_episodes.jsonl`
- `audit/corp_official_p0/install.log`

## 6. 当前 RoboCasa 自定义 flow 与官方 CoRP 的差异

| 项目 | 官方 CoRP | 当前自定义 flow |
|---|---|---|
| 语言 | T5-Efficient-Tiny，32 token + pooled FiLM | CLIP-large 预计算 768 embedding |
| hidden width | 768 | resampler 64 / DiT 512 |
| resampler | K=48/view，768 维 | K=48/view，64 维 |
| action head | 1-layer DiT, 768, 12 heads | 12-layer DiT, 512, 8 heads |
| state | 8D | 9D |
| action | 7D, horizon 12 | 12D, horizon 10 |
| views | 2 | 3 |
| image | resize 224，无 crop | center crop 224 |
| benchmark | LIBERO | RoboCasa365 |

因此当前工程只能叫：

> **CoRP-like RoboCasa flow policy**

不能叫：

> **CoRP official baseline**

## 7. P1 适配边界

如果建立 `CoRP-RoboCasa`，只能修改接口：

```text
views 2→3
state 8→9
action 7→12
horizon 12→10
RoboCasa action/state normalizer
RoboCasa language instruction → T5 tokenizer
```

必须保留：

```text
DINOv2 ViT-S/14 + T5-tiny
K=48/view learned-query resampler
FiLM 在 resampler query 上
1-layer DiT, hidden 768
flow matching, 10 Euler steps
DINOv2/T5 联合训练
```

需要明确标记为适配差异：

```text
view 2→3：第 3 个 view query 没有官方 checkpoint 权重
action projection 7→12：没有官方权重
state projection 8→9：没有官方权重
RoboCasa normalizer：不是 LIBERO normalizer
```

因此 P1 不得声称“直接使用官方 checkpoint”。合理做法是：

```text
官方 DINOv2/T5/resampler 共同权重初始化
→ 新增/适配输入输出投影
→ RoboCasa 严格 split 重训
```

## 8. P0 最终判断

> **官方 CoRP 架构、checkpoint 和 LIBERO inference 已通过真实 episode 验证。**
>
> **官方训练协议没有随仓库发布，因此 CoRP-RoboCasa 训练必须标为我们的 reproduction/adaptation，不能标为 official training。**
>
> **下一步可以做 P1 接口适配，但不要先装 gate。**
