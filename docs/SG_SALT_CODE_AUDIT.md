# 工单 0：CoRP / SALT / 现有代码 逐行审计（Phase 0，只读）

> 状态：**只读审计，未修改模型或实验代码**。本文件回答"我们改哪个文件、改哪一行、公式对应什么"；
> 未读到的代码不写结论。所有引用均指向源码路径:行号。

## 0. 环境与 checkpoint 映射（必须先固定，不能混用）

| 执行链 | 模型 | checkpoint | 环境入口 |
|---|---|---|---|
| 本项目 TTA 研究（本文档使用） | CoRP（LIBERO 版） | `repos/CoRP/checkpoints/corp_libero.pt` | `repos/CoRP/scripts/evaluate_libero.py` / 我们封装 `scripts/libero_shift_eval.py`、`scripts/salt_eval.py` |
| 仓库 README 主链（历史） | CoRP-RoboCasa | 本地 RoboCasa checkpoint（不在本审计范围） | `scripts/eval_corp_robocasa*.py` |

- `checkpoints/config.json` 与 `corp_libero.pt` 同目录；`SHA256SUMS` 存在（`repos/CoRP/checkpoints/`）。
- **Phase 1 只用 LIBERO + `corp_libero.pt`**；RoboCasa 链不参与本方法实验。

## 1. CoRP 结构事实（实测，非推测）

`load_policy(corp_libero.pt)` 读出：

| 项 | 值 |
|---|---|
| 总参数 | **48.69 M** |
| hidden_dim | 768 |
| action_dim / horizon | 7 / 12 |
| state_dim | 8 |
| view_names | `['agentview', 'eye_in_hand']` |
| DiT blocks | **1** |
| num_heads / mlp_ratio / adaln_rank | 12 / 4.0 / 64 |
| num_sample_steps | 10 |
| SelfAttention 参数 | `qkv.weight [2304,768]`、`qkv.bias [2304]`、`projection.weight [768,768]`、`projection.bias [768]` |
| Resampler | `CrossAttentionResampler`：`queries [2,48,768]`、`view_embedding [2,1,768]`、`input_projection [768,384]` |

模块定义位置：
- `repos/CoRP/corp/policy.py:11`（CoRP 类）、`:22`（`predict`）
- `repos/CoRP/corp/representation/extractor.py:21`（`RepresentationExtractor`）、`:39`（`forward`）
- `repos/CoRP/corp/representation/cross_attention_resampler.py:9`（Resampler）、`:41`（`forward`）
- `repos/CoRP/corp/action_generator.py:36`（`SelfAttention`）、`:53`（`attend`）、`:91`（`DiTBlock`）、`:106`（`encode_prefix`）、`:128`（`DiTActionGenerator`）、`:154`（`prepare_prefix`）、`:169`（`forward`）、`:182`（`sample`）
- `repos/CoRP/corp/preprocessing.py:34`（`unnormalize_action`）

## 2. 调用链与张量 shape（从观测到动作）

```
policy.predict(observation)                         policy.py:22
  prepare_observation(...)                          preprocessing.py:42   -> images bf16, state 归一化
  RepresentationExtractor.forward                   extractor.py:39
      visual_encoder(images)                        -> patch tokens
      language_encoder(instruction)                 -> language_tokens, pooled_language [B,768]
      cross_attention_resampler(patch_tokens, pooled_language)  -> visual_tokens [B, 96, 768]
      cat([visual_tokens, language_tokens]) + state_token
                                                    -> tokens [B, T_rep, 768]（T_rep≈129）
  DiTActionGenerator.sample                         action_generator.py:182
      action = randn(B, 12, 7)
      prepare_prefix(representation)                action_generator.py:154
          block.encode_prefix(tokens, pooled_language, mask)   :106
             attention.attend(...)   <-- call 0（T_rep 长度，prefix；不可改）
             -> PrefixCache(key,value,adaln)
      for step in range(10):                        :199-215
          tau = step/10
          v = self(action, tau, prefix)             :169
              action_projection(action)+position+time_embedding
              block(action tokens 12x768, prefix)   :115
                  attention.attend(...)  <-- call step（T=12；写回发生处）
              velocity_projection                    -> [B,12,7]
          action = action + v * (1/10)
  normalizer.unnormalize_action(actions)            preprocessing.py:34
```

**"写回接口"的准确定义（我们此前研究的 A 所在）**：
`action_generator.blocks[0].attention.attend` 的**返回值**，shape `[B, 12, 768]`；
最后一次 Euler 调用的 `mean over 12 tokens` 得到 `w ∈ R^768`。prefix 调用（call 0）绝不能改。

## 3. Flow Matching 目标：官方仓库只有推理，没有训练 loss

- 官方 repo `corp/` 内**没有训练代码/损失函数**（只有 `policy.py` 推理与 `action_generator.py` 采样）。
- 训练目标须按论文公式实现（SALT 论文 Eq.）：
  - `X_τ = (1-τ) Ỹ + τ ε`
  - `v_target = ε - Ỹ`
  - `L = E[ || M ⊙ ( v_θ(X_τ, τ, o, s, g) - v_target ) ||² ]`
- 与官方采样循环的**时间方向对应**：采样用 `tau = step/N`、`action += v/N`，即 τ 从 0 到 (N-1)/N；
  实现 loss 时必须用同一约定（τ~U[0,1) 或与采样对齐），并用单元测试核对。
- `M` 只作用于 leftover 对齐的时间位置（`chunk[0:H-E]` vs `previous_chunk[E:H]`）。

**动作空间**：模型输出（`velocity`/`sample`）在**归一化动作空间**；`unnormalize_action` 只对前 6 维
做 `[low,high]` 变换，**gripper 维（第 7 维）不变**（`preprocessing.py:34-40`）。
→ SALT loss 应在**归一化空间**计算（与模型输出一致），不能直接拿 env 单位动作做 MSE。

## 4. SALT 需要更新的参数（CoRP 上的候选）

SALT 冻结主干、只更新选定注意力投影，使用 bounded multiplicative singular-value scaling。
CoRP 里可选的 Linear：
1. `block.attention.projection`：`W [768,768]`（对 attend 输出做投影）
2. `block.attention.qkv`：`W [2304,768]`（**fused q,k,v**，缩放会同时作用于三者）

不确定点（需以 SALT 原文/附录核实，不能猜）：
- SALT 更新的是 q/k/v/out 投影中的哪些；
- 是否对 fused qkv 分开处理；
- 缩放上界与初始化（我们的实现取 `s=1+bound*tanh(phi)`，init=0 严格恒等）。

## 5. 动作敏感子空间 A 的定义（现有资产）

`scripts/build_interface_stats.py`：
- `WriteHook`（:38-77）挂在 `blocks[0].attention.attend`；prefix 调用重置计数，最后 Euler 步
  记录 `mean over tokens` 的 `w ∈ R^768`（:65-66）。
- 有限差分（:150-168）：在 `w` 上加 `eps * d`（d 为单位方向），固定 RNG，测动作块变化 / eps；
  `J = lstsq(pushes, responses)`（:192-199），SVD 取能量 90% 的前 r 个方向 → `action_subspace [768, r]`。

**空间说明（关键）**：
- A 定义在**写回空间（R^768，token 均值）**；
- CPIA 的低秩变换直接作用在写回（`dacsc/leftover_tta.py:29` 的 `_apply`）；
- SALT 更新的是**参数空间**（qkv/projection 权重）。

→ **不能把 A 直接乘到 SALT 的参数更新上**。SG-SALT 的保护项必须先测量"参数更新引起的写回变化
`Δz = z_adapted − z_frozen`（同一观测、同一 RNG、同样 token 均值/最后 Euler 步）"，再在写回空间
施加 `λ||P_A Δz||²`。这一步目前**未实现、未验证**。

## 6. 现有可复用代码（已读）

| 资产 | 位置 | 复用方式 |
|---|---|---|
| policy loader / checkpoint 校验 | `dacsc/corp_robocasa.py`、`dacsc/upstream.py` | 直接用 |
| `sample_with_grad` | `dacsc/leftover_tta.py:88` | 复刻自官方 `sample`（:182-215），需单元测试证明恒等 |
| leftover loss（旧，env 单位） | `dacsc/leftover_tta.py:108` | **不用于 SALT**；SALT 用 Flow Matching loss |
| CPIA 低秩接口变换 | `dacsc/leftover_tta.py:29` | 参数高效更新的对照实现 |
| episodic reset | `dacsc/leftover_tta.py:85`、`scripts/leftover_tta_eval.py` | 复用重置逻辑 |
| 统计工具 | `scripts/analyse_matrix.py` | Newcombe + cluster bootstrap + paired/cluster TOST |
| A 子空间文件 | `/hy-tmp/validation/interface_stats_spatial_k32_fixed.json` 等 | SG-SALT 候选保护方向 |

## 7. 审计中发现的代码问题（Phase 1 必须先修）

1. **`scripts/salt_eval.py` 崩溃**（实测日志）：
   `RuntimeError: Input type (unsigned char) and bias type (c10::BFloat16) should be the same`
   原因：直接把 `batch`（uint8 图像）送 `representation_extractor`，未走 `prepare_observation`
   （正确做法见 `policy.predict` `policy.py:30-37` 或 `leftover_tta_eval.py` 的 prepared 路径）。
2. `sample_with_grad` 与官方 `sample` 的恒等性**尚未做单元测试**（需固定 RNG 比较）。
3. leftover 目标当前是 env 单位（旧 CPIA 实现）；SALT 需要归一化空间 + mask，禁止混用。
4. `SaltTTA` 目前只实现了参数选择（projection/qkv）与恒等初始化；**Flow Matching loss、mask、gate 未实现**。

## 8. Phase 1 最小文件清单与验收

新增/修改（Phase 1 才允许动代码）：
- `dacsc/salt_tta.py`：补 `flow_matching_leftover_loss(...)`（归一化空间 + mask + τ/ε）
- `scripts/salt_eval.py`：修 prepare_observation；接入 flow loss；实现 frozen/salt/self-training 三模式
- `tests/test_salt.py`：恒等性、梯度只到 phi、τ/ε 方向、leftover 对齐、episodic reset

验收（全部通过才允许进入 Phase 2）：
1. Frozen 与官方推理逐条一致（10/10）；
2. 零初始化时 SALT 输出与 Frozen **逐条一致**（已在审计中用 `projection` 验证 `max|d|=0.0`）；
3. 只有 `phi` 有梯度；`torch.autograd.grad` 检查冻结参数为 None；
4. `sample_with_grad` 与官方 `sample` 在同 RNG 下 `max|d|=0`；
5. leftover 索引：`chunk[0:H-E]` 对应 `previous_chunk[E:H]`（构造已知张量验证）；
6. 每 episode reset 后 `phi=0`、optimizer 新实例；
7. 动作范围：unnormalize 后与 Frozen 同范围。

## 9. 未决问题（需读 SALT 原文/附录确认，不猜）

1. SALT 的 gate 具体判别量与阈值；
2. SALT 更新参数集合（q/k/v/out 全更新还是仅 out projection）；
3. `M` 的 action 维度 mask（7 维是否全监督；gripper 是否特殊）；
4. SALT 是否在归一化空间定义 Ỹ（我们推断应为归一化空间）；
5. 更新后是否重新生成 chunk 再执行（我们按 SALT 描述实现：更新→重新采样→执行）。
