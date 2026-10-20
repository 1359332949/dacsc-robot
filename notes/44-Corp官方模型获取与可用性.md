# CoRP 官方模型获取与可用性（2026-10-20）

## 官方来源

- 项目页：https://corp-policy.github.io/
- 代码：https://github.com/corp-policy/CoRP
- 论文：https://arxiv.org/abs/2610.08183
- License：Apache-2.0
- 最新 commit：`52445740aa9517ae396cc8fb14bae8fe5922f118`

## 服务器路径

```text
/hy-tmp/models/corp/CoRP
/hy-tmp/models/corp/corp_libero.pt
```

权重：

```text
size: 227,747,291 bytes (~218 MiB)
sha256: f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6
```

GitHub LFS 在服务器上约 2.5 MB/s，约 1.5 分钟完成，不需要第三方镜像。

## 官方 smoke

```text
device=cuda:0 precision=fp32 parameters=48689159 action_shape=(1,12,7) smoke_ok
device=cuda:0 precision=bf16 parameters=48689159 action_shape=(1,12,7) smoke_ok
```

## 官方配置

```yaml
image_size: 224
action_dim: 7
action_horizon: 12
state_dim: 8
hidden_dim: 768
views: [agentview, eye_in_hand]
visual_encoder: DINOv2 ViT-S/14
language_encoder: t5-efficient-tiny
resampler: 48 tokens/view
action_generator: 1-layer DiT flow matching, 10 sampling steps
```

## 与 RoboCasa 的接口差异

| 项目 | CoRP LIBERO | RoboCasa365 当前 |
|---|---|---|
| action dim | 7 | 12 |
| action horizon | 12 | 10 |
| state dim | 8 | 9 (3 pos + 4 quat + 2 gripper) |
| views | 2 | 3 |
| language | T5-tiny | CLIP-large / 768 |
| image | 224 | 256 crop 224 |
| action head | 1-layer DiT, 768 | 12-layer DiT, 512 |
| action normalization | LIBERO 6D/relative | RoboCasa 12D |

## 结论

可以**直接获取并运行官方 CoRP 模型**，但不能直接把 LIBERO checkpoint 当 RoboCasa 策略推理。

可直接复用：

- DINOv2 ViT-S/14 visual encoder 权重；
- cross-attention resampler 结构；
- T5 language encoder / FiLM 机制；
- 1-layer DiT flow-matching action generator 代码；
- 官方 checkpoint 作为 LIBERO baseline。

需要改：

- action projection 7→12；
- state projection 8→9；
- view queries 2→3；
- action horizon 12→10；
- RoboCasa action/state normalizer；
- 训练数据改成 RoboCasa365。

因此正确路线是：

```text
CoRP official baseline on LIBERO
→ 适配 RoboCasa 接口
→ 用 RoboCasa 微调/重训 action head
→ 再验证 token-level gate
```

不要直接把当前 dimension gate 装到 CoRP 上。
