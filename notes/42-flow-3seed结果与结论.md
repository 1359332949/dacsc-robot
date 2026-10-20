# Flow K48 3-seed 结果与结论（2026-10-20）

## 实验

- 新主基线：DINOv2 ViT-S/14 + learned-query resampler K=48 + DiT flow matching
- 条件 token：每观测步 `1 + 3×48 = 145`，To=2 → 290
- groups：flow_no_mask / flow_random / flow_domain
- seeds：42/43/44
- 每个 51 步，step50 gate probe

## 结果

| group | drop gap | sample gap | gate on/off | v raw rel std |
|---|---:|---:|---:|---:|
| flow_no_mask | — | — | — | — |
| flow_random | +0.0000 ± 0.0007 | +0.0002 ± 0.0008 | **-5.68% ± 0.51%** | 0 |
| flow_domain | **+0.5095 ± 0.0091** | **+0.7453 ± 0.0056** | **+3.63% ± 1.44%** | 0.376 ± 0.017 |

flow_no_mask 的 `train_action_mse_error`：

```text
0.8098, 0.8085, 0.6325
mean = 0.7503
```

这个值与 gate probe 的 `gate_off_mse` 不是同一个计算路径，不能直接横比。

## 结论

1. K-token 架构和 flow matching 都已经正确运行；
2. flow 下域结构仍然很强；
3. **但 domain gate 在 flow 下使动作 MSE 变差 +3.6%**；
4. random mask 反而改善 -5.7%；
5. 因此 flow 主基线的门控性能验收 **不通过**。

当前不能进入 200 步或 2k。下一步应在 DDPM 阶段一继续验证门控，或重新设计 flow 下的门控接口，而不是继续调 51 步的 loss。

## 失败原因假设

- flow matching 的 velocity target 与 DDPM epsilon target 的噪声尺度不同；
- 当前 gate 的 hard dropout/renormalization 可能不适合 flow 的连续速度场；
- `two_stage` 动作保护是基于 DDPM 的动作梯度统计，迁移到 flow 后需要重新校准；
- flow 的动作头更强，可能已经把 shortcut 吸收到动作头，视觉 token gate 的边际收益下降。

这些只是假设，尚未验证。
