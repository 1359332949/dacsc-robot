# SG-SALT 任务书公式更正（重要）

此前任务书中的 Flow Matching 方向：

```
X_tau = (1-tau) * Y + tau * eps      v* = eps - Y      （错误：与 CoRP sampler 相反）
```

**自 7c5452a 起统一采用 CoRP-native 方向（已由解析积分测试验证）**：

```
X_tau = (1-tau) * eps + tau * Y      v* = Y - eps
```

理由：CoRP `DiTActionGenerator.sample` 从 `randn` 出发、`tau=step/N`、`action += v/N`，
即 τ=0 是噪声、τ=1 是数据；训练目标必须与推理采样方向一致。

任何后续交给其他 AI/成员的任务书必须使用上述 CoRP-native 公式；当前算法代码不再修改。
