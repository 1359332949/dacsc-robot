# B2 报告：候选更新 vs 不更新的成对反事实（第一轮）

> **后续状态**：第一轮的 save/restore 配对已被预检否决（恢复误差 1e-5 m，control vs zero 4×6 轨迹差 0.043）；
> 双环境重放被验证为逐位确定。正式协议见 `SG_CPA_B2_PROTOCOL_V3_REPORT.md`，正式实验待授权。

> 授权范围：模拟器内成对反事实分支，非物理 rollout；不调 η、不改 A-projection、不做适配训练。
> 代码：`scripts/sg_b2_counterfactual_v2.py`（单环境状态保存/恢复，消除跨环境非确定性）。
> 数据：`/hy-tmp/validation/b2v2/b2.jsonl`（288 行 = 24 episodes × 2 branches × 3 candidates × 2 modes）。

## 1. 协议与代码校验（先于实验）

- 分支点：固定 replan **2 和 4**；每个分支：
  1. 从同一状态 `env.sim.get_state()` 保存；
  2. 用**同一观测、同一 action latent seed** 采样 control chunk（β=0）与 update chunk（β=Δβ）；
  3. 分别 `set_state` 后执行完整 12 步 chunk；
  4. 记录 object→target proximity progress `p_before/p_after`、success。
- 候选更新（由同一梯度/J_A 计算）：
  - `proj_005 = −0.05 P_null g`
  - `full_005 = −0.05 g`
  - `sens_005 = −0.05 (I−P_null) g`
- **预检（zero update）**：24 个分支对中 control 与 zero-update 的 `dp` 差 **0.0**、动作块差 **0.0**，
  说明单环境 save/restore、latent 与动作配对正确，排除了跨环境非确定性。

## 2. 结果

| candidate | n | mean τ | frac τ>0 | dp_control | success control/update | action change (update vs control) | ‖Δβ‖ |
|---|---|---|---|---|---|---|---|
| proj_005 | 48 | **0.0** | 0.0 | 1.69e-12 | 0 / 0 | 1.56e-6 | 5.3e-5 |
| full_005 | 48 | **0.0** | 0.0 | 1.69e-12 | 0 / 0 | 3.93e-6 | 8.2e-5 |
| sens_005 | 48 | **0.0** | 0.0 | 1.69e-12 | 0 / 0 | 2.86e-6 | 6.2e-5 |

（τ = Δp_update − Δp_control；所有 144 对的 τ 精确为 0，dp 量级 1e-12。）

## 3. 诊断：为什么 τ 测不到

1. **动作变化量级正确但太小**：update/control 动作块差 1.6e-6–3.9e-6，与 η=0.05 的 β~5e-5 一致；
   但 proximity 是 object→target 距离，分支点 replan 2/4（约 12–24 步）时物体**尚未被抓起**，
   object 位置在 12 步内几乎不变 → `dp ≈ 1e-12`，τ 落在数值噪声内。
2. **分支窗口太早/太短**：12 步 horizon 内 control 与 update 都未产生可测的物体位移；
   因此 τ=0 不是"更新无益"的证据，而是**当前测量窗口对该候选尺度不敏感**。
3. success 全 0 也符合"早段执行 12 步不可能完成任务"。

## 4. 结论

> 在 η=0.05 的单步候选更新、分支点 2/4、H=12 步的设置下，更新对 object→target proximity 的
> 反事实效应低于测量分辨率（τ≡0）。这**不能**用来判断更新有益/有害，只说明当前 B2 设计
> 没有覆盖"更新能影响动作"的时段与幅度。

## 5. 下一轮 B2 修订（需授权）

1. **分支点**：固定为 replan 8/12（物体已被抓取/移动的阶段），并对短 episode 记录跳过原因；
2. **horizon**：H=24 步（两个 chunk），允许物体位移累积；
3. **候选更新**：预注册 η∈{1, 10, 100}（proj/full/sens 三族），覆盖 action change 1e-4–3e-3；
4. 仍然：单环境 save/restore、配对 latent/RNG、先 zero-update 预检、不调正式方法参数；
5. 评价：τ 分布、τ>0 比例、信号（多视图/历史连续性）对 τ>0 的预测（探索性）。

**本轮结论只作为"测量窗口无效"的证据；不进入任何 rollout 或方法结论。**

## 6. 交付

- 代码：`scripts/sg_b2_counterfactual_v2.py`、`scripts/analyse_b2.py`
- 数据：`/hy-tmp/validation/b2v2/`；预检：`/hy-tmp/validation/b2v2_pre/`
- 本报告；commit 见本次提交。
