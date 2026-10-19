# XPolicyLab 是什么（由浅入深）

> 2026-10-16。全部结论来自本地源码阅读（`repos/XPolicyLab`，7795 个文件）。

## 第 0 层：一句话

**XPolicyLab 是机器人策略的「USB-C 标准 + 转接头集合」。**

- 以前：N 个策略模型 × M 个评测环境，要写 N×M 份对接胶水代码
- 现在：定义**一个统一接口**，策略侧装一个转接头、环境侧装一个插座，就能互相插
- 官方原话：把 **O(N×M) 降到 O(N+M)**

具体数字：63 个策略 × 3 类环境（RoboTwin / RoboDojo / 真机）= 189 份对接工作 → **66 份**。

## 第 1 层：它由三部分组成

```
XPolicyLab/
├── policy/                 ← ① 69 个策略文件夹
│   ├── DP/                 ←    每个 = 完整模型代码 + 一个薄壳
│   ├── Pi_0/  GR00T_N17/  ACT/  SmolVLA/  Awomo05/ ... （共 69 个）
│   └── demo_policy/        ←    最小参考示例
│
├── client_server/          ← ② 通信层（websocket RPC）
│   ├── ws/protocol/        ←    通信契约（schemas.py 93 行 + messages.py 35 行）
│   ├── ws/model_server.py  ←    服务端（635 行）
│   ├── ws/model_client.py  ←    客户端（219 行）
│   └── tcp/                ←    TCP 版（可选）
│
└── model_template.py       ← ③ 转接头接口定义（120 行，7 个方法）
```

**注意第三块不在这个仓库里**：环境侧（仿真器怎么接）的代码在**基准仓库**里，比如
`repos/RoboTwin/scripts/eval_policy_xpolicylab.py`（1444 行）。

## 第 2 层：一个策略"文件夹"里到底是什么

以 DP 为例：

```
policy/DP/
├── model.py                (141 行)  ★ 转接头：继承 ModelTemplate，实现 7 个方法
├── deploy.py               (41 行)   部署入口
└── diffusion_policy/                 ★ 完整的一份 Diffusion Policy 代码（自己的副本）
```

所以一个"策略适配器" = **完整模型代码 + ~140 行薄壳**。

## 第 3 层：薄壳要实现哪 7 个方法（这就是全部）

读 `model_template.py`：

| 方法 | 作用 |
|---|---|
| `update_obs(obs)` | 收单帧观测 |
| `update_obs_batch(obs_list)` | 批量收观测 |
| `get_action()` | 出动作 |
| `get_action_batch(env_idx_list)` | 批量出动作 |
| `reset()` | 新 episode 开始，清内部状态 |
| `prepare_case(case_meta)` | 可选钩子（一次 case 开始前） |
| `on_trial_end(result)` | 可选钩子（一次 trial 结束后） |

**XPolicyLab 的价值不在模型，就在这 7 个方法的约定 + 底下那条 websocket 通路。**

## 第 4 层：一次推理的完整生命线

```
   仿真器进程（环境侧）                     策略服务进程（XPolicyLab 侧）
   ─────────────────────                   ──────────────────────────────
   render 出图像
        │
        ├── RESET  frame ─────────────────────►  adapter.reset()
        │
        ├── INFER  frame ─────────────────────►  解码图像
        │   payload = {                            adapter.update_obs()
        │     vision: {3 个相机 ...},               adapter.get_action()
        │     state:  {...}                    │
        │   }                                  │
        │◄───── INFER_RESULT frame ─────────────┤
        │      payload = {action: [...]}        │
        │
   把 action 施加给机器人
```

通信契约是 `client_server/ws/protocol/schemas.py` 里的 `Frame`：

```python
class Frame(BaseModel):
    message_type: MessageType      # hello / reset / infer / trial_end / heartbeat / ...
    request_id: str
    evaluation_id: str
    action_case_id / trial_id / repeat_index / step
    payload: dict                  # 真正的数据（图像、状态、动作）都在这里
```

**为什么要跨进程**：63 个模型的依赖互相冲突（有的要 torch 2.5，有的要 flash-attn），
环境隔离（各自 conda/uv 环境）+ websocket 通信，就不用在同一个进程里塞 63 套依赖。
还能**跨机器**：策略服务器放在 GPU 机上，仿真在另一台。

## 第 5 层：它在我们项目里能干什么、不能干什么

| 我们要的东西 | 谁提供 | XPolicyLab 有吗 |
|---|---|---|
| 数据 + 域标签（`style_id`） | RoboCasa365 | ❌ **完全不含** |
| 仿真环境 + 闭环评测 | RoboCasa365 / robosuite | ⚠️ 只有 RoboTwin / RoboDojo 的 client，**没有 RoboCasa365 的** |
| **我们的方法（通道门控）** | **要自己写** | ❌ 它不含任何域泛化方法 |
| 基线 DP | 官方 RoboCasa fork | ⚠️ 有一份 `policy/DP/`，但**是它自己的版本**，不是我们的 RoboCasa fork |
| 多策略对比（π0/GR00T/ACT/SmolVLA…） | XPolicyLab | ✅ **这才是它唯一不可替代的价值** |
| 统一评测协议（可复现/上榜） | XPolicyLab | ✅ |

### 一个容易踩的坑

`XPolicyLab/policy/DP/` **不是**我们的基线。
- 我们的基线（`repos/diffusion_policy`）= RoboCasa 官方 fork：3 相机 256×256、LeRobot 数据、`ResNet18ConvFiLM` + 768 维语言
- XPolicyLab 的 DP = 它自己那份通用 DP

**两者不能混用**，我们的门控是插在 `ResNet18ConvFiLM` 上的。

## 结论：XPolicyLab 是"后期加分项"，不是"起步必需品"

| 阶段 | 需不需要 XPolicyLab | 理由 |
|---|---|---|
| **第一阶段**：验证我们自己的方法（DP vs DP+门控） | ❌ **不需要** | 对手是"不加门控的 DP"，不是 π0。用官方 fork + RoboCasa365 就够 |
| **第二阶段**：证明机制可迁移 / 投稿 | ✅ **需要** | 用来在 π0 / GR00T 上跑同一套门控（方法文档 §8.3 的 K 行） |

**用它要付的代价**：写一个 RoboCasa365 的 env client。参照物现成——
`repos/RoboTwin/scripts/eval_policy_xpolicylab.py`（1444 行）就是"环境侧该怎么接"的完整范例。

## 想真懂它，按这个顺序读（总共约 2000 行，一天能读完）

| 顺序 | 文件 | 行数 | 看什么 |
|---|---|---|---|
| 1 | `model_template.py` | 120 | 7 个方法的接口约定 |
| 2 | `policy/DP/model.py` | 141 | 一个真实转接头长什么样 |
| 3 | `client_server/ws/protocol/schemas.py` | 93 | 通信契约 `Frame` |
| 4 | `client_server/ws/protocol/messages.py` | 35 | 7 种消息类型 |
| 5 | `client_server/ws/model_client.py` | 219 | 客户端怎么发请求 |
| 6 | `repos/RoboTwin/scripts/eval_policy_xpolicylab.py` | 1444 | 环境侧怎么接（最关键） |

**中文读者提示**：仓库里有 `README_zh.md`（36KB，中文完整版），先看它。
