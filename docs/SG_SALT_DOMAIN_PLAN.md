# SG-SALT 评测域规划：不只 Camera（7 个扰动的因果分类与实验选择）

> 结论先行：**Camera 只是 7 个扰动之一，而且它从 episode 开始就是静态的，天然不适合 SALT 的
> "相邻 replan 变化" gate；主实验应放在"视觉外观类"且有成功率余量的域上（Camera / Background / Noise），
> Light 作为高基线对照；Robot / Layout / Language 属于状态/语义变化，不进入主结论。**

## 1. 7 个扰动的因果分类

| 扰动 | 改变了什么 | 与"接口 TTA"的关系 | 是否进入 SG-SALT 主实验 |
|---|---|---|---|
| **Camera**（视角） | 观测坐标系 | 观测变化；但从 episode 开始静态 → change-gate 失效 | **主实验（用固定触发）** |
| **Background**（背景纹理） | 观测外观 | 观测外观变化，SALT 可直接利用 leftover | **主实验** |
| **Sensor Noise**（高斯噪声） | 观测质量 | 观测外观变化，且与"自洽信号"关系最直接 | **主实验** |
| **Light**（光照） | 观测外观 | 外观变化，但基线已高（80%）→ 提升空间小 | 对照（high-baseline control） |
| **Objects Layout**（布局） | 世界本身 | 物体位置变化，不是观测误差 | 不进入主结论（单独分析） |
| **Robot Initial State** | 机器人真实状态 | 状态变化，proprioception 直接可见 | 不进入主结论 |
| **Language**（指令） | 任务语义 | 语义变化，与 TTA 无关 | 排除 |

**理由**：SALT/SG-SALT 是**观测侧**的测试时适配；对"世界本身/状态/语义"变化，正确做法是状态条件化/语言 grounding，
不是接口适配。把它们混进主表会让结论不可解释。

## 2. 各域冻结基线（120 任务采样，episode 0）

来自本项目全量官方基线（10,030 episodes）匹配：

| 域 | 匹配任务 | 冻结成功 | 成功率 | 分套件（10/goal/object/spatial） |
|---|---|---|---|---|
| Camera | 120 | 52 | **43.3%** | 13 / **1** / 19 / 19 |
| Background | 120 | 46 | **38.3%** | 8 / 9 / 15 / 14 |
| Noise | 120 | 43 | **35.8%** | 9 / **2** / 21 / 11 |
| Robot | 120 | 46 | 38.3% | 13 / 4 / 13 / 16 |
| Light | 120 | 96 | 80.0% | 13 / 26 / 30 / 27 |
| Layout | 120 | 85 | 70.8% | 24 / 12 / 25 / 24 |
| Language | 120 | 93 | 77.5% | 27 / 8 / 30 / 28 |

**观察**：
- **goal 套件在 Camera/Noise 下接近全失败**（1/30、2/30）→ 不适合做配对比较；
- **spatial 与 object** 在 Camera/Background/Noise 下都有"部分成功"，适合做配对筛选；
- Light 基线 80%，提升空间小；可作"高基线对照"验证适配不会伤害好条件。

## 3. 推荐实验矩阵（Phase 3 小规模筛选前）

### 主条件（3 个视觉外观域 × 2 个套件）

| 域 | 套件 | 候选任务数（30 中 frozen 成功数） |
|---|---|---|
| Camera | spatial / object | 19 / 19 |
| Background | spatial / object | 14 / 15 |
| Noise | spatial / object | 11 / 21 |

每格选 **3 个任务**（优先 frozen 成功率在 ~20–80% 的任务，保证有翻转空间），
每任务 **5 个配对 episode**（同 seed、同初始状态、同扰动）：

```
3 domains × 2 suites × 3 tasks × 5 episodes = 90 paired episodes / method
× 3 methods (Frozen / SALT / SG-SALT) = 270 rollouts
```

### 对照条件
- Light（spatial/object，3 任务 × 5 episodes）：验证高基线域不被适配伤害（clean-like safety proxy）。

### 不进入主表
- Robot / Layout / Language：单独报告为"非观测侧扰动，不适用接口 TTA"的边界分析。

## 4. 触发条件（与域无关的统一设置）

- 首轮 SG-SALT 验证：**固定触发（always-on）**，避免 gate 变量混淆；
- 对 **Camera 这类静态扰动**：change-gate 原理上不触发（Phase 1.2 已实测 gate 分数 median 0），
  因此必须在文档中声明"本实验使用固定触发，不代表 SALT 原版 gate 行为"；
- 后续若要恢复 gate：只在"mid-episode shift（A′）"协议下有意义；静态域用 episode 级 shift 检测，另行研究。

## 5. 评测与统计

- 配对：同任务、同初始状态、同 seed、同扰动；
- 指标：成功率、配对成功率差、McNemar、任务级 cluster bootstrap、clean/高基线域非劣效；
- 同时记录：β 饱和比例、每次适配耗时、更新参数量；
- 不以单一域（尤其 Camera）的成功率作为唯一结论。

## 6. 对 SG-SALT 的直接含义

1. **保护项 A 来自 Camera/spatial 的写回统计**：在 Background/Noise 上是否同样成立需要记录，
   但 A 定义在接口空间，与扰动类型无直接绑定；这本身是一个可检验点（跨域使用同一 A）。
2. 若 A 只在 Camera 上有效、在其他域无效 → 结论应写成"Camera-specific protection"，不能泛化。
3. 主实验至少 **2 个非 Camera 的视觉外观域**，否则不能支撑"泛化到视觉扰动"的结论。
