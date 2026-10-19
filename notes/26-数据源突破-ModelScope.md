# 数据源突破：ModelScope 上就有完整的 RoboCasa365（2026-10-19）

> 你找到的这个链接是**关键突破**：不用 Box、不用翻墙、不用手动上传，
> AutoDL 上直接下，**22.6 MB/s**。

---

## 一、这是什么

**数据集**：`nv-community/robocasa365-datasets`
https://www.modelscope.cn/datasets/nv-community/robocasa365-datasets

| 项 | 值 |
|---|---|
| 下载量 | **55,950** |
| 许可 | **CC-BY-4.0**（与官方一致） |
| 内容 | RoboCasa365 v1.0 的**人类示范**（LeRobot 格式） |
| README | 与官方 `datasets_overview` 一字不差（300 任务 / 2500 场景 / 482 小时） |

**它是官方数据集的完整镜像**，不是第三方转换。

---

## 二、⭐ 最关键的一点：**`ep_meta.json` 在**

我下载了 `OpenDrawer` 实测，包内结构：

```
lerobot/
├── data/chunk-000/episode_XXXXXX.parquet
├── videos/...（3 路相机）
├── meta/...
└── extras/episode_XXXXXX/
     ├── ep_meta.json      ← ★★ 在这里
     ├── states.npz
     └── model.xml.gz
```

`ep_meta.json` 内容实测：
```json
{"layout_id": 16, "style_id": 17,
 "object_cfgs": [...], "fixtures": {...},
 "gen_textures": {}, "lang": "Open the left drawer.", ...}
```

**`style_id` / `layout_id` 都在** → 我们的方法有原料了。

> 对比：HF 镜像上的那些 RoboCasa 数据集**全都没有 extras/**，所以不能用。
> ModelScope 这个是**原样镜像**，所以 extras 完整保留。

---

## 三、体积与速度（实测）

| 类别 | 包数 | 体积 |
|---|---|---|
| **pretrain/atomic** | **61** | **5.63 GB** ← 我们要的 |
| pretrain/composite | 240 | 107.81 GB |
| target/atomic | 18 | 9.17 GB |
| target/composite | 32 | 52.37 GB |
| **合计** | **351** | **175 GB** |

**速度实测**：71.3 MB 用 **3.2 秒**（22.6 MB/s）；解析后另一次 71 MB 用 6.2 秒（11 MB/s）。

**所以**：
- 全部 atomic 预训练（5.63 GB）≈ **4–8 分钟**
- 全部 target（61.5 GB）≈ **45–90 分钟**
- 全部（175 GB）≈ **2–4 小时**

> 注：ModelScope 版**不含 MimicGen 合成数据**（官方 Box 上是 360 个包，这里 301 个）。
> 对我们没影响——人类示范质量更高，且我们的方法只需要 100 demo/任务。

---

## 四、⭐ 第一次拿到真实数据的审计结果（2 个任务）

```
episodes = 210    tasks = 2    styles = 49
缺失 layout_id/style_id: 0
解析失败: 0
```

| 指标 | 值 | 含义 |
|---|---|---|
| 每任务 styles | 41 / 42 | 覆盖很广 ✓ |
| **每 style episodes** | min 1 / **median 4** / max 10 | **稀疏**（2 任务时必然） |
| **单例场景占比** | 94% / 96% | 几乎每个 episode 是唯一的 (layout,style) 对 |
| **Cramér's V(task, style)** | **0.577** | **严重耦合** ← 只有 2 个任务，属预期 |
| layout × style V | 0.474 | 同上 |
| 泄漏检查 | **PASS** | ✓ |
| 分组检查（pretrain） | **PASS** | ✓ |

### 这组数字告诉我们什么

1. **标签完全可靠**：210 条全部有 style_id，0 缺失 ✓
2. **每 style 4 条（2 任务）→ 61 任务时约 122 条/style** → **域分类器可行** ✓
3. **耦合度 0.577 是"任务数太少"造成的假象** → 必须下满 61 个 atomic 任务再审计
4. **单例场景 94–96%** → 说明 layout×style 是**逐 episode 随机抽的**，不是固定配对 ✓
   （这正是我们需要的：域与场景解耦）

---

## 五、下载器（已写好并实测）

`scripts/fetch_robocasa365_modelscope.py`

```bash
# 看清单
python3 scripts/fetch_robocasa365_modelscope.py --list --split pretrain --kind atomic

# 探针（2 个任务，0.17 GB，约 15 秒）
python3 scripts/fetch_robocasa365_modelscope.py --preset probe --root /root/autodl-tmp/datasets/robocasa365

# ★ 全量 atomic（5.63 GB，约 4-8 分钟）-- 推荐
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all \
    --root /root/autodl-tmp/datasets/robocasa365 --audit
```

**落盘布局与官方 `get_ds_meta()` 完全一致**：
```
<root>/v1.0/pretrain/atomic/<Task>/<date>/lerobot/{data,videos,meta,extras}
```

---

## 六、对 GPU 部署方案的影响（重要）

**之前担心的三件事全部消失**：

| 原担心 | 现在 |
|---|---|
| Box 不通 → 拿不到数据 | ✅ **走 ModelScope，不用 Box** |
| 需要从 Mac 上传数据 | ✅ **不需要，服务器直接下** |
| 数据体积 TB 级、磁盘不够 | ✅ **atomic 全套只要 5.63 GB** |

**新的部署流程简化成**：
```bash
# AutoDL 上，无卡模式
git clone / scp 代码（264 KB）
bash scripts/setup_server.sh --no-gpu
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all --root /root/autodl-tmp/datasets/robocasa365 --audit
# → 5.63 GB，几分钟
# 然后切 GPU 模式，开始训练
```

**磁盘需求重新估算**：
- 代码 + 环境：~10 GB
- atomic 全套：5.63 GB
- 训练产物（checkpoint）：每个 ~700 MB（85.84M 参数 × 3 组状态）
- **建议 50 GB 数据盘足够**
