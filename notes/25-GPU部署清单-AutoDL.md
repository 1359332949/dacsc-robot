# GPU 部署清单（AutoDL）

> 2026-10-19。按需付费，所以每一步都标了**要花多少钱、能不能在无卡模式下做**。

---

## 一、先回答你最关心的：数据集在哪、能不能拿到

### 1.1 官方数据在 **Box**（UT Austin 的网盘），不是 HF

```
410 个 tar 包 = pretrain 360（含 MimicGen 60） + target 50
下载入口：https://utexas.box.com/shared/static/<id>.tar
        ↓ 302
        https://utexas.app.box.com/public/static/<id>.tar   ← 真正的下载主机
```

**从你的 Mac 上 `utexas.app.box.com` 是不可达的**（连接直接超时）。
所以「先在 Mac 下好再上传」这条路走不通。

### 1.2 ⚠️ HF 镜像上的 RoboCasa 数据集**都不能用**

我查了 `hf-mirror.com`（国内可用）上所有 robocasa 相关数据集：

| 仓库 | 有什么 | 能不能用 |
|---|---|---|
| `haosulab/RoboCasa` | **场景素材**（材质/纹理，给 ManiSkill 用） | ❌ 不是示范数据 |
| `daixianjie/robocasa_human_lerobot` | LeRobot 转换版 | ❌ **没有 ep_meta.json** |
| `daixianjie/robocasa_mg_lerobot` | MimicGen 版（7 万文件） | ❌ 同上 |
| `kinam0252/robocasa_all` | 同上 | ❌ 同上 |

**为什么不能用**：我们的方法需要 `style_id`（外观域标签），它只存在于
`extras/episode_XXXXXX/ep_meta.json`。第三方做 LeRobot 转换时把 `extras/` **整个丢了**。
没有 style_id → 我们的方法没有原料。

### 1.3 所以只有三条路（按优先级）

| 路径 | 做法 | 前提 | 成本 |
|---|---|---|---|
| **A（首选）** | **在 AutoDL 上下 Box** | AutoDL 能访问 `utexas.app.box.com` | 只花流量/时间 |
| **B（兜底）** | **自己用仿真生成数据** | 有 robocasa 环境；我们**自己控制 style_ids** | GPU/CPU 时间 |
| **C** | 找人代下（有 Box 访问权的同学/VPN） | — | 人情 |

**必须先做的第一件事就是测 A 能不能通**（见下面第 3 步）——这决定后面所有事。

---

## 二、AutoDL 上怎么省钱的三个关键点

### ① 用「无卡模式」做所有不需要 GPU 的事

AutoDL 支持**无卡模式开机**（约 0.1 元/小时，具体见控制台）。
以下事情**全都不需要 GPU**，务必在无卡模式下做：

- 上传代码
- `pip install` 依赖（这一步很慢）
- **下载数据集**（最耗时）
- 跑 4 个单元测试 + 合成数据烟测（CPU 也能跑，只是慢）

装好、数据齐了 → **关机 → 切 GPU 模式** → 只做训练。

### ② 数据放 `/root/autodl-tmp`，代码放 `/root`

- `/root`（系统盘）：只放代码（我们的代码才 264 KB）
- `/root/autodl-tmp`（数据盘）：放 RoboCasa 资产（~10GB）+ 数据集

**注意**：AutoDL 的数据盘扩容要额外付费，租之前先看清你选的机型给了多少。

### ③ 第一次训练**不要**跑完整配置

官方配置是 `1000 epochs × 500 steps = 50 万步`。
在 4090 上按 ~0.5–1 s/步估算，**单次就是 70–140 小时（140–280 元）**。

**第一轮只跑 1–2 万步**（约 2–3 小时）：
- 够看出"加门控 vs 不加门控"的趋势
- 不够得出最终结论（最终要跑满 + 多种子）

---

## 三、执行清单（照做即可）

### 第 0 步（Mac 上，2 分钟）

```bash
cd ~/dacsc-robot
bash scripts/pack_for_server.sh
```
产出三个文件在 `.cache/upload/`：
- `dacsc_code.tgz`（264 KB）← **必须传**
- `dacsc_repos.tgz`（约 50 MB）← 服务器能 clone 就不用传
- `dinov2.zip`（2.8 MB）← 可选，服务器能下也不用传

### 第 1 步（AutoDL 控制台）

1. 选机型：**RTX 4090 / 24GB**（够用且便宜）；要跑大 batch 再选 A100 40G
2. 镜像：**PyTorch 2.x + CUDA 12.x + Python 3.10/3.11**
3. 开机方式选 **无卡模式**（setup 阶段）
4. 记下 SSH 的 **host / port / 密码**

### 第 2 步（Mac 上，上传）

```bash
# 换成 AutoDL 给你的端口和地址
scp -P <PORT> .cache/upload/dacsc_code.tgz root@<HOST>:/root/
# 可选：服务器 clone 不通时再传
scp -P <PORT> .cache/upload/dacsc_repos.tgz root@<HOST>:/root/
```

### 第 3 步（服务器上）—— **★ 最重要的一步：测 Box 能不能通**

```bash
ssh -p <PORT> root@<HOST>

# 解包
mkdir -p /root/dacsc-robot && tar -xzf /root/dacsc_code.tgz -C /root/dacsc-robot
# 如果传了仓库
tar -xzf /root/dacsc_repos.tgz -C /root/dacsc-robot 2>/dev/null || true

# ★ 测 Box 下载主机
curl -sI --max-time 20 https://utexas.app.box.com | head -3
echo "退出码: $?"
```

**判读**：
- 返回 `HTTP/2 200/302` → **路径 A 可行**，继续第 4 步
- 超时/000 → **Box 不通**，跳到第 7 步（自己生成数据）

### 第 4 步（服务器）装环境（无卡模式，约 10–20 分钟）

```bash
cd /root/dacsc-robot
bash scripts/setup_server.sh --no-gpu
```

它会：装依赖 → 写 `.pth` → 下 DINOv2（87MB）→ 跑 4 个单元测试 → 跑合成数据烟测。

**期望看到**：4 个测试 ✓、`loss 下降`、`门控被更新`。

### 第 5 步（服务器）先下 2 个任务"试水"

```bash
cd /root/dacsc-robot
# 先看清单（不发请求）
python3 scripts/fetch_robocasa365.py --dryrun --preset probe

# 真下（约 2 个包）
python3 scripts/fetch_robocasa365.py --preset probe --root /root/autodl-tmp/datasets/robocasa365
```

**这一步的目的**：**测出单个任务的实际体积**，才能决定"下多少个任务"。
官方没公布体积，之前只有我的粗估（可能 TB 级）。

下完后：
```bash
du -sh /root/autodl-tmp/datasets/robocasa365/v1.0/pretrain/atomic/*
```

### 第 6 步（服务器）规模决策

拿到单任务体积后按这个表决定：

| 单任务体积 | 能下多少任务 | 建议 |
|---|---|---|
| < 1 GB | 20–50 个 | `--preset style-check`（20 个） |
| 1–5 GB | 5–10 个 | 自选 5 个短任务 |
| > 5 GB | 2–3 个 | 只做单任务可行性验证 |

### 第 7 步（Box 不通时的兜底）自己生成数据

**这条路其实有优势：style_id 是我们自己控制的，标签 100% 准确。**

```bash
# 需要先装好 robocasa 的仿真依赖
pip install robosuite mujoco gymnasium lerobot

# robocasa 自带数据生成脚本
python -m robocasa.scripts.dataset_scripts.convert_hdf5_lerobot --help
```

具体生成命令要等 box 测试结果出来后再定（因为方式取决于能否拿到现有的 HDF5）。

### 第 8 步（切 GPU 模式）训练

```bash
# 关机 → 控制台切"GPU 模式"开机
cd /root/dacsc-robot

# 确认 GPU
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"

# 短训练（先跑这个看趋势）
python3 train.py --config-name=train_dinov2_bs192 \
    task=robocasa/pretrain_human300 \
    dataloader.batch_size=32 \
    training.max_train_steps=2000 \
    training.num_epochs=1 \
    logging.mode=offline
```

**注意**：`train.py` 在 `repos/diffusion_policy/` 下，需要先
```bash
cd /root/dacsc-robot/repos/diffusion_policy
```
并确保 `dacsc` 在 path 上（`.pth` 已处理）。

---

## 四、成本估算（按 4090 ≈ 2 元/小时）

| 阶段 | 模式 | 时长 | 费用 |
|---|---|---|---|
| 装环境 + 下 DINOv2 | 无卡 | 0.3 h | ~0.03 元 |
| 下 2 个任务试水 | 无卡 | 0.5–3 h（看体积） | ~0.05–0.3 元 |
| 下 20 个任务 | 无卡 | 数小时 | ~1 元 |
| 短训练 2000 步 | GPU | 0.3–1 h | 0.6–2 元 |
| **单次完整训练 50 万步** | GPU | **70–140 h** | **140–280 元** |
| 消融 6 组 × 短训练 | GPU | 2–6 h | 4–12 元 |

**结论：先花 5 元做验证，别一上来就跑完整训练。**

---

## 五、三个必须提前知道的风险

| 风险 | 影响 | 应对 |
|---|---|---|
| **Box 不通** | 拿不到官方数据 | 走第 7 步（自己生成）或找人代下 |
| **磁盘不够** | 下到一半失败 | 先 `df -h` 看 `/root/autodl-tmp`，按第 6 步的表决定下多少 |
| **训练超时/欠费** | checkpoint 丢 | 用 `checkpoint_every=100`，并确认输出写在 `/root/autodl-tmp` |

---

## 六、我现在就要你做的三件事

1. **在 AutoDL 租一台 4090，用「无卡模式」开机**（先别开 GPU）
2. 把 SSH 的 host/port 发我（**不要发密码**）
3. 在服务器上跑这两条，把输出发我：
```bash
curl -sI --max-time 20 https://utexas.app.box.com | head -3
df -h /root /root/autodl-tmp
```

**这两条输出决定后面全部方案**：Box 通不通、磁盘够不够。
