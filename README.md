# DACSC-Robot 工作区

机器人视觉策略在**未见外观域**下的动作泛化研究。

## 关键结论

- **数据源（已定）**：**RoboCasa365**，使用官方 `pretrain/atomic` 人类示范；每条 episode 都有
  `style_id`/`layout_id`，第一阶段 61 个任务、6,935 条 episode。
- **基线（已锁定）**：RoboCasa 官方 Diffusion Policy fork（`robocasa-benchmark/diffusion_policy` +
  `robomimic@robocasa-dp`），`H=10 / n_obs=2 / n_action=8`。
- **我们的主干**：DINOv2 ViT-S/14（多视角共享、联合微调）→ learned-query resampler → DiT/L1 动作头。
- **RoboSaGA 不是 base**，是对比方法；**XPolicyLab 是后期扩展基线的工具箱**，不是第一阶段的运行依赖。

## 目录约定（数据与代码分离）

```text
~/dacsc-robot/                         # 代码、配置、脚本
~/datasets/robocasa365/                # 数据集（本机；5.3 GB）
  v1.0/pretrain/atomic/<Task>/<date>/lerobot/
~/dacsc-robot/runs/                    # 训练输出/checkpoint（后续创建）
~/.cache/dacsc-robot/                  # 可删除缓存

# AutoDL 上：
/root/dacsc-robot/                     # 代码放系统盘
/root/autodl-tmp/datasets/robocasa365/ # 数据放数据盘
/root/autodl-tmp/runs/dacsc-robot/     # 训练产物放数据盘

# 恒源云上：
/root/dacsc-robot/                     # 或你的 HOME 下
/hy-tmp/datasets/robocasa365/          # 数据放数据盘
/hy-tmp/runs/dacsc-robot/              # 训练产物放数据盘
```

统一用 `DACSC_DATA_ROOT` 指定数据集根目录；不设置时，脚本会自动选择
`~/datasets/robocasa365`（本机）、`/root/autodl-tmp/datasets/robocasa365`（AutoDL）
或 `/hy-tmp/datasets/robocasa365`（恒源云）。

## 我们的代码（官方仓库零修改）

```
dacsc/                       我们的包
  compat.py                  diffusers 版本差异的兼容 shim
  dinov2_encoder.py          DINOv2 观测编码器（多视角共享权重）
  policy.py                  RoboCasaDINOv2Policy（继承官方策略）
  gating.py                  通道门控（ChannelGate + 注入/卸载）
  domain_stats.py            跨域激活统计 → u^disc（KL）
  losses.py                  L_gate 正则 + 渐进预算 + 动作重要性 EMA
  dataset.py                 域标签接入（style_id / layout_id）
  stub_dataset.py            合成数据集（外观依赖域、动作不依赖域）
```

## 目录

```
notes/
  01-参考代码调研.md           参考仓库全清单、成熟度评级、六个坑
  02-Robo家族2025-2026调研.md  ★ 为什么选 XPolicyLab，2025-2026 Robo* 生态全图
  03-已拉取仓库清单.md         已拉取的 17 个仓库与体积
docs/
  DACSC-Robot-方法定义与实验设计.md     当前方法文档（v0.2）
  baseline_audit_real/                  阶段 1 基线审计（真实仓库 + 真实 torch）
scripts/
  fetch_repos.py             拉取参考仓库（可续传、gzip 校验、tarball 缓存）
  fetch_robosaga_subset.py   只抓 RoboSaGA 核心源码（~200KB，替代 680MB 全量）
  audit_dataset.py           阶段 0：数据审计（纯标准库）
  audit_baseline.py          阶段 1：基线静态审计（torch 可选）
  fetch_robocasa365_modelscope.py ★ RoboCasa365 数据下载（ModelScope，可续传/自动审计）
  check_data_layout.py       ★ 数据路径与 style/layout 标签完整性检查
  audit_baseline_runtime.py  ★ 运行时审计：CPU 上真实实例化基线并前向
  fetch_dinov2.py            ★ 拉取+验证 DINOv2 backbone（CoRP 用的那个）
  setup_env.sh               ★ 一键建环境（venv + .pth 让三个包可导入）
  test_dinov2_policy.py      ★ CPU 冒烟测试（4 项检查 + 与官方基线对比）
  test_gating.py             ★ 门控 6 项测试（含"恒等门控不改变基线"）
  test_domain_labels.py      ★ 域标签 7 项测试（含标签错位安全网）
  train_smoke_test.py        ★ 本机 MPS 端到端训练烟测（走官方 workspace.run()）
repos/                       17 个已拉取仓库（见 notes/03）
audit/                       审计输出
```

## 已验证的基线事实（真实仓库 + 真实 torch 运行）

```yaml
camera_views: [robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand]
image_size: [256, 256]      # → crop [224, 224]
visual_encoder: ResNet18ConvFiLM   # native_import 实测，8 个 conv block
horizon: 10 / n_obs_steps: 2 / n_action_steps: 8 / batch_size: 192
policy_backbone: DiT (12 层, n_cond_layers 4, n_head 8, n_emb 512)
action_dim: 12
```

8 个 conv block 输出（224×224 输入，hook 实测）：
`[64,56,56] [64,56,56] [128,28,28] [128,28,28] [256,14,14] [256,14,14] [512,7,7] [512,7,7]`
每块后接 FiLM（语言 768 维，episode 级）。**SpatialSoftmax 在 block 8 之后** —— 门控必须插在它之前。

每任务 horizon：317 个任务，300–7200 步（v1.0.1 起全部 ×1.5，均为 150 的倍数）。

## 常用命令

```bash
python3 scripts/fetch_repos.py --list                 # 看清单
python3 scripts/fetch_repos.py --group core           # 拉官方基线栈
python3 scripts/fetch_robosaga_subset.py              # RoboSaGA 核心源码
python3 scripts/check_data_layout.py                               # 检查现有数据
python3 scripts/fetch_robocasa365_modelscope.py --list --split pretrain --kind atomic
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all --audit
/tmp/venv_torch/bin/python scripts/audit_baseline.py \
    --dp-repo repos/diffusion_policy --robomimic-repo repos/robomimic \
    --robocasa-repo repos/robocasa --out audit/baseline
```

## 已验证可用（2026-10-16）

- **本机 MPS 完整训练跑通**：官方 `workspace.run()`，8 步 loss 1.2855→1.2202，
  **4607/4608 个门控被更新**，u^disc 成功识别域敏感通道（~2.3 s/步 @ bs=8）（notes/21）

- **数据集域标签**：batch 含 `domain_id`（style_id）/ `layout_id`，7 项测试通过，
  含**标签对齐安全网**（能检出 extras 序号与 episode_index 错位）（notes/20）

- **DINOv2 策略 v1 跑通**：编码器换成 DINOv2 ViT-S/14，总参数 85.84M，
  `predict_action` 输出 `[1,8,12]` 与官方一致，**梯度能流到 DINOv2（联合微调生效）**（notes/18）

- **官方 DP 基线**：CPU 上跑通，106.03 M 参数，输出 `action [1,8,12]`（notes/06）
- **DINOv2 ViT-S/14**：本地加载跑通，22.1 M 参数，`224×224 → [1,256,384]`（notes/16）
  - 代码从 codeload、权重从 Meta CDN（**4.9 MB/s**）拉取，绕过被拦的 github.com
- **网络快慢取决于主机**：GitHub ~38 KB/s，Meta CDN ~4.9 MB/s

## 环境限制（重要）

- 本机到 `github.com` 的 git 协议不可达，仅 `codeload`/`api`/`raw` 可达 → 用 tarball 快照，`git clone` 需换网络
- 本机到 `huggingface.co` **完全不可达** → RoboCasa 数据与官方 checkpoint 需在能上 HF 的机器上取
- 实测 GitHub 吞吐约 27–55 KB/s，大仓库耗时长；tarball 缓存在 `repos/.cache/`（~200MB，可删）
