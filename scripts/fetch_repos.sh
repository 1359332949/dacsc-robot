#!/usr/bin/env bash
# 拉取 DACSC-Robot 需要的全部参考仓库。
#
# 用法：
#   bash scripts/fetch_repos.sh            # 默认：tarball 快照（无 git 历史，体积最小）
#   bash scripts/fetch_repos.sh --git      # 用 git clone（有历史，可 git pull 更新）
#   bash scripts/fetch_repos.sh --list     # 只打印清单
#
# 说明：本脚本在带宽受限环境下也可用；用 --only <名字> 可只拉其中一个。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/repos"
mkdir -p "$DEST"

# 格式: 目录名|owner/repo|分支|类别|体积|说明
MANIFEST='
robocasa|robocasa/robocasa|main|A 仿真环境|47.7MB|RoboCasa365 本体（任务/场景/数据下载脚本）
diffusion_policy|robocasa-benchmark/diffusion_policy|main|A 官方基线|12.8MB|★主基线：官方 Diffusion Policy fork
robomimic|robocasa-benchmark/robomimic|robocasa-dp|A 官方基线|53.4MB|ResNet18ConvFiLM 定义所在（必须用 robocasa-dp 分支）
openpi|robocasa-benchmark/openpi|main|A 官方基线|0.8MB|官方 pi0 fork
Isaac-GR00T|robocasa-benchmark/Isaac-GR00T|main|A 官方基线|47.8MB|官方 GR00T N1.5 fork
mimicgen|robocasa-benchmark/mimicgen|main|A 数据生成|15.6MB|MimicGen，合成数据（pretrain MG 数据集）
leaderboard|robocasa-benchmark/leaderboard|main|A 评测记录|2.7MB|官方排行榜提交记录（含全部基线数值）
RoboSaGA|zheyu-zhuang/RoboSaGA|master|B 对比方法|680MB|★显著性引导增广（CoRL2024）；内嵌 robosuite 故很大
RoboAug|Open-X-Humanoid/RoboAug|main|B 对比方法|0.1MB|★区域对比增广（CoRL2026），仅数据生成部分
roboengine|michaelyuancb/roboengine|main|B 对比方法|27.5MB|★即插即用增广工具箱（IROS2025）
RoboVIP_VDM|RoboVIP/RoboVIP_VDM|main|B 对比方法|2.6MB|多视角视频增广（仿真训练/评测未发布）
NICE-Scene-Surgery|SajjadPSavoji/NICE-Scene-Surgery|main|B 对比方法|4.7MB|场景编辑增广（CoRL2025 Workshop，成熟度低）
any4lerobot|Tavish9/any4lerobot|main|C 工具|0.3MB|LeRobot 数据格式转换/可视化工具
lerobot|huggingface/lerobot|main|C 工具|30MB|LeRobot 官方框架（RoboCasa365 数据格式基座）
lerobot-dataset-visualizer|huggingface/lerobot-dataset-visualizer|main|C 工具|21.6MB|LeRobot 数据集网页可视化（阶段0 审计有用）
Awesome-WAM|OpenMOSS/Awesome-WAM|main|D 综述|1.8GB|World Action Model 综述（当前 SOTA 家族）
awesome-embodied-vla-va-vln|jonyzhang2023/awesome-embodied-vla-va-vln|main|D 综述|0.6MB|VLA/VA/VLN 综述列表
Awesome-RL-VLA|Denghaoyuan123/Awesome-RL-VLA|main|D 综述|1.2MB|VLA 强化学习综述
Awesome-Imitation-Learning|kristery/Awesome-Imitation-Learning|master|D 综述|0.1MB|模仿学习综述列表
AwesomeWorldModels|Li-Zn-H/AwesomeWorldModels|main|D 综述|4.2MB|世界模型综述

注：仓库名 DRAIL 存在同名冲突 —— GitHub 上 NVlabs/DRAIL 是 "Diffusion Rewards Adversarial
Imitation Learning" (NeurIPS 2024)，与本项目参考文献 9 的 "Dual-Region Augmentation for
Imitation Learning"（农业操作，arXiv 2603.04845）不是同一篇，故不下载。
'

MODE="tarball"; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --git) MODE="git"; shift ;;
    --list) MODE="list"; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

printf '%s\n' "$MANIFEST" | while IFS='|' read -r name slug branch cat size desc; do
  [[ -z "${name:-}" || "$name" == 注* ]] && continue
  [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
  if [[ "$MODE" == "list" ]]; then
    printf '%-30s %-42s %-12s %-12s %-8s %s\n' "$name" "$slug" "$branch" "$cat" "$size" "$desc"
    continue
  fi
  if [[ -d "$DEST/$name" ]]; then echo "[skip] $name 已存在"; continue; fi
  echo "[get ] $name  <- $slug@$branch  ($size)"
  if [[ "$MODE" == "git" ]]; then
    git clone --depth 1 --branch "$branch" "https://github.com/$slug.git" "$DEST/$name" || echo "[fail] $name"
  else
    tmp="$(mktemp -d)"
    if curl -sL --retry 3 --max-time 3600 \
        "https://codeload.github.com/$slug/tar.gz/refs/heads/$branch" \
        | tar -xz -C "$tmp" --strip-components=1 2>/dev/null; then
      mv "$tmp" "$DEST/$name"
    else
      echo "[fail] $name"; rm -rf "$tmp"
    fi
  fi
done
echo "完成。目录：$DEST"
