# VLA 测试时自适应：完整研究总结（2026-11）

> 本文件汇总本阶段全部研究逻辑、代码与实验。日志见 [`LIBERO_ROUTE_A.md`](LIBERO_ROUTE_A.md)。
> 结论先行：**"在动作接口上用源域统计做测试时自适应"这一类方法在 LIBERO-Plus 上系统性无效；
> 根因是接口偏移不可识别（模型误差地板 ≈ 真实偏移），且动作敏感方向的偏移是策略的"补偿"而非噪声。**

---

## 1. 研究问题

在官方 CoRP（LIBERO checkpoint）上，研究**测试时自适应（TTA）**：
- 站点：DiT 动作生成器的 cross-attention **写回**（动作接口）；
- 约束：不重训、不硬删通道、变换有界可逆；允许测试时少量训练；
- 评测：LIBERO-Plus 官方 7 个扰动维度（Camera / Robot / Language / Light / Background / Noise / Layout）。

**官方全量基线（10,030 episodes）**：总体 **53.4%**；
维度：Language 74.3 / Light 72.8 / Objects Layout 71.1 / Camera 44.2 / Sensor Noise 39.6 / Background 37.8 / Robot 35.5；
套件：object 71.0 / spatial 68.3 / libero_10 51.4 / goal 24.5；层级：L1 72.0 / L2 63.0 / L3 53.8 / L4 46.3 / L5 33.1。

---

## 2. 逻辑演进（6 个阶段）

| 阶段 | 做法 | 结果 | 教训 |
|---|---|---|---|
| ① 通道删除（V4） | 写回上删"外观通道" | 硬投影 **−21.8** | 分类的通道删除经验不能迁移到动作（动作无冗余） |
| ② 软映射（PSC 式） | 子空间投影/软缩放 | **±2（不显著）** | 方向由源域方差定，不等于动作需要 |
| ③ 谱诊断 + 统计三件套 | top-k 压方差 / 去偏 / CORAL 仿射 | 小子集 +13.3 → **全维 ≈0**（camera 去偏 n=291 Δ+0.3, p=1.0） | 小样本正结果不可信；口径必须一致 |
| ④ 动作敏感子空间 | Jacobian 找 A，动作子空间对齐 | A 对齐 **−15.9（p=0.0043）** | A 方向的偏移是"补偿"，不能擦 |
| ⑤ 状态条件化 + 离线协议 | 本体状态+步数条件、adapt-then-freeze | clean 60–90%，目标域有害 | 条件模型误差地板 3.11 ≈ 偏移 3.2 → **不可识别** |
| ⑥ 接口低秩 + leftover（CPIA） | `z'=z+UVᵀz`，leftover 梯度，零空间保护 | n=60 +5（p=0.375）；n=120 **−0.8**；加强 −1.7；clean −10~−20 | leftover 损失 clean 0.0137 vs camera 0.0118 → **信号不敏感** |

### 关键机制发现（本阶段最有价值的结论）

1. **补偿**：配对实验（同任务、同动作，clean 重放 vs 相机扰动）：
   - 写回差异 ‖Δw‖≈7，其中 **~39% 落在动作敏感子空间 A**；
   - **成功轨迹的 A 偏移显著大于失败**（3.59 vs 3.17，Mann-Whitney p<0.0001；任务内配对 +0.33）；
   - 把 A 方向"对齐回源域" = 撤掉补偿 → **−15.9pp**。
2. **不可识别**：留一任务交叉验证的条件模型误差 **3.11**，真实域偏移 **3.2** → 误差≈信号。
3. **信号不敏感**：leftover 自洽损失在 clean 与相机下几乎相同 → 自洽 ≠ 正确。
4. **容量-信号权衡**：安全档动作变化 4e-4（无效），有效档 0.016（clean −10~−20）→ 无"中间档"。

---

## 3. 代码地图

### 3.1 方法模块（`dacsc/`）

| 文件 | 类/函数 | 作用 | 状态 |
|---|---|---|---|
| `dacsc/residual_debias.py` | `ResidualDebias` | 残差均值去偏 | 已弃用（≈0） |
| `dacsc/residual_shrink_topk.py` | `ResidualShrinkTopK` | 残差 top-k 谱缩放 | 已弃用（≈0） |
| `dacsc/residual_shrink.py` | `ResidualShrink` | 全残差谱缩放 | 有害 |
| `dacsc/residual_affine.py` | `ResidualAffine` | CORAL 仿射映射 | 已弃用（≈0） |
| `dacsc/adaptive_residual.py` | `AdaptiveResidual` | 模式选择+锁 | 不可靠 |
| `dacsc/interface_anchoring.py` | `ISA` | 有界缩放+leftover+锚 | ±2 |
| `dacsc/interface_calibration.py` | `InterfaceCalibration` | 接口标定 | 探索 |
| `dacsc/action_subspace_alignment.py` | `ActionSubspaceAlignment` | 只校正 A 方向均值 | **−15.9（有害）** |
| `dacsc/null_space_alignment.py` | `NullSpaceAlignment` | 只校正 A 零空间（ANST） | 安全但无效 |
| `dacsc/full_mean_alignment.py` | `FullMeanAlignment` | 整块写回均值对齐 | clean 50–60% |
| `dacsc/conditional_alignment.py` | `ConditionalAlignment` | query 条件回归 | 条件被污染，≈0 |
| `dacsc/state_conditional_alignment.py` | `StateConditionalAlignment` | 本体状态+步数条件 | clean 80–90%，有害 |
| `dacsc/state_residual_mean.py` | `StateResidualMeanAlignment` | 条件残差均值 | 有害 |
| `dacsc/offline_state_alignment.py` | `OfflineStateAlignment` | 离线 adapt-then-freeze | clean 60% |
| `dacsc/leftover_tta.py` | `LowRankInterfaceTransform`, `leftover_loss`, `sample_with_grad` | **CPIA**：接口低秩变换 + leftover | pilot 无稳定收益 |
| `dacsc/psc.py`, `psc_tta.py`, `psc_weighted.py` | PSC 系列 | 子空间对齐基线 | 已弃用 |
| `dacsc/invariance.py` | `appearance_consistency_loss` 等 | 训练期外观一致性（早期） | 探索 |

### 3.2 驱动与工具（`scripts/`）

| 文件 | 作用 |
|---|---|
| `scripts/libero_shift_eval.py` | 官方协议（exec full chunk）+ 各统计方法开关（主驱动） |
| `scripts/leftover_tta_eval.py` | 修改协议（exec_len<H，保留 leftover）+ CPIA 适应循环 |
| `scripts/sca_eval.py` | 官方协议 + 状态条件化模块注入（state/step） |
| `scripts/build_interface_stats.py` | 源域统计：写回均值/支撑基/协方差/Jacobian 动作子空间 |
| `scripts/build_state_cond_source.py` | 状态条件化校准对（phi=[state, step]） |
| `scripts/build_conditional_source.py` | query 条件化校准对 |
| `scripts/write_spectrum_probe.py` | 写回能量谱（支撑/残差/动作敏感能量） |
| `scripts/compensation_probe.py` | 补偿探针（成功/失败 A 偏移对比） |
| `scripts/verify_impl.py` | 11 项本地实现校验（钩子序列/恒等性/正交性等） |
| `scripts/analyse_*.py` | 配对统计（bootstrap CI、McNemar、Holm） |

### 3.3 验证与审计（重要）

- `scripts/verify_impl.py`：11/11 PASS（钩子序列 11 次/prefix 129/Euler 12；恒等性；正交性；仿射恒等 2.4e-7；独立 base 与官方基线 60/60 一致）。
- 不干预对照：`c_max=0` / `adapt_steps=0` 与 base **逐条一致**（20/20、10/10）。
- CPIA base（exec_len=12）与官方 baseline **10/10 一致**（需 `torch.autocast` bf16）。

---

## 4. 实验清单（按阶段）

| 实验 | 规模 | 结论 |
|---|---|---|
| 官方全量 baseline | 10,030 eps | 53.4%（见上） |
| 相机全维（debias） | spatial 291 | +0.3pp, p=1.0 |
| 变体重测（重验口径） | 120/变体 | topk +0.8 / affine +0.8 / start5 −3.3 / α0.5 −0.8 / 全步 α0.3 −1.7 / α1.0 −7.5 |
| 动作子空间对齐 | 88 | **−15.9, p=0.0043** |
| 补偿探针 | 60 相机 + 12 clean | 成功 A 偏移 3.59 vs 失败 3.17, **p<0.0001** |
| 状态条件化 / 离线 | 10–120 | clean 60–90%，目标域有害 |
| 7 维统一方法扫描 | 各 120（Noise 部分） | 无显著改善；Language/Light 趋势 −4.2（ns） |
| CPIA pilot | 60 / 120 | +5.0（p=0.375）→ **−0.8**（p=1.0） |
| CPIA 加强 | 60 | −1.7；clean −10~−20 |

---

## 5. 本阶段修复的 Bug（全部有记录）

1. 源统计采集 **Euler 步 off-by-one**（采集第 9 步、模块用第 10 步）→ 统计错位；
2. 步数归一化不一致（校准 k/9 vs 驱动 k/15）→ 条件外推；
3. 在线逐帧校正的**闭环反馈**（校正改变轨迹→污染下一帧统计）；
4. 驱动缺 `torch.autocast` bf16 → 与官方协议不一致（2/10）；
5. CPIA 预/后采样随机噪声不同 → 把噪声误判为"更新效果"；
6. `loss_trace`/`adapted_once` 属性缺失、`fork_rng` 作用域错误、任务范围 `0-9` 解析；
7. 早期 Jacobian 估计未固定 RNG → 动作子空间被噪声污染（rank 11→7）。

---

## 6. 与文献的对比（为什么"他们行、我们不行"）

| 要素 | 有效论文 | 我们（历史做法） |
|---|---|---|
| 信号带"对错" | SALT（leftover 是动作目标）、Verify-Act-Distill（验证器）、COAST（成功方向） | 统计量/自洽，偏移下不变 |
| 容量 | 全参数 / 隐藏态 | rank2、最后一步（动作变化 4e-4） |
| 门控 | 特征变化检测 / 参数恢复 | 统计门控不可分（clean≈camera） |
| 样本 | 上千 / 多 episode | 几十帧 |
| 目标 | 优化任务目标 | **估计偏移**（不可识别） |

关键文献：SALT [2610.07946](https://arxiv.org/html/2610.07946v1)、Verify-Act-Distill [2610.14885](https://arxiv.org/html/2610.14885v1)、
COAST [2605.17144](https://export.arxiv.org/abs/2605.17144)、ActMAD (CVPR 2023)、SSA [2502.02801](https://arxiv.org/abs/2502.02801)、
PSC [2603.09338](https://arxiv.org/abs/2603.09338)、Tent [2006.10726](https://arxiv.org/abs/2006.10726)、CoTTA [2203.13591](https://arxiv.org/abs/2203.13591)、
TTA 可识别性 [2609.11235](https://arxiv.org/abs/2609.11235)。

---

## 7. 复现方式

```bash
# 环境（服务器）
cd /hy-tmp/dacsc-robot-784fe2e
source repos/CoRP/.venv/bin/activate

# 官方协议 + 方法
python scripts/libero_shift_eval.py --suite libero_spatial --tasks 0-9 \
  --residual-debias <stats> --residual-shrink <spectrum> ...

# CPIA（修改协议，exec_len<horizon）
python scripts/leftover_tta_eval.py --suite libero_spatial --tasks <ids> \
  --exec-len 6 --mode tta --rank 2 --adapt-steps 2 --lr 1e-3 \
  --action-basis <stats> --last-step-only --adapt-once --uv-clip 0.05 \
  --libero-src /hy-tmp/libero_plus --libero-config-path /hy-tmp/libero_plus_config \
  --output-dir <out>

# 本地实现校验
python scripts/verify_impl.py
```

**注意**：CPIA 为修改协议（exec_len=6），其 base 与 CPIA 用同一驱动对比；
官方协议结果为 `libero_shift_eval.py`（exec full chunk）。

---

## 8. 当前状态与建议

- 代码：所有模块可运行；协议保真已修（base 与官方 10/10 一致）；`verify_impl.py` 11/11 PASS。
- 有效资产：**补偿机制发现**、**可识别性诊断**、**接口低秩可逆基础设施**、**严格评测驱动**。
- 未解决：接口统计类方法在该基准无可见收益；leftover 信号对偏移不敏感。
- 下一步（二选一）：
  - **A**：换"带对错"的信号（Verify-Act-Distill 式验证器 + 蒸馏），配合更大容量与门控；
  - **B**：以"补偿 + 不可识别"为核心写机制/负结果论文（证据已足够）。

---

_文档生成：2026-11-11。所有数字来自 `research/server_runs/LIBERO_ROUTE_A.md` 记录与配对统计。_
