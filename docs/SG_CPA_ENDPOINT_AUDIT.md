# 终点审计（第一步）与 A 跨域稳定性（第二步）状态

> 授权：只做离线分析，不跑新 rollout、不改方法。  
> 第一步已完成（基于归档 B1 数据与本地几何模块）；第二步因远程 GPU 主机不可达而阻塞。

## 一、第一步：On/In 几何终点审计

### 1.1 数据与预注册

- 数据：B1 capture v2 归档（24 episodes，12 tasks），最终 success 来自 episode summary；
- 几何定义（在看结果前固定，复用 `dacsc/b1_progress_geometry.py`）：
  - `On`：物体底部 vs 支撑面顶部 contact × 水平支撑区域 inside；
  - `In`：物体中心在**容器定向内壁盒**内的平滑 inside × below-rim；
- 同时记录旧 `proximity`（object→target 中心距离，以首帧为 d0）作为对照；
- 指标：final、最后 3 个 replan 的最大值；AUC 只作**描述性诊断**，不用来选择公式。

### 1.2 结果

| suite | n | success | 指标 | median（成功） | median（失败） | AUC（描述性） |
|---|---|---|---|---|---|---|
| spatial | 12 | 7 | geom_final | 0.197 | 0.0 | 0.63 |
| spatial | 12 | 7 | geom_last3_max | 0.327 | 0.0 | 0.80 |
| spatial | 12 | 7 | prox_final | 0.778 | 0.103 | 0.83 |
| object | 12 | 6 | geom_final | **0.0** | 0.0 | **0.50** |
| object | 12 | 6 | geom_last3_max | **0.0** | 0.0 | **0.50** |
| object | 12 | 6 | prox_final | 0.677 | -0.003 | 1.00 |

全体 24 episodes：geom_last3_max AUC 0.70；prox_final AUC 0.90。

### 1.3 结论

1. **`In`（object suite）几何终点完全无效**：所有成功与失败 episode 的几何分数都为 0（AUC 0.50）；
   说明当前定向盒/rim 公式没有表达 “basket In” 的真实判定语义。
2. **`On`（spatial suite）几何终点有部分信息但很弱**：last3 max 的 median 成功 0.327 vs 失败 0.0，
   AUC 0.80；final 更弱（AUC 0.63）。仍不足以作为任务完成终点。
3. **`proximity` 在 object suite 上 AUC=1.00，但这是中心距离代理**：
   object suite 的成功 episode 通常把物体移向 basket 中心；这不能证明物体进入容器或位置正确。
   spatial suite 的 proximity 也受“接近/抬起”影响，不能当任务完成。
4. 因此 B2 的 `τ_prox` 只能支持“中心距离代理的局部效应”，**不能支持 task utility**；B2 冻结结论不变。
5. 若未来重新设计终点，`In` 必须使用容器内部/BDDL success 语义，`On` 必须使用支撑面接触 + 区域，
   且必须用 simulator success/几何检查验证误判率，而不是用 success 挑公式。

### 1.4 预注册的阈值分类（描述性）

阈值 0.5/0.8 的混淆矩阵见 `/tmp/endpoint_audit.json`（归档 `docs/data/endpoint_audit.tar.gz`）。
这些阈值未用于选择指标；仅用于说明“几何分数不能直接当成功概率”。

## 二、第二步：A 的跨域稳定性——阻塞

### 2.1 计划（已冻结，未执行）

在 Camera / Background / Noise / Layout 的目标观测上分别估计 \(A_d\)：

1. 用与 `build_interface_stats.py` 相同的有限差分 + Jacobian/SVD 方法；
2. 只用 B1 已有目标域观测，不跑新 rollout（policy forward/Jacobian 需要 GPU）；
3. 比较：
   - 主子空间夹角（principal angles）；
   - 重叠度 \(\|A_d^\top A_{d'}\|_F^2/5\)；
   - 奇异值谱与覆盖的动作敏感能量；
   - 各域 A 对源域 A 的投影保留率。
4. 判定：
   - 高重合 → 固定 A 是合理近似；
   - 仅相似视觉域重合 → A 应域条件化；
   - 任务间差异大 → 固定五维 A 的跨域保护假设不成立。

### 2.2 阻塞原因

远程 GPU 主机 `i-2.gpushare.com:20119` **连接被拒**（多次重试），无法加载 CoRP/checkpoint 做目标域 Jacobian 采样。
第一步完全基于本地归档数据完成；第二步需要在主机恢复后执行，脚本尚未运行、不产生任何结论。

### 2.3 恢复后需要的最小产物

- `A_camera / A_background / A_noise / A_layout`（各 768×5）；
- 6 对 domain 的主夹角与重叠度矩阵；
- 每域 A 相对源域 A 的覆盖率/奇异值表；
- 判定“固定 A 是否成立”的报告。

## 三、当前状态

| 项 | 状态 |
|---|---|
| 终点审计 | ✅ 完成（本报告 §1） |
| A 跨域稳定性 | ⛔ 阻塞：GPU 主机不可达 |
| B2 | 冻结，不重跑 |
| B3 | 不启动 |
| η/A-projection/rollout | 未授权 |

**建议**：先以终点审计结论修订未来 endpoint 设计；待 GPU 主机恢复后只做第二步离线 A 估计，不扩展 B2。
