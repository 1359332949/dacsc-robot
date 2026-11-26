# CoRP Route A：同期双目公平物理五臂主表

主问题仅含Clean、双目Sensor、物理Lighting与Background；没有单眼Sensor。每套100条件×3种子×5臂=1500局，EGL/FP32，H=E=12，原生Euler10，220步上限。
Spatial严格外9/内8+1任务LOTO；Ours沿用指定冻结full.npz。Object四个适配臂仅使用Spatial10全局参考，0条Object轨迹拟合。
ActMAD-v2是明确标注的ActMAD-inspired VLA移植。旧ActMAD/污染语言/错误shared阈值的历史数字不补填。

## spatial

| 组 | Frozen | Pixel-Align | Deep B12 | ActMAD-inspired v2 | Ours V1 |
|---|---:|---:|---:|---:|---:|
| clean | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 |
| sensor_bilateral | 15/90 | 17/90 | 15/90 | 14/90 | 25/90 |
| lighting | 74/90 | 84/90 | 74/90 | 68/90 | 80/90 |
| background | 59/90 | 26/90 | 53/90 | 58/90 | 58/90 |
| table_texture | 25/45 | 5/45 | 19/45 | 23/45 | 23/45 |
| scene_texture | 34/45 | 21/45 | 34/45 | 35/45 | 35/45 |
| total | 177/300 | 156/300 | 171/300 | 169/300 | 192/300 |

### 配对比较：相对 frozen

| 组 | 臂 | 救回/误伤 | 净增 | Δpp | 精确双侧p | 任务聚类95%区间pp |
|---|---|---:|---:|---:|---:|---:|
| clean | pixel_align | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | deep_block12 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | actmad_inspired_viewwise_v2 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | ours_v1 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | pixel_align | 5/3 | +2 | +2.2 | 0.726562 | [-5.1, +10.1] |
| sensor_bilateral | deep_block12 | 5/5 | +0 | +0.0 | 1 | [-5.2, +6.9] |
| sensor_bilateral | actmad_inspired_viewwise_v2 | 2/3 | -1 | -1.1 | 1 | [-9.0, +4.9] |
| sensor_bilateral | ours_v1 | 11/1 | +10 | +11.1 | 0.00634766 | [+0.0, +24.7] |
| lighting | pixel_align | 10/0 | +10 | +11.1 | 0.00195312 | [+3.6, +22.2] |
| lighting | deep_block12 | 7/7 | +0 | +0.0 | 1 | [-12.7, +11.1] |
| lighting | actmad_inspired_viewwise_v2 | 3/9 | -6 | -6.7 | 0.145996 | [-14.7, +0.0] |
| lighting | ours_v1 | 11/5 | +6 | +6.7 | 0.210114 | [-4.6, +23.2] |
| background | pixel_align | 4/37 | -33 | -36.7 | 1.02584e-07 | [-51.4, -19.8] |
| background | deep_block12 | 7/13 | -6 | -6.7 | 0.263176 | [-16.3, +5.3] |
| background | actmad_inspired_viewwise_v2 | 6/7 | -1 | -1.1 | 1 | [-10.8, +11.1] |
| background | ours_v1 | 9/10 | -1 | -1.1 | 1 | [-10.8, +10.3] |
| table_texture | pixel_align | 1/21 | -20 | -44.4 | 1.09673e-05 | [-60.0, -22.2] |
| table_texture | deep_block12 | 3/9 | -6 | -13.3 | 0.145996 | [-21.2, +4.2] |
| table_texture | actmad_inspired_viewwise_v2 | 3/5 | -2 | -4.4 | 0.726562 | [-15.9, +15.2] |
| table_texture | ours_v1 | 5/7 | -2 | -4.4 | 0.774414 | [-15.0, +12.5] |
| scene_texture | pixel_align | 3/16 | -13 | -28.9 | 0.00442505 | [-50.9, +0.0] |
| scene_texture | deep_block12 | 4/4 | +0 | +0.0 | 1 | [-13.3, +13.3] |
| scene_texture | actmad_inspired_viewwise_v2 | 3/2 | +1 | +2.2 | 1 | [-6.7, +14.8] |
| scene_texture | ours_v1 | 4/3 | +1 | +2.2 | 1 | [-7.4, +17.8] |
| total | pixel_align | 19/40 | -21 | -7.0 | 0.0086415 | [-16.0, +3.4] |
| total | deep_block12 | 19/25 | -6 | -2.0 | 0.451381 | [-6.9, +3.2] |
| total | actmad_inspired_viewwise_v2 | 11/19 | -8 | -2.7 | 0.200488 | [-6.2, +1.2] |
| total | ours_v1 | 31/16 | +15 | +5.0 | 0.0399861 | [-1.4, +14.6] |

### 配对比较：相对 actmad_inspired_viewwise_v2

| 组 | 臂 | 救回/误伤 | 净增 | Δpp | 精确双侧p | 任务聚类95%区间pp |
|---|---|---:|---:|---:|---:|---:|
| clean | frozen | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | pixel_align | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | deep_block12 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | ours_v1 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | frozen | 3/2 | +1 | +1.1 | 1 | [-4.9, +9.0] |
| sensor_bilateral | pixel_align | 4/1 | +3 | +3.3 | 0.375 | [-1.1, +8.3] |
| sensor_bilateral | deep_block12 | 5/4 | +1 | +1.1 | 1 | [-4.9, +8.1] |
| sensor_bilateral | ours_v1 | 11/0 | +11 | +12.2 | 0.000976562 | [+2.7, +23.5] |
| lighting | frozen | 9/3 | +6 | +6.7 | 0.145996 | [+0.0, +14.7] |
| lighting | pixel_align | 16/0 | +16 | +17.8 | 3.05176e-05 | [+6.2, +33.3] |
| lighting | deep_block12 | 11/5 | +6 | +6.7 | 0.210114 | [-3.5, +18.1] |
| lighting | ours_v1 | 17/5 | +12 | +13.3 | 0.0169005 | [+3.3, +29.2] |
| background | frozen | 7/6 | +1 | +1.1 | 1 | [-11.1, +10.8] |
| background | pixel_align | 3/35 | -32 | -35.6 | 6.67787e-08 | [-48.7, -20.8] |
| background | deep_block12 | 3/8 | -5 | -5.6 | 0.226562 | [-9.5, -1.6] |
| background | ours_v1 | 7/7 | +0 | +0.0 | 1 | [-7.9, +6.3] |
| table_texture | frozen | 5/3 | +2 | +4.4 | 0.726562 | [-15.2, +15.9] |
| table_texture | pixel_align | 0/18 | -18 | -40.0 | 7.62939e-06 | [-59.5, -24.2] |
| table_texture | deep_block12 | 2/6 | -4 | -8.9 | 0.289062 | [-16.7, -1.9] |
| table_texture | ours_v1 | 5/5 | +0 | +0.0 | 1 | [-13.9, +11.9] |
| scene_texture | frozen | 2/3 | -1 | -2.2 | 1 | [-14.8, +6.7] |
| scene_texture | pixel_align | 3/17 | -14 | -31.1 | 0.00257683 | [-51.0, -2.6] |
| scene_texture | deep_block12 | 1/2 | -1 | -2.2 | 1 | [-8.3, +0.0] |
| scene_texture | ours_v1 | 2/2 | +0 | +0.0 | 1 | [-4.5, +7.7] |
| total | frozen | 19/11 | +8 | +2.7 | 0.200488 | [-1.2, +6.2] |
| total | pixel_align | 23/36 | -13 | -4.3 | 0.117477 | [-12.7, +5.6] |
| total | deep_block12 | 19/17 | +2 | +0.7 | 0.867939 | [-3.1, +5.0] |
| total | ours_v1 | 35/12 | +23 | +7.7 | 0.00108854 | [+2.3, +15.8] |

### 各种子Ours对Frozen净局数

| 组 | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| clean | +0 | +0 | +0 |
| sensor_bilateral | +2 | +3 | +5 |
| lighting | +4 | -1 | +3 |
| background | +4 | -2 | -3 |
| table_texture | +3 | -2 | -3 |
| scene_texture | +1 | +0 | +0 |
| total | +10 | +0 | +5 |
## object

| 组 | Frozen | Pixel-Align | Deep B12 | ActMAD-inspired v2 | Ours V1 |
|---|---:|---:|---:|---:|---:|
| clean | 30/30 | 30/30 | 25/30 | 30/30 | 30/30 |
| sensor_bilateral | 20/90 | 21/90 | 19/90 | 20/90 | 22/90 |
| lighting | 86/90 | 84/90 | 77/90 | 87/90 | 87/90 |
| background | 41/90 | 43/90 | 43/90 | 44/90 | 49/90 |
| table_texture | 26/66 | 22/66 | 30/66 | 27/66 | 31/66 |
| scene_texture | 15/24 | 21/24 | 13/24 | 17/24 | 18/24 |
| total | 177/300 | 178/300 | 164/300 | 181/300 | 188/300 |

### 配对比较：相对 frozen

| 组 | 臂 | 救回/误伤 | 净增 | Δpp | 精确双侧p | 任务聚类95%区间pp |
|---|---|---:|---:|---:|---:|---:|
| clean | pixel_align | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | deep_block12 | 0/5 | -5 | -16.7 | 0.0625 | [-40.0, +0.0] |
| clean | actmad_inspired_viewwise_v2 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | ours_v1 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | pixel_align | 3/2 | +1 | +1.1 | 1 | [-3.3, +6.7] |
| sensor_bilateral | deep_block12 | 1/2 | -1 | -1.1 | 1 | [-6.7, +3.3] |
| sensor_bilateral | actmad_inspired_viewwise_v2 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | ours_v1 | 2/0 | +2 | +2.2 | 0.5 | [+0.0, +6.7] |
| lighting | pixel_align | 4/6 | -2 | -2.2 | 0.753906 | [-7.8, +2.2] |
| lighting | deep_block12 | 1/10 | -9 | -10.0 | 0.0117188 | [-22.2, +0.0] |
| lighting | actmad_inspired_viewwise_v2 | 2/1 | +1 | +1.1 | 1 | [-3.3, +6.7] |
| lighting | ours_v1 | 4/3 | +1 | +1.1 | 1 | [-2.2, +4.4] |
| background | pixel_align | 14/12 | +2 | +2.2 | 0.845019 | [-8.9, +14.4] |
| background | deep_block12 | 13/11 | +2 | +2.2 | 0.83882 | [-8.9, +14.4] |
| background | actmad_inspired_viewwise_v2 | 8/5 | +3 | +3.3 | 0.581055 | [-2.2, +10.0] |
| background | ours_v1 | 13/5 | +8 | +8.9 | 0.0962524 | [-4.4, +22.2] |
| table_texture | pixel_align | 8/12 | -4 | -6.1 | 0.503445 | [-15.9, +3.9] |
| table_texture | deep_block12 | 11/7 | +4 | +6.1 | 0.480682 | [-4.8, +19.2] |
| table_texture | actmad_inspired_viewwise_v2 | 5/4 | +1 | +1.5 | 1 | [-6.2, +9.0] |
| table_texture | ours_v1 | 8/3 | +5 | +7.6 | 0.226562 | [-2.0, +17.5] |
| scene_texture | pixel_align | 6/0 | +6 | +25.0 | 0.03125 | [+7.4, +41.7] |
| scene_texture | deep_block12 | 2/4 | -2 | -8.3 | 0.6875 | [-33.3, +7.4] |
| scene_texture | actmad_inspired_viewwise_v2 | 3/1 | +2 | +8.3 | 0.625 | [+0.0, +18.5] |
| scene_texture | ours_v1 | 5/2 | +3 | +12.5 | 0.453125 | [-20.8, +37.5] |
| total | pixel_align | 21/20 | +1 | +0.3 | 1 | [-3.3, +4.3] |
| total | deep_block12 | 15/28 | -13 | -4.3 | 0.065994 | [-12.3, +2.7] |
| total | actmad_inspired_viewwise_v2 | 10/6 | +4 | +1.3 | 0.454498 | [-1.0, +4.0] |
| total | ours_v1 | 19/8 | +11 | +3.7 | 0.052239 | [-0.7, +8.3] |

### 配对比较：相对 actmad_inspired_viewwise_v2

| 组 | 臂 | 救回/误伤 | 净增 | Δpp | 精确双侧p | 任务聚类95%区间pp |
|---|---|---:|---:|---:|---:|---:|
| clean | frozen | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | pixel_align | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| clean | deep_block12 | 0/5 | -5 | -16.7 | 0.0625 | [-40.0, +0.0] |
| clean | ours_v1 | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | frozen | 0/0 | +0 | +0.0 | 1 | [+0.0, +0.0] |
| sensor_bilateral | pixel_align | 3/2 | +1 | +1.1 | 1 | [-3.3, +6.7] |
| sensor_bilateral | deep_block12 | 1/2 | -1 | -1.1 | 1 | [-6.7, +3.3] |
| sensor_bilateral | ours_v1 | 2/0 | +2 | +2.2 | 0.5 | [+0.0, +6.7] |
| lighting | frozen | 1/2 | -1 | -1.1 | 1 | [-6.7, +3.3] |
| lighting | pixel_align | 3/6 | -3 | -3.3 | 0.507812 | [-11.1, +4.4] |
| lighting | deep_block12 | 2/12 | -10 | -11.1 | 0.0129395 | [-24.4, +0.0] |
| lighting | ours_v1 | 3/3 | +0 | +0.0 | 1 | [-4.4, +5.6] |
| background | frozen | 5/8 | -3 | -3.3 | 0.581055 | [-10.0, +2.2] |
| background | pixel_align | 11/12 | -1 | -1.1 | 1 | [-15.6, +11.1] |
| background | deep_block12 | 10/11 | -1 | -1.1 | 1 | [-12.2, +11.1] |
| background | ours_v1 | 10/5 | +5 | +5.6 | 0.301758 | [-4.4, +15.6] |
| table_texture | frozen | 4/5 | -1 | -1.5 | 1 | [-9.0, +6.2] |
| table_texture | pixel_align | 7/12 | -5 | -7.6 | 0.359283 | [-22.2, +3.7] |
| table_texture | deep_block12 | 9/6 | +3 | +4.5 | 0.607239 | [-8.7, +19.7] |
| table_texture | ours_v1 | 7/3 | +4 | +6.1 | 0.34375 | [-4.0, +14.8] |
| scene_texture | frozen | 1/3 | -2 | -8.3 | 0.625 | [-18.5, +0.0] |
| scene_texture | pixel_align | 4/0 | +4 | +16.7 | 0.125 | [+3.7, +29.6] |
| scene_texture | deep_block12 | 1/5 | -4 | -16.7 | 0.21875 | [-38.9, -3.0] |
| scene_texture | ours_v1 | 3/2 | +1 | +4.2 | 1 | [-25.0, +25.0] |
| total | frozen | 6/10 | -4 | -1.3 | 0.454498 | [-4.0, +1.0] |
| total | pixel_align | 17/20 | -3 | -1.0 | 0.742829 | [-5.3, +3.0] |
| total | deep_block12 | 13/30 | -17 | -5.7 | 0.0137182 | [-13.0, +1.7] |
| total | ours_v1 | 15/8 | +7 | +2.3 | 0.21004 | [-1.3, +6.3] |

### 各种子Ours对Frozen净局数

| 组 | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| clean | +0 | +0 | +0 |
| sensor_bilateral | +1 | +0 | +1 |
| lighting | +0 | -3 | +4 |
| background | +2 | +2 | +4 |
| table_texture | +0 | +2 | +3 |
| scene_texture | +2 | +0 | +1 |
| total | +3 | -1 | +9 |

## 更新后的效率实测

CUDA-synchronized steady replan; ActMAD includes 4-frame SGD+parameter-step diagnostic+prediction, excludes episode construction/prefill/close

| 臂 | 中位ms | 相对Frozen | 峰值allocated MiB中位 |
|---|---:|---:|---:|
| frozen | 44.660 | 1.000× | 262.81 |
| pixel_align | 45.727 | 1.024× | 263.09 |
| deep_block12 | 44.999 | 1.008× | 262.81 |
| actmad_inspired_viewwise_v2 | 243.756 | 5.458× | 1144.17 |
| ours_v1 | 45.304 | 1.014× | 262.81 |

## 审计与解释边界

- 所有统计直接读取完整原始JSON，binomtest().pvalue不再乘2；bootstrap10,000次按任务整组重采样，保留各任务全部条件与种子。
- 每个case/seed五臂一起原子提交；断点仅跳过通过指纹/完整性核验的完整记录，未完成job按原种子重跑，不按成败筛选。
- 初始物理状态、机器人读数、双目原始输入像素及Flow噪声前缀逐对检查。旧协议限定同物理状态下≤64像素、≤1灰度的EGL舍入统一，逐项记录；不同状态绝不复用图片。
- 参考来自同十条Frozen源轨迹101推理块；第3层统计逐位复现，并与冻结参考逐位复算一致。拟合全部在原生Clean前向，未叠适配。
- 只有十个任务簇，结果不是未见任务总体保证。Clean误伤、负结果、近地板及无显著性均如实保留；不得只凭total最高宣称安全或普适。
- Object零样本指零Object校准/选参；官方池历史已访问，不能冒称未触碰盲测。保护池未读取。
