# V0 + V1.1 实现说明

> 范围：只实现 gate 接口与同状态配对审计；**未训练、未跑正式 rollout、未做 V1.2–V1.4/V2**。

## 1. 新增代码

| 文件 | 作用 |
|---|---|
| `dacsc/visual_gate.py` | `VisualTokenGate`（identity/scale，按 view 分参数）+ `GatedRepresentationExtractor`（在 DINO patch tokens 后、Resampler 前插入 gate） |
| `dacsc/state_pairing.py` | MuJoCo 全状态快照/恢复、数值比较、camera 比较、state digest |
| `scripts/v0_identity_gate_check.py` | V0：identity gate 全链路 no-op 检查 |
| `scripts/v11_state_pair_audit.py` | V1.1：Clean/Shift 同物理状态配对审计与 manifest |
| `tests/test_visual_gate.py` | gate identity/scale/异常 view 单测 |
| `tests/test_state_pairing.py` | snapshot 比较、digest、camera 单测 |

## 2. V1.1 hardening v6（Preflight A 冻结）

已修复：

- `compare()` fail-closed：缺失字段、单侧空数组、shape 不一致、非 finite 一律失败；不再用默认 0.0 表示未比较。
- 增加 `body_xquat`（四元数 q/−q 等价角误差）、`eef_pos`、`eef_name` 必需检查。
- `model_signature` / `model_compatible()`：restore 前比较 nq/nv/nu、关节名/类型/地址、actuator、body names。
- `camera_compare()`：names/shape/finite 检查；区分“应相同”和“应发生 camera 变化”；NaN 不会误判为相同。
- V1.1 脚本 fail-closed：物理状态、模型兼容、camera 预期、必需视角、实际图像差异任一失败 → 非零退出。
- 图像用无损 PNG；计算 mean abs diff 与 changed fraction；不再只用 SHA 判断。
- `init_index` 严格范围检查；记录 init 文件 hash；输出目录按 task/init 唯一化；manifest 单行 JSONL。
- 测试改为 unittest，覆盖缺失字段、单侧空 act、body 四元数、NaN camera、shape mismatch、EEF 缺失、model mismatch。
- **四元数单位修复**：`_quat_angle_rad()` 现在返回 radian，`body_xquat_rad` 容差不再被误读为 degree；增加 q/−q 等价测试。
- **Camera 验收改为 target cameras**：不再要求所有 MuJoCo camera 都变化；新增 `--policy-views` 与 `--target-cameras`；Camera 变体只要求目标视角 camera 参数和图像变化。
- **Task identity preflight**：新增 `dacsc/task_pairing.py`，比较 language、canonical base name、BDDL goal text、object set、BDDL SHA；不通过直接 fail。`--allow-language-diff` 已删除。
- **Manifest fail-closed**：每次运行有 `run_id`，JSONL 追加写入；成功/失败都写 manifest；失败路径不再只留在 stderr。
- **图像 dtype/range 检查**：`to_hwc()` 检查 dtype、范围、通道数；float [0,1] 显式转换，异常立即报错。
- **render GPU 参数**：`--render-gpu` 控制 `render_gpu_device_id`，不再与 `--device` 混淆。
- **settling 检查**：settling 期间 episode 提前结束 → fail。
- **任务身份记录**：记录 BDDL SHA、goal/objects/language、LIBERO git rev、task classification SHA、版本、seed、thresholds。
- **官方 Python task map loader 修复**：读取 `module.libero_task_map`，不再要求外层字典值全是字符串；新增官方结构测试。
- **vanilla Clean BDDL 不再被覆盖**：`--clean-source vanilla` 使用显式 `--clean-bddl` / `--clean-init-file`，resolve 后校验文件存在；manifest 记录输入路径与实际路径。
- **实际 instruction 修复**：`canonical_instruction()` 同时处理下划线后缀和空格形式（`view 46 0 100 0 0 initstate 0`）；manifest 分开记录 task.language、canonical_language、clean policy_instruction。
- **分组对象声明**：`parse_objects()` 支持 `name_1 name_2 - type`，所有 pending names 获得同一 type。
- **三方身份一致性**：task map name vs benchmark task name vs classification name（若提供）；不一致 fail。
- **category 别名**：Camera Viewpoints / Background Textures / Light Conditions 精确映射；Clean 排除规则覆盖 light/sensor/noise。
- **Camera 模式强制显式 `--target-cameras`**；不再默认所有策略视角。
- **失败 Manifest 更稳**：环境 close 单独捕获；cleanup_errors 写入；setup/import 在 try 内；vanilla 的 clean task id 记为 null。
- **Clean 文件身份硬校验**：vanilla 要求 `clean_bddl.stem == clean_task_name`，init 文件名必须是 `<name>.pruned_init` / `.init` / `.pruned_init.pkl`；不再只比较 CLI 参数与自身副本。
- **共同 policy instruction 冻结**：Clean/Shift 使用同一 `policy_instruction`；要求 `canonical_instruction(clean_policy) == canonical_instruction(shift_task.language)`；manifest 分别记录 raw/canonical/actual policy instruction。
- **classification 严格唯一 + 完整 variant name**：必须恰好一条记录；若有 name，必须与 task map 完整任务名一致（不只是 base name）；category 精确匹配。
- **物理几何签名**：`model_signature` 记录 geom names/bodyid/type/size/pos/quat；Camera 模式 `strict_geometry=True`，防止 Camera 变体同时改变物理几何。
- **顶层 import 移入事务**：`evaluate_libero` 在 run_id/record 创建后导入；import/config 失败也写失败 manifest。
- **物理模型签名扩展**：geom collision/friction/solref/margin/gap、body mass/inertia、dof damping/frictionloss/armature、actuator gain/bias/gear/ctrlrange；Camera 模式 `strict_physics=True`。
- **Clean 身份/prompt 抽成纯函数**：`dacsc/pairing_checks.py`；测试覆盖错误 BDDL/init、错误 vanilla prompt。
- **Camera 非目标视角**：Camera 模式要求非目标相机参数保持不变，并拒绝重复 camera-map。
- **vanilla prompt 独立锚定**：由 task name 生成 `expected_vanilla_instruction`，与用户输入比较后才冻结。

## 2. V0 结果（本机 CPU，官方 checkpoint）

命令：

```
.venv/bin/python scripts/v0_identity_gate_check.py --device cpu \
  --out /tmp/v0_identity_gate.json
```

结果：

```json
{
  "patch_dim": 384,
  "identity_raw_patches_equal": true,
  "identity_representation_equal": true,
  "identity_pooled_equal": true,
  "identity_dtype_equal": true,
  "identity_action_equal": true,
  "suppression_tokens_changed": true,
  "suppression_action_changed": true
}
```

- identity gate 直接返回原 tensor，不乘 1；
- patch → resampler token → action chunk 全链路一致；
- 非 identity（suppression gamma=0.5）会改变 token 和动作，说明插入点有效；
- V0 退出条件包含 raw patches/representation/action 一致 + suppression token 变化。

> 这是接口验收，不是方法有效性结论。

## 3. Preflight A（只读，不建环境）

```bash
python scripts/v11_state_pair_audit.py --preflight-only \
  --clean-source vanilla --clean-suite libero_spatial \
  --clean-task-name pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate \
  --clean-bddl /path/to/vanilla/bddl_files/libero_spatial/<name>.bddl \
  --clean-language "pick up the black bowl on the cookie box and place it on the plate" \
  --clean-init-file /path/to/vanilla/init_files/libero_spatial/<name>.pruned_init \
  --vanilla-src /path/to/vanilla \
  --shift-suite libero_spatial --shift-task 705 --shift-kind camera \
  --task-map /path/to/libero_suite_task_map.py \
  --task-classification /path/to/task_classification.json \
  --target-cameras agentview --init-index 0 --out /hy-tmp/validation/v11_preflight_a
```

`--preflight-only` 只做身份/文件/prompt 检查，不创建 `OffScreenRenderEnv`、不 reset、不 render；通过后退出 0，并写 manifest。

## 3.1 V1.1 使用方式（需 GPU/LIBERO 环境）

旧命令 `--clean-task 0 --shift-task 705` 已废弃：Plus task 0 与 705 不是同一基础任务。  
当前必须提供 task map、classification 和显式 Clean 来源。示例（路径按服务器 revision 替换）：

```bash
python scripts/v11_state_pair_audit.py \
  --clean-source vanilla \
  --clean-suite libero_spatial \
  --clean-task-name pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate \
  --clean-bddl /path/to/vanilla/bddl_files/libero_spatial/<name>.bddl \
  --clean-language "pick up the black bowl on the cookie box and place it on the plate" \
  --clean-init-file /path/to/vanilla/init_files/libero_spatial/<name>.pruned_init \
  --shift-suite libero_spatial --shift-task 705 --shift-kind camera \
  --task-map /path/to/libero_suite_task_map.py \
  --task-classification /path/to/task_classification.json \
  --policy-views agentview eye_in_hand \
  --camera-map eye_in_hand=robot0_eye_in_hand \
  --target-cameras agentview \
  --init-index 0 \
  --out /hy-tmp/validation/v11_pair_audit
```

脚本做：

1. 校验 task map、classification、Clean 文件身份和共同 policy instruction；
2. 建 Clean / Shift 两个 `OffScreenRenderEnv`；
3. Clean 环境 `set_init_state` + settling，取全状态快照；
4. 把快照恢复进 Shift 环境；
5. 比较物理状态、body pose/EEF、Camera 参数与物理几何/动力学签名；
6. 保存 policy views 的**无损 PNG** + sha256、像素差异和 pair manifest。

输出：

```
/hy-tmp/validation/v11_pair_audit/pair_manifest.jsonl   # append-only
/hy-tmp/validation/v11_pair_audit/pair_manifest_<run_id>.json
/hy-tmp/validation/v11_pair_audit/<kind>/<task-pair>/run_<run_id>/*.png
```

## 4. 已确认：libero_spatial 0 → 705 不是有效配对

根据当前 LIBERO-Plus task map：

- Plus `libero_spatial` task 0 = `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1`；
- Plus `libero_spatial` task 705 = Camera View 变体，基础任务是 `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`。

两者基础操作不同，**不是同一任务的 Clean/Shift 配对**。旧命令 `--clean-task 0 --shift-task 705` 已废弃。

当前正确的候选 Clean reference 是 **vanilla LIBERO spatial task 3**（同样的 cookie-box 基础任务），而不是 Plus 的 task 0。  
V1.1 现在要求显式 `--task-map` 与 `--task-classification`，并比较 base name / goal / objects / canonical language；它不会再把 0→705 当作合法配对。

### Camera suffix canonicalization

真实 Camera 名称如：

```
..._view_46_0_100_0_0_initstate_0
```

现在 `canonical_base_name()` 只剥离已知参数化 variant 后缀（`_view_<params>...`、`_table_...`、`_initstate_N` 等），**不再使用宽泛的 `_\d+$`**。测试覆盖真实 Camera 后缀和同名不同物体类型。

### 新的 Clean/Shift 输入

- Shift：Plus suite/task index + `task_map` + `task_classification`；
- Clean：
  - Plus 来源：`--clean-source plus`，走 `clean_suite.get_task_init_states()`；
  - Vanilla 来源：`--clean-source vanilla`，必须显式提供 `--clean-task-name`、`--clean-bddl`、`--clean-language`、`--clean-init-file`，避免把 Plus variant 当 Clean。

### Camera 名称拆分

```text
CoRP policy view   : agentview / eye_in_hand
LIBERO obs key     : agentview_image / robot0_eye_in_hand_image
MuJoCo camera name : agentview / robot0_eye_in_hand
```

默认映射：`--camera-map eye_in_hand=robot0_eye_in_hand`；`--target-cameras` 现在按 **policy view** 指定，内部再映射到 MuJoCo camera。Camera 模式不再默认要求所有视角都变化。

## 5. 运行前必读

V1.1 现在**必须**提供：

```
--task-map <installed libero_suite_task_map.py>
--task-classification <installed task_classification.json>
```

Vanilla Clean 示例（路径按服务器实际安装替换）：

```
python scripts/v11_state_pair_audit.py   --clean-source vanilla   --clean-suite libero_spatial   --clean-task-name pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate   --clean-bddl /path/to/libero/bddl_files/libero_spatial/<name>.bddl   --clean-language "pick up the black bowl on the cookie box and place it on the plate"   --clean-init-file /path/to/libero/init_files/libero_spatial/<name>.pruned_init   --shift-suite libero_spatial --shift-task 705 --shift-kind camera   --task-map /path/to/libero_suite_task_map.py   --task-classification /path/to/task_classification.json   --target-cameras agentview --init-index 0
```

## 6. 验收标准

| 检查 | 标准 |
|---|---|
| body_names_equal | true |
| qpos/qvel/act | max abs err ≤ 预注册容差（默认 1e-6） |
| body_xpos | ≤ 1e-5 |
| time | ≤ 1e-9 |
| camera_ok | Background/Lighting 相同；Camera 允许不同 |
| state digest | 记录 clean/restored 两个 digest；不要求相同，用于追踪 |

如果 qpos/body 误差超过容差，说明“同状态配对”没有成立，不能进入 V1.2。

## 7. 当前阻塞与未做项

- V1.1 尚未在服务器运行（GPU 主机当前不可达）；
- V1.2–V1.4、V2 未实现；
- 没有 teacher-student、可学习 gate、delayed-feedback β、reward model；
- 没有正式 rollout。

## 8. 单测

已统一为 unittest（符合仓库 `python -m unittest discover -s tests -v`）：

- `test_visual_gate.py`：5 tests passed；
- `test_state_pairing.py`：15 tests passed（含 strict geometry/physics）；
- `test_task_pairing.py`：10 tests passed；
- `test_pairing_checks.py`：3 tests passed（Clean 文件身份 + prompt 源锚定）；
- 新增 `--preflight-only` 与 `--vanilla-src`；source binding 检查 task map/classification/vanilla 文件是否在指定 checkout 下；Camera 模式严格要求 target cameras。
- 覆盖 identity no-op、错误 view/dim、suppression bound、缺失字段、单侧空 act、四元数、NaN camera、shape mismatch、EEF 缺失、model mismatch。

## 9. 仍未闭合的 P1

- 顶层 `dacsc`/torch import 仍在 main 前；但 `evaluate_libero` 已移入事务。
- vanilla Clean 的 BDDL/init 资产与本机 Plus 包混用；需要在服务器 revision 上确认资产一致。
- `task_classification` 的真实 schema 尚未在服务器验证；本地测试是合成结构。
- 未执行真实 MuJoCo restore。
- 未做 V1.2、训练或 rollout。

## 10. 下一步

1. 服务器先跑 Preflight A：确认 LIBERO-Plus revision、task_classification、clean/shift 是否同 base task；
2. 再跑 V1.1 单 pair，只做一个 task × 一个扰动 × 一个 init state；
3. 检查 qpos/qvel/body/EEF、target camera、policy views、PNG 像素差异、manifest；
4. 通过后再做重复性检查，再扩 Background/Lighting；
5. V1.1 不过，不进入 V1.2。


## 11. First end-to-end server run（2026-11-16）

服务器：`i-2.gpushare.com:26990`，RTX 5090 32GB，Python 3.11，torch 2.9.1+cu128。
LIBERO-plus 主仓 + 9.5GB assets 已安装到 `/hy-tmp/libero_plus`；数据盘 100GB。

### V0

`python3 scripts/v0_identity_gate_check.py --device cuda`：通过。
identity raw/representation/pooled/action 全部相等；suppression 改变 token/action。

### Preflight A

使用 LIBERO-plus `libero_suite_task_map.py` + `task_classification.json`，
Clean = vanilla spatial base task（cookie-box 任务），Shift = Camera task 705。

结果：

- task map base name / BDDL goal / objects / canonical language：通过；
- 官方 classification id 为 **1-based**：benchmark index 705 → classification id 706；
- full variant name 匹配通过；
- Clean/Shift 共用 policy instruction；
- `audit.pass=true`，退出码 0。

### V1.1 单 pair

Clean base BDDL/init 与 Camera 705（virtual `_view_` BDDL）各建一个 env；
恢复 Clean 物理状态到 Shift env。

结果：

- qpos/qvel/act/mocap/time：误差 0；
- body_xpos max error ≈ 1.23e-5 m；body_xquat ≈ 3.74e-5 rad；
- Camera target `agentview`：cam_pos diff ≈ 0.474，cam_xmat diff ≈ 0.719；
- 非目标 `robot0_eye_in_hand`：差异 < 1e-4；
- 图像：agentview mean abs diff ≈ 0.142，changed fraction ≈ 0.808；
- `audit.pass=true`，退出码 0。

注意：这只是**单 task × 单 Camera 变体 × 单 init state 的静态配对**，不是 V1.2 数据，更不是方法有效性。

## 12. 必记的 LIBERO-plus 实现细节

- `task_classification.json` 的 `id` 是 **1-based**；benchmark/task-map index 是 0-based。
- Camera 变体的 BDDL 文件不存在，`OffScreenRenderEnv` 需要传入 **virtual `_view_...bddl` 文件名**；
  它自己会 split `_view_` 并加载 base BDDL，同时设置 camera 参数。
- `get_task_init_states` 在 torch>=2.6 需要 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
- robosuite 1.4.0 与 mujoco 4.x 不兼容；已回钉 mujoco 2.3.7。
- 运行环境：`LIBERO_CONFIG_PATH=/hy-tmp/libero_config MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。


## 13. V1.2 pilot capture（2026-11-16）

在 RTX 5090 上采集：

- Base task：cookie-box spatial task；
- Shift：Camera task 705；
- 50 个 Clean init states（实际 49 个通过状态恢复，state 42 body quat 误差 1.12e-4 略超 1e-4，被跳过并记录）；
- 每状态双视角 Clean/Shift PNG + DINO patch features `[256,384]`；
- 固定 split seed 1234：30 dev / 19 held-out（按有效状态）；
- 无策略动作、无 rollout。

输出：

```text
/hy-tmp/validation/v12_pilot/features.npz   69MB
/hy-tmp/validation/v12_pilot/manifest.json
/hy-tmp/validation/v12_pilot/scores.json
```

### s_pair 初步结果

- `s_pair(d) = E|F_clean - F_shift|`，跨 49 states、双视角、256 patches；
- pair mean ≈ 1.2327；
- dev vs held-out Spearman ≈ 0.970；
- split-half Spearman ≈ 0.952；
- agentview vs eye-in-hand pair-score correlation ≈ 0.070（视角差异明显，不能混成一个分数）。

### 当前限制

- 只有一个 task、一个 Camera 变体；
- `s_stat` 只是跨状态 std proxy，不是已证明的 nuisance 分数；
- `s_aug` 尚未实现；
- V1.3 action propagation 尚未执行；
- 49 个状态只能作为 pilot，不能作为 V1.4 结论。

结果文件：`docs/data/v12_scores.json`、`docs/data/v12_manifest.json`。


## 14. V1.2 score correction（针对 71ed644 复审）

旧 `v12_scores.py` 的全局 `pair_mean` 是两个视角的平均，不能作为正式 V1.2 结果；已改为 **per-view** 统计。

### 修正后的 pilot 结果（49 valid；30 dev / 19 held-out）

| 视角 | s_pair dev mean | s_pair held-out mean | dev-vs-held Spearman | top-20 overlap | repeated split-half median |
|---|---:|---:|---:|---:|---:|
| agentview（目标） | 2.4430 | 2.4455 | 0.9698 | 0.80 | 0.9532 |
| eye-in-hand（对照） | 0.02173 | 0.02098 | 0.7465 | 0.45 | 0.6366 |

- agentview 的 bootstrap 95% CI（s_pair mean）：[2.4243, 2.4668]；
- eye-in-hand：[0.02049, 0.02245]；
- `s_stat` 现在是 **每个视角 384 维**：
  - primary：`std_state( mean_patch F_shift )`
  - alt：`mean_patch( std_state F_shift )`
- 不再跨视角平均。

### 解释边界

- agentview 是目标 Camera 视角；eye-in-hand 是未受 Camera 扰动的对照。
- agentview 与 eye-in-hand 的 s_pair 尺度相差约 114×，排序相关性很低；**不能合并**。
- 49 个状态仍是单 task / 单 Camera 变体的 pilot，不能外推到其他任务/扰动。
- `s_aug` 仍未实现；V1.3 action propagation 未启动。
- `s_stat` 仍是跨状态波动 proxy，不是已证明的 nuisance sensitivity。

### Provenance

当前 manifest 追加了：

- Clean init 文件路径与 SHA256；
- `features.npz` SHA256；
- 196 张 PNG 的组合 hash；
- Python/torch/numpy/mujoco/robosuite 版本；
- LIBERO-plus tarball SHA256。

文件：`docs/data/v12_scores_per_view.json`、`docs/data/v12_manifest_provenance.json`。
旧 `v12_scores.json`（全局平均版）只作历史记录，不再作为正式结果。


## 15. V1.2 s_aug pilot（定义冻结后）

定义：

- `s_aug_photo`：brightness / contrast / saturation，各 0.8、1.2；
- `s_aug_geo`：translate ±8 px、crop 95%、scale 1.05/0.95；
- 只增强目标视角，另一个视角保持原图；
- 双视角完整输入 `prepare_observation`；
- 在 Shift 原始图像上做增强，之后统一 180° rotate + resize + ImageNet normalize；
- Dev 只用于定义/开发，Held-out 只用于评价。

### Pilot 结果（49 valid；30 dev / 19 held-out）

| 视角 | photo mean | geo mean | photo dev-vs-held | geo dev-vs-held |
|---|---:|---:|---:|---:|
| agentview | 0.3924 | 1.0082 | 0.9332 | 0.9015 |
| eye-in-hand | 0.2387 | 1.2497 | 0.9372 | 0.9798 |

与 Held-out `s_pair` 的 Spearman：

| 视角 | photo vs pair | geo vs pair |
|---|---:|---:|
| agentview | 0.2267 | 0.1921 |
| eye-in-hand | 0.4291 | 0.3018 |

解释：

- 两种增强代理在跨状态上自身是稳定的（dev/held 0.90–0.98）；
- 但它们**不能复现 Camera 配对敏感度**，agentview 上相关性只有约 0.19–0.23；
- 因此当前 `s_aug_photo/geo` 不能直接作为 V1.4 的可部署 gate 代理；至少对 Camera 变体不成立；
- 这不是“增强无效”，而是“增强代理与真实 Camera 变化的机制不同”。

### 当前边界

- 只有单 task + 单 Camera 变体；
- 未做 V1.3；
- 未用 Held-out `s_pair` 调整增强参数；
- `s_aug` 结论只适用于本次 Camera 变体，不能外推到 Background/Lighting。


## 16. V1.3 action propagation pilot（2026-11-16）

使用 V1.2 的 49 个有效状态、保存的 DINO features 和图片；重新从 clean env 取得真实 8D policy state；
使用 common initial action noise z0；不执行 simulator action，不跑 rollout。

对照：

- `pair`：A_shift(z0) - A_clean(z0)；
- `gate_oracle`：在 shift features 上抑制 Dev s_pair(agentview) Top-20 DINO dims，A_gate(z0) - A_shift(z0)；
- `gate_random`：同预算随机 20 dims，A_random(z0) - A_shift(z0)，20 组随机 gate 平均；
- `noise`：同一 shift 输入、不同 z1/z2，A_shift(z1)-A_shift(z2)。

单位为反归一化 controller units 的平均绝对差。

| 指标 | xyz | rot | gripper | all | norm_all |
|---|---:|---:|---:|---:|---:|
| pair | 0.1483 | 0.02000 | 0.002576 | 0.07251 | 0.04710 |
| gate_oracle | 0.02327 | 0.002629 | 0.000590 | 0.01118 | 0.006984 |
| gate_random | 0.008393 | 0.001508 | 0.000439 | 0.004306 | 0.002860 |
| noise | 0.07070 | 0.01335 | 0.002677 | 0.03640 | 0.02440 |

初步解释：

- Camera Clean→Shift 的动作差异（all 0.0725）约为采样噪声（0.0364）的 2 倍，存在动作传播信号；
- Oracle Gate 的动作改变（0.01118）约为 Matched Random（0.00431）的 2.6 倍，方向选择并非完全随机；
- 但 Oracle Gate 的改动幅度远小于 Clean→Shift 差异，且尚未比较 Gate 是否让动作更接近 Clean；
- 仍是单 task / 单 Camera / 49 states pilot，不能作为方法有效性或成功率证据。

未做：

- ΔA_GC（Gate 后动作到 Clean 的距离）；
- 按 state 的 bootstrap / 置信区间；
- V1.4 代理验证；
- V2 完整任务成功率与 Clean retention。


## 17. 修正记录（针对 8325adf/eb7d2e7 复审）

### 17.1 s_aug_geo 修正

- `scale` 用“resize 两次”并未改变视野；已改为真正的 center-crop + resize zoom-in（1.05、1.10）；
- `translate` 改为 reflect padding，避免黑色边缘成为额外扰动；
- 分别输出 `geo_groups.translate/crop/zoom`，不再把三者混成一个不可解释的 geo 分数。

修正后（held-out）：

| 视角 | photo | geo | photo vs pair_held | geo vs pair_held | geo dev/held |
|---|---:|---:|---:|---:|---:|
| agentview | 0.3924 | 1.3097 | 0.2267 | 0.1811 | 0.8849 |
| eye-in-hand | 0.2387 | 1.8267 | 0.4291 | 0.2683 | 0.9808 |

结论不变：增强代理自身稳定，但不能复现 Camera 配对敏感度；geo 修正后仍弱。

### 17.2 s_stat_alt 拆分

已增加 `stat_alt_dev` / `stat_alt_heldout`。与 held-out `s_pair` 的 Spearman：

| 视角 | stat_dev | stat_heldout | stat_alt_dev | stat_alt_heldout |
|---|---:|---:|---:|---:|
| agentview | 0.1256 | 0.1618 | 0.1713 | 0.1951 |
| eye-in-hand | 0.3375 | 0.2842 | 0.3909 | 0.3377 |

agentview 上当前 `s_stat` 仍不能作为可靠代理。

### 17.3 V1.3 修正

- Gate 维度只来自 Dev `s_pair`；Dev/Held-out 分开聚合；
- 增加 `D_GC = ||A_G - A_C||` 和 `align_cos`；
- 10 组 common z0；8 组随机 gate；torch seed 固定；
- 当前为 FP32 协议（未用官方 BF16 autocast）。

Held-out（19 states）：

| 指标 | all | norm_all |
|---|---:|---:|
| pair (S-C) | 0.07095 | 0.04592 |
| oracle gate (G-S) | 0.01073 | 0.00665 |
| random gate (G-S) | 0.00415 | 0.00275 |
| gate-to-clean (G-C) | 0.06218 | 0.04107 |
| noise | 0.03549 | 0.02367 |

- `D_GC ≈ 0.06218 < D_SC ≈ 0.07095`，约降低 12.4%；
- `align_cos ≈ 0.815`，Gate 动作方向与 Clean 方向接近；
- Oracle Gate 动作影响约为 Random 的 2.6 倍；
- 但 Oracle 抑制的特征能量约 59.8k，Random 约 27.4k（2.18×），因此**不能排除能量差异**；还需 energy-matched random。

Dev（30 states）：pair 0.07406，oracle 0.01160，gate-to-clean 0.06455，random 0.00427，noise 0.03647。

### 17.4 当前可支持的最强结论

> Dev 选的配对敏感维度，在 held-out 初始状态上造成了比随机维度更大的动作扰动，并使动作略向 Clean 靠近；但干预能量不匹配、只有单 task/单 Camera、且无成功率证据。

不能说：TTA Gate 有效、动作被修复、成功率提高。
