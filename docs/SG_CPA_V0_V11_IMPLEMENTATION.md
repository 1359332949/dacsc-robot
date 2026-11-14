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

## 2. V1.1 hardening v4（针对 7d79d28 复审）

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

## 3. V1.1 使用方式（需 GPU/LIBERO 环境）

```
python scripts/v11_state_pair_audit.py \
  --clean-suite libero_spatial --clean-task 0 \
  --shift-suite libero_spatial --shift-task 705 \
  --shift-kind camera --init-index 0 \
  --out /hy-tmp/validation/v11_pair_audit
```

脚本做：

1. 建 Clean / Shift 两个 `OffScreenRenderEnv`；
2. Clean 环境 `set_init_state` + settling，取全状态快照；
3. 把快照恢复进 Shift 环境；
4. 比较 qpos/qvel/act/mocap/time/body_xpos、body names；
5. 比较 camera 参数，并按 `shift-kind` 检查预期（Background/Lighting 应相同，Camera 应不同）；
6. 保存双视角 JPEG + sha256、pair manifest。

输出：

```
/hy-tmp/validation/v11_pair_audit/pair_manifest.json
/hy-tmp/validation/v11_pair_audit/<kind>/*.jpg
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
- `test_state_pairing.py`：14 tests passed；
- `test_task_pairing.py`：10 tests passed；
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
