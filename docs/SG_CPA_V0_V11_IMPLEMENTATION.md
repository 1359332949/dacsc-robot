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

## 2. V1.1 hardening（针对静态审查）

已修复：

- `compare()` fail-closed：缺失字段、单侧空数组、shape 不一致、非 finite 一律失败；不再用默认 0.0 表示未比较。
- 增加 `body_xquat`（四元数 q/−q 等价角误差）、`eef_pos`、`eef_name` 必需检查。
- `model_signature` / `model_compatible()`：restore 前比较 nq/nv/nu、关节名/类型/地址、actuator、body names。
- `camera_compare()`：names/shape/finite 检查；区分“应相同”和“应发生 camera 变化”；NaN 不会误判为相同。
- V1.1 脚本 fail-closed：物理状态、模型兼容、camera 预期、必需视角、实际图像差异任一失败 → 非零退出。
- 图像用无损 PNG；计算 mean abs diff 与 changed fraction；不再只用 SHA 判断。
- `init_index` 严格范围检查；记录 init 文件 hash；输出目录按 task/init 唯一化；manifest 单行 JSONL。
- 测试改为 unittest，覆盖缺失字段、单侧空 act、body 四元数、NaN camera、shape mismatch、EEF 缺失、model mismatch。

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

## 4. 验收标准

| 检查 | 标准 |
|---|---|
| body_names_equal | true |
| qpos/qvel/act | max abs err ≤ 预注册容差（默认 1e-6） |
| body_xpos | ≤ 1e-5 |
| time | ≤ 1e-9 |
| camera_ok | Background/Lighting 相同；Camera 允许不同 |
| state digest | 记录 clean/restored 两个 digest；不要求相同，用于追踪 |

如果 qpos/body 误差超过容差，说明“同状态配对”没有成立，不能进入 V1.2。

## 5. 当前阻塞与未做项

- V1.1 尚未在服务器运行（GPU 主机当前不可达）；
- V1.2–V1.4、V2 未实现；
- 没有 teacher-student、可学习 gate、delayed-feedback β、reward model；
- 没有正式 rollout。

## 6. 单测

已统一为 unittest（符合仓库 `python -m unittest discover -s tests -v`）：

- `test_visual_gate.py`：5 tests passed；
- `test_state_pairing.py`：12 tests passed；
- 覆盖 identity no-op、错误 view/dim、suppression bound、缺失字段、单侧空 act、四元数、NaN camera、shape mismatch、EEF 缺失、model mismatch。

## 7. 下一步

1. 在服务器跑 V1.1，先只做 1 个 task × 1 个扰动 × 1 个 init state；
2. 确认 qpos/body/EEF/camera 误差；
3. 通过后再扩到 Background/Lighting 多个 state；
4. V1.1 不过，不进入 V1.2。
