# 87 - V10 Atomic Target Protocol and Checkpoint Provenance

日期：2026-11-06

## 1. 当前结论

v9 pilot 已完成并停止扩展：

```text
baseline clean 6/20 + 6/20, jitter 2/20 + 2/20
random   clean 8/20 + 10/20, jitter 2/20 + 2/20
cross    clean 9/20 + 10/20, jitter 2/20 + 0/20
```

cross_style 没有超过 random control，停止旧实现的方法叙事。

## 2. v10 冻结内容

- 5 个官方 target Atomic-Seen 任务
- pretrain train styles 11–50
- source holdout residue 9
- internal val styles 51–60
- official target scenes via `split="target"`
- target pairs `(1,1)...(10,10)`
- 30k custom pilot budget

机器可读配置：

```text
configs/v10_atomic_target_pilot.json
```

协议说明：

```text
docs/V10_PROTOCOL.md
```

## 3. 修复内容

- training task list 可显式冻结，不再只能靠 max_tasks
- train/val/test/style ranges 可配置，并强制 train/val 不重叠
- checkpoint 格式升级为 `corp_robocasa_v7`
- checkpoint 保存 `training_origin`，包括 train/source/val selection hash、
  数据集 metadata hash、帧数/ episode 数、任务列表、style ranges 和项目 commit
- SR schema 升级为 `corp_robocasa_sr_v10`
- target manifest 记录 `scene_split=target`，target scene 使用官方固定 pairs
- target evaluator 只接受官方 Atomic-Seen tasks
- aggregate 校验 target pair schedule、layout/style order、style range 和 split identity
- raw RGB 训练增广入口显式标记，不再以 style11–30 硬编码作为唯一允许范围
- 离线评测和 SR 聚合会从 checkpoint 的 training origin 读取允许的
  train/val style range；命令行覆盖必须与 checkpoint 一致
- checkpoint 的 comparison identity 加入 `completed_steps`，5k 与 30k
  子检查点不会被误认为同一预算
- `pack_for_server.sh` 在 Git archive 中写入 `CODE_IDENTITY.json`；
  无 `.git` 的训练目录必须通过文件 SHA256 校验后才能记录项目身份

## 4. 本地校验

完整回归共 139 tests：

```text
PYTHONPATH=tests .venv/bin/python -m unittest \
  tests.test_corp_ffn_capture tests.test_corp_pipeline \
  tests.test_corp_regressions tests.test_gpu_environment \
  tests.test_official_libero_adapter tests.test_pixel_augmentation \
  tests.test_readonly_validation tests.test_standard_metrics \
  tests.test_upstream_contracts
```

覆盖：

- target manifest scene_split / official pair
- target SR aggregate schedule
- explicit 5-task selection
- v10 frozen config
- checkpoint training origin / selection hash
- v10 source/val style ranges
- training augmentation separation
