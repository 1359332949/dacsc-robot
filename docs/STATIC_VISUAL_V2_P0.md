# Static Visual V2：实现状态与P0真实核查

日期：2026-11-23。**已实现并完成静态P0/单元测试；GPU Clean Smoke和Dev尚未运行，120/400名单未冻结。**

## 本轮不能直接执行原120/400的原因

从官方 `sylvestf/LIBERO-plus` 下载并固定源码commit：
`4976dc30028e805ff8094b55501d532c48fec182`。
这是本地核验的候选基准版本，尚未证明与当前不可连接服务器的安装版本相同。服务器运行前会再次比较registry/wrapper指纹。

| Spatial因素 | 官方实例数 | 最小task格子原始容量 | 本轮历史隔离后候选组数 |
|---|---:|---:|---:|
| Lighting | 292 | 4 | 290 |
| Background | 258 | 6 | 258 |
| Sensor Noise | 351 | 28 | 351 |
| Camera | 376 | 6 | 0：历史范围quarantine |

以上候选组仍未经完整服务器历史扫描，不称为“已确认从未访问”。共10个base task、1277个四因素实例，解析失败0。分类文件的ID为1-based，评测列表索引为ID−1；主身份使用完整logical name，避免编号错位。

即使不考虑历史，`next_to_the_plate` 的Lighting仅4、Background仅6；`next_to_the_ramekin` 两者各10，另一个任务Camera仅6。它们天然不能分出10 Blind+3 Dev。不得为了凑数重复seed或跨任务补位。

旧用户指定DeepSeek会话中，记录过四suite全Camera评测（总1599条计划），并有Spatial约258条已执行的具体进度。由于目前拿不到原始逐实例运行日志，不能声称1599条全部完成，也不能把没查到具体ID的Camera当未访问。暂将Camera范围quarantine。证据摘录和原压缩会话SHA256见 `protocols/static_visual_v2/history_camera_evidence.json`。

## 外部执行阻碍

旧GPU地址 `140.210.209.209:30977` 多次在SSH密钥交换前断开。不是密码/密钥被拒，也不是GitHub权限问题。GitHub插件已确认账号1359332949，可正常核对仓库身份；它不提供这台GPU主机的SSH连接。

本地只有源码、BDDL和初始化文件，没有完整仿真资产；未把本地静态检查包装成服务器模型实测。没有新训练，没有Blind或新Dev rollout。

## 已实现的精简入口

一个CLI：`scripts/static_visual_v2.py`，子命令：

- `audit`：核对checkpoint、registry ID/name对应、wrapper参数、BDDL/init资产、历史引用及范围排除，产出容量表、Source角色预留和P0清单。
- `freeze`：必须历史审计完整、容量足够、资产存在、每任务5条指定Smoke完成且非0/5、独立Clean Holdout审核通过，才可冻结目标名单。不满足即拒绝。
- `run`：仅Frozen/Source Smoke或正式Dev，默认不支持任何Blind解锁。先检查角色和hash，再加载模型/环境。实现当前CoRP路径及物理配对检查。
- `summarize`：按manifest保留缺失和异常，输出每task×factor配对转移表；未完成时宏平均为空，不从分母消失。

辅助模块 `dacsc/static_visual_v2.py` 包含纯函数的ID、runtime解析、RNG与执行锁，便于测试。不是新数据库或评测平台。

## 三条硬底线的实现

1. **身份与等价组**：`(commit,suite,logical_task_name)`作为实例身份；解析 `_view_..._initstate_..._noise_...`，严格区分机器人起始姿态参数与episode初始状态索引；等价组组合base task、实际BDDL hash、真实runtime参数、初始化资产hash。相同BDDL但Camera/Sensor参数不同不会混并。反过来，已解析资产内容相同的别名不会仅因编号不同就冒充新实例。
2. **历史排除**：2292/2312直接排除；旧object范围quarantine；Camera全量任务范围quarantine。扫描本地现有两份仓库docs/scripts的精确逻辑名引用，保守视为待排除引用，不声称每个引用都确实rollout。大文件/缺失来源列出，`history_complete=false`禁止冻结。服务器历史账本仍待合并。
3. **独立随机流与Blind锁**：environment RNG、sensor step seed、policy flow Generator和预留method augmentation seed分离。Sensor沿用官方变换函数，仅改agentview；在reset之前关闭wrapper内置corruption，避免它消耗环境随机流，再在观测返回时用独立流施加噪声。Frozen与后续方法可以保持同一step-indexed噪声。当前版本只有Frozen，不宣称已经验证TTA随机性。

`authorize`在checkpoint、环境或GPU加载前拒绝Blind；空Blind清单也明确拒绝。账本追加开始/结束事件。无`--force-blind`绕过参数；最终授权流程需要后续独立实现，不在本轮开放。

## Clean初始化与配对

四因素解析为对应base task的 `.pruned_init`；计划按官方默认索引0。保存 `raw_init_hash`，**不把它称为已验证物理状态hash**。`physical_init_hash=null`直到允许的Source/Dev环境reset后，以列举的qpos/qvel/act/time和静态模型/物理结构进行配对核查。不匹配记异常，不换初始化。

Source Fit/Calibration/Holdout从1..49按固定种子预留互斥索引，Source每角色10个。独立Clean Holdout的历史新鲜度尚未核实；不能靠索引不同就宣布全新。

由于同任务视觉变体复用索引0，Dev/Blind的paired Clean twin会重叠。正式生成器因此：Dev保存去重paired Clean；Blind的最终Clean保持性只用独立核验后的Clean Holdout。不会把Dev看过的Clean twin重新标为独立Blind Clean。

## 当前两份Manifest的真实状态

目录：`protocols/static_visual_v2/p0/`。

- `dev_manifest.json`：`P0_SMOKE_ONLY`，包含50个Source Fit Smoke计划，没有Target Dev执行授权。
- `blind_manifest.json`：`NOT_FROZEN_BLOCKED`，cases为空，列出阻碍。**它不是已生成的400条Blind名单。**
- `p0_audit.json`：原始registry解析、容量、资产指纹、扫描范围和Source预留。
- `manifest_hashes.json`：上述文件hash。

保留目标配额3/10，不擅自改成另一个试验范围。Camera历史核清前，降额本身不能解决“候选0”的问题。

## 实际验证

18项CPU测试通过：runtime串扰、initial-pose/index混淆、别名确定性、配额不能用重复seed补齐、历史隔离、RNG额外调用不干扰/异常后恢复、仅agentview受噪声影响、Blind提前拒绝、清单修改拒绝、缺失结果保留、独立Clean Holdout不复用Dev对照。

使用真实官方registry/BDDL/init资产完成1277条静态解析，不是仅对合成样例测试。GPU模型和真实native sensor函数的执行验收仍待服务器恢复，单元测试中的sensor调用替身不能替代这一步。

本轮新增CI仅跑协议契约测试，不下载模型或访问Blind。

## 后续需要的明确输入

1. GPU服务器当前可用的SSH地址/端口（或恢复旧实例）；GitHub插件不能替代。
2. 从原服务器取回旧Camera逐实例日志，确定哪些组可以解除quarantine；历史不能澄清的组继续排除。
3. 按真实容量审查配额：原3/10不能覆盖所有40格。可选择保留任务范围但预先降低稀缺格Blind配额并固定宏权重，或调整确认范围；**不能根据新Dev成功率选择如何降额**。
4. 核清独立Clean Holdout历史访问；之后才能冻结正式名单。

未满足这些条件前，继续编造一个120/400“正式划分”反而违反本轮三条硬底线。

## 复现命令

需使用同一固定commit的LIBERO-Plus源码与模型。将路径换成实际安装位置；输出目录必须不存在。服务器实例若是无Git的解压目录，应先建立可核对的固定revision源码目录并使用匹配资产，不能虚填commit。

```bash
python scripts/static_visual_v2.py audit --libero-src /path/to/pinned/LIBERO-plus --checkpoint /path/to/corp_libero.pt --settings protocols/static_visual_v2/settings.json --history-root /path/to/project/docs --history-root /path/to/project/scripts --history-root /hy-tmp/validation --out /path/to/new_p0
python -m unittest tests.test_static_visual_v2 -v
```

P0运行示例（仅Source，真实case-id从dev_manifest中读取）：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=/hy-tmp/libero_config OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/static_visual_v2.py run --audit /path/to/new_p0 --manifest /path/to/new_p0/dev_manifest.json --case-id SOURCE_CASE_ID --libero-src /path/to/pinned/LIBERO-plus --corp-src /path/to/CoRP --checkpoint /path/to/corp_libero.pt --out /path/to/smoke_results
```

该GPU runner已实现但未在当前失联服务器上执行；异常、配对失败和代码指纹变化会显式失败并留痕。此次提交不含任何新的模型成功率结果。
