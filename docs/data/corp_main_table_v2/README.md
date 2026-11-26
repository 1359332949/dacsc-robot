# Route A 证据入口

本轮闭环评测：3,000局，100.9分钟；600组完整五臂配对。代码冻结提交 a0e2091，Spatial中期报告17555f1。

五臂定义、完整分组成绩、救回/误伤、配对p值与任务聚类区间见 ../../CORP_MAIN_TABLE_V2.md。

## 校准与数据范围

Spatial采用严格任务外9/内8+1 LOTO；Object四个适配臂共用Spatial10全局包，Object拟合帧数为0。两套均为历史已访问的开发/重测池，不是未触碰的独立盲测。

| 套件 | 桌面纹理局数 | 场景纹理局数 | Ours/Frozen Clean |
|---|---:|---:|---:|
| spatial | 45 | 45 | 29/30 vs 29/30 |
| object | 66 | 24 | 30/30 vs 30/30 |

分组按真实BDDL资产解析；Object不是人为假定的桌面/场景各半。

## 结果解释

两套Ours的总成功数均在五臂中最高，但各套相对Frozen的任务聚类95%区间均包含0；不宣称稳健显著或三种子全部净赢。Clean逐例一致只描述本次样本，不是普适无害保证。

ActMAD臂是ActMAD-inspired VLA移植：全视觉编码器更新、12层逐视角矩损失、4帧滑窗、每chunk一SGD步、每局恢复；不是原版CIFAR空间L1方法。

## 效率口径

CUDA-synchronized steady replan; ActMAD includes 4-frame SGD+parameter-step diagnostic+prediction, excludes episode construction/prefill/close. 五轮交错测量，原始样本保存在latency.json。ActMAD时间包含参数变化审计，不能当作去掉诊断开销后的原版ActMAD耗时；旧3.22×或Consensus历史值未混入本次表。

## 完整性与复跑

final_audit.json：独立复核全部原始记录哈希、初始物理/机器人/双目像素/Flow噪声配对、校准包与代码哈希、278个依赖/任务输入文件及ActMAD非零更新。运行中逐局断言权重恢复、动作finite；审计记录说明了二者与离线哈希复核的区别。

完整原始JSON、冻结代码、源轨迹矩、日志及校准包保存在永久remote_artifacts/corp_main_table_v2/corp_main_table_v2_evidence.tar.gz；输入代码/BDDL/init补充包为reproduction_inputs.tar.gz。模型权重及完整纹理/网格资产不在压缩包内，需按记录的路径与校验码准备。

按原环境复跑：source env.sh，运行source_code/scripts/run_corp_main_table_campaign.py --out对应运行目录；完整job在指纹匹配时跳过。恢复本次实验须保留归档记录的校准路径。fit使用指定的新包文件名并拒绝覆盖；另起新实验须先配置独立校准路径与输出目录，再执行prepare/source/fit/preflight/latency，不能混入已有成败记录。
