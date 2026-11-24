# 三域 t=0 诊断：源域偏离、视角替换与在线可用的拉回

100/100 个冻结 Dev 实例完成四层扫描；13 条 Background 失败、6 条 Lighting 失败和 10 条 Clean 各执行 12 项离线探针。没有新 rollout、权重更新或 Blind 访问。原生单噪声动作与官方预测最大误差 0，与上一轮四噪声控制器动作最大差 3.40e-6。每个任务的参考只取另外九个任务各一条 Dev Clean。Suffix 已融合双视角，不赋予虚假的单视角归属。这里的“距离缩小”是相对同任务 Clean 策略预测的距离，不能换算成成功率。

## 失败排序与视角识别

桌面纹理（15条，10败）Flow 四噪声 AUROC .540；主视角 DINO 余弦和腕部 Key 余弦均 .680。腕部 Key 同任务排序 6/7，AUROC 对 Flow 的配对任务 bootstrap 差区间 [-.182,.371]。光照 Flow .799，最佳观察到的源距离为腕部 DINO 均值 Z 距离 .774；Sensor Flow .873，主视角 Resampler Z 距离 .828。场景纹理仅3败，Flow 与主视角 Value 余弦均 1.000。大量候选中没有充分依据宣称已显著补齐两类盲区。

Key 主/腕偏离比对 Sensor vs 其余两种 Shift 的 AUROC .936；四组中位数 Clean .876，Sensor 2.327，Background .710，Lighting .758。用其他任务的 Clean 内层留一最高分作为逐眼阈值，Sensor 仅14/30被分为仅主视角异常，Background 与 Lighting 均无一条被分为双眼异常。域类别排序与可部署的逐眼损坏判定不同。逐指标 Spearman、同任务配对、置信区间、四分位数和分类计数在 summary.json。Lee 等（NeurIPS 2018，arXiv:1807.03888）的多层马氏距离用于分布外检测；这里另行检验扰动域内部的成败，而非以域检测代替风险预测。

## 替换上限及无需同任务 Clean 的干预

下表为逐实例 clipped controller 全 chunk L2 距离的平均缩小百分比，XYZ / rotation；后者是旋转坐标距离，不是 SO(3) 测地距离。

|操作|Background 13败|Lighting 6败|
|---|---:|---:|
|主视角 Clean Value|37.6 / 28.8|5.0 / −4.0|
|腕部 Clean Value|44.7 / 29.5|16.1 / −13.3|
|主视角 Clean K+V|51.4 / 32.1|14.2 / −0.04|
|腕部 Clean K+V|44.3 / 30.8|7.7 / −11.7|
|双视角 Clean Value|80.2 / 78.9|59.5 / 79.9|
|双视角 Clean K+V|100 / 100|100 / 100|
|双视角 RGB 对齐 α=.25|3.1 / 10.0|20.9 / 10.2|
|双视角 RGB 对齐 α=.50|2.8 / 21.5|31.8 / 19.2|

双视角 K+V 复现 Clean 动作是单层结构的完整替换对照，并不定位上游病因。双视角 Value 已解释大部分参考差异，单眼替换（尤其 Lighting）经常补不回两眼联合作用。四组无标签 K/V 软截断均使 Background XYZ 变远（平均 6.0%–26.5%）；每组都有两条 XYZ 恶化至少10%却通过曲率不恶化检查。光度 α=.50 时十条 Clean 的平均绝对 XYZ / rot 漂移 .010 / .002；非零，不能推断 Clean 成功率保持。

## 决策

不把源距离提升为统一失败检测器，也不因 K/V 软截断失败断言必须深入 DINO。优先用全部30条 Lighting（含24条原成功）及10条 Clean 检验固定光度操作的获益与损害，再决定闭环。Background 的桌面和场景纹理分开处理；Sensor 继续保留曲率排序与视角退权作为候选。没有把本轮挑出的最佳指标视为独立发现。

`docs/data/global_shift_diag_v1/summary.json` 包含完整数值。`evidence.tar.gz` 包含 LOTO 参数、逐例打分、替换/拉回的四噪声动作与回执；`feature_sketches.tar.gz` 包含每例矩、来源及完整 NPZ 的哈希，完整张量位于服务器 `/hy-tmp/patch-diag/runs/global_shift_diag_v1/features/`。
