# 外部对照实现审计与修复 V2

本轮先修对照与固定Frozen复现，不设计新TTA公式，不改原V1，不访问保护的Blind/Held-out清单。

## 实际修复

`dacsc/baselines_tta.py` 的输入是view-major（所有agentview帧后接所有wrist帧），旧reshape却假设frame-major。n=1掩盖错误；n=2/4时，主眼全0、腕眼全100会被错误聚合成两个50。现先按[view,frame,patch,channel]恢复，再在各视角内聚合。新增形状及相机顺序检查，去掉256/384硬编码。

`ActMADInspired` 是明确的VLA移植；保留旧 `ActMAD` 导入别名仅为兼容旧脚本。实现标识 `actmad_inspired_viewwise_v2`。旧 `actmad_true` 分数保持历史记录，不追认为已修复分数。

## 与作者实现的差异仍存在

作者公开CIFAR路径保留空间位置，使用均值/方差的L1匹配；我们端采用跨patch/近期帧聚合的均值/标准差源尺度标准化平方差。更新12层DINO、最多4帧、每chunk一步、每episode重置是移植选择。修正相机索引不自动消除这些方法差异。

原作者代码：https://github.com/jmiemirza/ActMAD/blob/master/methods/actmad_cifar.py
统计代码：https://github.com/jmiemirza/ActMAD/blob/master/utils/utils.py
核查commit：401544d94072c13d6c07c916bbc18d5358c0237f。

## 验证范围

- CPU单测3项通过：1/2/3/4帧的相机统计及反向梯度；更新、缓冲上限、参数/优化器重置、移除hooks；错误数量拒绝。
- RTX3090真实CoRP权重：使用已校验Dev缓存，统计严格排除该任务，仅其余9任务。没有用目标成功标签。H=E=12。
- 未更新时动作逐位恒等；4次更新均finite并改变参数及动作；reset和close后动作逐位恢复，参数再次冻结。
- 动作变化L2=0.0115688108，只证明改动实际进入执行链，不能解释为动作改善或成功率提升。
- 尚未重跑这份移植的完整闭环；尚未完成原作者算法在VLA上的忠实移植与正式比较。

## 网上仓库核验

- TTA-VLA目前项目页Code按钮指回首页；本轮未找到算法源仓库。论文的Entropy对照针对OpenVLA、pi0、OFT分别定义，不等同于统一Tent代码直接套用。
- VANE所检checkout主要为网页、PDF和展示素材，未找到训练/评测Python源码，不能把项目页当完整可复现实现。
- IDR有核心refiner及模型接入代码，但OpenVLA-OFT的run_libero_eval_cf.py明确仅执行baseline动作、只记录反事实；直接跑该入口不等于执行完整IDR。

来源：https://tta-vla2.github.io/assets/tta-vla_paper.pdf 、https://github.com/MindVLA-Team/VANE 、https://github.com/floy4/IDR-Framework 。
核查VANE commit d65f05134cca8c825df8ac684e4dc5cc5d059d55；IDR commit 429989aa7d42cc9a1eca9d054c502ed0d5f7b166。

## 正式比较前必须做到

先验收同一固定Frozen底座；对照各自标明原版/移植/内部消融/引用数字。检查adapter开关确实改变最终执行动作、关闭时回到Frozen。固定checkpoint、参考数据预算、渲染器、执行频率与测试清单；额外交互、梯度、学习模块成本单列。修复后的胜率必须实测，不能由旧结果推断。
