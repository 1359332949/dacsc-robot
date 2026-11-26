# SmolVLA Clean 复现诊断：从环境错误到可运行的固定版本

**当前结果：** 原始 vanilla LIBERO + 社区配方所用 LeRobot 旧版，在预定十个基础任务各一个 init0/固定种子上 8/10；当前 LeRobot0.4.4 是6/10。旧版通过≥7/10的起步检查，现代版尚未过关。不是完整100回合的79%复现，不能用10局包装为SmolVLA论文成绩。本轮没有TTA、层位搜索、超参搜索或新checkpoint挑选。

## 1. 两种不同错误

运行故障：旧sitecustomize在Python启动时导入LIBERO，而GLFW探测会启动python-c子进程，继承同一sitecustomize，导致递归。已停止247个精确识别的该探测链进程（仅匹配命令特征、corp解释器和继承phase3_patch路径），保留PID/命令hash收据；新增-c防递归分支。新评测不导入旧桥接，并显式指定PYGLFW_LIBRARY；回归检查显示python-c启动不再导入LIBERO，最后精确探测进程数为0。宿主load平均不等于本容器GPU忙，未宣称其所有负载均来自此故障。

条件错配：原桥接换了十个任务名，仍加载Plus代码和Plus纹理。真正vanilla独立克隆固定到8f1084e3132a39270c3a13ebe37270a43ece2a01，使用独立config目录，不改/root/.libero。十个基础BDDL和pruned_init在两树逐字节相同，不能把问题笼统归为这两类文件。编译相机和纹理有差异：主视角pos(0.658613,0,1.61035)与(0.6586,0,1.6104)，quat也有舍入差；四个默认纹理中地面255→512、桌面及桌腿1270→512、墙880→512。物体qpos/qvel、质量、摩擦、灯光相同。

## 2. 相同起点的场景反事实（单案例，非总体胜率）

|条件|结果|步数|
|---|---|---:|
|真vanilla基线|成功|72|
|原Plus基础任务|失败|280|
|identity|失败|280|
|camera|失败|280|
|textures|成功|69|
|both|成功|68|

identity/camera/textures/both四组都经同一XML重编译路径，四组起始模拟状态哈希及8维本体读数相同。原始不重载的Plus与重载后有约3.4e-7的EEF读数差，因此归因主要使用identity对照而不假装重载前后逐位相同。恢复纹理后本例救回，仅修相机未救回：支持纹理/渲染条件影响该checkpoint，但不能外推所有任务或所有纹理变换。原Plus与vanilla同物理state的首帧agentview像素MAE0.324、最大55；腕部MAE0.163、最大3。不要把这些当作噪声舍入来强行统一成相同图片。

## 3. 真实完整Clean逐任务结果

|vanilla task ID|任务|LeRobot0.4.4|社区旧版483be9aa|
|---:|---|---|---|
|0|pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate|1 / 72步|1 / 71步|
|1|pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate|1 / 111步|1 / 106步|
|2|pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate|1 / 95步|1 / 94步|
|3|pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate|1 / 83步|1 / 89步|
|4|pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate|0 / 280步|0 / 280步|
|5|pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate|0 / 280步|1 / 108步|
|6|pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate|1 / 100步|1 / 105步|
|7|pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate|0 / 280步|0 / 280步|
|8|pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate|1 / 96步|1 / 103步|
|9|pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate|0 / 280步|1 / 115步|

合计：modern 6/10；legacy 8/10。两者同任务、同起点、同种子；API路径和Euler实现不同。没有证明这2局差距是统计显著或定位到唯一API错误。选择社区旧版继续是因为它是事前指定的参考配方，不从结果重新挑alpha、种子、相机方向或夹爪映射。

checkpoint的788个权重张量在两路径均逐值核对匹配（按目标dtype转换），不是仅匹配键名。官方Hub LFS SHA与本地71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f一致。num_vlm_layers=0在源码中表示不截断默认层数，不是VLM零层；load_vlm_weights=False仅用于离线构建后加载完整checkpoint，未随机留置模型权重。现代processor与旧normalize模块都显式载入同一组公开统计。两路径均180°翻图一次，像素0..1一次，SigLIP归一化由模型处理，动作无夹爪反号，仅[-1,1]裁剪，checkpoint默认E1、Euler10。CoRP的H=E12不能硬套到另一个策略的官方基线。

## 4. 官方CLI及来源证据

同一真正vanilla条件下，官方lerobot-eval的task0为1/1成功，约62.45s；先前Plus/task-map桥接0/1仍保留为旧环境条件下的记录。社区79/100来自ActuallyIR/roboeval公开结果（100条，RTX5090，lerobot483be9aa，vanilla8f1084e3，MuJoCo3.2.7），不是本checkpoint模型卡承诺的83%。本机Torch2.10/cu128、Transformers4.57.6、MuJoCo3.3.0、RTX3090与其不完全相同，且只测init0各一次；继续正式第二底座前应固定此兼容profile并做多起点验证。

检查顺序参考公开复现清单与源码：先固定资产与task/init ID，再核相机/state/归一化/动作约定，接着同输入核前向，再Clean闭环，最后才适配。以键数、finite或一次官方CLI能启动代替Clean验收，会错过环境语义问题。没有使用新任务失败结果为适配方法调参。

## 5. 同时查出的CoRP校准实现问题

scripts/aggregate_calib.py的旧式是((mu_t-mu_s)^2 + sigma_t - sigma_s)^2 / (sigma_s^2+eps)；正确的既定式是((mu_t-mu_s)^2 + (sigma_t-sigma_s)^2)/(sigma_s^2+eps)。均值差2、两sd均1时，旧式给16，既定式给4。回归测试先失败后通过。服务器原spatial_shared_calib.npz的tau_b3与错误式完全一致(max error0)，与正确式最大差0.309779，故不能把读取这份共享参考的后续结果当作原冻结V1的等实现复现。

已修代码，输出改为新文件spatial_shared_calib_corrected.npz，拒绝覆盖已有文件；旧reference/实验结果保留，本轮没有拿新阈值重跑或偷偷改写CoRP胜率。原冻结全轨迹Spatial/Object参考以及封存方法校验仍通过，不是所有CoRP结果作废。另，libero10的init0共享校准测试是state-disjoint同任务校准，不等于严格外9/内8+1 LOTO；98/100只能按其实际权限报告，任务隔离结论仍须独立复算。

## 6. 当前可用结论和下一步

SmolVLA不是因为没接对SigLIP层而低分：冻结Clean基线已有可运行兼容版本；已有观测/资产错配和启动递归实证。它尚不构成V1在第二底座的收益。下一步只在旧版+真vanilla固定profile补多起点Frozen验收，之后单独预注册SigLIP适配接口与源统计；不要把原Plus低分当成方法提升空间。CoRP保留先前固定版本证据，同时给后续共享参考结果加版本标签并先修复公式/任务隔离，再讨论是否真的跨房间有效。
