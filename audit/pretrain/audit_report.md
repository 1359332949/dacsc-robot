# DACSC-Robot 阶段 0 数据审计报告

- episode 总数：**6935**
- 缺少 layout_id/style_id 的 episode：**0**
- 解析失败：**0**
- 任务数：**61**，数据中出现的 style 数：**50**

## 1. 每个任务覆盖了多少 style

| task | episodes | styles | layouts | scenes | 每 style 最大条数 | 单例场景占比 |
|---|---|---|---|---|---|---|
| AdjustToasterOvenTemperature | 107 | 45 | 40 | 105 | 6 | 0.9626 |
| AdjustWaterTemperature | 106 | 43 | 47 | 101 | 6 | 0.9057 |
| CloseBlenderLid | 106 | 40 | 43 | 105 | 7 | 0.9811 |
| CloseCabinet | 105 | 43 | 46 | 104 | 5 | 0.981 |
| CloseDishwasher | 111 | 45 | 48 | 111 | 6 | 1.0 |
| CloseDrawer | 110 | 43 | 43 | 109 | 7 | 0.9818 |
| CloseElectricKettleLid | 102 | 43 | 45 | 100 | 7 | 0.9608 |
| CloseFridge | 106 | 44 | 40 | 104 | 5 | 0.9623 |
| CloseFridgeDrawer | 111 | 36 | 42 | 106 | 7 | 0.9099 |
| CloseMicrowave | 105 | 42 | 44 | 99 | 6 | 0.8857 |
| CloseOven | 107 | 41 | 11 | 98 | 7 | 0.8505 |
| CloseStandMixerHead | 110 | 44 | 43 | 108 | 7 | 0.9636 |
| CloseToasterOvenDoor | 110 | 48 | 46 | 106 | 6 | 0.9273 |
| CoffeeServeMug | 108 | 39 | 45 | 108 | 9 | 1.0 |
| CoffeeSetupMug | 105 | 41 | 43 | 98 | 5 | 0.8762 |
| NavigateKitchen | 503 | 50 | 50 | 452 | 19 | 0.8091 |
| OpenBlenderLid | 106 | 40 | 44 | 101 | 7 | 0.9057 |
| OpenCabinet | 107 | 43 | 45 | 106 | 7 | 0.9813 |
| OpenDishwasher | 111 | 43 | 45 | 110 | 7 | 0.982 |
| OpenDrawer | 102 | 41 | 40 | 99 | 5 | 0.9412 |
| OpenElectricKettleLid | 110 | 44 | 42 | 102 | 6 | 0.8545 |
| OpenFridge | 105 | 45 | 41 | 101 | 6 | 0.9238 |
| OpenFridgeDrawer | 107 | 37 | 42 | 102 | 8 | 0.9065 |
| OpenMicrowave | 105 | 45 | 47 | 103 | 7 | 0.9619 |
| OpenOven | 108 | 45 | 11 | 99 | 6 | 0.8426 |
| OpenStandMixerHead | 109 | 44 | 44 | 104 | 6 | 0.9083 |
| OpenToasterOvenDoor | 105 | 43 | 45 | 101 | 6 | 0.9333 |
| PickPlaceCabinetToCounter | 106 | 43 | 45 | 105 | 6 | 0.9811 |
| PickPlaceCounterToBlender | 106 | 37 | 42 | 101 | 6 | 0.9057 |
| PickPlaceCounterToCabinet | 108 | 42 | 46 | 106 | 7 | 0.963 |
| PickPlaceCounterToDrawer | 107 | 44 | 44 | 104 | 6 | 0.9439 |
| PickPlaceCounterToMicrowave | 110 | 45 | 40 | 107 | 5 | 0.9455 |
| PickPlaceCounterToOven | 111 | 47 | 11 | 104 | 4 | 0.8829 |
| PickPlaceCounterToSink | 108 | 42 | 44 | 105 | 6 | 0.9537 |
| PickPlaceCounterToStandMixer | 107 | 41 | 42 | 104 | 5 | 0.9439 |
| PickPlaceCounterToStove | 108 | 46 | 43 | 107 | 5 | 0.9815 |
| PickPlaceCounterToToasterOven | 108 | 46 | 37 | 107 | 7 | 0.9815 |
| PickPlaceDrawerToCounter | 103 | 44 | 44 | 100 | 7 | 0.9417 |
| PickPlaceFridgeDrawerToShelf | 106 | 40 | 34 | 100 | 7 | 0.8868 |
| PickPlaceFridgeShelfToDrawer | 111 | 36 | 44 | 103 | 9 | 0.8739 |
| PickPlaceMicrowaveToCounter | 112 | 42 | 43 | 108 | 6 | 0.9286 |
| PickPlaceSinkToCounter | 108 | 44 | 43 | 106 | 6 | 0.963 |
| PickPlaceStoveToCounter | 109 | 47 | 47 | 107 | 5 | 0.9633 |
| PickPlaceToasterOvenToCounter | 106 | 43 | 37 | 101 | 6 | 0.9057 |
| PickPlaceToasterToCounter | 105 | 37 | 43 | 103 | 7 | 0.9619 |
| PreheatOven | 107 | 41 | 11 | 98 | 8 | 0.8505 |
| SlideDishwasherRack | 100 | 46 | 46 | 98 | 7 | 0.96 |
| SlideOvenRack | 113 | 42 | 11 | 99 | 8 | 0.7522 |
| SlideToasterOvenRack | 105 | 46 | 43 | 103 | 5 | 0.9619 |
| StartCoffeeMachine | 108 | 46 | 43 | 105 | 9 | 0.9444 |
| TurnOffMicrowave | 108 | 44 | 42 | 107 | 5 | 0.9815 |
| TurnOffSinkFaucet | 106 | 40 | 45 | 105 | 6 | 0.9811 |
| TurnOffStove | 109 | 44 | 47 | 109 | 6 | 1.0 |
| TurnOnBlender | 107 | 37 | 42 | 106 | 7 | 0.9813 |
| TurnOnElectricKettle | 108 | 43 | 43 | 107 | 6 | 0.9815 |
| TurnOnMicrowave | 107 | 45 | 45 | 103 | 6 | 0.9439 |
| TurnOnSinkFaucet | 107 | 42 | 44 | 107 | 6 | 1.0 |
| TurnOnStove | 104 | 43 | 42 | 103 | 5 | 0.9808 |
| TurnOnToaster | 108 | 42 | 45 | 105 | 7 | 0.9537 |
| TurnOnToasterOven | 103 | 39 | 44 | 101 | 7 | 0.9612 |

（仅显示前 60 个任务，完整表见 task_style_counts.csv）

## 2. 每个 style 有多少 episode

- style 类别数：**50**，随机水平：**0.02**
- 每 style episode 数：min=98 / median=140.0 / max=169
- 达到阈值（每 style >= 32 条）的 style 数：**50**
- 若 style 在任务内均匀抽取，每 style 期望 episode 数约为 **138.7**；实测中位数 140.0（偏差 0.9%）→ 两者接近，符合均匀抽取
- 结论：**PASS**

## 3. task / layout / style 是否严重耦合

- task x style 的 Cramér's V = **0.1014**（chi2=3492.33, df=2940）
- layout x style 的 Cramér's V = **0.0868**
- 仅出现在单一任务中的 style 数：**0** / 50
- 判读：V < 0.10 视为基本独立；0.10–0.25 轻度耦合；> 0.25 视为严重耦合，此时 style 与 task 不可分离，域分类器会把任务信息当作域信息。

## 4. train/val/test 泄漏检查

- 划分定义：`{'train': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'val': [41, 42, 43, 44, 45, 46, 47, 48, 49, 50], 'test': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]}`
- 跨 split 重复的 style：**0**
- 出现在多个 split 的 scene：**0**
- 未映射到任何 split 的 episode：**0**
- 同时跨多个 split 的任务：**61**（这是**预期现象**：split 由 style_id 决定，而 style 在每个任务内逐 episode 抽取，因此每个任务天然横跨 train/val/test；只有 scene 级重复才是真泄漏）
- 结论：**PASS**

## 5. 分组归属检查（expect_group=pretrain）

- 违反分组的 episode：**0**
- 违规 style：[]
- 违规 layout：[]
- 结论：**PASS**

## 6. 域分类器可行性

- 结论：**PASS**
- 域分类器在 50 类、随机水平 2.00% 下，若多数风格样本量不足，域梯度分数不可信；应先把 style 归并为粗粒度外观域。

## 7. 门槛判定

**可以进入阶段 1/3**（泄漏=PASS，缺失=0，解析错误=0，分组=PASS）
