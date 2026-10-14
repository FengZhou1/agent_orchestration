# 逐轮进度、运行时缓存与等价性核验（2026-10-14）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized runtime optimization and reproducibility verification
- Source: 当前源码、用户已中断的单槽场景/轨迹、改动前157ccc3实现
- Verification Status: 原值缓存、数值摘要和回归已核验；下面的短程计时不是收敛证据或多场景速度保证
- Protocol: slot_joint_v7，参数、数学关系、动作和学习日程不变

## 1. 变更及边界

1. 接入已有 TrainingProgressReporter：默认纯文本逐轮/分阶段消息，可选`--progress-bar`。新增`training_status.json`与追加式`training_history.jsonl`，保留原检查点和`history.json`。状态中记录已完成更新数、当前物理槽、部署目标、模型选择组；长时间计算前显示physical_routing、analytical、workflow阶段。恢复后的速度估算不把历史完成量计入当前进程速度，跨阶段ETA仍只是粗略工作量估计。中断记录为interrupted，不自动重试或重新启动长训练。
2. 分开记录策略推断、环境step、reset、观测构造、固定预训部署构造、物理路由、解析性能、未服务检查、工作流评价和指标整理。计时保存在rollout/history，不加入模型输入、奖励、SLO或指标对象。`env_step`是包含内部环节的总时间，`observation`等是子项，不能将所有列直接相加。初始化和首轮固定基线在收集计时前单独执行。
3. 按原始有序节点/边缓存DAG根节点、父节点顺序与lexicographical拓扑顺序（上限4096项）。不缓存跨评价的节点时延。路径是否可承载只在当前一次工作流评价内复用，每次evaluate重新清空，避免负载改变后误用旧可行性。
4. 按模型、完整实例配置、有序token类别、有序权重、总到达率、chunk和计算模式缓存LLM解析结果（上限256项）。键不舍入、不分桶、不重排，不把相近状态视为相同。可变的类性能记录返回独立副本，调用者修改结果不能污染缓存。既有参考容量缓存仍保留。
5. 同一次观测只计算一遍全局特征，再形成两分支观测；构造特征时到达率字典只获取一次，而非每个入口条目重新生成。没有改变特征顺序、值、缩放、截断或形状，也没有把所有训练观测指向同一个可变数组。
6. 物理映射采样的概率数组在一次采样调用内预备一次，不再为每个样本重复构造、归一化。保留相同发生器、种子、逐样本/逐节点的choice调用及调用次序，仍使用128映射上限，没有减少样本或修改分布。

本轮没有改数据生成、SLO、负载、成本/时延定义、同槽差分、绝对约束、两分支actor/critic、GAE/PPO损失、冻结行为或三阶段预算；没有引入蒸馏、教师或额外评价轨迹。参数来源摘要保持原值，未清除正式训练pending。也没有实现整个rollout的紧凑存储，因此此前最大规模多槽内存瓶颈不能据此宣布解决。

## 2. 核验输入与方法

使用真实预实验的20台服务器、50个应用、77类工具、304个正到达率应用—入口组、279个pattern。场景`results/slo_calibrated_main_20261014/servers_20_applications_50.yaml`的SHA256为`4813b11b72a3a1aed04069e6a2276a778f2849cd0f628cbb75aea93259c405a3`。

直接读取用户已保存的单槽`results/single_slot_n20_a50_s0/train_trajectory.json`，内容摘要`9c5b0cd15400a72d9b30fe209349282c225fd77f38e1e310e8834d60b740dd75`。使用4层×256、seed0的随机策略、CPU、模型选择阶段和128次映射上限，采集一次完整rollout，不做PPO更新，不覆盖原运行目录。函数级计时使用cProfile；另执行最终实现的不带分析器计时。

`benchmark_slot_runtime.py`在修改生产代码前先运行，载入157ccc3的原实现；随后在新进程运行优化实现。中间对照是第一批缓存，final是包含到达率单次构造和评价内路径可行性缓存的最终实现。所有输出为新results目录，原用户运行未重启、未覆盖。最终报告还记录输入/实现源码开始执行时的SHA256。

## 3. 数值一致性

四次探针（before、after、final_profile、final）均为304条转移，得到完全相同摘要：

- 观测、下一观测、随机动作、log-prob、价值、奖励和折扣：`2a9158991712610f7d22a35df57c194c1fda6457fb975e9a38343a5eef864fa5`。
- 完整SlotMetrics（含逐应用、逐flow及部署/模型选择诊断）：`918a535e5cda7f1ae3b0bc4233cad47f0c84ecab7bdc47fd615bb49b5dfa1bdf`。

这是一条给定轨迹与一组动作上的严格摘要一致，不等于穷尽所有输入。补充单元测试覆盖五种解析模式的缓存/无缓存结果一致、输入单比特变化触发miss、可变结果隔离、图顺序与结构变更、逐元素观测一致、链路负载变化后的失效、状态打印、heartbeat去重和恢复后速度估算。全部回归为 **308 passed**，另有11条既有NetworkX hash版本提醒。

## 4. 时间结果

| 同轨迹单次探针 | 采集总时间 | 工作流评价 | 观测构造 |
|---|---:|---:|---:|
| 修改前，cProfile | 297.033 s | 250.513 s | 23.192 s |
| 第一批优化，cProfile | 142.444 s | 111.336 s | 8.650 s |
| 最终优化，cProfile | 131.379 s | 105.827 s | 3.413 s |
| 最终优化，不启用cProfile | 70.965 s | 57.982 s | 1.401 s |

带相同分析器的修改前/最终探针约为2.26倍速度差。cProfile本身会改变运行时间，不能将297秒直接与不带分析器的71秒比较后宣称日常训练有4倍提升。原用户运行前两轮不带分析器的采集时间为138.404/138.712秒，只能作为约束相同情况下的参考，并非多次配对速度实验。

最终不带分析器的一次探针另记录：策略推断0.309秒，物理路由3.210秒，解析性能5.854秒，未服务检查0.520秒，reset1.383秒。工作流仍是主要开销，未通过降低映射上限或改变关键路径/平均次序消除它；后续轮次随部署/模型选择变化，时间也会不同。这里没有给出2000轮完成时间或收敛保证。

探针输出分别在`results/runtime_exact_cache_20261014_{before,after,final_profile,final}/`。三阶段端到端功能试跑在`results/runtime_exact_cache_smoke_20261014/`：toy、1槽、3次更新、2次映射、2层×32，三阶段和新日志均正常结束，不作为论文效果。

## 5. 恢复原单槽任务

原运行的参数来源摘要仍匹配当前配置；特征与网络尺寸未变，旧v7检查点结构保持兼容。本轮只核对兼容性，没有代用户恢复2000轮。运行原命令并附加`--resume`：

```powershell
python -u scripts/run_slot_joint.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_50.yaml `
  --output results/single_slot_n20_a50_s0 `
  --slots 1 --updates 2000 --route-updates 500 --deploy-updates 500 `
  --mapping-samples 128 --seed 0 --trajectory-seed 101 --device cpu --resume
```

保持同一环境与原参数，不能同时开两个进程向同一目录训练；总预算、阶段、轨迹或PPO参数变化仍由原恢复检查拒绝。新JSONL只追加恢复后的轮次，之前的完整历史仍在原`history.json`和检查点中。日志中的完成轮数不是检查点已落盘的替代证据。

复核一次rollout的命令（使用新目录）：

```powershell
python scripts/benchmark_slot_runtime.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_50.yaml `
  --trajectory results/single_slot_n20_a50_s0/train_trajectory.json `
  --output results/runtime_probe_new
```
