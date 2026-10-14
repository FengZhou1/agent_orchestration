# 整数索引与批量工作流评价核验（2026-10-14）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: author-approved indexed/batched implementation and exactness verification
- Source: 1eede8c标量实现、旧实验代码的数据组织方式、固定20台/50应用场景及已有单槽轨迹
- Verification Status: 逐样本、完整指标和rollout摘要已核对；330项回归通过；性能探针不代表收敛或所有场景速度
- Protocol: slot_joint_v7，采样器、数学口径、动作、奖励和三阶段学习不变

## 1. 本轮只实施前两项优化

借鉴`old_exp/ppo_experiment/MSPPO/MST_PPO/mstppo/`用服务/服务器整数编号组织数据表的方式，新增`performance/workflow_batch.py`，但不导入旧代码，不照搬其串行链时延公式、实例数规划或路径穷举。

1. 编译pattern的节点列号、父节点列号、根节点、边和最终节点。静态索引按实际有序节点/边/最终节点建键，不只按应用或flow编号。保持原NetworkX词典序拓扑和父节点顺序。
2. 每次工作流评价内，将当前LLM响应/TTFT/TBT/可服务状态、工具池时延/可服务状态、网络路径时延/可承载状态整理成数值表。表不跨负载评价复用；只保留当前应用的LLM表，所有模型的实例以候选编号区分。编号0表示缺失位置/实例。评价结束释放运行时表，避免保留整个解析结果或形成强引用环。
3. **原采样器不变**。它产出的映射和概率转换为整数数组后，按拓扑顺序一次更新当前批次的所有样本。每个样本仍独立等待最长上游依赖、计算工作流TTFT/完整响应时间/中间阶段完成时间并判断SLO。
4. 对样本的完成比例、时延、达标比例和阶段时间仍使用原样本顺序逐项加权累加，不使用`np.mean`或改变浮点求和顺序。完成条件、失稳、缺失部署与非有限时延的边界保留；未完成请求不进入时延均值，也不因SLO放宽而计入goodput。

使用float64，不降低精度，不以平均节点时延代替逐映射关键路径，不先平均再判断SLO。三类合同、单输出token的TBT=0约定及应用等权完成请求时延均保持原口径。

本轮未优化第三项“采样算法”，未减少128映射上限、改变随机种子/choice调用次序、合并接入节点或更改模型选择。未改两类actor/critic、PPO/GAE、差分奖励、绝对违约惩罚、训练轮数/阶段/轨迹、数据、负载、冻结SLO或质量。也未压缩整个PPO rollout缓存，不能据此声称解决最大多槽规模的内存问题。

## 2. 标量对照与边界测试

原`_flow_summary`正文保留为`_flow_summary_scalar`，`WorkflowEvaluator(..., batch_evaluation=False)`供测试使用；主流程默认批量实现。基准脚本的`--scalar-workflow`只在独立诊断进程选用该标量对照，不是论文中的另一个学习方法或新实验参数。

新增22个测试用例，覆盖：并行汇聚、多个图起始节点、三类SLO的恰等于阈值及相邻浮点值、非均匀样本权重、部分缺失映射、未知服务器、LLM/工具/链路失稳、KV不可用、非有限首token时延、中间LLM首token非有限但完成时间有限、纯推理、单输出token、空映射，以及跨评价负载改变后的表重建。保留旧边界谓词的含义，不借优化修订其他计算规则。

各有效样本的e2e、TTFT、TBT及中间阶段完成时间与标量函数进行float64字节比较；完整WorkflowResult与完成概率字典逐项相等。全量回归 **330 passed**，仅11条既有NetworkX hash版本提醒。另做toy、2槽、3次更新的三阶段端到端试跑，输出`results/indexed_workflow_smoke_20261014/`，不是收敛结果。

用AST核对确认`_physical_mapping_outcomes`与本轮改动前HEAD实现完全相同；新增批量模块没有随机数调用。图与模型/工具/网络数据不同不被当成同一动态结果。

## 3. 真实预实验场景的严格对照

沿用已存在的20服务器、50应用、77工具、279 pattern、304应用—入口组场景：

- 场景文件SHA256：`4813b11b72a3a1aed04069e6a2276a778f2849cd0f628cbb75aea93259c405a3`。
- 保存的单槽轨迹内容摘要：`9c5b0cd15400a72d9b30fe209349282c225fd77f38e1e310e8834d60b740dd75`。
- 4层×256、CPU、seed0、128映射上限，模型选择阶段，单次rollout，不更新PPO。

标量对照、初版批量、最终批量、最终批量cProfile及最终批量补测，均得到304条转移及相同摘要：

| 核对对象 | SHA256摘要 |
|---|---|
| 观测/下一观测、动作、log-prob、价值、奖励、折扣 | `2a9158991712610f7d22a35df57c194c1fda6457fb975e9a38343a5eef864fa5` |
| 完整SlotMetrics与逐应用/逐flow诊断 | `918a535e5cda7f1ae3b0bc4233cad47f0c84ecab7bdc47fd615bb49b5dfa1bdf` |

以上也与上一轮1eede8c探针相同。没有使用数值容差掩盖差异，但有限对照和边界测试不是穷尽所有输入的形式证明。

## 4. 耗时与剩余热点

| 不带cProfile的单次探针 | 采集总时间 | 工作流评价 |
|---|---:|---:|
| 标量对照 | 66.912秒 | 54.824秒 |
| 初版批量 | 55.282秒 | 42.874秒 |
| 最终批量（与诊断/回归并发） | 58.218秒 | 45.341秒 |
| 最终批量补测（未再并发另一个诊断/回归） | 53.859秒 | 41.391秒 |

以标量对照和最终补测为参考，采集时间减少约19.5%，工作流评价减少约24.5%。这是当前机器上的少量功能探针，不是多次隔离硬件计时或所有训练阶段的加速保证。不能与更早138秒的原运行相除后，把前一轮缓存收益也全部归到本轮。

最终实现的cProfile探针总采集87.026秒、工作流59.022秒，其中未改动的`_physical_mapping_outcomes`累计51.191秒，批量数值内核累计6.080秒；两者均调用6328次。现在评价循环的开销已减少，**剩余主要耗时转移到位置采样及映射字典构造**。本轮按用户范围没有继续改采样器，也没有承诺单轮几秒或2000轮收敛。

产物目录`results/indexed_workflow_20261014_{scalar,batch,final,final_profile,final_solo}/`保存输入/实现摘要、完整指标和计时；这些新目录不覆盖原任务。cProfile本身有开销，不能直接与普通运行时间比较。

## 5. 运行与恢复

原单槽任务未恢复，原参数来源摘要仍匹配，网络/观测尺寸和检查点协议未变。用原命令加`--resume`即可使用默认批量评价；不改2000轮及500/500/1000阶段预算。

复核命令在仓库根目录、agent-orch环境下执行，输出必须使用新目录：

```powershell
python scripts/benchmark_slot_runtime.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_50.yaml `
  --trajectory results/single_slot_n20_a50_s0/train_trajectory.json `
  --output results/indexed_probe_scalar --scalar-workflow

python scripts/benchmark_slot_runtime.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_50.yaml `
  --trajectory results/single_slot_n20_a50_s0/train_trajectory.json `
  --output results/indexed_probe_batch
```
