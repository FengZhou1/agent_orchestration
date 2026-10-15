# Agent 应用图生成器与边缘联合编排仿真

本仓库为 Agent 应用的 LLM 服务与无状态工具服务联合编排提供合成工作负载、解析性能评价和分层 PPO 训练。**先生成一个固定应用池，再从中导出不同应用数量、工具规模和服务器数量的场景。** 不需要具体业务、工具 API、真实任务数据集或调用真实 LLM，也不限定四种应用模板。

本文档重点介绍新生成器如何使用。当前实验协议是 `slot_joint_v7`，参数仍为允许用于预实验的候选设置，**生成成功不等于 SLO 全部可满足，也不等于训练已经收敛**。

## 工作流运行加速（不改变实验参数）

训练入口、快照评价和模型选择基线支持 `--workflow-engine reference|array` 与
`--workflow-workers 1|2|4`。`reference` 保留原采样器与评价实现；`array`
直接生成整数位置数组，仍按原顺序精确枚举或使用原来的固定样本，并在每个位置方案下先计算关键路径和判断 SLO。
多进程仅可用于 `array`，全网负载先统一计算，再分配同槽的应用评价。
新 `run_slot_joint.py` 训练默认 `array + 1进程`，保留 `reference` 回退。
20/50应用的完整32槽回放从450.20/949.80秒降至105.18/271.75秒，指标和转移摘要完全一致。
2/4进程在这两场景均未达到额外15%的端到端耗时下降门槛，因此不推荐设为默认。

```powershell
# 在原有训练指令末尾追加；其他科学参数不变
# --workflow-engine array --workflow-workers 1

# 绑定已有运行的真实检查点、轨迹及预训部署；不训练、不覆盖原运行
python scripts/benchmark_slot_runtime.py `
  --run results/composition_n20_a20_32slots_s0 `
  --workflow-engine array --workflow-workers 1 `
  --output results/runtime_probe_array1

# 固定输入的五轮对照，包括 reference、array 1/2/4 进程
python scripts/validate_workflow_runtime.py `
  --run results/composition_n20_a20_32slots_s0 `
  --output results/runtime_validation
```

无运行时记录的旧 manifest 按 `reference + 1进程` 解释。`--resume` 不指定这两个选项时沿用保存值；显式切换会记录运行时变更，保留轨迹、部署库、优化器及奖励参照。
因此，恢复旧训练并启用加速时，在**原训练指令**后追加 `--resume --workflow-engine array --workflow-workers 1`，其余参数仍须与原 manifest 一致。
这些选项不进入科学参数摘要、策略观测或奖励。验证探针的内存监测需要 `psutil`；结果目录必须是新目录。
详细边界与验收结果见 [运行加速核验](docs/workflow_runtime_2026-10-15.md)。

### 整流程去重复计算（第二轮优化）

在上面的数组引擎基础上，进一步复用槽内不变观测/掩码、共享LLM求解的chunk数据，
索引非零路由支持，并按原顺序进行原生float64前缀累加。扫描范围、稳态初值与收敛阈值、
动作顺序、128个固定样本及奖励口径均不改变；无新增科学参数。

训练与各评价环境共用固定贪心参照。新运行会在结果目录写入小型
`fixed_reference_cache.json`，按场景、轨迹、目标/尺度、采样规则、实现和依赖环境校验，
不按部署相同就复用动态指标。旧检查点缺该文件时仍可恢复，首次计算后保存。
该缓存不是教师动作、策略蒸馏或完整rollout缓存。

```powershell
# 对已保存运行做隔离的旧版/优化版32槽对照，不更新PPO、不覆盖原运行
python scripts/benchmark_pipeline_runtime.py `
  --run results/composition_n20_a20_32slots_s0 `
  --output results/pipeline_probe_new `
  --reference-source results/pipeline_reference_de4a017
```

对照代码从Git提交 `de4a017` 导出，需本地具备该提交；不混用两版模块。
运行器冻结输入检查点、轨迹和部署库，并要求完整指标和转移摘要相同。
analytic内部计时、容量/实例缓存状态分别记录于性能日志，不进入策略状态或奖励。
详细结果见 [整流程优化核验](docs/pipeline_runtime_2026-10-15.md)。
本轮在上一版数组单进程基础上，20/50应用32槽回放分别从106.66/275.49秒降至44.90/97.31秒；
对应加速2.38×/2.83×，完整指标与转移摘要一致。均不含PPO更新，不作为收敛结论。

## 1. 先区分四类数据

| 对象 | 包含什么 | 如何变化 |
|---|---|---|
| 应用池 `pool.json` | 完整图、pattern、概率、token、工具绑定、质量、原始 SLO、基准到达率及生成记录 | 生成一次后保存；不同应用规模取固定嵌套子集 |
| 场景 `servers_N_applications_A.yaml` | 从池中选择的应用、该子集实际使用的工具，以及网络、服务器、GPU、模型配置和冻结 SLO | 服务器规模改变时显式重映射入口和工具服务器参数，不重新抽图或调整应用总负载 |
| 时隙轨迹 `train_trajectory.json` | 各槽到达率、带宽、CPU/内存可用量 | 槽间可变化；每个训练回合从槽0重放相同数值 |
| 预训部署 `warmup_deployments.json` | 每槽预先构造的模型实例和工具副本 | 只为模型选择预训提供部署上下文；不属于应用图生成器，不是教师动作 |

**图节点代表一次推理或工具请求；工具类型代表可部署的服务。** 同一种工具可以绑定多个节点，也可以跨应用复用；允许大量专属、不复用工具。应用编号不同不自动算不同应用：生成器按属性图去重，并另外统计不同拓扑数。

## 2. 安装与快速开始

以下命令均在仓库根目录执行，示例使用 PowerShell。已有环境时跳过环境创建与安装。

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e ".[rl,dev,plot]"
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
```

### 2.1 生成主应用池和场景

```powershell
python scripts/build_registered_scenarios.py `
  --servers 20 --subsets 20,50,100,200 `
  --output results/agent_scenarios

python scripts/audit_registered_parameters.py `
  --scenario results/agent_scenarios/servers_20_applications_50.yaml
```

使用当前配置生成 **200个应用、200种实际使用工具类型的主池**，再导出20/50/100/200应用的嵌套场景。50应用场景不保证使用200种工具，实际工具数会写入manifest。20是服务器数，不是GPU数。

所有生成命令要求使用**不存在的新输出目录**，不会覆盖已有池、场景或实验。`results/`默认不进Git；其他机器需按配置重新生成或自行取得保存池，不能假定本机历史结果目录随仓库推送。

首次生成会打印多次 `generation pass`，并可能出现 `input-refined`：它们是**合成长度分位数的预校准**，不是PPO轮次、策略训练或蒸馏。生成期间不计算策略成绩，也不按SLO达标率筛图。

### 2.2 同一个池，导出不同服务器规模

```powershell
python scripts/build_registered_scenarios.py `
  --pool results/agent_scenarios/pool.json `
  --servers 10,20,30,40,50 --subsets 20,50,100,200 `
  --output results/agent_scale_matrix
```

导出5种服务器规模×4种应用数量，共20个场景。图、pattern概率、token、质量、工具身份及每应用总请求率来自同一池；入口重新映射到各网络，工具的同类处理率映射到新服务器。**不根据资源容量缩放负载，也不在每种规模重新生成应用。** 各场景之后独立训练，不从最大规模继续训练。

复用 `--pool` 时不能同时用 `--applications`、`--tool-types`、`--calibration-passes`修改生成设置。需要新工作负载时生成新池；只需要较少应用时改 `--subsets`。

## 3. 完整图与执行 pattern 怎样生成

先抽取全图的LLM、工具节点预算，再从入口LLM进行受预算约束的概率扩展，组合继续推理、串行工具、并行工具、互斥片段、提前响应和有限重复。重复会展开成新节点，**不生成环**；工具执行分支最终由LLM响应。

完整图包含所有允许的依赖。一次请求只执行其中一个合法pattern：串行片段顺序执行，并行片段保留全部必要分支，互斥片段只选择一个选项。互斥概率归一化，pattern概率由控制选择累乘得到；并行分支不能当作互斥选项归一化。

构图时就预留pattern组合预算，不在生成后删除部分pattern再重新归一化。所有节点都必须出现在某个正概率pattern中。完整图按节点和边保存，避免为嵌套并行枚举所有路径。

**“每次执行的LLM数量P50=2”不等于“完整图只有2个LLM”。** 长分支可以存在于完整图中，但只有部分请求经过。执行LLM数量按“应用等权、pattern按概率加权”统计，另报告到达率加权分布；应核对生成后的实际P50，不能把目标当作已实现的结果。

## 4. 生成参数

参数依据是[实验参数登记表](docs/experiment_parameter_register_2026-10-13.md)。多数生成参数在[agent_generator.yaml](configs/agent_generator.yaml)；网络/GPU与训练配置在[experiment_settings.yaml](configs/experiment_settings.yaml)。以下是当前文件的值，非真实数据拟合结果或正式冻结的最优设置。

### 4.1 应用规模与结构

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `seed` | 20261013 | 主生成种子；结构、长度、绑定和属性使用分域随机种子 |
| `applications` | 200 | 主池应用数，不是每个子场景的应用数 |
| `tool_types` | 200 | 主池真正使用的工具类型数，不是工具请求节点数 |
| `pure_fraction` | 0.05 | 纯推理应用数取 `round(applications × pure_fraction)`；默认10个 |
| `llm_bounds` | [2,15] | 含工具应用的全图LLM节点预算；纯推理例外为1个 |
| `tool_bounds` | [1,30] | 含工具应用的全图工具节点预算；纯推理例外为0个 |
| `llm_log_mu` / `llm_log_sigma` | ln(4) / 0.8 | LLM预算：对数正态提案、取整并按边界拒绝采样 |
| `tool_log_mu` / `tool_log_sigma` | ln(5) / 0.9 | 工具预算：同上，不是保证最终P50为5 |
| `max_depth` | 30 | 最长依赖路径上的请求节点数上限；纯推理深度为1 |
| `max_parallel` | 4 | 单次局部展开分支上限，不限制整张图最大层宽为4 |
| `max_patterns` | 32 | 每应用执行pattern上限，在构图前约束组合数 |
| `operation_weights` | serial/parallel/exclusive/repeat各1 | 在预算允许的操作中归一化抽样，不代表四种拓扑各占25% |
| `root_stop_range` | [0.05,0.30] | 入口直接响应概率的抽样范围 |
| `short_stop_range` | [0.60,0.85] | 短执行结束概率范围；没有剩余推理时直接结束 |
| `later_stop_range` | [0.15,0.55] | 后续推理中提前响应的概率范围 |
| `structural_attempts` | 200 | 每应用结构构造尝试上限 |
| `length_attempts` | 200 | 固定结构下长度联合生成的尝试上限 |
| `calibration_passes` | 20 | 长度分位数预校准轮数；实现还可进行最多8步输入分位数细化 |

节点预算是右偏抽样，深度和宽度由扩展与联合合法性约束形成，不另独立抽一个深度再强行补图。含工具路径需要入口、工具及后继LLM；设置最大深度不代表范围内每个深度都会出现。

### 4.2 Token、历史上下文与预算

长度分布及预算目前是[agent_lengths.py](src/agent_orch/workload/agent_lengths.py)和[agent_pool.py](src/agent_orch/workload/agent_pool.py)中的实现定义，**不是 `agent_generator.yaml` 已支持的额外字段**。不能向YAML添加 `input_p50` 等不存在的键，或把所有长度写成独立抽样。

| 对象 | 合成目标 | 硬范围/预算 |
|---|---|---|
| 每次LLM完整输入，含保留历史 | P50=2,048；P90=8,192 | 256–16,384 token |
| 每次LLM输出 | P50=200；P90=1,200 | 16–2,048 token |
| 工具返回可见文本 | P50=256；P90=1,500 | 48–5,000 token |
| 单次LLM窗口 | 输入＋输出 | ≤32,768，且不超过基础配置最小上下文窗口 |
| 每个实际执行pattern累计量 | 全部LLM输入＋输出之和，重复prefill重复计入 | ≤128,000 token |

使用截断对数正态提案。先抽输出和工具文本，再按消息身份累计历史，最后条件抽取非负新增输入。并行汇聚时共同历史只计一次，各分支结果分别加入；新增消息在后续输入中重复出现所需的预算会提前预留。所有pattern都要通过检查，不能只检查平均执行。

同一节点在不同pattern中可能携带不同历史：`pool.json`保留每个pattern的详细长度，场景YAML使用经过该节点的pattern概率条件加权的输入代表值。当前合成长度对各模型不额外添加长度倍率，也未调用真实tokenizer。

预算与历史会改变最终长度分布，必须查看输出统计。修改长度范围需要同步检查生成、校准和验证实现，不应只改一个常量后沿用旧合同或旧结果。

### 4.3 工具绑定、处理资源与通信

| 配置项/对象 | 当前值 | 含义 |
|---|---|---|
| `shared_fraction` | 0.5 | 工具目录中允许共享的类型比例，不是共享请求占比 |
| `private_request_probability` | 0.5 | 有所属私有工具且存在共享目录时，剩余请求选择私有工具的概率 |
| `shared_zipf_exponent` | 1.1 | 共享工具非均匀选择权重，部分工具更热门 |
| 每副本CPU | 1核 | 当前构造固定值，不是YAML可选的1/2/4核 |
| `tool_memory_gb` | [0.5,3.0] | 每类工具抽取的单副本内存 |
| `tool_service_rate_rps` | [2.0,8.0] | 每类工具单副本处理率；同一类型在各服务器取相同值 |
| 到达间隔SCV | 1.0 | 当前工具构造固定值 |
| `tool_cpu_cost_per_s` | 0.02 | 每核相对成本率系数 |
| `tool_memory_cost_per_gb_s` | 0.001 | 每GB内存相对成本率系数 |
| `tool_start_cost_ratio_s` | 0.1 | 启动费用=单副本运行成本率×该系数 |
| `nontext_return_mb` | [2.0,10.0] | 每个工具请求额外非文本返回量，十进制MB |
| `text_bytes_per_token` | 4.0 | 合成文本长度换算为通信字节量的假设 |

共享工具可能跨应用使用，私有工具只属于一个应用。不强制每个应用都有共享工具；“允许共享”不保证某类型在每个子集中都真正被多个应用使用。完整池中每种工具都必须被使用，子场景自动剔除不使用的目录项。

通信量与可见文本分开保存。非文本回包占用网络，但不会全部变成LLM输入token；不能由回包长度直接推断CPU需求。

### 4.4 应用属性和SLO

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `application_rate_rps` | [0.005,0.02] | 每应用总基准请求率，不是每入口各分配这么多 |
| `ingress_probability` | 0.3 | 每服务器进入非零接入支持集的抽样概率；至少一个入口 |
| 接入权重 | Dirichlet(1,…,1) | 在非零入口分配应用总率；其他服务器允许零率 |
| `slo_type_weights` | [1,1,1] | lat/ddl/cmp三类SLO的抽样权重 |
| `ttft_s_range` | [2,20]秒 | 原始池占位阈值范围，不是当前训练场景最终合同 |
| `tbt_s_range` | [0.02,0.2]秒 | 同上 |
| `deadline_s_range` | [20,180]秒 | 同上 |
| `quality_base` / `quality_size_gain` | 0.58 / 0.24 | 按模型参数量排序构造合成质量均值 |
| `quality_noise` | 0.1 | 各应用—模型独立高斯噪声标准差 |
| `quality_range` | [0.4,0.95] | 最终质量代理分数截断范围 |

合成质量不是任务准确率实测；均值偏置不保证每个应用上大模型质量都更高。

当前主入口按 `experiment_settings.yaml` 的 `slo_contracts`，用已冻结合同覆盖导出场景的占位阈值：固定 **Qwen3-8B＋单L20** 的低负载解析工作流，pattern加权q95的TTFT/完整时延乘1.5，TBT乘1.25；紧档为主档0.8倍。类型、图、概率、token或通信需求不匹配时会报错，不自动放宽。详见[SLO标定说明](docs/slo_calibration_2026-10-14.md)。

### 4.5 与应用池组合的网络和模型

这些参数来自 `experiment_settings.yaml`，不是应用结构生成器内部规则。

| 项目 | 当前设置 |
|---|---|
| 服务器候选规模 | 10/20/30/40/50，默认20 |
| 网络种子 | 20261014，与应用池种子分开 |
| 拓扑 | 固定种子、MST保证连通后加边；平均度目标4 |
| CPU / 内存 | 每服务器8–64核 / 16–80GB |
| GPU | A10/L20/H20；每机0/1/2张等概率，型号等概率；显存档24/48/96GiB |
| 模型 | Qwen3-4B/8B/14B/32B，BF16，固定revision清单 |
| 链路带宽 / 每跳传播 | 50–100Mbps / 0.2–2ms |
| 实例候选 | 同配置可有多个副本候选，总占用受GPU/显存约束；TP=2不是两个副本 |
| 批处理 | 固定vLLM v0.31.0在线默认档：A10/L20为2048 token、256序列；H20为8192 token、1024序列 |
| 前缀缓存 / 解析chunk | 关闭 / 512 token；固定chunk实机核验仍待完成 |

有效算力、带宽效率、价格和部分资源分布是明示的合成假设。4090/A6000用于之后的真实服务器校验，不限制本阶段仿真。完整硬件参数、价格与来源仍以登记表和配置为准。

## 5. 输出文件怎么用、怎么检查

```text
results/agent_scenarios/
  pool.json                              # 原始完整应用池与实际统计
  manifest.json                          # 来源摘要、各场景规模/ID/文件哈希
  infrastructure_n20.yaml                # 纯基础设施，供映射/工具重绑定
  servers_20_applications_20.yaml         # 可由ScenarioLoader读取的场景
  servers_20_applications_20_audit.json    # 登记参数检查及未确认项
  servers_20_applications_50.yaml
  servers_20_applications_50_audit.json
  ...
```

`pool.json`不是训练入口直接读取的场景文件；训练传入 `servers_N_applications_A.yaml`。原始池保存生成阈值，当前训练合同在导出场景中应用，不要混为一份最终SLO。

```powershell
$taskPool = Get-Content results/agent_scenarios/pool.json -Raw | ConvertFrom-Json
$taskPool.statistics | ConvertTo-Json -Depth 8

$taskManifest = Get-Content results/agent_scenarios/manifest.json -Raw | ConvertFrom-Json
$taskManifest.scenarios | ConvertTo-Json -Depth 4
```

重点查看：

- `applications`、`unique_topologies`：不同应用配置数和不同拓扑数。
- `actual_tool_types`、`actually_shared_types`、`non_reused_types`：实际使用、真实跨应用复用及未复用工具数。
- `full_graph`：全图LLM/工具数、深度、宽度和pattern数分布。
- `application_equal_pattern_weighted`：执行LLM数、输入、输出及工具文本的概率加权分布。
- `arrival_weighted`：加入实际应用总率后的分布。
- `structural_rejections`、`length_rejections`：生成尝试的拒绝统计。
- `provenance.length_calibration`、`input_quantile_refinement`、`length_distributions`：长度预校准过程和最终提案参数。

`pool.statistics`是整个主池的统计；各场景实际应用/工具数量查看 `manifest.scenarios`。单网络入口还在manifest保存每个子集的完整统计。合法性检查包括DAG、控制树重放、pattern概率、正概率覆盖、末级LLM、消息合并、逐pattern预算和应用配置去重，**不要求贪心部署满足全部SLO**。

当前默认配置的一次核对示例（2026-10-15，本机软件环境）：主池有200个应用配置、172种拓扑、200种实际工具类型。

| 概率加权统计 | 实际P50 | 实际P90 |
|---|---:|---:|
| 每次执行LLM数 | 2 | 4 |
| 完整输入token | 2,105 | 8,075 |
| 输出token | 200 | 1,177 |
| 工具文本token | 256 | 1,493 |

这是生成后统计，不是新增的硬目标或性能结果。更换种子、配置、依赖版本后应重新核对，以自己的输出文件为准。

## 6. 常用变体与自定义

### 6.1 只生成一个网络上的固定池

```powershell
python scripts/generate_agent_pool.py `
  --config configs/agent_generator.yaml --subsets 20,50,100,200 `
  --output results/agent_pool_single_network
```

默认构造登记的20服务器网络；也可用 `--infrastructure` 提供兼容基础设施。场景命名为 `applications_20.yaml`、`applications_50.yaml` 等，而不是主入口的 `servers_N_applications_A.yaml`。

支持 `--seed`、`--applications`、`--tool-types`、`--calibration-passes`。**多网络主入口没有 `--seed` 或 `--config`**，通过 `--settings` 引用的生成器配置修改种子/结构规则。

### 6.2 保持图和长度，改变工具种类及复用

```powershell
python scripts/rebind_agent_tools.py `
  --pool results/agent_scenarios/pool.json `
  --infrastructure results/agent_scenarios/infrastructure_n20.yaml `
  --tool-counts 20,50,100,200 --output results/agent_tool_scale
```

输出 `tools_20.yaml` 等及manifest；还可显式设置 `--shared-fraction`。图、pattern、概率、token、质量和到达率不重抽，原请求冻结合同继承；工具绑定及工具profile改变。**当前入口使用统一工具profile**：CPU1核，内存/处理率取配置区间中点，以隔离种类数和复用的影响。

默认导出**整个输入池**，没有 `--subsets` 参数；上述例子是200应用的工具规模对照，不是50应用。基础设施必须与保存池服务器映射兼容；换网络先显式映射，不直接套不匹配的入口ID。

工具类型数不能超过可绑定的工具请求节点总数；全私有目录需要每个含工具应用至少有一个所属类型。遇到绑定不可行应修改明确的生成设置或规模，不以闲置目录凑数。

### 6.3 同一合同使用紧SLO档

```powershell
python scripts/build_registered_scenarios.py `
  --pool results/agent_scenarios/pool.json `
  --servers 20 --subsets 20,50,100,200 --slo-variant tight `
  --output results/agent_slos_tight
```

这是原冻结合同的紧档，不是根据训练成绩重新标定。

### 6.4 改种子、结构或长度：新池与新SLO合同

默认冻结合同对应特定200应用池。改变种子、主池数量或长度预校准设置，可能连同应用ID的图/token一起改变。**`--applications 50`不是默认200应用池的前50个应用；使用原池子集应写 `--pool ... --subsets 50`。**

自定义时保留原配置和正在运行的实验，使用独立配置：

```powershell
Copy-Item configs/agent_generator.yaml configs/my_agent_generator.yaml
Copy-Item configs/experiment_settings.yaml configs/my_experiment_settings.yaml
```

手动编辑 `my_agent_generator.yaml`；在 `my_experiment_settings.yaml` 设置 `generator: configs/my_agent_generator.yaml` 和 `slo_contracts: null`。以下以仍有200应用的自定义池为例：

```powershell
# 此时导出的是声明范围内的占位SLO。
python scripts/build_registered_scenarios.py `
  --settings configs/my_experiment_settings.yaml `
  --servers 20 --subsets 20,50,100,200 --output results/custom_agent_raw

# 为包含全部应用的场景单独标定一次，不改训练流量。
python scripts/calibrate_synthetic_slos.py `
  --settings configs/my_experiment_settings.yaml `
  --scenario results/custom_agent_raw/servers_20_applications_200.yaml `
  --output results/custom_agent_slos
```

然后将自定义settings的 `slo_contracts` 指向 `results/custom_agent_slos/contracts.json`，用同一 `custom_agent_raw/pool.json`重新导出到新目录：

```powershell
python scripts/build_registered_scenarios.py `
  --settings configs/my_experiment_settings.yaml `
  --pool results/custom_agent_raw/pool.json `
  --servers 20 --subsets 20,50,100,200 --output results/custom_agent_ready
```

旧场景、轨迹、检查点的来源摘要不会自动更新；不要覆盖旧场景后直接 `--resume`。自定义规模仍应与登记扫描点、预算和绑定条件兼容，不能借关闭合同绕过正式参数确认。

## 7. 从场景衔接模型选择预训

优先做较小规模定位时，已另准备[20服务器、20应用的模型选择诊断](docs/small_scale_n20_a20_2026-10-15.md)：实际47种工具，32槽固定多样部署，100轮示例预算。它从原池取子集，不重抽工作负载、不复用50应用的权重；只缩小规模，不表示此前学习稳定性问题已经修复。

应用池与场景生成完后，才生成固定时隙轨迹及多样部署：

```powershell
python scripts/build_warmup_deployments.py `
  --scenario results/agent_scenarios/servers_20_applications_50.yaml `
  --slots 32 --trajectory-seed 101 --deployment-seed 20261014 `
  --output results/agent_warmup_n20_a50

# 500轮只是示例预算，不是已经验证的收敛保证。
python -u scripts/run_slot_joint.py `
  --scenario results/agent_scenarios/servers_20_applications_50.yaml `
  --warmup-bundle results/agent_warmup_n20_a50 `
  --output results/agent_composition_n20_a50_s0 `
  --slots 32 --trajectory-seed 101 --warmup-seed 20261014 `
  --updates 500 --route-updates 500 --deploy-updates 0 `
  --mapping-samples 128 --seed 0 --device cpu
```

模型选择阶段安装每槽固定部署，只更新模型选择分支。部署/联合阶段仍由部署PPO构造方案，不从库中选部署。无教师、蒸馏或默认ICM/RND；同一请求各LLM阶段使用同一模型，物理路由仅容量比例。

### 已完成预算后的续训

`--updates`是**包括已完成轮数在内的总目标**，不是新增轮数。普通`--resume`仍要求参数与原预算一致；增加预算时须同时添加`--extend-budget`。只能延长最后一个既有训练阶段，不能改变此前阶段的长度、顺序、网络、奖励或数据。只训练模型选择时，`--route-updates`必须与`--updates`相同，`--deploy-updates 0`保持不变。

当前20服务器、20应用的100轮运行可从第101轮续到总计1000轮，完整命令见[小规模训练说明：扩展预算](docs/small_scale_n20_a20_2026-10-15.md#41-从100轮续到总计1000轮)。该命令保留原目录和检查点，不重新初始化。添加`--check-resume`可先做只读核验，不计算固定参照、不训练，也不修改运行文件。

实际扩展开始前，脚本将原`checkpoint.pt`、`manifest.json`和`history.json`复制到运行目录的`budget_extensions/<时间戳>/`，并记录预算变更；每轮的新检查点继续保存策略、Adam状态、随机数状态、乘子和上一轮同槽奖励参照。中断后重复同一扩展命令即可。增加轮数不表示保证收敛，也不取消当前参数的预实验标记。

图、token、pattern概率、工具绑定、质量与SLO跨槽固定。到达率乘U(0.6,1.4)，带宽乘U(0.7,1.0)，CPU/内存乘U(0.8,1.0)；GPU库存主场景不变。应用池、网络、轨迹、部署和PPO种子分别记录，不能把修改PPO种子当作重新生成应用。

评价只重放保存的训练轨迹，不创建额外评价轨迹。模型选择预训、冻结该分支训练部署、最后联合更新的框架不变；各场景独立初始化。当前学习稳定性仍需改进，本README不宣称已有正式收敛结果。长轨迹初始化/工作流评价和部署阶段rollout内存仍有成本，不能把生成器验收当作整套大规模训练已就绪。

## 8. 验证、常见报错与实现位置

```powershell
python -m pytest tests/test_agent_generator.py tests/test_registered_parameters.py `
  tests/test_datacenter_gpu.py tests/test_slo_calibration.py

# 全部已跟踪测试，不包含本机未提交的历史探针。
$taskTests = @(git ls-files -- 'tests/test_*.py')
python -m pytest @taskTests
```

| 报错/现象 | 处理 |
|---|---|
| 输出目录已存在 | 换新目录，不覆盖历史生成物或运行中的训练文件 |
| `SLO contracts do not match ...` | 新图/概率/token/通信需求不能复用旧合同，按6.4生成并冻结新合同 |
| 工具数超过可绑定请求数 | 降低类型数或明确增加图/应用规模，不加入闲置类型 |
| 新增YAML字段被拒绝 | 检查 `GeneratorConfig`；不是所有实现参数都暴露为配置字段 |
| 分位数未恰好等于目标 | 核对合法生成后统计、预算及校准记录，不把提案当最终分布 |
| 主入口参数审计不通过 | 对照登记表、模型清单与来源摘要，不回退到Abilene默认或删掉检查 |
| `--formal`拒绝启动 | 当前仍有 `pending`；候选可用于预实验不等于正式参数已冻结 |

| 实现/文档 | 职责 |
|---|---|
| [agent_graph.py](src/agent_orch/workload/agent_graph.py) | 预算扩展、控制树、pattern语义及DAG合法性 |
| [agent_lengths.py](src/agent_orch/workload/agent_lengths.py) | 消息合并、条件长度抽样、窗口及累计预算 |
| [agent_pool.py](src/agent_orch/workload/agent_pool.py) | 工具绑定、属性、去重、分层排序与统计 |
| [infrastructure.py](src/agent_orch/workload/infrastructure.py) | 随机网络、GPU候选及跨网络重映射 |
| [build_registered_scenarios.py](scripts/build_registered_scenarios.py) | 主池、多网络、嵌套子集、冻结合同与审计 |
| [生成器方法说明](docs/agent_generator.md) | 生成语义和来源记录；历史状态以本页及当前参数表为准 |
| [固定部署说明](docs/warmup_deployments_2026-10-14.md) | 模型选择预训部署，与应用生成器区分 |
| [时隙训练协议](docs/slot_joint_protocol_2026-09-23.md) | 顺序决策、奖励、冻结、同轨迹评价及恢复 |
| [计算口径](docs/metric_conventions_2026-10-14.md) | 成本率、固定网络窗口及完成请求时延 |
| [脚本入口](scripts/README.md) | 当前入口与历史工具边界 |

`configs/benchmarks/`的Abilene/GEANT、旧四类固定图、旧矩阵/蒸馏脚本与历史结果不作为当前生成器主入口。合成图覆盖声明分布中的有限样本，不表示穷尽所有Agent应用。真实LLM profile与K8s具体应用实验另行开展，不是生成合成仿真图的前置条件。
