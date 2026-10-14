# Agentic 应用的边缘 LLM–无状态服务联合编排实验

当前主线为 **slot_joint_v7**。参数以[当前登记表](docs/experiment_parameter_register_2026-10-13.md)为依据，统一机器配置为[experiment_settings.yaml](configs/experiment_settings.yaml)，另引用[图/工具分布](configs/agent_generator.yaml)与[固定版本模型清单](configs/qwen3_model_catalog.yaml)。主流程不再继承Abilene、GEANT或旧GPU档案。

用户允许候选用于预实验，正式训练仍须确认。未冻结项由`pending`列出，`--formal`会拒绝启动。新网络、GPU效率、非KV显存、价格、传播时间映射均是显式合成假设，不是实测数据。接线与验收边界见[全参数核对记录](docs/parameter_alignment_2026-10-14.md)。

主仿真A10/L20/H20，真机4090/A6000只留之后校验。A10/L20批预算/序列上限为2048/256，H20（96GiB档）为8192/1024，来自固定v0.31.0在线默认。相同配置允许多个实例候选，`replica_index`区分副本，GPU/显存仍共享原掩码；TP=2不是两个副本。每跳传播为0.2–2ms。**负载不按资源规模缩放，已有池的应用总率不重抽。**详情见[GPU与副本改动记录](docs/datacenter_gpu_update_2026-10-14.md)。

随后SLO已按用户确认完成[固定参照标定](docs/slo_calibration_2026-10-14.md)：8B＋单L20，pattern加权参考q95的TTFT/完整时延乘1.5、TBT乘1.25；tight为main的0.8倍。合同由`configs/experiment_settings.yaml`的`slo_contracts`引用，生成场景自动应用并校验；同一逻辑请求跨资源规模不重标。原pool.json仍保存原始生成参数/占位合同，实际训练以生成场景和独立冻结合同为准。训练负载未改。

## 1. 训练与评价

每回合从槽0重放同一条预采样轨迹。到达率、带宽、可用CPU/内存可以跨槽变化；图、pattern概率、token代表值、工具绑定、质量及SLO固定。每槽从空部署表顺序确定LLM候选开关、工具池整数副本数，再逐正到达率的(应用,接入节点)进行请求级模型选择，最后计算容量比例物理路由、完整工作流指标和共同奖励。切换费用仍与上一槽最终部署比较，不表示真实服务每槽全部重启。

| 阶段 | 更新分支 | 另一分支 |
|---|---|---|
| composition | 模型选择 | 安装各槽固定多样部署，不调用部署actor |
| deployment | 部署 | 模型选择冻结，采用Dirichlet均值 |
| joint | 两分支 | 顺序决策并共同更新 |

保持两个条件化actor/critic分支、共同槽末差分效用、当前绝对违约惩罚及首轮固定贪心参照。无教师、蒸馏或ICM/RND。所有场景独立随机初始化，不从最大规模续训。同一请求的各LLM阶段使用同一模型；物理路由仅容量比例，不含Softmin。

评价读取保存的`train_trajectory.json`并核对摘要，不生成另一条评价轨迹。结果是训练场景内评价，不作为未见场景泛化证据。指标口径见[时隙协议](docs/slot_joint_protocol_2026-09-23.md)和[计算口径记录](docs/metric_conventions_2026-10-14.md)：固定网络计量窗口、成本率、仅完成请求的应用等权时延；无完成请求时延为null，另报完成/失败率。

## 2. 安装与测试

在本仓库执行：

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e ".[rl,dev,plot]"
$env:PYTHONIOENCODING='utf-8'
$taskTests = @(git ls-files -- 'tests/test_*.py')
python -m pytest @taskTests
```

已有环境时跳过创建。未editable安装时先设`$env:PYTHONPATH='src'`。测试列表不包含未提交的旧Softmin探针；其文件未被本轮修改。

## 3. 一次应用池、多网络、嵌套子集

当前表中候选服务器点为10/20/30/40/50，默认20；未自动加入128台或改成40台主场景。生成器不调用真实LLM、不训练策略、不按PPO或SLO成绩筛图。

```powershell
python scripts/build_registered_scenarios.py `
  --servers 10,20,30,40,50 --subsets 20,50,100,200 `
  --output results/registered_scenarios

# 同一冻结合同的紧档敏感性场景，不是重新标定。
python scripts/build_registered_scenarios.py `
  --pool results/registered_scenarios/pool.json --servers 20 `
  --subsets 20,50,100,200 --slo-variant tight --output results/registered_slos_tight

python scripts/audit_registered_parameters.py `
  --scenario results/registered_scenarios/servers_20_applications_50.yaml
```

从一个固定池保留图、pattern、token、SLO、质量、工具绑定和每应用总请求率，仅显式映射不同网络的入口、工具服务器率。更多入口不自动增加总负载。更换网络时不会重新抽图或缩短token；保存每个子集的真实使用工具数，不能把闲置目录计入规模。

输出含`pool.json`、各网络`infrastructure_nN.yaml`、可训练的`servers_N_applications_A.yaml`、逐场景审计JSON和manifest。用`--pool <saved pool.json>`可复用池，不重复生成结构/长度；输出目录须为新目录，不覆盖已有结果。生成后的实际分位数和拒绝率必须复核，不能只引用提案分布。

单网络接口仍可用：

```powershell
python scripts/generate_agent_pool.py --output results/one_network_pool
python scripts/rebind_agent_tools.py `
  --pool results/registered_scenarios/pool.json `
  --infrastructure results/registered_scenarios/infrastructure_n10.yaml `
  --tool-counts 20,50,100,200 --output results/tool_scale
```

工具重绑定的基础网络必须与保存池兼容，或先用多网络构建器显式映射。其默认统一工具profile用于隔离种类数/复用的作用。

## 4. 预实验训练

以下是功能试跑，不是正式收敛预算：

```powershell
python scripts/run_slot_joint.py `
  --scenario results/registered_scenarios/servers_20_applications_50.yaml `
  --output results/registered_trial_s0 --updates 12 `
  --route-updates 4 --deploy-updates 4 --trajectory-seed 101 --seed 0 --device cpu
```

主入口统一读取预实验候选：4层×256、两个分支学习率1e-4、PPO小批量32、epoch3、共享条件化模型选择头、跨槽gamma0.99、当前GAE1。32槽、映射128和阶段比例1:1:2也只是登记候选。总更新数必须明确给出，没有自动决定的正式训练预算。

toy和运行时学习参数覆盖必须显式标记为`--smoke`：

```powershell
python scripts/run_slot_joint.py --scenario configs/toy.yaml `
  --output results/toy_smoke --slots 1 --updates 3 `
  --route-updates 1 --deploy-updates 1 --mapping-samples 2 `
  --hidden-size 32 --hidden-layers 2 --update-epochs 1 --minibatch-size 32 --smoke
```

启动前检查数值与结构，旧Abilene或引擎参数冲突会阻止主运行。审计不会把人工/运行时/实机核验项伪装成自动通过。恢复仅限同目录、场景、轨迹、种子、协议、完整配置和原定阶段/总预算，参数来源摘要不同会拒绝`--resume`。

## 5. 评价与记录

训练结束自动生成`evaluation.json`和`baselines.json`；纯模型选择阶段另有同部署基线。快照诊断仍重放训练轨迹，网络结构从训练manifest恢复，不重新用默认层数：

```powershell
python scripts/evaluate_slot_joint_snapshot.py --run results/registered_trial_s0 --baselines
python scripts/evaluate_slot_joint_routing_baselines.py --run results/registered_trial_s0 --mapping-samples 128
```

manifest记录完整PPO/目标/阶段设置、参数表/配置/模型清单摘要、场景和轨迹摘要；`parameter_audit.json`记录检查及待定项，`checkpoint.pt`仅供同场景恢复，`policy.pt`是最终权重。结果与权重被Git忽略，只有checkpoint不代表训练完成。旧v6及更旧协议不续训、不重评分。

### 运行进度与分项耗时

主入口默认逐轮打印阶段、效用、约束和采集/优化、物理路由、解析计算、工作流评价、观测及reset耗时。`training_status.json`保存当前阶段和槽/部署目标/模型组位置，`training_history.jsonl`逐轮追加，原`history.json`与`checkpoint.pt`继续保存。需要动态进度条时加`--progress-bar`，不影响训练参数。PPO完成后仍会打印同轨迹评价和固定基线的执行提示；训练状态完成不代表后续基线评价已经结束。

```powershell
Get-Content results/single_slot_n20_a50_s0/training_status.json -Raw
Get-Content results/single_slot_n20_a50_s0/training_history.jsonl -Tail 3
```

原2000轮单槽任务中断后，用原命令原参数加`--resume`，不覆盖场景或改变阶段预算。新日志记录后续轮次，旧轮次仍保留在原`history.json`，实际恢复以检查点为准。缓存与等价性核验见[运行时优化记录](docs/runtime_exact_cache_2026-10-14.md)；这些改动降低重复计算，不包含最大规模PPO整轮缓存的存储压缩。

工作流评价现默认采用[整数索引与样本批量计算](docs/indexed_workflow_2026-10-14.md)：保留同一采样器，每个映射独立求关键路径/判断SLO，再按原样本顺序汇总。`benchmark_slot_runtime.py --scalar-workflow`仅用于与保留的标量实现对照，不是另一种训练算法。原训练命令无需新增开关。

## 6. 主代码与历史边界

| 文件 | 职责 |
|---|---|
| src/agent_orch/parameters.py | 统一读取、数值/结构审计、正式冻结检查、来源摘要 |
| src/agent_orch/workload/infrastructure.py | 连通随机网络、GPU候选、跨网络入口映射 |
| src/agent_orch/workload/agent_*.py | 概率扩展、工具绑定、联合消息/预算、固定池 |
| src/agent_orch/workload/slot_trajectory.py | 固定工作负载与可变需求/资源轨迹 |
| src/agent_orch/envs/slot_joint_env.py | 槽内顺序部署和模型选择、同槽反馈 |
| src/agent_orch/routing/capacity.py | 唯一物理路由 |
| src/agent_orch/agents/ | 两分支PPO、critic与乘子更新 |
| configs/slo_calibration.yaml | 已声明的固定参照、q95与倍率 |
| data/catalogs/slo_reference_20261014/ | 冻结阈值及1171条pattern参考记录 |
| src/agent_orch/workload/slo_calibration.py | 参考工作流、条件阶段分位数及固定合同校验 |

`configs/benchmarks/`、旧矩阵/蒸馏/内层求解器脚本及旧结果仅为历史资产，不是参数依据或主训练流程。未删除它们，也未把旧结果重标为新协议结果。单网络生成器只有显式`--legacy-smoke`才允许不符合登记表的旧基础模板。固定chunk实机、实际KV、质量、价格和正式预算仍待确认/校验；合成SLO已标定，但不代表真实服务器SLA已经验证。

旧console接口`agent-orch-sim`也必须显式`--legacy-smoke`，防止误跑旧环境/奖励；当前各脚本角色见[scripts/README](scripts/README.md)。
