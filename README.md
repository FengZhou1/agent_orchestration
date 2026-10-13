# Agentic 应用的边缘 LLM–无状态服务联合编排实验

当前主线为 **slot_joint_v6**：仅容量比例物理路由、token与分支概率跨时隙固定、评价重放保存的训练轨迹。入口为 [run_slot_joint.py](scripts/run_slot_joint.py)，流程见[时隙协议](docs/slot_joint_protocol_2026-09-23.md)，参数来源与待办见[参数登记表](docs/experiment_parameter_register_2026-10-13.md)。

> 2026-10-13：Softmin实现和专用槽跳变诊断脚本已删除；旧v3–v5检查点不能续接或用本入口重新评分。历史结果与未提交的独立探针原文件保留，不作为当前实验结论。大规模随机应用池和联合token生成目前仅完成规则设计，尚未接入生成器；Toy/Main配置仅用于代码验证。

## 1. 训练与评价流程

只有物理时隙t。训练前采样一次轨迹，保存为train_trajectory.json；每个episode从槽0重放。到达率、带宽、可用CPU/内存可以跨槽变化；图结构、pattern概率、各LLM节点输入输出token、质量和SLO固定。

每槽依次顺序部署LLM候选与工具副本，再逐(应用,接入节点)确定请求级模型选择比例，之后计算容量比例路由并评价一次完整工作流。同一请求的各LLM阶段使用同一模型。部署/模型子步不推进物理时间；从空表构造当期方案，不表示实际服务每槽全部重启，切换费用仍与上一槽最终部署比较。

LLM在在线、可达、正容量同模型实例之间按参考容量分配；工具按副本数乘单副本处理率分配。空支持集标为不可服务，不用均匀路由补齐。四项效用与上一回合同槽比较，首轮使用固定贪心参照；惩罚使用当前绝对违约量。

| 阶段 | 更新分支 | 另一类决策 |
|---|---|---|
| composition | 模型选择 | 安装各槽固定、多样可行部署，不调用部署actor |
| deployment | 部署 | 模型选择冻结，采用Dirichlet均值 |
| joint | 两分支 | 依次产生动作，共同更新 |

--route-updates、--deploy-updates及剩余预算决定阶段。随机初始化，无教师或蒸馏权重，贪心只作奖励参照。当前需求/资源合成扰动范围为到达率U(0.6,1.4)、带宽U(0.7,1.0)、CPU/内存U(0.8,1.0)，尚未实测标定。token与pattern概率直接复制基础配置，不再通过随机扰动或大Dirichlet浓度近似固定。

评价直接读取保存的训练轨迹并核对摘要；不生成evaluation_trajectory.json，没有额外评价种子。策略与基线使用相同轨迹及评价精度，结果属于训练场景内评价，不作为未见场景泛化证据。

## 2. 安装与测试

在仓库目录执行：

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e ".[rl,dev,plot]"
# 当前Git管理的正式测试，不运行未提交的旧Softmin探针。
$taskTests = @(git ls-files -- 'tests/test_*.py')
python -m pytest @taskTests
```

已有环境时跳过创建。未editable安装时先设PYTHONPATH为src。Windows绘图前应激活环境以加载DLL路径。

## 3. 运行示例

以下为功能验证，不是冻结的正式参数，输出须为新目录：

```powershell
python scripts/run_slot_joint.py `
  --scenario configs/toy.yaml --output results/slot_joint_v6_toy_01 `
  --slots 2 --updates 2 --route-updates 1 --deploy-updates 0 `
  --trajectory-seed 101 --seed 0 --mapping-samples 2 --device cpu
```

主场景模型选择诊断仍使用现有配置，不代表随机应用池已构建：

```powershell
python scripts/run_slot_joint.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --output results/slot_joint_v6_capacity_s0 `
  --slots 8 --updates 120 --route-updates 120 --deploy-updates 0 `
  --trajectory-seed 101 --seed 0 --mapping-samples 8 `
  --update-epochs 3 --minibatch-size 128 `
  --shared-model-head --exploration none --device cpu
```

完整三阶段试跑可设--updates 12 --route-updates 4 --deploy-updates 4，剩余4次为联合更新。不同正式场景独立随机初始化，不从最大规模续训。中断后--resume仅用于同目录、场景、轨迹、种子、协议和训练配置，包括原定总更新数和阶段划分；不支持直接修改总更新数后无损延长。

## 4. 快照评价与基线

训练结束自动生成evaluation.json与baselines.json；模型选择单阶段另生成同部署的routing_baselines_same_placement.json。额外快照诊断仍重放训练轨迹：

```powershell
python scripts/evaluate_slot_joint_snapshot.py `
  --run results/<run-directory> --mapping-samples 128 --baselines

python scripts/evaluate_slot_joint_routing_baselines.py `
  --run results/<run-directory> --mapping-samples 128
```

输出已存在时报错，不静默覆盖。比较时核对场景与训练轨迹摘要、映射精度、部署条件及更新数。提高映射样本数只改变数值评价精度，不产生新业务轨迹。旧v4专用绘图脚本仅处理历史文件。

## 5. 文件与边界

| 文件 | 用途 |
|---|---|
| src/agent_orch/workload/slot_trajectory.py | 固定工作负载、可变需求/资源的轨迹 |
| src/agent_orch/envs/slot_joint_env.py | 顺序部署、模型选择、同槽反馈 |
| src/agent_orch/routing/capacity.py | 唯一物理路由实现 |
| src/agent_orch/agents/ | 两个actor/critic分支、PPO及乘子 |
| scripts/profile_capacity_routing.py | 容量路由计时及容量扫描统计 |
| docs/experiment_parameter_register_2026-10-13.md | 参数、联合长度规则与待办 |

运行目录只有一条train_trajectory.json。manifest记录协议、场景/轨迹摘要、PPO种子、训练设置和evaluation_protocol=training_trajectory_replay。checkpoint.pt供恢复，policy.pt为最终权重，history及评价/基线JSON记录实际过程；仅有checkpoint不表示训练完成。结果、原始数据、权重被Git忽略。

正式实验仍需随机应用池生成器、工具文本48–5,000 token与历史/预算联合生成、参数冻结，以及网络时间单位修正。网络修正本轮只给方案，未改性能代码；slot_seconds=1为旧占位值，不是1秒完成部署的声明。vLLM动态chunk与现有固定chunk解析也仍待校验。

旧部署库、内层求解器、蒸馏和矩阵脚本不是主流程。删除的源码及专用诊断可从Git历史或修改前恢复点找回；未提交旧探针未改动，但其Softmin依赖已不受支持。
