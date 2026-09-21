# 代码结构（2026-09-20 重构后）

本次重构把实验流水线拆成职责单一的模块，并新增三个此前缺失的能力：**可审计的目标函数**、**离线分层可行部署库**、**TensorBoard + SwanLab 双写记录**。

`gym_env.py` 与 `structured_ppo.py` 退化为转发层，历史导入路径全部保留。

## 模块地图

```
src/agent_orch/
├── objective/                 ★ 新增：目标与约束的唯一实现处
│   ├── spec.py                ObjectiveSpec：权重、约束阈值、归一化方式；两个档案
│   ├── references.py          ReferenceScales：成本界（取自部署库）与各应用时延尺度
│   ├── evaluator.py           ObjectiveEvaluator：SlotMetrics → (utility, components, c_k, diagnostics)
│   └── audit.py               ObjectiveAudit：逐项"决策间跨度"与死项检测
├── deployment/                ★ 新增：可行部署库
│   ├── library.py             DeploymentEntry / DeploymentLibrary（JSON + manifest + 覆盖审计）
│   ├── builder.py             DeploymentLibraryBuilder：L1–L5 分层构造
│   └── sampler.py             StratifiedSampler（分层循环）/ FixedDeploymentSampler
├── telemetry/                 ★ 新增：实验记录
│   ├── writers.py             MetricWriter 协议；TensorBoard / SwanLab / Multi / Null；make_writer
│   └── run_logger.py          RunLogger：按前缀分组的 log_update / log_validation / log_action_distribution
├── envs/
│   ├── layout.py              动作空间索引 ↔ 身份映射
│   ├── base.py                BaseOrchestrationEnv：掩码、路由解码、势函数、特征向量、目标接线
│   ├── orchestration_env.py   AgentOrchestrationEnv（部署 + 组成）
│   ├── composition_env.py     CompositionLibraryEnv（组成，部署库驱动）← 原 RoutingOnlyEnv
│   ├── deployment_env.py      DeploymentOnlyEnv（仅部署）
│   └── gym_env.py             转发层（历史导入路径）
├── agents/
│   ├── config.py              PPOConfig + for_constraint_count(n) + to_dict
│   ├── distributions.py       Dirichlet / 类别采样与评估、优势与掩码工具
│   ├── networks.py            StructuredActorCritic
│   ├── rollout.py             RolloutBatch + collect_rollout
│   ├── ppo.py                 GAE、PPO 更新循环、拉格朗日更新、record 构造
│   ├── training.py            train_ppo（签名不变，新增 run_logger）
│   └── structured_ppo.py      转发层
└── （performance/ capacity/ routing/ baselines/ simulator/ schema/ workload/ validation/ 未改动）
```

## 目标函数：两个档案

| | `legacy` | `slo_constrained`（**新默认**） |
|---|---|---|
| 目标项 | goodput 0.25 + quality 0.25 − cost 0.25 − latency 0.25 | quality 0.5 − cost 0.25 − latency 0.25 |
| 约束 | `c_llm`、`c_service` | `c_llm`、`c_service`、`c_attainment = (0.9 − G^req)^+` |
| 用途 | 复现既有结果 | 新实验 |

**为什么改**：SLO 达成率本身就是各应用时延与截止期满足度的聚合。把它同时当作 0.25 权重的目标项，会让它与它本该约束的质量互相抵消——实测在主场景上，"永远用最小模型"成为占优策略，退化基线 `static` 在四个负载档全部第一。改成约束后排序反转为 `greedy` 第一、`static` 落到均匀分配以下。

**成本归一化界**：改为取自可行部署库（`library+switch-on`），而不是"全部候选的理论上限"。主场景上界从 3.96 降到 1.22，成本项的影响度从 3.4% 升到 30.4%。这是独立于权重选择的修复。

## 可行部署库

`scripts/build_deployment_library.py` 离线构造、落盘、可审计：

| 层 | 内容 | 主场景数量 |
|---|---|---|
| `L5_extreme` | 全押最大/最小模型、成本最小、容量最大 | 4 |
| `L1_model_subset:<集合>` | 4 个模型的全部 14 个非空真子集，各一套最便宜摆放 | 14 |
| `L2_capacity:<档>` | 全模型集在 GPU 预算 25/50/75/100% | 1 |
| `L3_replicas:<n>` | 每服务 1/2/4 副本 | 4 |
| `L4_placement:<档>` | 便宜/均衡/昂贵摆放 | 105 |
| **合计** | | **128 套，21 层，distinct 激活模型集 = 15** |

对比：重构前的运行时贪心目录是 8 套，**全部为"4 模型各 1 实例 + 6 服务各 1 副本"，只有摆放不同**，导致组成策略的动作掩码与质量函数在每个 episode 恒定（distinct 激活模型集 = 1）。

## 训练入口的新参数

```powershell
python scripts/run_rl_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --objective-profile slo_constrained `   # 或 legacy
  --attainment-target 0.9 `
  --deployment-library data/processed/deployment_library_agent-abilene-20.json `
  --library-sampler cycle `               # 组成阶段：cycle | uniform | fixed
  --potential-cost-weight 0.05 `          # 势函数含成本项（0 = 关闭）
  --telemetry both `                      # none | tensorboard | swanlab | both
  --swanlab-project agent-orch `
  ...
```

每个 run 目录新增：

| 产物 | 内容 |
|---|---|
| `tensorboard/events.out.tfevents.*` | 按组归类：`objective/` `loss/` `constraint/` `exploration/` `timing/` `train/` `validation/` `action/model_share/<app>/<model>` `episode/` |
| `swanlab/run-*/` | 同样的指标（offline 模式；`--swanlab-online` 上传） |
| `run_config.json` | 完整 run spec + 目标档案 + 约束名 + 部署库路径与分层计数 |

约束相关的标签名随目标档案动态生成：`slo_constrained` 下会多出 `lagrange_attainment`、`mean_constraint_attainment`、`next_lagrange_attainment`。

## 诊断脚本

```powershell
# 逐项奖励灵敏度：哪些项在决策间真的有区分度
python scripts/audit_objective_sensitivity.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --load-levels data/processed/load_levels.json `
  --periods 40 --library-deployments 24 `
  --output results/objective_audit

# 构造/重建部署库（含覆盖审计 CSV 与 manifest）
python scripts/build_deployment_library.py `
  --scenario configs/benchmarks/main_abilene.yaml --min-entries 128
```

`audit_objective_sensitivity.py` 输出 `objective_audit.md/.json` 与逐档案的 `terms_<profile>.csv`，含每项的 `between_label_spread`、`weighted_spread`、`influence` 与 `dead` 标记。**建议在任何正式训练前先看这张表**：`influence` 低于 5% 的项等于没有梯度。

## Stage A：组成策略的内层优化与门禁

Stage A 的目标是让组成策略真正学会"部署 → 最优组成"这个响应函数。在外层部署策略开始训练之前，必须先证明这一点，否则外层是在给一个答错问题的下层做补偿。

新增三件东西：

| 组件 | 文件 | 作用 |
|---|---|---|
| 内层求解器 | `src/agent_orch/routing/composition.py` | 给定部署，用 canonical 候选 + 逐组贪心坐标上升近似求最优组成 |
| 参考解生成 | `scripts/solve_composition_reference.py` | 对部署库每套部署求解，落盘为教师数据与门禁参考 |
| 门禁 | `scripts/evaluate_composition_response.py` | 在留出 split 上逐层对比启发式，算 Spearman ρ 与提升捕获率 |
| 蒸馏预热 | `scripts/pretrain_composition.py` | 把求解器的组成克隆进策略的 Dirichlet 均值，给 PPO 一个非平凡的起点 |

求解器的 canonical 候选集：`uniform`、`quality_greedy`、`cost_greedy`、`latency_greedy`、`quality_softmax`、`quality_uniform_mix`。求解过程先评估全部 canonical 候选取最优作起点，再做逐组贪心坐标上升，全程遵守评估预算。**每次评估前 reset 仿真器且路由不看上一周期利用率**，保证候选之间是冷启动可比的。

### 完整流程

```powershell
# 1) 部署库（若还没有）
python scripts/build_deployment_library.py `
  --scenario configs/benchmarks/main_abilene.yaml --min-entries 128

# 2) 内层参考解（128 套；支持 --resume 断点续跑，默认开启）
python scripts/solve_composition_reference.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --split all --budget 48
#   -> data/processed/composition_reference_agent-abilene-20.json + .manifest.json

# 3) 蒸馏预热（Teacher = 上一步的参考解）
python scripts/pretrain_composition.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --reference data/processed/composition_reference_agent-abilene-20.json `
  --epochs 300 --validation-fraction 0.2 `
  --output results/stage_a/pretrain
#   -> results/stage_a/pretrain/policy_pretrained.pt （可直接用于 --initial-policy）

# 4) Stage A：组成策略 PPO（从蒸馏权重出发，只用训练 split 的上下文）
python scripts/run_rl_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --seeds 0 --modes route --variants no-rnd --training-phase composition `
  --initial-policy results/stage_a/pretrain/policy_pretrained.pt `
  --objective-profile slo_constrained `
  --library-split train `
  --updates 20 --rollout-periods 16 --rollout-contexts 2 `
  --train-slots 3600 --eval-slots 100 --arrival-scale 7.165234375 `
  --device cpu --resume --telemetry both `
  --output results/stage_a/composition

# 5) 门禁 G_A（留出 split；退出码非 0 表示未通过）
python scripts/evaluate_composition_response.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --policy results/stage_a/composition/<run>/policy_best.pt `
  --reference data/processed/composition_reference_agent-abilene-20.json `
  --split test --periods 8 --warmup 3 `
  --output results/stage_a/gate
```

### 门禁判据（三条全过才 PASS）

| 判据 | 默认阈值 | 为什么 |
|---|---|---|
| Spearman ρ(策略, 求解器参考) | ≥ 0.8 | 测"响应函数"是否学到：策略必须能把部署按可达性能正确排序 |
| 捕获提升率 = (策略均值 − uniform 均值) / (参考均值 − uniform 均值) | ≥ 80% | 测是否真的把内层问题解开了，而不是退化成均匀组成 |
| 逐层策略均值不低于该层最好启发式 | 容差 = 10% × 可达提升 | 测是否被任何闭式启发式打败 |

三者都在**同一个稳态多周期协议**下测量（策略、启发式、求解器参考都跑同样的周期数与到达强度），因此可直接比较。

### 一个已知的参数化上限

Dirichlet 头的浓度被 `_concentrations` 夹在 `[minimum, max(100, 2*minimum)]`，所以一组 k 个激活模型的**均值份额上限**是 `100 / (100 + (k-1)*minimum)`：

| 激活模型数 | 上限（minimum=1.0） |
|---|---|
| 1 | 1.0000 |
| 2 | 0.9901 |
| 3 | 0.9804 |
| 4 | 0.9709 |

求解器的目标经常是近似 one-hot 的（把全部份额给最强模型），因此会**超出这个上限**。门禁报告里的 `dirichlet_ceiling` 段会统计"有多少组的目标超出上限"以及最差超出量——这部分差距是**构造上不可达的**，不是学习失败。toy 场景实测最差超出 +0.0099，与"策略比最优低 0.4%"完全吻合。

如果后续要缩小这个差距，正确的做法是放宽 `_concentrations` 的上夹（会同时加宽探索），而不是把门禁阈值放宽。

