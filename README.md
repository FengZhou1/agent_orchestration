# Agent 联合编排仿真器

`latex/agent_system_model_service_mesh_style.tex` 中系统模型的独立实验实现：在边缘网络上联合编排 **LLM 服务实例的部署**、**无状态服务副本的部署**、**应用→模型的组成决策**，以及**由解析式 Softmin 给出的物理路由**，目标是在 SLO 约束下最大化质量与 goodput、压低成本与时延。

`old_exp/` 是实验室上一代微服务实验代码，**只读参考，本项目不 import 其中任何模块**。

---

## 1. 当前状态（协作者请先读这一节）

**能跑通、有实测结果的**

| 能力 | 状态 | 入口 |
|---|---|---|
| 解析式仿真（LLM 稳态 / 无状态服务排队 / 多跳网络 / 关键路径） | 可用，170 个测试 | `src/agent_orch/simulator`、`performance/` |
| 5 个基线策略（static / equal / least_load / random / greedy） | 可用 | `scripts/run_baseline_matrix.py` |
| **目标函数与约束的可插拔实现 + 灵敏度审计** | 可用 | `scripts/audit_objective_sensitivity.py` |
| **离线分层可行部署库**（128 套 / 21 层 / 15 个不同激活模型集） | 可用 | `scripts/build_deployment_library.py` |
| **内层组成求解器**（canonical 候选 + 逐组贪心坐标上升） | 可用 | `scripts/solve_composition_reference.py` |
| **Stage A 门禁 G_A**（留出划分 + 秩相关 + 提升捕获） | 可用，有 PASS/FAIL 退出码 | `scripts/evaluate_composition_response.py` |
| 组成策略的蒸馏预热 | 可用 | `scripts/pretrain_composition.py` |
| PPO 训练（组成 / 部署 / 联合） | 可用 | `scripts/run_rl_matrix.py` |
| TensorBoard + SwanLab 双写记录 | 可用 | `--telemetry both` |

**关键实测结论（决定后续工作方向）**

- 组成决策本身有真实价值：128 套部署上，均匀组成 0.1317 → 内层最优 0.1763，**平均提升 +0.0446（+34%）**。
- **蒸馏（内层求解器做教师）能通过门禁**：留出测试集上 ρ=0.8630、提升捕获 136.1%、28/30 套严格打赢全部闭式启发式。
- **PPO 从随机初始化单独训练不能通过门禁**：最好一次（200 轮，`results/stage_a/ppo_tuned`）为 ρ=0.7384、捕获 49.0%、10/30 打赢全部启发式。方向正确、幅度不足。
- 因此门禁 G_A 的通过目前**依赖"内层求解器 + 蒸馏"这条路**。PPO 能否作为独立贡献来写，是一个尚未解决的开放问题（见 §7 的进行中实验）。

**尚未实现**

- Stage B（冻结组成策略、训练部署策略）与 Stage C（真正的双时间尺度联合微调）
- 论文的 `T^dep` 部署冻结约束与 LLM 实例 `start_cost`
- 设计化的 train/test 变化轴（负载档 × 组成偏斜 × profile 留出点 × scale/stress）
- 仿真器映射结果缓存（当前 `performance/workflow.py` 的 `rng.choice` 是热点）
- 质量分与 SLO 仍是占位/参考值（`quality_status: reference values`），正式实验前必须替换

---

## 2. 快速开始

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e ".[rl,dev,plot,telemetry]"
```

GPU 环境用 `environment-gpu.yml`。**所有脚本都必须在 `conda activate agent-orch` 之后运行**——直接调该环境的 `python.exe` 时 `Library\bin` 不在 `PATH` 上，matplotlib 的 `savefig` 会以 `Windows fatal exception: code 0xc06d007f` 崩溃。

冒烟与回归：

```powershell
pytest                                    # 170 个测试，约 30 s

agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 10 --seed 7
agent-orch-sim train --scenario configs/toy.yaml --mode joint --exploration rnd `
  --updates 10 --rollout-periods 4 --device auto

python scripts/run_rl_matrix.py --scenario configs/toy.yaml --updates 4 `
  --rollout-periods 8 --device cpu --telemetry both --output results/smoke
```

`configs/toy.yaml`（4 节点 / 2 模型 / 2 应用）**只用于冒烟，论文结果一律用 `configs/benchmarks/`**。

---

## 3. 仓库结构

```
src/agent_orch/
├── objective/        ★ 目标与约束的唯一实现处
│   ├── spec.py         ObjectiveSpec：两个档案（legacy / slo_constrained）、权重、约束阈值
│   ├── references.py   ReferenceScales：成本界（取自部署库）与各应用时延尺度
│   ├── evaluator.py    ObjectiveEvaluator：SlotMetrics → (utility, components, c_k, diagnostics)
│   └── audit.py        ObjectiveAudit：逐项"决策间跨度"与死项检测
├── deployment/       ★ 离线可行部署库
│   ├── library.py      DeploymentLibrary / DeploymentEntry，JSON + manifest + 覆盖审计 + train_test_split
│   ├── builder.py      分层构造 L1–L5
│   └── sampler.py      StratifiedSampler（分层循环）/ FixedDeploymentSampler
├── telemetry/        ★ 实验记录
│   ├── writers.py      MetricWriter 协议；TensorBoard / SwanLab / Multi / Null；make_writer
│   └── run_logger.py   RunLogger：按前缀分组写标量、动作分布、验证事件
├── envs/
│   ├── layout.py               动作空间索引 ↔ 身份映射
│   ├── base.py                 BaseOrchestrationEnv：掩码、路由解码、势函数、特征向量、目标接线
│   ├── orchestration_env.py    AgentOrchestrationEnv（部署 + 组成）
│   ├── composition_env.py      CompositionLibraryEnv（仅组成，部署库驱动）
│   ├── deployment_env.py       DeploymentOnlyEnv（仅部署）
│   └── gym_env.py              转发层，保留历史导入路径
├── agents/
│   ├── config.py         PPOConfig（含 for_constraint_count）
│   ├── distributions.py  Dirichlet / 类别采样与评估、优势归一化
│   ├── networks.py       StructuredActorCritic
│   ├── rollout.py        RolloutBatch + collect_rollout
│   ├── ppo.py            GAE、PPO 更新循环、拉格朗日更新、record 构造
│   ├── training.py       train_ppo（唯一入口，签名稳定）
│   └── structured_ppo.py 转发层
├── routing/
│   ├── physical.py       PhysicalRouter（Softmin 解析路由）
│   └── composition.py  ★ 内层组成求解器
├── performance/      LLM Roofline 与稳态、排队、网络、工作流关键路径
├── capacity/         容量规划与可行性检查（planner / stability）
├── baselines/        5 个基线策略
├── simulator/        时隙主循环
├── schema/           场景与决策数据类、YAML 加载
└── workload/         到达强度

scripts/   39 个脚本（见 §5、§6）
docs/      architecture.md ★、experiment_protocol.md ★、formula_contract.md ★、
           preconstructed_agent_pattern_graphs.md、llm_*_validation*.md、dataset_update_*.md
configs/   toy.yaml（冒烟）、benchmarks/（main_abilene / scale_geant / stress_×3）
data/      catalogs（进版本控制）、raw / processed（gitignored）
results/   gitignored
tests/     25 个文件 / 170 个测试
```

详细模块说明见 [docs/architecture.md](docs/architecture.md)。

---

## 4. 核心概念

### 4.1 目标函数：两个档案

| | `legacy` | `slo_constrained`（**当前默认**） |
|---|---|---|
| 目标项 | goodput 0.25 + quality 0.25 − cost 0.25 − latency 0.25 | quality 0.5 − cost 0.25 − latency 0.25 |
| 约束 | `c_llm`、`c_service` | 同左，另可加 `c_attainment = (0.9 − G^req)^+` |

**为什么换档案**：SLO 达成率本身就是各应用时延与截止期满足度的聚合，把它同时当作 0.25 权重的目标项会让它与它本该约束的质量互相抵消。实测在主场景上，"永远用最小模型"因此成为占优策略，退化基线 `static` 在四个负载档全部第一；改成约束后 `greedy`（质量优先）反超、`static` 落到均匀组成以下。

**成本归一化界取自可行部署库**，不是"全部候选的理论上限"。后者会让成本项的影响度掉到 3.4%（等于没有梯度），取自部署库后是 30.4%。这条修复独立于权重选择。

`--attainment-target` / `--no-attainment-constraint` 控制第三个约束。**部署外生的阶段（Stage A）必须用 `--no-attainment-constraint`**：attainment 由部署决定，单模型弱部署对任何组成都达不到目标（实测 17 个部分模型上下文里 7 个的最优 attainment < 0.9，两个恒为 0.000），拉格朗日乘子会追一个消不掉的违约而无界增长，奖励退化成惩罚。

### 4.2 可行部署库

组成策略的输入里包含"哪些模型可用"，所以训练它的上下文**必须张成模型子集这个维度**。库按五层离线构造、落盘、可审计：

| 层 | 内容 | 主场景数量 |
|---|---|---|
| `L5_extreme` | 全押最大/最小模型、成本最小、容量最大 | 4 |
| `L1_model_subset:<集合>` | 4 个模型的全部 14 个非空真子集，各一套最便宜摆放 | 14 |
| `L2_capacity:<档>` | 全模型集在 GPU 预算 25/50/75/100% | 1 |
| `L3_replicas:<n>` | 每服务 1/2/4 副本 | 4 |
| `L4_placement:<档>` | 便宜/均衡/昂贵摆放 | 105 |
| **合计** | | **128 套 / 21 层 / 15 个不同激活模型集** |

对比：重构前的运行时贪心目录是 8 套，**全部为"4 模型各 1 实例"，distinct 激活模型集 = 1**，组成策略的掩码与质量函数在每个 episode 恒定，没有任何可条件化的东西。

### 4.3 门禁 G_A

在外层（部署策略）开始训练之前，必须先证明内层（组成策略）真的解开了自己的问题。三条判据全部通过才算 PASS：

| 判据 | 默认阈值 | 测什么 |
|---|---|---|
| Spearman ρ（策略 vs 求解器参考） | ≥ 0.8 | 响应函数是否学到：策略必须能把部署按可达性能正确排序 |
| 提升捕获率 = (策略 − uniform) / (参考 − uniform) | ≥ 80% | 是否真的解开了内层问题，而不是退化成均匀组成 |
| 逐层策略均值不低于该层最好启发式 | 容差 = 10% × 可达提升 | 是否被任何闭式启发式打败 |

三者都在**同一个协议**下测量。`scripts/evaluate_composition_response.py` 退出码非 0 表示未通过。

### 4.4 协议必须对齐（踩过的坑，见 §8）

内层求解器与门禁各自支持 `--protocol-periods` / `--protocol-warmup`。**参考解是在哪个协议下求的，比较就必须在哪个协议下做**：冷启动单周期与稳态多周期的**最优组成不同**，混用会得到互相矛盾的结论（我先后撞到"策略低于参考 16%"和"高于参考 31%"两个方向）。

---

## 5. 奖励灵敏度审计（**任何正式训练前先跑这个**）

它测量每个奖励项在**决策之间**的跨度，`influence` = 加权跨度 / 权重。`influence` 低于 5% 的项等于没有梯度。

```powershell
python scripts/audit_objective_sensitivity.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --load-levels data/processed/load_levels.json `
  --periods 40 --library-deployments 24 `
  --output results/objective_audit
```

输出 `objective_audit.md/.json` 与逐档案的 `terms_<profile>.csv`，含每项的 `between_label_spread`、`weighted_spread`、`influence`、`dead` 标记，以及各约束的**绑定比例**与超额量。主场景实测：四项影响度分别为 goodput 98.4%、latency 59.4%、quality 36.8%、cost 30.4%，`attainment` 约束 86.4% 的时间在起作用。

---

## 6. 实验流水线

### 6.0 生成基准场景（若尚无）

```powershell
$env:PYTHONPATH='src'
python scripts/build_benchmark_scenarios.py
python scripts/export_reference_catalogs.py
```

Main = Abilene 12 节点 / 15 链路 / 4 模型（Qwen3-4B/8B/14B/32B BF16）/ 20 应用 / 6 无状态服务 / 33 候选 / 65 pattern flow；Scale = GEANT 22 节点 / 36 链路 / 50 应用；Stress ×3 = 链路容量减半 / 服务能力减半 / 每 4 个候选丢弃 1 个。

### 6.1 可行部署库

```powershell
python scripts/build_deployment_library.py `
  --scenario configs/benchmarks/main_abilene.yaml --min-entries 128
# -> data/processed/deployment_library_<scenario_id>.json
#    + .manifest.json（含 scenario hash / 分层计数 / json 的 sha256）
#    + .coverage.csv（每套部署的激活模型集、GPU、成本）
```

### 6.2 内层参考解（教师 + 门禁锚点）

```powershell
python scripts/solve_composition_reference.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --split all --budget 48 --protocol-periods 6 --protocol-warmup 2
# -> data/processed/composition_reference_<scenario_id>[_steady].json
```

默认开启断点续跑（`--no-resume` 关闭）。主场景 128 套约 22 分钟（10.4 s/套）；稳态协议 30 套约 23 分钟。求解器先评估 6 个 canonical 候选（uniform / quality_greedy / cost_greedy / latency_greedy / quality_softmax / quality_uniform_mix），取最优作起点，再做逐组贪心坐标上升，全程遵守评估预算。

### 6.3 Stage A：组成策略

```powershell
# (a) 蒸馏预热：把求解器的组成克隆进策略的 Dirichlet 均值
python scripts/pretrain_composition.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --reference data/processed/composition_reference_agent-abilene-20_steady.json `
  --epochs 400 --validation-fraction 0.2 `
  --output results/stage_a/pretrain
# -> results/stage_a/pretrain/policy_pretrained.pt（可直接给 --initial-policy）

# (b) 组成策略 PPO（从零训练就去掉 --initial-policy）
python scripts/run_rl_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --seeds 0 --modes route --variants no-rnd --training-phase composition `
  --objective-profile slo_constrained --no-attainment-constraint `
  --library-split train `
  --initial-policy results/stage_a/pretrain/policy_pretrained.pt `
  --updates 200 --rollout-periods 8 --rollout-contexts 8 `
  --update-epochs 10 --minibatch-size 32 `
  --composition-learning-rate 3e-3 --composition-group-relative `
  --baseline-periods 3 --baseline-warmup 1 `
  --train-mapping-samples 64 --validation-mapping-samples 64 --eval-mapping-samples 128 `
  --validation-interval 20 --validation-periods 12 --validation-warmup-periods 2 `
  --train-slots 3600 --eval-slots 40 --arrival-scale 7.165234375 `
  --device cpu --no-progress --telemetry both --resume `
  --output results/stage_a/composition

# (c) 门禁 G_A（退出码非 0 = 未通过）
python scripts/evaluate_composition_response.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --policy results/stage_a/composition/<run>/policy_final.pt `
  --reference data/processed/composition_reference_agent-abilene-20_steady.json `
  --no-attainment-constraint --split test --periods 6 --warmup 2 `
  --output results/stage_a/gate
```

关键参数的理由：

- `--library-split train`：训练只用训练侧部署，门禁在 `test` 侧评分，两侧不重叠。用 `train_test_split(group_by="n_models")` 可让留出侧覆盖模型数这个维度（默认按 stratum 分组时，单例 stratum 会整层进训练侧）。
- `--composition-group-relative`：按 episode 组相对归一化优势，**只减组均值、不除组标准差**。一条 rollout 跨多个外生上下文，全局归一化会把"我在哪个上下文"当成主要信号。
- `--update-epochs 10`：每次 update 只有几十个样本、`clip_ratio=0.2`，约 10 个 epoch 后 ratio 就长期落在裁剪区外，多出来的 epoch 是白跑。
- `--composition-learning-rate 3e-3`：实测优于 1e-4/1e-3。
- `--baseline-periods 3 --baseline-warmup 1`：控制变量与奖励必须同协议；系统一步就到不动点，3 期与 10 期结果逐位相同，省 3 倍开销。
- `--no-attainment-constraint`：见 §4.1。

查看策略学到的组成（在多个上下文上聚合，而不是只看一个 rollout 的最后一步）：

```powershell
python scripts/describe_composition_policy.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --policy results/stage_a/composition/<run>/policy_final.pt `
  --split test --output results/stage_a/composition_description
```

### 6.4 基线矩阵

```powershell
python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml --slots 600 `
  --load-levels data/processed/load_levels.json `
  --seeds 0,1,2,3,4 --resume --output results/baseline_levels
```

**注意**：现有 `results/figures_baseline_levels/` 的扫描覆盖 `arrival_scale` 0.52–1.37，而 Stage A 训练用 7.165——**两者没有共同工作点，不能直接比较**。要对齐需按同一 `--arrival-scale` 重跑。

### 6.5 Stage B / C（尚未实现）

Stage B 需要三件事：reset 从部署库分层抽初始部署（当前恒为 `initial_deployment()`）、实现 `T^dep` 冻结与 LLM `start_cost`、打开含成本项的 potential shaping（`--potential-cost-weight`）。参考门槛是在同一 CRN 轨迹、同一工作点上打赢 static/equal/greedy。

---

## 7. 进行中的实验（接手前请先看）

| 实验 | 目的 | 目录 |
|---|---|---|
| `ppo_tuned` | PPO 单独最佳基线（已完成 200/200） | `results/stage_a/ppo_tuned` |
| `ppo_long` | 600 轮，钉死"训练长度是否瓶颈" | `results/stage_a/ppo_long` |
| `ppo_biggroup` | 组内样本 8→32，测优势估计的天花板 | `results/stage_a/ppo_biggroup` |
| `ppo_nocon` | 去掉 llm/service 约束的对照（已完成，更差） | `results/stage_a/ppo_nocon` |
| `ppo_random_init` | 随机初始化检查点（起点对照，非训练运行） | `results/stage_a/ppo_random_init` |

**已排除的假设**（不要重试）：去掉约束更差（捕获 14.4% vs 49.0%，惩罚项起正则作用）；提高 Dirichlet 浓度下限降噪更差（**动作噪声本身在充当探索**，压小它上下文内的效用变化也变小）；加大 rollout/epoch/学习率到某点后无进一步收益。

---

## 8. 踩过的坑（必读）

这些都是真实发生过、代价很高的错误，改代码前请对照：

1. **协议必须对齐。** 内层求解器/门禁各自支持 `--protocol-periods`；参考解在冷启动协议下求、比较在稳态协议下做，会得出矛盾结论。另外求解器的目标是**效用**，约束从不进入效用——所以改变约束不影响效用数值，门禁数值仍可比。

2. **控制变量必须与它减去的对象同协议、同归一化。** 冷启动单周期基线比稳态效用低 0.13–0.23，且**部分模型部署上份额未按激活模型归一化**会让基线偏低最多 0.15——而整个效用范围只有 0.12–0.26。验证基线时**务必用部分模型部署**，4 模型全激活的部署上这个 bug 不可见。

3. **一条 rollout 跨多个外生上下文时，优势要在组内归一化。** 实测跨上下文 std 0.020 而上下文内 0.030，全局归一化让 2/3 的方差在编码"我在哪个上下文"。且**只减组均值、不除组标准差**：各组组内标准差相差 20 倍，除标准差会把"组成几乎不起作用"的上下文放大数百倍。

4. **约束必须在该阶段可满足。** 部署外生的阶段不能用 attainment 约束（见 §4.1），否则乘子无界增长、奖励退化成惩罚。

5. **`policy_best.pt` 不经门禁不能信。** 验证协议只用 3 个上下文 × 12 周期，实测它在训练效用单调上升时中途掉到 0.146 又回升；600 轮运行的最佳验证检查点在测试集上**比 200 轮的终值更差**（38.2% vs 49.0%）。

6. **判断一次运行是否跑完，看 `run_complete.json` 是否存在。** `training_status.json` 是持续覆盖的进度文件；进程被强杀时它不会更新，会停留在 `status=running`。训练中途被中止的运行，其 `policy_best.pt` / `policy_final.pt` 仍然是完整可用的检查点，但**报告时必须标注"该运行被中止于第 N 轮"**。

7. **`action/model_share/*` 只记录每个 rollout 最后一步的组成**，那是某一个上下文下的组成，不是策略的一般行为。要看聚合视图请用 `scripts/describe_composition_policy.py`。

8. **Dirichlet 均值有构造上限。** `_concentrations` 把浓度夹在 `[floor, max(100, 50×floor)]`，k 个激活模型的均值份额上限约为 `cap/(cap+(k−1)floor)`。求解器目标常近似 one-hot 而超出该上限，这部分差距是参数化限制而非学习失败；门禁报告里的 `dirichlet_ceiling` 段会统计它。

9. **画图必须在 `conda activate agent-orch` 之后运行**（见 §2）。

10. **`configs/generated/` 现在存在**（只有一份说明；生成脚本自己会建父目录）。`scripts/aggregate_llm_steady_state_validation.py` 的输入是远端跑出来的 LLMServingSim 校验结果，本地没有；该脚本现在**列清缺哪些目录并以非零码退出**，不再写出空报告。

11. **准入等待项没有不一致**（此前的说法已核实为过时）：正文式 `\eqref{eq:compact-llm-ttft}`、`docs/experiment_protocol.md` 与实现三者一致，都是 `TTFT = 首次准入等待 + prefill`（代码里是 `ttft_s = macro_wait + prefill`，见 `performance/analytical.py`）；`docs/formula_contract.md` 完全不提这一项，因此不构成冲突。

12. **`data/raw/burstgpt_v2/`（193 MB）决定为不使用**：它没有被任何代码、配置或文档引用，也不在 `data/raw/` 之外的任何清单里；保留在磁盘上但不纳入实验，理由是到达过程已定为参数化混合负载族（见 `docs/stage_a_ppo_plan_2026-09-23.md`），BurstGPT 的真实轨迹与当前的时间轴语义（任意负载族、可复现种子）不兼容。若将来要做 trace-driven 对照，再单独引入。

---

## 9. 结果产物与日志

每次 `run_rl_matrix.py` 运行在输出目录下建一个 `run_id` 子目录：

| 文件 | 内容 |
|---|---|
| `training_history.jsonl` | 每个 PPO update 一行，含 `mean_utility`、`mean_learning_utility`、`mean_loss`、约束项与乘子、耗时 |
| `validation_history.jsonl` | 每次验证一行（`--validation-interval` 控制） |
| `checkpoint.pt` | update 级恢复状态，`--resume` 用它继续 |
| `policy_best.pt` / `policy_final.pt` / `policy.pt` | 检查点（**注意 §8.5、§8.6**） |
| `run_complete.json` | **存在即表示该运行正常结束** |
| `run_config.json` | 完整 run spec + 目标档案 + 约束名 + 部署库路径与分层计数 |
| `tensorboard/events.out.tfevents.*` | 按组归类：`objective/` `loss/` `constraint/` `exploration/` `timing/` `train/` `validation/` `action/model_share/<app>/<model>` `episode/` |
| `swanlab/run-*/` | 同样的指标（offline 模式；`--swanlab-online` 上传） |

约束相关的标签名随目标档案动态生成（`slo_constrained` 下会多出 `lagrange_attainment` 等）。

```powershell
tensorboard --logdir results/stage_a     # 然后打开 http://127.0.0.1:6006
```

汇总与绘图：

```powershell
python scripts/summarize_results.py `
  --input results/baseline_matrix.summary.parquet `
  --group-columns policy `
  --metrics mean_cost,mean_latency_s,mean_goodput_rps,mean_quality,mean_slo_attainment,mean_violations,violation_slot_fraction `
  --baseline static

python scripts/plot_results.py `
  --results results --output results/figures --analysis-output results/analysis

# 单次运行的收敛曲线（第一个参数是 training_history.jsonl 的路径）
python scripts/plot_last_training_curve.py `
  results/stage_a/ppo_tuned/<run>/training_history.jsonl `
  --output results/figures
```

`results/` 被 gitignore，**结果不进版本控制**；正式训练一律用新目录。

---

## 10. 场景、数据与标定

### 时间模型（2026-09-22 起，语义已锁）

**决策单元就是时隙**：时隙是唯一时钟，也是唯一决策单元。不存在单独的编排周期符号——`orchestration_period_s` 已从代码中移除，场景文件里遗留的该键由加载器丢弃（因此冻结的 SLO 校准不受影响）。由此推出三条：

1. **成本按时隙计价**，用 `slot_seconds`。部署库的字段是 `cost_per_slot`（库 schema v2，**旧库需要重建**）。
2. **目标为累积量** $J=\mathbb{E}[\sum_t u(t)]$：env 在每个 episode 内累计效用/成本/时延，更新记录里带 `episode_cumulative_*`。单周期绝对值在恒定 λ 下必然退化成静态优化。
3. **episode = 一条时间轴**：所有模式的 episode 都是 `--train-slots` 个时隙；`--rollout-periods` 只表示一次 update 收集多少决策步。组成与部署共用一条轨迹与系统折扣（`composition_gamma` 默认 `None`）。

> ⚠ **特征宽度已变**（状态新增"下一时隙 λ"、"剩余冻结"、"轨迹内进度"），**所有 2026-09-21 及更早的 checkpoint 都无法加载**。新实验必须用新目录。

### 到达过程 λ(t)

分析型仿真把场景的平均到达率作为排队模型的稳态强度，`--arrival-scale` 统一缩放，**不把一秒内的离散请求计数反推为稳态到达率**。Main 场景基准总到达率 0.004 request/s。

到达强度**逐时隙可变**，默认是周期性高斯突发（`--arrival-pattern bursty`，周期 60 时隙、σ=8、低位 0.4×、带 5% 抖动）。**这一步是必需的而非可选**：恒定强度下每时隙效用只取决于部署本身，最优就是"永远用同一套部署"，部署子问题退化成静态优化、RL 无从下手。训练/验证/评估分别取**同一过程的独立实现**（`realization=0/1/2`），所以验证测的是"换一条轨迹还能不能适应"，而不是在同一个静态点上重复抽样。

`configs/benchmarks/stress_arrival_burst.yaml` 补齐了协议里列过但从未生成的"到达突发"压力场景；它由 `scripts/add_arrival_burst_scenario.py` 从**已校准的** main 场景派生（直接重跑场景生成器会抹掉冻结的 SLO 块，因为生成器产出的永远是校准前的文件）。

先量化"时变到底带不带得来可学的东西"：

```powershell
python scripts/quantify_reactivity_headroom.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --composition-policy results/stage_a/pretrain/<run>/policy_pretrained.pt `
  --load-levels 0.4,0.7,1.0 --periods 14
```

它给出"最好固定部署"与"按 λ 反应的最优（完全预见、无切换成本，上界）"的差距；**差距≈0 就说明这个负载分布下部署策略没有可学的东西**。

四档负载见 `data/processed/load_levels.json`：

| 档位 | target_load | rate_scale | 总到达率 (req/s) |
|---|---|---|---|
| low | 0.40 | 3.371875 | 0.013488 |
| medium | 0.65 | 5.479297 | 0.021917 |
| high | 0.85 | 7.165234 | 0.028661 |
| overload | 1.05 | 8.851172 | 0.035405 |

参考稳定容量 `stable_capacity_rps = 0.03371875`（`arrival_scale = 8.4296875`）。

```powershell
python scripts/calibrate_slos.py --scenario configs/benchmarks/main_abilene.yaml `
  --low-load-fraction 0.20 --output configs/benchmarks/main_abilene.yaml `
  --propagate-to configs/benchmarks/stress_network_0p5.yaml `
                 configs/benchmarks/stress_service_0p5.yaml `
                 configs/benchmarks/stress_gpu_unavailable.yaml

python scripts/calibrate_load_levels.py --scenario configs/benchmarks/main_abilene.yaml `
  --output data/processed/load_levels.json
```

### 数据来源

| 用途 | 来源 | 状态 |
|---|---|---|
| GPU 服务器 | Alibaba GPU Trace v2026 | `data/raw/`（gitignored） |
| 拓扑 | SNDlib 1.0（Abilene / GEANT） | 已固化进 configs |
| 无状态服务图 | Alibaba Microservices v2021 + DeathStarBench | 参考 catalog 已生成 |
| Agent 工作流 | TraceLab v2（8,058 sessions / 665,453 steps） | `data/preconstructed_agent_workloads.yaml` |
| 质量分 | BFCL V3/V4、LongBench v2、SWE-bench Verified | ⚠ **占位参考值，必须替换** |
| LLM profile | LLMServingSim 2.0 + 少量 vLLM 实测 | 见 `docs/llm_*_validation*.md` |

### LLM 性能数据的准备与校验

```powershell
python scripts/prepare_llm_profiles.py --input <normalized.csv> --source-version <pinned> `
  --output data/processed/llm_profile.csv
python scripts/validate_llm_model.py --scenario configs/benchmarks/main_abilene.yaml `
  --profile data/processed/llm_profile.csv
```

论文中的宏观稳态模型可用 LLMServingSim 2.0 做分解验证（Roofline 工作量、稳态有效并发度、KV 驻留边界、请求时延趋势），步骤见 [docs/llmservingsim_queue_validation.md](docs/llmservingsim_queue_validation.md)。

### 量纲

时间以秒、到达率与处理率以请求/秒、单请求数据量以 MB、链路负载与容量以 Mbit/s、计算量以 FLOPs、显存访问量以字节。控制时隙长度不改变 Mbps 负载；链路传输时延按一个时隙内的聚合数据量与链路速率之比计算。

---

## 11. 协作约定

- **改代码前先跑 `pytest`（170 个，约 30 s）；提交前必须全绿。** 现有测试是行为的唯一护栏：`legacy` 目标档案的四项加权等价性、库成本界收紧、约束向量随档案变形、部署库分层与可行性、采样器均衡性、优势归一化都已被钉住。
- **保持向后兼容的导入路径。** `envs/gym_env.py` 与 `agents/structured_ppo.py` 是转发层，历史脚本与检查点靠它们解析；拆模块时请保留。
- **改目标函数或奖励，必须同步更新门禁与参考解。** 约束向量是观测特征的一部分，约束个数变化会改变特征宽度，旧检查点将无法加载。
- **新增模块请配测试**，并优先复用 `objective/`、`deployment/`、`telemetry/` 的既有抽象，不要在 env 里另写一套奖励或日志。
- **任何涉及模型/公式的判断，回到 `latex/agent_system_model_service_mesh_style.tex` 与 `docs/formula_contract.md` 核对**，不要凭记忆。
- 结果与检查点不进 git；正式训练用新目录，不复用旧归一化奖励产生的检查点。
- 提交信息沿用版本号风格（`5.1`、`5.2`…）。

### 常见任务的入口

| 我想…… | 从这里开始 |
|---|---|
| 加一个目标项或约束 | `objective/spec.py` + `objective/evaluator.py`，然后跑审计脚本 |
| 加一个基线策略 | `baselines/policies.py` 的 `make_policy`，再跑 `run_baseline_matrix.py` |
| 改部署库的分层 | `deployment/builder.py`，重建后看 `.coverage.csv` 与 `active_model_sets` 数量 |
| 加一个内层候选组成 | `routing/composition.py` 的 `canonical_candidates` |
| 改训练协议 | `agents/ppo.py` / `agents/rollout.py`，注意 §8.2、§8.3 |
| 换日志后端 | `telemetry/writers.py` 的 `make_writer` |
| 定位性能瓶颈 | `performance/workflow.py` 的映射采样循环（当前热点） |
