# Agent 联合编排仿真器

本项目是 `latex/agent_system_model_service_mesh_style.tex` 中系统模型的独立实验实现。`old_exp/` 下的旧版代码仅供参考，当前实现不会导入其中的模块。

当前版本提供以下功能：

- 具有类型约束的场景与决策数据结构；
- 基于分析模型和性能 profile 的 LLM、无状态服务及网络后端；
- 模型选择、LLM 实例分配和无状态服务路由；
- 链式调用与 fork--join 关键路径计算；
- Static、EqualSplit、LeastLoad、Random 和 Greedy 等基线策略；
- 由随机种子控制、可复现的命令行实验。

## 环境配置

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e .
```

如需使用 NVIDIA GPU 训练策略网络和 RND 网络，可创建 CUDA 环境：

```powershell
conda env create -f environment-gpu.yml
conda activate agent-orch-gpu
python -m pip install -e .
```

## 生成基准场景

`toy.yaml` 仅用于冒烟测试。论文实验采用自动生成的 Abilene Main、GEANT Scale 和各类 Stress 场景。

```powershell
conda activate agent-orch
$env:PYTHONPATH='src'
python scripts/build_benchmark_scenarios.py
python scripts/export_reference_catalogs.py
```

固定基础设施数据版本并完成无状态服务测量后，可使用处理后的服务器联合记录和服务 profile 重新生成场景：

```powershell
python scripts/build_benchmark_scenarios.py `
  --infrastructure-catalog data/processed/alibaba_gpu_server_catalog.yaml `
  --service-profile data/processed/stateless_service_profile.csv
```

生成的场景固定了 Qwen3-4B/8B/14B/32B 的 BF16 配置、vLLM 引擎参数、六类或八类无状态服务、SNDlib 网络拓扑、归一化 GPU 成本、参数量纲和数据来源。A10 运行 4B，L20 运行 4B/8B/14B 或双卡 32B，H20 运行 8B/14B/32B。场景内预置的质量与 SLO 均标记为参考值；正式论文实验前，必须使用固定版本的评测结果和校准结果进行替换。

## 运行 Toy 场景

```powershell
conda activate agent-orch
agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 10 --seed 7
```

执行结构化 PPO 冒烟训练：

```powershell
agent-orch-sim train `
  --scenario configs/toy.yaml `
  --mode joint `
  --exploration rnd `
  --updates 10 `
  --rollout-periods 4 `
  --device auto
```

使用 `--mode joint`、`--mode deploy` 或 `--mode route` 选择联合优化、仅部署优化或仅路由优化。联合模式默认运行带拉格朗日约束和 RND 探索的 DTS-PPO-RND；`--exploration none` 关闭内在奖励。ICM 和 Potential Shaping 通过后述矩阵脚本作为显式对照运行：

```powershell
agent-orch-sim train --scenario configs/toy.yaml --mode joint --exploration rnd
```

运行多随机种子基线矩阵：

```powershell
python scripts/run_baseline_matrix.py --scenario configs/toy.yaml --slots 600
```

运行 DTS-PPO-RND。默认仅执行随机种子 0 和联合算法，不展开消融矩阵；`--device auto` 在 CUDA 可用时选择 `cuda:0`，否则使用 CPU：

```powershell
python scripts/run_rl_matrix.py `
  --scenario configs/toy.yaml `
  --updates 100 `
  --rollout-periods 16 `
  --device auto `
  --resume
```

训练时，终端进度条分别显示矩阵、轨迹收集、PPO 优化和评估阶段。每个运行目录中的 `training_status.json` 持续覆盖当前进度、耗时和预计剩余时间，`training_history.jsonl` 在每次 PPO update 后立即追加训练指标，`checkpoint.pt` 保存 update 级恢复状态；矩阵根目录中的 `matrix_status.json` 和 `rl_experiment.log` 记录整体进度。`--resume` 从最近完成的 PPO update 继续并跳过已完成的种子—算法组合。使用 `--no-progress` 可关闭终端进度条而保留状态与日志文件。需要完整消融时，显式传入 `--seeds 0,1,2,3,4 --modes joint,deploy,route --variants auto`；Potential Shaping 与 ICM 可通过 `--modes joint --variants potential,icm` 运行。

分阶段训练使用同一策略结构和检查点格式。模型组成预训练采用 `RoutingOnlyEnv`：系统预构造多套资源可行部署，每个 episode 内固定一套部署，不同 episode 按随机种子轮换部署。route-only 训练的 episode 长度与 `--rollout-periods` 相同，因此每次 PPO update 后都会进入新的固定部署上下文。主场景默认构造 8 套可行部署，若场景可行组合不足则使用实际生成数量。

正式训练使用全新的 revision 4 输出目录，不能从旧归一化奖励生成的检查点继续训练。下面给出随机种子 0 的完整三阶段命令：

```powershell
conda activate agent-orch
cd C:\Users\01\Desktop\agent\experiments\agent_orchestration

$scenario = "configs/benchmarks/main_abilene.yaml"
$hash = ((Get-FileHash $scenario -Algorithm SHA256).Hash.Substring(0,16)).ToLower()
```

第一阶段只训练模型组成策略：

```powershell
python scripts/run_rl_matrix.py `
  --scenario $scenario `
  --seeds 0 `
  --modes route `
  --variants no-rnd `
  --training-phase composition `
  --updates 100 `
  --rollout-periods 16 `
  --update-epochs 4 `
  --minibatch-size 256 `
  --train-mapping-samples 128 `
  --validation-interval 5 `
  --validation-periods 20 `
  --validation-warmup-periods 5 `
  --validation-mapping-samples 128 `
  --eval-mapping-samples 512 `
  --train-slots 3600 `
  --eval-slots 100 `
  --arrival-scale 7.165234375 `
  --device cuda `
  --status-interval-steps 1 `
  --resume `
  --output results/formal_v3/stage1_composition

$routeRun = "agent-abilene-20-route-no-rnd-composition-s0-$hash"
$routePolicy = "results/formal_v3/stage1_composition/$routeRun/policy_best.pt"
Test-Path $routePolicy
```

第二阶段加载模型组成策略，冻结组成分支并训练部署策略。该阶段使用 `joint` 环境，使已训练的模型组成策略参与部署方案的周期性能评估：

```powershell
python scripts/run_rl_matrix.py `
  --scenario $scenario `
  --seeds 0 `
  --modes joint `
  --variants rnd `
  --training-phase deployment `
  --initial-policy $routePolicy `
  --updates 100 `
  --rollout-periods 16 `
  --update-epochs 4 `
  --minibatch-size 256 `
  --train-mapping-samples 128 `
  --validation-interval 5 `
  --validation-periods 20 `
  --validation-warmup-periods 5 `
  --validation-mapping-samples 128 `
  --eval-mapping-samples 512 `
  --train-slots 3600 `
  --eval-slots 100 `
  --arrival-scale 7.165234375 `
  --device cuda `
  --status-interval-steps 20 `
  --resume `
  --output results/formal_v3/stage2_deployment

$deployRun = "agent-abilene-20-joint-rnd-deployment-s0-$hash"
$deployPolicy = "results/formal_v3/stage2_deployment/$deployRun/policy_best.pt"
Test-Path $deployPolicy
```

第三阶段解冻两个策略分支并联合微调：

```powershell
python scripts/run_rl_matrix.py `
  --scenario $scenario `
  --seeds 0 `
  --modes joint `
  --variants rnd `
  --training-phase joint `
  --initial-policy $deployPolicy `
  --updates 100 `
  --rollout-periods 16 `
  --update-epochs 4 `
  --minibatch-size 256 `
  --train-mapping-samples 128 `
  --validation-interval 5 `
  --validation-periods 20 `
  --validation-warmup-periods 5 `
  --validation-mapping-samples 128 `
  --eval-mapping-samples 512 `
  --train-slots 3600 `
  --eval-slots 100 `
  --arrival-scale 7.165234375 `
  --device cuda `
  --status-interval-steps 20 `
  --resume `
  --output results/formal_v3/stage3_joint
```

没有 CUDA 环境时，将上述命令中的 `--device cuda` 改为 `--device cpu`。`--validation-warmup-periods` 指定不计入检查点选择分数的预热周期，`--validation-periods` 指定随后用于选择最佳检查点的周期数。

根据多随机种子汇总结果生成可复现的置信区间和配对显著性检验：

```powershell
python scripts/summarize_results.py `
  --input results/baseline_matrix.summary.parquet `
  --group-columns policy `
  --metrics mean_cost,mean_latency_s,mean_goodput_rps,mean_quality,mean_slo_attainment,mean_violations,violation_slot_fraction `
  --baseline static
```

生成统一的固定负载统计、IEEE 风格 PDF/PNG 图、RL 消融图、收敛曲线和决策开销图：

```powershell
python scripts/plot_results.py `
  --results results `
  --output results/figures `
  --analysis-output results/analysis
```

绘图程序会先核对基线与 RL 实验的场景哈希、随机种子集合和评估时长，确认一致后再合并结果。

在总请求率保持不变的条件下生成工作负载组成实验：

```powershell
python scripts/generate_composition_sweep.py `
  --scenario configs/toy.yaml `
  --short-app chat-search `
  --long-app research-agent `
  --long-fractions 0,0.25,0.5,0.75,1
```

论文使用的四类应用 60/20/10/10 偏斜组成场景通过以下命令生成：

```powershell
python scripts/generate_composition_sweep.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --family-sweep --output configs/generated/family_composition
```

## 预构建 Agent 负载与泊松到达

四类应用的调用图、概率选择、并行分支以及 LLM 节点输入输出 token 特征均由 `data/preconstructed_agent_workloads.yaml` 给出。每类应用按配置中的截断对数正态分布生成五个分位档位；每个档位都会将所有概率选择组合展开为 pattern flows，并保留其中的并行调用链。

到达过程统一采用平稳强度过程。分析型仿真直接把场景配置中的平均到达率作为排队模型的稳态强度，`--arrival-scale` 对全部强度作统一缩放，不将一秒内的离散请求计数反推为稳态到达率。Main 场景的基准总到达率为 0.004 request/s。实验先使用参考容量 20% 的低负载窗口冻结默认 SLO，并将相同阈值传播到三个资源压力场景：

```powershell
python scripts/calibrate_slos.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --low-load-fraction 0.20 `
  --output configs/benchmarks/main_abilene.yaml `
  --propagate-to configs/benchmarks/stress_network_0p5.yaml `
                 configs/benchmarks/stress_service_0p5.yaml `
                 configs/benchmarks/stress_gpu_unavailable.yaml
```

随后基于正式入口场景求稳定容量，并按其 0.40、0.65、0.85 和 1.05 倍生成四档负载：

```powershell
python scripts/calibrate_load_levels.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --output data/processed/load_levels.json
```

运行基线矩阵，`--load-levels` 会依次执行四档负载：

```powershell
python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml --slots 600 `
  --load-levels data/processed/load_levels.json `
  --seeds 0,1,2,3,4 `
  --resume `
  --output results/baseline_levels
```

Baseline 按负载档位、随机种子和策略保存独立运行文件；`baseline_status.json` 给出总体进度和预计剩余时间，`baseline_experiment.log` 记录各组合的开始、完成与异常。中断后使用相同命令和 `--resume` 即可跳过已经完成的组合。

仅绘制 baseline 结果：

```powershell
python scripts/plot_baseline_load_sweep.py `
  --input results/baseline_levels `
  --output results/figures_baseline_levels
```

绘图脚本必须在 `conda activate agent-orch` 之后运行。直接调用该环境的 `python.exe` 时，`Library\bin` 不在 `PATH` 上，matplotlib 的延迟加载原生库无法解析，`savefig` 会以 `Windows fatal exception: code 0xc06d007f` 崩溃；激活环境后同一脚本可正常输出 PDF 与 PNG。

## 准备和校验 LLM 性能数据

分析式仿真不再读取离线 profile 表；以下命令仅把 LLMServingSim 或 vLLM 输出规范化，并生成留出点校验报告：

```powershell
python scripts/prepare_llm_profiles.py `
  --input <normalized-LLMServingSim-output.csv> `
  --source-version <pinned-commit> `
  --output data/processed/llm_profile.csv
python scripts/validate_llm_model.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --profile data/processed/llm_profile.csv
```

验证程序报告 TTFT、TBT、完整响应时延和稳定容量的误差，其有效性判据与 `docs/experiment_protocol.md` 一致。

论文中的 LLM 宏观稳态模型可使用 LLMServingSim 2.0 进行分解验证，分别检查 Roofline 工作量、稳态有效并发度、KV 驻留边界和请求时延趋势。完整步骤和结果口径见 [docs/llmservingsim_queue_validation.md](docs/llmservingsim_queue_validation.md)。

## 输出与量纲

实验结果以 JSON Lines、Parquet 和运行清单等形式写入 `results/`。所有时间均以秒为单位，到达率与处理率均以请求/秒为单位，单请求数据量以 MB 为单位，链路负载和链路容量以 Mbit/s 为单位，计算量以 FLOPs 为单位，显存访问量以字节为单位。控制时隙长度不改变 Mbps 负载；链路传输时延按一个时隙内的聚合数据量与链路速率之比计算。

联合控制器按编排周期运行。每个周期开始时，Agent 感知混合容量规划器根据节点访问概率、到达率和候选实例能力，确定各无状态服务的副本需求及各模型的有效容量需求。带资源掩码的分类策略依次处理每个 LLM 候选实例和每个“无状态服务–服务器”副本池；每个部署项只决策一次，部署子步不推进物理业务时隙。部署完成后，分组 Dirichlet 策略生成一次应用到模型的工作负载比例，LLM 实例分流和无状态服务路由依据预测服务与网络时延通过 Softmin 解析得到。

周期奖励使用相邻周期的原始性能增量。成本和时延分别取上一周期值减当前周期值，请求级 goodput 和系统质量分别取当前周期值减上一周期值；四项增量按照场景中的目标权重相加。奖励不使用成本参考值或时延参考值归一化。某项相对变化不超过 0.1% 时，该项取 $10^{-3}$ 的小额正奖励；首个周期用于建立参考，各项同样取该小额正奖励。状态向量仍对成本和时延进行固定尺度处理，以保持神经网络输入稳定，该处理不参与奖励计算。

LLM 实例和无状态服务池分别形成稳定性约束，并由独立拉格朗日乘子更新。主算法通过阶段相关 GAE 将周期末效用传递至周期内的部署动作；Potential shaping 作为显式对照，RND 用于训练阶段的部署状态探索并随训练进度衰减。部署与模型组成策略采用独立编码器，并按“模型组成预训练、冻结组成分支训练部署策略、联合微调”的顺序训练。
