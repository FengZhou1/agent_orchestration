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
agent-orch-sim train --scenario configs/toy.yaml --updates 10 --rollout-steps 256
```

使用 `--mode joint`、`--mode deploy` 或 `--mode route` 选择联合优化、仅部署优化或仅路由优化。联合模式默认运行带拉格朗日约束和 RND 探索的 DTS-PPO-RND；`--exploration none` 关闭内在奖励，`--exploration icm` 和 Potential Shaping 仅作为显式对照：

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
  --rollout-steps 1024 `
  --device auto
```

训练时，终端进度条分别显示轨迹收集和 PPO 优化阶段。每个运行目录中的 `training_status.json` 持续覆盖当前进度、耗时和预计剩余时间，`training_history.jsonl` 在每次 PPO update 后立即追加训练指标；训练完成后仍会生成完整的 `training_history.json`。使用 `--no-progress` 可关闭终端进度条而保留在线文件。需要完整消融时，显式传入 `--seeds 0,1,2,3,4 --modes joint,deploy,route --variants auto`；Potential Shaping 与 ICM 可通过 `--modes joint --variants potential,icm` 运行。

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

到达过程统一采用平稳强度过程。分析型仿真直接把场景配置中的平均到达率作为排队模型的稳态强度，`--arrival-scale` 对全部强度作统一缩放，不将一秒内的离散请求计数反推为稳态到达率。Main 场景的基准总到达率为 0.004 request/s。实验先求固定参考部署的稳定容量，再按该容量的 0.40、0.65、0.85 和 1.05 倍四档设定负载：

```powershell
python scripts/calibrate_load_levels.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --output data/processed/load_levels.json
```

在测试集开放前，使用参考容量 20% 的低负载窗口冻结默认 SLO：

```powershell
python scripts/calibrate_slos.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --low-load-fraction 0.20 `
  --output configs/benchmarks/main_abilene_calibrated.yaml
```

运行基线矩阵，`--load-levels` 会依次执行四档负载：

```powershell
python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene_calibrated.yaml --slots 3600 `
  --load-levels data/processed/load_levels.json `
  --seeds 0,1,2,3,4 `
  --output results/baseline_levels
```

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

论文中的 LLM 排队近似可使用 LLMServingSim 2.0 进行分解验证，包含 Roofline 处理时延、稳态有效并发度和 Allen--Cunneen 排队时延。完整步骤和结果口径见 [docs/llmservingsim_queue_validation.md](docs/llmservingsim_queue_validation.md)。

## 输出与量纲

实验结果以 JSON Lines、Parquet 和运行清单等形式写入 `results/`。所有时间均以秒为单位，到达率与处理率均以请求/秒为单位，单请求数据量以 MB 为单位，链路负载和链路容量以 Mbit/s 为单位，计算量以 FLOPs 为单位，显存访问量以字节为单位。控制时隙长度不改变 Mbps 负载；链路传输时延按一个时隙内的聚合数据量与链路速率之比计算。

联合控制器在两个时间尺度上运行。每个部署周期开始时，Agent 感知混合容量规划器根据节点访问概率、到达率和候选实例能力，确定各无状态服务的副本需求及各模型的有效容量需求。带资源掩码的分类策略随后逐个选择副本服务器或 LLM 候选实例；这些顺序决策共同构成当期部署方案，不推进物理业务时隙。在后续物理时隙内，分组 Dirichlet 策略仅生成应用到模型的工作负载比例，LLM 实例分流和无状态服务路由依据预测服务与网络时延通过 Softmin 解析得到。

时隙效用由固定尺度归一化后的成本和时延、SLO 满足率及质量构成。LLM 服务强度、KV cache、无状态服务稳定性和链路带宽分别形成四类动态约束，并由独立拉格朗日乘子更新。一个部署周期结束后，当前方案相对保留上一方案的约束效用增益被均分给本周期的顺序部署动作。RND 仅用于慢时标部署状态探索，其权重随训练进度线性衰减；确定性评估不使用内在奖励。
