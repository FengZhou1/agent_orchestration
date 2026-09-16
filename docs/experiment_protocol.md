# 实验协议

## 数据依据与参数化方法

实验采用分层参数化方法：公开工作负载统计确定 LLM 请求特征，泊松过程生成应用请求到达，仿真器用于评估编排策略，受控基准测试用于校准组件性能。

- `data/preconstructed_agent_workloads.yaml` 提供四类 Agent 的调用图、概率选择、并行分支、LLM 节点输入输出 token 的 P50/P95 统计以及无状态服务处理时间。
- TraceLab v2 和 BFCL V3/V4 提供 Agent 执行模式、概率 pattern flow 以及调用质量任务。
- Alibaba Microservices v2021 提供无状态服务图的深度、扇出、复用、调用率和响应时间参考分布；DeathStarBench 或等价的本地测试程序提供低负载处理时间、SCV、稳定处理率和序列化数据量。
- Alibaba GPU Trace v2026 提供 GPU 类型、GPU 数量和 CPU 容量的服务器级联合记录，不用于生成请求到达或 LLM 推理时延。
- SNDlib 提供 Abilene 和 GEANT 的网络拓扑及节点坐标。
- LLMServingSim 2.0 提供完整的模型—GPU 性能网格，并使用少量 vLLM 实测验证低负载、接近容量和长请求占优三类运行点。

每个处理后产物均记录源文件校验和、版本、许可证、预处理命令、随机种子和数据划分边界。原始公共数据集不提交至代码仓库。

## 实验场景

| 场景 | 物理网络 | 模型数 | 应用数 | 无状态服务数 | 用途 |
|---|---:|---:|---:|---:|---|
| Toy | 4 个节点 | 2 | 2 | 3 | 冒烟测试、公式验证、路由验证和穷举检查 |
| Main | Abilene：12 个节点、15 条链路 | 4 | 20 | 6 | 主要算法对比和消融实验 |
| Scale | GEANT：22 个节点、36 条链路 | 4 | 50 | 8 | 可扩展性和决策开销实验 |
| Stress | Abilene/GEANT | 4 | 20/50 | 6/8 | 到达突发、服务降速、链路降容和 GPU 不可用实验 |

代码仓库中的基准场景由 `scripts/build_benchmark_scenarios.py` 生成。四个 BF16 模型档位分别为 Qwen3-4B、Qwen3-8B、Qwen3-14B 和 Qwen3-32B。候选部署配置包括 A10 上的 4B，L20 上的 4B、8B 和 14B，H20 上的 8B、14B 和 32B，以及双卡 L20 上的 32B。Main 场景交错配置 4 台 A10、5 台双卡 L20 和 3 台 H20；Scale 场景配置 7 台 A10、9 台双卡 L20 和 6 台 H20。vLLM 的显存利用率为 0.9，上下文窗口为 32768，批处理 token 上限为 8192，序列数上限为 128，开启 chunked prefill，prefill chunk 为 512 tokens，关闭 prefix caching。

主实验以 4B 作为模型规模下界，使候选模型能够覆盖多步规划、工具选择和结果综合等 Agent 任务；0.6B 和 1.7B 模型不进入主实验。各候选配置在预留模型权重和运行时显存后，均可容纳至少一个长度为 32768 tokens 的请求。A10 节点保留为低成本 4B 服务节点，双卡 L20 与 H20 节点为 8B--32B 模型提供主要部署容量。

Main 场景包含四类应用，每类设置五个模板：

1. 交互式检索：latency-sensitive，包含一至两个 LLM 阶段；
2. 事务型调用：deadline-sensitive；
3. 深度研究：compound，包含两条并行无状态服务分支；
4. 编码 Agent：compound，包含串行和并行测试模式。

四类应用分别对应预构建负载文件中的四类 Agent。每类应用按截断对数正态分布的五个分位档位生成负载模板，并依据配置中的概率选择将所有可行调用组合展开为 pattern flows；并行分支作为同一 pattern flow 中的多条调用链保留。每个模板在优化模型中作为一种应用类型，因此不增加请求级优化变量。

## 到达过程与数据划分

- 链路负载的分析参考间隔：1 秒；
- 宏观编排周期：60 秒；
- 训练或评估窗口：3,600 个编排周期；
- Main 场景基准总到达率：0.004 request/s；
- 负载档位：固定参考部署稳定容量的 0.40、0.65、0.85 和 1.05 倍；
- 分析型仿真在全部对比算法间共享相同的泊松到达强度；随机种子用于算法训练与随机路由。

应用请求服从平稳泊松过程。若应用 $a$ 在接入节点 $g$ 的平均到达率为 $\lambda_a^g$，分析型仿真将 $\lambda_a^g$ 输入 LLM 的 mixed/decode 稳态模型和无状态服务的 GI/M/$c$ 排队模型。负载档位通过统一缩放全部 $\lambda_a^g$ 得到，不改变应用组成与接入位置。离散请求计数仅用于事件驱动仿真，不作为稳态排队模型的瞬时到达率。`slot_seconds=1` 用于将通信数据量换算为链路负载，部署成本与运行指标按 60 秒编排周期累计。

均衡组成中四类应用各占 25%。另设置四组偏斜组成实验，每次令一类应用占 60%，其余三类按 20%、10% 和 10% 分配，同时保持总到达率不变。性能 profile 网格还改变长请求比例和总调用率。

## 组件参数

LLMServingSim profile 仅用于独立验证解析服务曲线和配置排序；主编排仿真使用第 3 节的 mixed/decode 稳态解析模型。验证运行点覆盖稳定容量的 20%、40%、60%、80%、95% 和 105%，并记录 TTFT、TBT、完整响应时延、稳定容量和 KV token 占用。

Main 场景包含 Web 检索、信息检索、代码执行、文件处理、结果验证和外部 API 六类无状态服务；Scale 场景增加知识图谱和数据转换服务。在条件允许时，每类服务分别测量 1、2 和 4 vCPU 配置。GI/G/c 近似分别保留到达 SCV 和服务时间 SCV。稳定处理率定义为：错误率低于 1%，且 P95 响应时间不超过低负载 P95 两倍时的最大负载。无法获得实测结果时，以 2--8 请求/秒/核作为初始参考范围，并执行 0.5、1 和 2 倍敏感性实验。

应用入口与出口数据量按每个 token 4 字节折算，中间调用采用服务 profile 中的序列化请求和响应字节数。2--10 MB/request 仅用于高通信量敏感性实验。网络负载以 Mbit/s 表示，不受控制时隙长度影响。Abilene 和 GEANT 链路根据距离设置约 50、150 或 400 Mbit/s 的容量；传播时延按地理距离除以 `2e8 m/s` 计算，并限制在 1--20 ms。链路容量敏感性实验采用 0.5、1 和 2 倍设置。

归一化 GPU-hour 成本设置为 A10=1、L20=2、H20=4。无状态服务成本与分配的 vCPU 数量成正比，并对所有成本系数执行 ±30% 敏感性分析。

## SLO 与质量校准

在打开测试集前，使用固定参考部署的流量加权低负载 P95 时延冻结 SLO。主场景的阈值原样用于网络降容、无状态服务降速和 GPU 不可用场景：

- latency-sensitive：TTFT 阈值为参考 P95 的 1.5 倍，TBT 阈值为参考 P95 的 1.25 倍；
- deadline-sensitive：完整响应 deadline 为参考 P95 的 1.5 倍；
- compound：各中间阶段和完整工作流 deadline 均为相应参考 P95 的 1.5 倍，最终生成阶段同时满足 TTFT 和 TBT 约束。

严格和宽松敏感性实验分别采用 1.25 倍和 2 倍。代码仓库中预置的阈值和质量均标记为参考值，正式论文实验前必须使用固定的校准结果替换。不同应用类别的质量分别采用归一化的 BFCL、LongBench v2 和 SWE-bench Verified 得分。

## 对比算法与评价指标

主要对比算法包括 Static-Homogeneous、Static-MultiModel、EqualSplit、LeastLoad、Greedy-SLO、PPO-Deploy、PPO-Route、DTS-PPO、DTS-PPO-RND 和无约束 DTS-PPO-RND。ICM 与 Potential Shaping 作为附加探索对照，穷举算法仅在 Toy 场景运行。

每种算法报告成本、请求流量加权平均响应时延、请求 goodput、质量、响应时延 P50/P95/P99、SLO 满足率、TTFT/TBT、GPU/KV/无状态服务/链路利用率、平均违约约束数、发生任意约束违约的时隙比例、重配置次数、决策时延和训练总时间。预实验使用三个随机种子，完整实验至少使用五个随机种子；资源允许时，主结果表使用十个随机种子。各算法共享相同的泊松到达强度。统计报告包括均值、中位数、效应量、95% bootstrap 置信区间，以及带 Holm 校正的配对 bootstrap 或 Wilcoxon 检验。

## 保真度与验收条件

- 每个应用输入分析型排队模型的泊松强度与场景设定值一致，各算法使用完全相同的到达强度。
- 各应用模板的 LLM 节点输入、输出 token 均处于预构建负载文件给定的 P50--P95 范围内。
- 工作流深度、无状态服务调用数、并行宽度和 pattern flow 频率与数据来源中的经验分布一致。
- 无状态服务处理时间来自低负载内部计时，不得将包含排队和网络的生产端到端 RT 直接作为处理时间。
- 在留出的 LLM profile 运行点上，中位绝对百分比误差不超过 10%，P95 误差不超过 20%。该指标用于验证解析模型的适用范围，不替换主实验中的解析模型。
- LLM 实例的稳态时延由服务曲线和首次准入等待共同给出：TTFT 为首次准入等待与 prefill 处理时间之和，TBT 为 decode 处理时间除以输出 token 数减一，响应时间为 TTFT 与后续 decode 时间之和；KV 与序列槽位约束常驻运行集，越界记为过载约束代价。
- 低负载下所有队列保持稳定；接近容量边界时，随并发增长的服务时延和 SLO 违约率应上升，LLM 调用率利用率趋于一；链路、无状态服务和长请求组成压力应分别反映在相应利用率和时延指标中。

## 可复现实验命令

```powershell
conda activate agent-orch
$env:PYTHONPATH='src'
python scripts/build_benchmark_scenarios.py
python scripts/build_benchmark_scenarios.py `
  --infrastructure-catalog data/processed/alibaba_gpu_server_catalog.yaml `
  --service-profile data/processed/stateless_service_profile.csv
python scripts/export_reference_catalogs.py
python scripts/prepare_llm_profiles.py `
  --input <normalized-LLMServingSim-output.csv> `
  --source-version <pinned-commit> `
  --output data/processed/llm_profile.csv
python scripts/calibrate_slos.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --slots 3600 --calibration-samples 16 `
  --low-load-fraction 0.20 `
  --output configs/benchmarks/main_abilene.yaml `
  --propagate-to configs/benchmarks/stress_network_0p5.yaml `
                 configs/benchmarks/stress_service_0p5.yaml `
                 configs/benchmarks/stress_gpu_unavailable.yaml
python scripts/calibrate_slos.py `
  --scenario configs/benchmarks/scale_geant.yaml `
  --slots 3600 --calibration-samples 16 `
  --low-load-fraction 0.20 `
  --output configs/benchmarks/scale_geant.yaml
python scripts/calibrate_load_levels.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --output data/processed/load_levels.json
python scripts/generate_composition_sweep.py `
  --scenario configs/benchmarks/main_abilene.yaml --family-sweep `
  --output configs/generated/family_composition
python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml --slots 600 `
  --load-levels data/processed/load_levels.json --seeds 0,1,2,3,4 `
  --resume `
  --output results/baseline_revised
```

长时间实验均采用增量落盘。Baseline 在每个负载档位—随机种子—策略组合结束后保存独立结果；PPO 在每个 update 后保存策略、优化器、探索模块、拉格朗日乘子和随机状态。使用同一输出目录并传入 `--resume` 时，程序从最近的完整单元继续运行。状态文件与追加日志用于监测进度和定位异常。
