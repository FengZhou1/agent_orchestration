# 数据集调研更新与仿真参数实现说明

本文档更新《数据集调研与实验组合建议_2026-07-15.md》，并说明其在当前仿真实现中的对应关系。

## 更新后的数据组合

实验不依赖单一的“万能数据集”，而是根据不同参数的实际语义组合多个数据来源：

| 参数层次 | 主要数据源 | 在仿真中的用途 |
|---|---|---|
| 请求到达 | 参数化平稳泊松过程 | 将场景平均到达率和负载缩放系数形成的泊松强度输入分析型排队模型 |
| LLM 请求特征 | preconstructed_agent_workloads.yaml | 四类 Agent 工作流、概率组合、并行分支和节点级输入输出 token 统计 |
| Agent 工作流 | TraceLab v2 | LLM 步骤、无状态服务调用和并行调用结构 |
| 调用质量 | BFCL V3/V4 | 多步、并行和可执行函数调用任务，以及应用相关质量得分 |
| 无状态服务结构 | Alibaba Microservices v2021 | 调用图深度、扇出、复用和调用率分布 |
| 无状态服务性能 | DeathStarBench 或本地轻量服务 | 低负载处理时间、SCV、稳定吞吐和接口数据量 |
| GPU 基础设施 | Alibaba GPU Trace v2026 | GPU 类型、GPU 数量和 CPU 容量的服务器级联合分布 |
| 物理网络 | SNDlib | Abilene 和 GEANT 的节点、链路和地理坐标 |
| LLM 性能 | LLMServingSim 2.0 与少量 vLLM 实测 | 不同模型—GPU—工作负载组成下的 TTFT、TBT、完整时延和稳定容量 |

TraceLab 已更新至 v2。当前公开数据池包含 8,058 个会话、665,453 个 Agent 步骤和 743,819 次调用，旧文档中的早期样本规模不再使用。

Alibaba GPU Trace v2026 覆盖六个月的集群资源与调度记录，仅用于生成服务器异构性和资源背景，不用于生成 Agent 请求到达、token 或 LLM 处理时延。Alibaba Microservices v2021 单独用于无状态服务图和响应分布；其中的生产 RT 可能同时包含排队和通信，因此不能直接替代低负载服务处理时间。

Mooncake 仅用于前缀缓存扩展实验。当前主实验不显式优化缓存命中率，因此不将 `hash_ids` 或缓存亲和性加入主要优化变量。Alpaca、ShareGPT 和框架示例只保留用于兼容性或冒烟测试，不作为生产负载依据。

## 已实现的参数与文件

- `configs/benchmarks/main_abilene_revised.yaml`：12 个物理节点、20 个应用模板和 6 类无状态服务；
- `configs/benchmarks/scale_geant.yaml`：22 个物理节点、50 个应用模板和 8 类无状态服务；
- `configs/benchmarks/stress_*.yaml`：链路容量减半、服务能力减半和 GPU 不可用等压力场景；
- `data/catalogs/main_abilene/`：基础设施、工作流、无状态服务及 LLM profile 请求网格；
- `agent_orch.workload.ArrivalTrace.stationary_poisson_intensity`：根据场景平均到达率和负载缩放系数生成分析型仿真使用的平稳泊松强度；
- `scripts/prepare_llm_profiles.py`：LLMServingSim/vLLM 输出规范化和留出点插值验证；
- `scripts/prepare_service_profiles.py`：根据低负载测量计算处理时间均值、服务时间 SCV 和稳定处理率；
- `scripts/prepare_infrastructure_catalog.py`：从 Alibaba GPU server-hour 数据中联合抽取 A10、L20 和 H20 服务器；
- `scripts/estimate_reference_capacity.py`：计算固定参考部署的稳定请求容量；
- `scripts/calibrate_slos.py`：使用低负载验证窗口冻结应用级和阶段级 SLO。

## 参数量纲

- 请求到达率和无状态服务处理率：request/s；
- 应用通信数据量：MB/request；
- 链路负载和容量：Mbit/s；
- 时延：秒；传播时延配置使用毫秒；
- 控制时隙只决定在线决策时刻，不参与 Mbps 的定义；
- LLM 入口和最终输出按 4 bytes/token 折算，中间调用采用接口请求和响应字节数；
- 主场景路由时隙为 1 秒，部署周期为 60 秒，训练或测试窗口为 3,600 秒。

## 正式论文实验前仍需完成的校准

当前生成场景中的质量与 SLO 明确标记为参考值。正式实验必须使用固定版本的 BFCL、LongBench v2 和 SWE-bench Verified 结果替换质量参数，并使用固定参考部署的低负载 P95 结果冻结 SLO。

LLMServingSim 2.0 profile 必须记录版本、配置和回放命令。在留出运行点上，中位误差超过 10% 或 P95 误差超过 20% 的区域直接采用 profile 后端结果，不声明为分析模型的有效预测范围。
