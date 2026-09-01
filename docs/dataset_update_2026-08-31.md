# 数据集调研更新与仿真参数落地说明

本说明更新《数据集调研与实验组合建议_2026-07-15.md》并对应当前仿真实现。

## 更新后的数据组合

实验不采用单一“万能数据集”，而是按参数语义组合数据：

| 层次 | 主数据源 | 在仿真中的用途 |
|---|---|---|
| 请求到达 | BurstGPT v2 | 秒级到达时间、周期性和突发结构 |
| 外部泛化 | Azure LLM Inference Trace 2024 | 独立的时间戳和输入—输出 token 配对样本 |
| Agent 工作流 | TraceLab v2 | LLM 步骤、无状态服务调用、并行调用和 token 特征 |
| 工具质量 | BFCL V3/V4 | 多步、并行、可执行函数调用及应用相关质量得分 |
| 无状态服务结构 | Alibaba Microservices v2021 | 调用图深度、扇出、复用和调用率分布 |
| 无状态服务性能 | DeathStarBench/本地轻量服务 | 低负载处理时间、SCV、稳定吞吐与接口字节数 |
| GPU 基础设施 | Alibaba GPU Trace v2026 | GPU 类型与数量、CPU 容量的服务器级联合分布 |
| 物理网络 | SNDlib | Abilene 和 GEANT 拓扑、节点坐标与链路关系 |
| LLM 性能 | LLMServingSim 2.0 + 少量 vLLM | 模型—GPU—负载组成对应的 TTFT、TBT、完整时延和容量 |

TraceLab 已更新至 v2。公开池当前为 8,058 个会话、665,453 个 Agent
步骤和 743,819 次无状态服务调用，旧文档中的早期样本规模不再使用。

Alibaba GPU Trace v2026 是六个月的集群资源与调度轨迹。它只用于构造
服务器异构性和资源压力，不用于生成 Agent 请求到达、token 或 LLM 处理时延。
Alibaba Microservices v2021 单独用于无状态服务图和响应分布；其中生产 RT
可能包含排队和通信，因此不会直接代替低负载处理时间。

Mooncake 只用于前缀缓存扩展实验。当前主模型不显式优化缓存命中，因此不把
`hash_ids` 或缓存亲和性加入主实验变量。Alpaca、ShareGPT 和框架示例只保留为
兼容性或冒烟测试，不作为生产负载证据。

## 已落地的参数与文件

- `configs/benchmarks/main_abilene.yaml`：12 节点、20 个应用模板、6 类无状态服务。
- `configs/benchmarks/scale_geant.yaml`：22 节点、50 个应用模板、8 类无状态服务。
- `configs/benchmarks/stress_*.yaml`：链路容量减半、服务能力减半和 GPU 不可用。
- `data/catalogs/main_abilene/`：基础设施、工作流、服务和 LLM profile 请求网格。
- `scripts/prepare_arrival_traces.py`：BurstGPT/Azure 轨迹标准化、应用映射、负载缩放和 60/20/20 划分。
- `scripts/prepare_service_profiles.py`：从低负载测量计算处理均值、SCV 和稳定处理率。
- `scripts/prepare_infrastructure_catalog.py`：从 Alibaba GPU server-hour 表联合抽取 A10/L20/H20 服务器。

## 参数口径

- 到达率和无状态服务处理率：request/s。
- 应用通信数据：MB/request。
- 链路负载和容量：Mbit/s。
- 时延：s；传播时延配置使用 ms。
- 控制时隙只确定决策时刻，不参与 Mbps 的定义。
- LLM 入口和最终输出按 4 bytes/token 折算；中间调用采用接口请求/响应字节数。
- 主场景路由时隙为 1 s，部署周期为 60 s，训练或测试窗口为 3,600 s。

## 最终论文实验前仍需填充的实测项

生成场景中的质量和 SLO 明确标记为 reference values。最终实验必须使用固定
版本的 BFCL、LongBench v2、SWE-bench Verified 结果替换质量值，并使用固定
参考部署在低负载下的 P95 结果冻结 SLO。LLMServingSim 2.0 profile 表必须记录
版本、配置和回放命令；留出点中位误差超过 10% 或 P95 误差超过 20% 的区域，
直接采用 profile 后端结果。
