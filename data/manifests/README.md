# 数据清单规范

原始公共数据集不提交至实验源码目录。每个处理后产物必须包含一份 JSON 数据清单，记录数据源 URL、固定的 release 或 commit、许可证、SHA-256 校验和、完整预处理命令、随机种子、转换参数以及按时间顺序划分的数据边界。数据清单由 `agent_orch.data.DatasetManifest` 负责校验。

不同类型的模型参数使用相互独立的数据来源：

| 产物 | 主要数据依据 | 不用于推断的内容 |
|---|---|---|
| 请求到达 | BurstGPT v2；Azure 2024 用于外部验证 | 工作流拓扑和 GPU 容量 |
| 工作流 | TraceLab v2；BFCL V3/V4 | 生产请求到达强度 |
| 无状态服务 profile | DeathStarBench 或本地低负载测量 | LLM 推理时延 |
| 基础设施 | Alibaba GPU Trace v2026 | 请求 token 和 LLM 时延 |
| 物理网络 | SNDlib Abilene/GEANT | Agent 任务需求语义 |

## 到达轨迹格式

统一的到达轨迹格式为：

```text
slot,application,ingress,rate_rps
```

到达率使用请求/秒。在轨迹覆盖的时隙内，缺失的应用记录解释为零流量，而不是场景中的默认到达率。

规范化后的请求分配审计表格式为：

```text
timestamp_s,session_id,prompt_tokens,output_tokens,length_class,
family,application,ingress
```

输入和输出 token 始终保留同一源请求中的配对关系。不同负载档位通过一次全局时间戳缩放生成，不对请求独立重采样，也不单独放大部分突发区间。

## LLM 性能 profile 格式

统一的 LLM profile 格式为：

```text
model,config,prompt_tokens,output_tokens,arrival_rate_rps,
long_request_fraction,interactive_retrieval_fraction,
transactional_tool_fraction,deep_research_fraction,coding_agent_fraction,
ttft_s,tbt_s,response_s,stable_capacity_rps,kv_tokens
```

四个应用类别比例字段必须同时出现或同时省略。省略这些字段的 profile 仍然有效，但表示后端不区分工作负载组成。

## 无状态服务测量格式

统一的无状态服务测量输入格式为：

```text
service,server,vcpu,arrival_rate_rps,latency_ms,error,
request_bytes,response_bytes
```

`prepare_service_profiles.py` 根据低负载测量计算平均处理时间和服务时间 SCV；到达 SCV 作为独立的场景参数保留。稳定处理率定义为错误率低于 1%，且 P95 响应时间不超过低负载 P95 两倍时的最大实测到达率。不得将包含排队和网络时延的生产端到端 RT 直接作为服务处理时间。
