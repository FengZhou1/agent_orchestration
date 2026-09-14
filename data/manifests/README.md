# 数据清单规范

原始公共数据集不提交至实验源码目录。每个处理后产物必须包含一份 JSON 数据清单，记录数据源 URL、固定的 release 或 commit、许可证、SHA-256 校验和、完整预处理命令、随机种子、转换参数以及按时间顺序划分的数据边界。数据清单由 `agent_orch.data.DatasetManifest` 负责校验。

不同类型的模型参数使用相互独立的数据来源：

| 产物 | 主要数据依据 | 不用于推断的内容 |
|---|---|---|
| 请求到达 | 场景平均到达率与平稳泊松过程 | 工作流拓扑和 GPU 容量 |
| LLM 请求特征 | preconstructed_agent_workloads.yaml | Agent 调用图、概率组合及节点级输入输出 token 特征 |
| 工作流 | TraceLab v2；BFCL V3/V4 | 生产请求到达强度 |
| 无状态服务 profile | DeathStarBench 或本地低负载测量 | LLM 推理时延 |
| 基础设施 | Alibaba GPU Trace v2026 | 请求 token 和 LLM 时延 |
| 物理网络 | SNDlib Abilene/GEANT | Agent 任务需求语义 |

## 泊松到达参数

场景文件的 `ingress_rates` 记录应用在各接入节点的平均请求率，单位为 request/s。给定时隙长度和随机种子，仿真器按平稳泊松过程生成每时隙请求数；不同负载档位通过统一的 `arrival_scale` 调整全部平均到达率。运行清单记录随机种子、负载缩放系数和场景校验和。

预构建负载文件的节点级输入、输出 token P50/P95 记录在应用配置中。场景生成器按五个分位档位编译应用模板，并将每个应用内的全部概率组合展开为 pattern flows，不生成逐请求 token 记录。

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
