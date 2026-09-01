# 论文公式与代码实现对应关系

系统模型以 `latex/agent_system_model_service_mesh_style.tex` 为准。

| 论文对象 | 代码实现 | 量纲或约束 |
|---|---|---|
| `y_l`、`z_hn` | `DeploymentDecision` | LLM 候选实例为二值变量；无状态服务副本数为非负整数 |
| `x_agm` | `RoutingDecision.model_share` | 对每个 `(a,g)`，模型选择比例满足单纯形约束 |
| `varphi_agil` | `RoutingDecision.llm_share` | 模型 `m` 的所有实例分配比例之和等于 `x_agm` |
| `p_aij^{u,v}` | `RoutingDecision.tool_route` | 在已部署的目标无状态服务副本之间满足单纯形约束 |
| `lambda_{a,i,l}` | `AnalyticalBackend.llm_arrivals` | 请求/秒 |
| LLM Roofline 计算负载 | `performance.llm.service_demand` | FLOPs、字节、秒 |
| `W_l^LLM` | `performance.queueing.llm_waiting_time` | 秒；过载时使用有限的预设时延 |
| `Lambda_{h,n}` | `AnalyticalBackend.tool_arrivals` | 请求/秒；实际执行的每个并行节点均计入一次 |
| GI/M/c 无状态服务时延 | `performance.queueing.tool_response_time` | 秒 |
| `D_e^con` 或链路负载 | `NetworkBackend.add_traffic` | Mbps；不受控制时隙长度影响 |
| `T_{u,v}^net` | `NetworkBackend.path_delay` | 秒 |
| 关键路径 | `WorkflowEvaluator` | 取完整调用链时延的最大值；多条链之间的共享段不重复求和 |
| SLO 事件 | `WorkflowEvaluator.slo_satisfied` | 按 `lat`、`ddl` 和 `cmp` 三类分别判断 |
| `G^req`、`Q^sys` | `SlotMetrics` | 请求/秒，以及取值范围为 `[0,1]` 的流量加权质量得分 |

## 算法实现

- 每个部署周期执行一次慢时间尺度动作，联合选择全部 `y_l` 和 `z_hn`；
- 快时间尺度 PPO 策略头在每个物理时隙选择 `x_agm`；
- `varphi_agil` 和 `p_aij^{u,v}` 由考虑容量、时延和利用率的物理路由器计算；
- PPO-Lagrangian 分别处理归一化目标效用和物理约束代价。

## 边界处理规则

- 零流量且未部署的对象产生零负载和零时延；
- 正流量无法映射至可行实例时记为服务失败；
- 队列不稳定时使用配置中的有限过载时延，并设置约束违例标志；
- 只有一个可选项的路由组采用确定性选择，不计入 RL 策略的对数概率；
- 分析后端不得返回 NaN 或无穷值。
