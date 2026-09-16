# 论文公式与代码实现对应关系

系统模型以 `latex/agent_system_model_service_mesh_style.tex` 为准。

| 论文对象 | 代码实现 | 量纲或约束 |
|---|---|---|
| `y_l`、`z_hn` | `DeploymentDecision` | LLM 候选实例为二值变量；无状态服务副本数为非负整数 |
| `x_agm` | `RoutingDecision.model_share` | 对每个 `(a,g)`，模型选择比例满足单纯形约束 |
| `varphi_agil` | `RoutingDecision.llm_share` | 模型 `m` 的所有实例分配比例之和等于 `x_agm` |
| `p_aij^{u,v}` | `RoutingDecision.tool_route` | 在已部署的目标无状态服务副本之间满足单纯形约束 |
| `lambda_{a,i,l}` | `AnalyticalBackend.llm_arrivals` | 请求/秒 |
| LLM Roofline 计算负载 | `performance.llm.prefill_work` / `decode_work` | FLOPs、字节、秒 |
| `mu_l^ana`、`rho_l^LLM` | `performance.llm.throughput_capacity` | 调用/秒；`mu = max_nu nu / mean_service(nu)`，利用率为 `Lambda / mu` |
| `bar B_l` | `performance.llm.steady_active_concurrency` | 请求；Little 定律的最小不动点 |
| `D_l^pre`、`D_l^dec` | `performance.llm.ServiceCurve.prefill_at` / `decode_at` | 秒；prefill 按单个代表性 chunk 与运行集合共同占用一次 iteration |
| `Lambda_{h,n}` | `AnalyticalBackend.tool_arrivals` | 请求/秒；实际执行的每个并行节点均计入一次 |
| GI/M/c 无状态服务时延 | `performance.queueing.tool_response_time` | 秒 |
| `D_e^con` 或链路负载 | `NetworkBackend.add_traffic` | Mbps；不受控制时隙长度影响 |
| `T_{u,v}^net` | `NetworkBackend.path_delay` | 秒 |
| 关键路径 | `WorkflowEvaluator` | 取完整调用链时延的最大值；多条链之间的共享段不重复求和 |
| SLO 事件 | `WorkflowEvaluator.slo_satisfied` | 按 `lat`、`ddl` 和 `cmp` 三类分别判断 |
| `G^req`、`Q^sys` | `SlotMetrics` | 请求/秒，以及取值范围为 `[0,1]` 的流量加权质量得分 |

## 算法实现

- `CapacityPlanner.plan` 根据 Agent 节点访问概率生成 `N_h^req`、`C_m^req` 及候选实例有效容量；
- 周期内分类 Actor 按资源池顺序逐个选择无状态服务服务器或 LLM 候选实例，资源掩码同步更新，部署子步不推进物理时隙；
- 部署完成后，分组 Dirichlet Actor 在每个编排周期只选择一次 `x_agm`；
- `varphi_agil` 和 `p_aij^{u,v}` 根据预测服务时延与多跳网络时延，经 Softmin 映射解析计算；
- LLM 实例和无状态服务池分别形成稳定性约束代价，并由对应拉格朗日乘子更新；
- 容量潜势差分奖励作用于周期内部署子步，周期性能效用在组合评估转移中计算；
- 共享编码器连接部署分类头、模型组成头和两个价值函数头，所有周期内转移按各自折扣联合计算 GAE。

## 边界处理规则

- 零流量且未部署的对象产生零负载和零时延；
- 正流量无法映射至可行实例时记为服务失败；
- 实例不稳定时记录 `llm_queue_overload` / `llm_kv_overload` 约束违例，不再引入过载下的等待时延；
- 只有一个可选项的路由组采用确定性选择，不计入 RL 策略的对数概率；
- 分析后端不得返回 NaN 或无穷值。
- `AnalyticalBackend._llm_performance` delegates each instance to `AnalyticalBackend.evaluate_llm_instance`; steady state carries no LLM admission-wait term.
- `bar K_l^act`, `C_l^run`, and `delta_l^KV` are computed before the steady-state concurrency fixed point; `C_l^run` is a KV/sequence residency limit, never a server count.
- `bar B_l` solves `bar B_l = sum(lambda_{a,i,l} D_{a,i,l}^{svc}(max(1,bar B_l)))` without clipping to the residency limit; crossing the limit is reported as overload.
- TTFT is the prefill part of the curve at `max(1,bar B_l)`, TBT is the decode part divided by `O-1`, and the response time is their sum.
- `service_demand` and `ServiceCurve` are two views of one model; a unit test asserts they agree to machine precision.
