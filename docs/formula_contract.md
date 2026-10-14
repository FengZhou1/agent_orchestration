# 论文公式与代码实现对应关系

系统模型与算法以当前 ACM 主稿 `latex/ACM_Conference_Proceedings_Primary_Article_Template__1_/sections/` 为准。当前训练主入口为`slot_joint_v7`，参数以`experiment_parameter_register_2026-10-13.md`为准、机器配置为`configs/experiment_settings.yaml`。固定网络计量窗口、成本率和完成请求应用等权时延见`metric_conventions_2026-10-14.md`；历史条目须结合最新接线读取，不用于恢复旧默认值。

| 论文对象 | 代码实现 | 量纲或约束 |
|---|---|---|
| `y_l`、`z_hn` | `DeploymentDecision` | LLM 候选实例为二值变量；无状态服务副本数为非负整数 |
| `x_agm` | `RoutingDecision.model_share` | 对每个 `(a,g)`，模型选择比例满足单纯形约束 |
| `varphi_agil` | `RoutingDecision.llm_share` | 模型 `m` 的所有实例分配比例之和等于 `x_agm` |
| `p_aij^{u,v}` | `RoutingDecision.tool_route` | 在已部署的目标无状态服务副本之间满足单纯形约束 |
| `lambda_{a,i,l}` | `AnalyticalBackend.llm_arrivals` | 请求/秒 |
| LLM Roofline 计算负载 | `performance.llm.prefill_work` / `decode_work` | FLOPs、字节、秒 |
| `mu_l^ana`、`rho_l^LLM` | macro 模式的 `performance.analytical.reference_llm_capacity` / `llm_queueing.two_mode_capacity` | 调用/秒；按 prefill/decode token-flow 平衡扫描并发；利用率为 `Lambda / mu` |
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
- 时隙内分类 Actor 每步确定一个候选的 `0/1` 状态或一个服务池的 `0…R_max` 副本数；当前部署从空表构造，上一部署参与观测与切换成本；
- 部署完成后，分组 Dirichlet Actor 逐 `(a,g)` 确定 `x_agm`；模型选择单阶段直接安装固定部署，每槽只执行模型子步；
- 当前唯一的 `CapacityProportionalRouter`：同模型推理流按 `alpha_ai * sum_g(lambda_ag * x_agm)` 形成 token 组成，`varphi_agil = x_agm * mu_ref_l / sum(mu_ref)`，只归一化可达、在线、正容量实例；服务池按 `z_hv * mu_hv` 分流。2026-10-13已删除旧Softmin实现；token与pattern概率跨槽固定，评价读取训练轨迹（slot_joint_v6）；
- LLM、服务、网络与总体 SLO 构成四维绝对违约量，乘子只按每槽最终评价记录更新；
- 部署与中间模型子步奖励为零，时隙终端用本轮对上一轮同槽的四项效用差分减当前绝对违约惩罚；首轮参考为同轨迹、同容量路由的固定贪心方案；
- 两个 Actor 使用独立编码器和价值头；槽内子步折扣取 1，槽间取系统折扣，当前入口的两个分支 GAE 系数均为 1，使时隙终端奖励完整回传；
- 训练接口支持模型组成策略预训练、冻结模型组成分支后的部署策略训练，以及两个分支解冻后的联合微调。

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
