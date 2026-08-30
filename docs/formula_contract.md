# Formula-to-code contract

Source of truth: `latex/agent_system_model_service_mesh_style.tex`.

| Paper object | Code owner | Unit / invariant |
|---|---|---|
| `y_l`, `z_hn` | `DeploymentDecision` | binary LLM candidate; nonnegative tool replicas |
| `x_agm` | `RoutingDecision.model_share` | simplex over models for every `(a,g)` |
| `varphi_agil` | `RoutingDecision.llm_share` | sums to `x_agm` over instances of model `m` |
| `p_aij^{u,v}` | `RoutingDecision.tool_route` | simplex over deployed destination replicas |
| `lambda_{a,i,l}` | `AnalyticalBackend.llm_arrivals` | requests/s |
| LLM Roofline work | `performance.llm.service_demand` | FLOPs, bytes, seconds |
| `W_l^LLM` | `performance.queueing.llm_waiting_time` | seconds; finite overload sentinel |
| `Lambda_{h,n}` | `AnalyticalBackend.tool_arrivals` | requests/s, every visited parallel node counted once |
| GI/M/c tool delay | `performance.queueing.tool_response_time` | seconds |
| `D_e^con` | `NetworkBackend.link_loads` | Mbit carried in one 1-s slot |
| `T_{u,v}^net` | `NetworkBackend.path_delay` | seconds |
| critical path | `WorkflowEvaluator` | maximum complete chain delay; shared segments are not summed across chains |
| SLO event | `WorkflowEvaluator.slo_satisfied` | piecewise by `lat`, `ddl`, `cmp` |
| `G^req`, `Q^sys` | `SlotMetrics` | requests/s and traffic-weighted score in `[0,1]` |

Boundary rules:

- zero-flow undeployed objects contribute zero load and delay;
- positive flow without a feasible instance is a service failure;
- unstable queues use a finite configured overload delay and set a violation flag;
- one-element routing groups are deterministic and do not enter an RL log-probability;
- the analytical backend never returns NaN or infinity.

