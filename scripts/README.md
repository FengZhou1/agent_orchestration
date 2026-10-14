# 实验脚本入口边界

本轮参数以docs/experiment_parameter_register_2026-10-13.md为依据。主入口：

| 脚本 | 当前职责 |
|---|---|
| build_registered_scenarios.py | 随机网络＋一次固定池＋多网络嵌套子集 |
| generate_agent_pool.py | 单网络固定池；默认登记基础网络 |
| rebind_agent_tools.py | 已保存池的工具规模变体；须保持网络兼容 |
| audit_registered_parameters.py | 登记表全行清单与数值/结构检查 |
| calibrate_synthetic_slos.py | 固定8B/L20参照、一次性冻结SLO；不运行PPO、不改训练负载 |
| run_slot_joint.py | 三阶段PPO与训练轨迹评价；正式参数冻结检查 |
| evaluate_slot_joint_snapshot.py | 保存配置下的快照评价，不采另一轨迹 |
| evaluate_slot_joint_routing_baselines.py | 同轨迹、同预训部署上下文的模型选择基线 |
| profile_capacity_routing.py | 显式指定场景的路由计算开销诊断 |
| benchmark_slot_runtime.py | 同场景/已保存轨迹的一次rollout计时与数值摘要，不更新PPO；可选cProfile，--scalar-workflow用于标量对照 |

旧build_benchmark_scenarios、export_reference_catalogs、矩阵训练、蒸馏、内层求解器、
reactivity与旧组成探针等不是当前主实验接口。它们保留为历史/回归工具，不应按默认
Abilene参数生成新论文结果。旧LLM校验脚本也不能替代新硬件实测参数冻结。
console接口agent-orch-sim仅在显式--legacy-smoke下允许旧环境诊断，不能冒充slot_joint_v7。

旧calibrate_slos.py会按容量缩放probe并原地覆盖场景，已要求显式--legacy-smoke；主标定使用新入口和独立合同产物。生成器与工具绑定变体继承合同，不按不同资源/策略表现重新放宽。
