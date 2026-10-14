# 历史场景，仅用于回归和追溯

本目录的Abilene/GEANT、四类固定应用与旧引擎配置不是当前实验依据。
当前仿真也使用A10/L20/H20，但不等于复用本目录的模型精度、批处理参数、拓扑或业务负载；
本轮仅明确登记复用的硬件标量锚点，不能把旧场景整套恢复为主场景。
主入口依据`docs/experiment_parameter_register_2026-10-13.md`及`configs/experiment_settings.yaml`，使用
`scripts/build_registered_scenarios.py`生成连通随机边缘网络、固定应用池与嵌套子集。
旧场景可以在显式`--smoke`中用于功能回归，不能作为正式主场景；原文件未删除，旧结果不重评分。
