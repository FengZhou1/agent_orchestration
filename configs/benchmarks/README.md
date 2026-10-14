# 历史场景，仅用于回归和追溯

本目录的Abilene/GEANT、A10/L20/H20、四类固定应用与旧引擎数值不是当前实验依据。
主入口依据`docs/experiment_parameter_register_2026-10-13.md`及`configs/experiment_settings.yaml`，使用
`scripts/build_registered_scenarios.py`生成连通随机边缘网络、固定应用池与嵌套子集。
旧场景可以在显式`--smoke`中用于功能回归，不能作为正式主场景；原文件未删除，旧结果不重评分。
