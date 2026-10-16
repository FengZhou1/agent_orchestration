## Material Passport

- ARS academic-research-suite / experiment-agent；用户授权的固定条件对照实现与检查。
- 输入：现有20服务器、20应用、47工具场景，以及原1000轮运行的32份固定部署。
- 范围：只固定外生条件；不改网络、PPO、奖励、约束、物理路由或三阶段框架。
- 状态：准备和接线验证；不是新收敛结果，不是实机测量。

# 32槽仅部署变化的模型选择对照

## 1. 固定口径

固定为场景文件中的**基准值**，不是原轨迹的槽0，也不是把各应用的到达率统一成一个数。

| 项目 | 新对照的32个槽 |
|---|---|
| 每个(应用,入口)到达率 | 保留其原基准值，跨槽相同；原零入口仍为零 |
| CPU、内存、GPU库存 | 每台服务器保留其原资源，跨槽相同 |
| 带宽、传播时延、拓扑 | 每条链路及路径条件跨槽相同 |
| 图、pattern概率、token、工具绑定、工具性能 | 原场景不变，不重抽应用池 |
| SLO、质量、价格、LLM配置、批处理参数 | 原场景不变，不重新标定 |
| 部署 | 逐槽复用原32份完整布局，包括模型实例与工具副本位置 |
| PPO、随机初始化方式和种子、奖励、乘子 | 原设置不变；新目录从零训练，无教师/蒸馏 |

时隙编号正常推进；上一部署、上槽性能、当期已选模型比例和奖励历史仍正常随决策演化。这些是算法的内部状态，不是要冻结的外生输入。部署变化仍会产生已有定义的切换费用。

这组对照能减少跨槽需求/资源波动对模型选择学习的干扰，但**不能保证**恢复应用分离或收敛；此前的编码器饱和问题没有在本次修改中处理。固定到达率仍保留应用间的负载差异；它们本来就是决策需要考虑的信息。因为到达率、CPU/内存和带宽一并冻结，若结果变化，不能单独归因于到达率。基准资源也不等于旧波动轨迹的平均可用资源。

## 2. 已准备的独立输入

场景：`results/slo_calibrated_main_20261014/servers_20_applications_20.yaml`。

新bundle：`results/warmup_n20_a20_32slots_fixed_base_20261016/`。

源布局：`results/composition_n20_a20_32slots_s0/`的`warmup_deployments.json`，按原槽号对应，不打乱、不修复、不按SLO筛选。复用前用源轨迹核验旧摘要，复用后重新计算容量、CPU/内存占用等诊断。新轨迹和新bundle有新的摘要，不能与原检查点混用。

生成与核验命令（本机已生成，重做须换新输出目录）：

```powershell
conda activate agent-orch
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'

python scripts/build_warmup_deployments.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_20.yaml `
  --slots 32 --trajectory-seed 101 --deployment-seed 20261014 `
  --fixed-exogenous --reuse-deployments results/composition_n20_a20_32slots_s0 `
  --output results/warmup_n20_a20_32slots_fixed_base_20261016

python scripts/verify_warmup_deployments.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_20.yaml `
  --bundle results/warmup_n20_a20_32slots_fixed_base_20261016 --regenerate `
  --output results/warmup_n20_a20_32slots_fixed_base_20261016/verification.json
```

本机另外保存`control_checks.json`，列出每类外生输入的不同取值数；均应为1。`verification.json`保存按源布局重放构造后摘要一致的检查。容量筛查通过不等于所有SLO可行，仍需观察训练与逐槽结果。

## 3. 新的模型选择训练命令

```powershell
cd C:\Users\01\Desktop\agent\experiments\agent_orchestration
conda activate agent-orch
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'

python -u scripts/run_slot_joint.py `
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_20.yaml `
  --warmup-bundle results/warmup_n20_a20_32slots_fixed_base_20261016 `
  --output results/composition_n20_a20_32slots_fixed_base_s0 `
  --fixed-exogenous --slots 32 `
  --updates 1000 --route-updates 1000 --deploy-updates 0 `
  --mapping-samples 128 `
  --seed 0 --trajectory-seed 101 --warmup-seed 20261014 `
  --device cpu --workflow-engine array --workflow-workers 1
```

1000轮沿用原运行的观察预算，不是收敛保证。本次没有启动这项长训。首次启动**不要加`--resume`**，也不要使用旧1000轮权重。中断后才在同一命令后加`--resume`，必须保留`--fixed-exogenous`；扩展预算仍用`--extend-budget`并同时调整两个轮数。

本次没有改变`configs/experiment_settings.yaml`中的主协议波动范围或来源摘要；固定条件是显式登记在manifest/checkpoint中的诊断模式。原波动轨迹、1000轮结果和图集保留。`--formal`仍不能绕过参数冻结门禁，本对照不能被自动升级为正式已冻结实验。

## 4. 实现与验收

- `SlotTrajectory.fixed`按基准值生成32个独立副本；序列化、加载和执行时检查固定声明，拒绝元数据称固定但实际仍变化的轨迹。
- `reuse_layouts_fixed_exogenous`保留所有物理部署动作，重算诊断；不生成模型选择标签，不更改负载以适配容量。
- bundle和训练入口都需显式`--fixed-exogenous`，错误的bundle、旧checkpoint或漏标续训会被拒绝。
- 测试覆盖固定输入、应用间差异保留、数据无别名、布局逐项一致、诊断重算、确定性复现、入口及中断恢复。原波动模式保持兼容。

已有输入核对：32/32个不同物理布局，在线2/3/4模型分别11/11/10槽，覆盖11种模型子集；20应用、129个正入口组、47种工具保持不变。具体实测与测试记录以bundle中的核验文件和本次交付说明为准。

2026-10-16核验记录：

- 全部32个执行场景按数据类逐项比较均等于基准场景；六类snapshot字段的不同取值数均为1。
- 全部32份部署与源运行逐项一致；布局绑定和诊断重算可确定性复现；无容量筛查风险标记，不据此保证全部SLO可行。
- 总基准到达率固定为0.26311917732499485 request/s；不同应用/入口的原有差异和零入口保留。
- 路由观测维度仍为5052，部署动作头宽度仍为63；同seed0下，模型选择编码器、动作头和价值头的随机初始张量与原组完全一致。
- 新轨迹摘要：`4a5ef738795383d5d4be03b499c95eabc71e68d2b4829f896cd52dcbcbf8c5c7`。
- 新部署库摘要：`acad85f764a9b20fc6614c108c99e17c72bd353ef667399cd394e927169b5b68`。
- 全部419项已跟踪测试及新增固定条件测试通过；原1000轮history、manifest和evaluation的文件摘要未变。
- 完整32槽环境接线回放通过：4,128个模型选择子动作，观测、奖励和约束均有限；各槽场景始终等于基准场景，安装布局与保存布局一致。该检查使用均匀动作、0次PPO更新，不是收敛实验。
