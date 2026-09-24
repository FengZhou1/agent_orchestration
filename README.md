# Agentic 应用的边缘 LLM–无状态服务联合编排实验

本仓库是论文 *SLO-Aware Joint Orchestration of LLM Services and Stateless Services with Dynamic Pattern Graphs for Agentic Applications in Edge Networks* 的解析仿真与强化学习实现。当前主线是 **`slot_joint_v4` 单时隙协议**，入口为 [`scripts/run_slot_joint.py`](scripts/run_slot_joint.py)，细节见 [`docs/slot_joint_protocol_2026-09-23.md`](docs/slot_joint_protocol_2026-09-23.md)。

> **截至 2026-09-24 的证据边界**：新协议已完成主场景模型选择阶段的两次从随机权重开始的 PPO 训练（种子 0、1，各 120 次更新）。部署 PPO 和联合更新的代码路径已实现，但**尚无对应的正式训练结果**。下面的数字仅是暂定场景与参数下的诊断，不能写成论文最终结论。旧 `stage_a/`、`stage_b/`、`formal_v*` 使用不同实验设置，不得与新协议混合比较。

## 1. 当前实验如何运行

物理时间轴只有时隙 `t`。先用固定种子预采样整条轨迹，保存为 `train_trajectory.json`；每个训练 episode 都从槽 0 重放同一组外生数值。时隙内的部署步和模型选择步只是顺序决策，**不会推进物理时间**：

1. 读取该时隙的应用到达率、pattern flow 概率、输入/输出 token 均值、可用带宽及服务器资源。
2. 顺序决定 LLM 候选实例是否启用，以及每个 `(无状态服务, 服务器)` 池的副本数。
3. 对每个 `(应用, 接入节点)` 分别采样一次上线模型上的比例向量；同一应用请求的所有 LLM 节点沿用所选模型。
4. 用解析 Softmin 完成 LLM 实例与无状态服务的物理路由；仿真一次，计算该时隙的效用、SLO 与资源约束。
5. 将本轮与上一轮**同一时隙**的四项性能指标作差；首轮使用同轨迹的固定贪心方案作参考。约束惩罚使用本轮的**绝对违约量**，而不是违约差分。

当前默认扰动范围是可运行占位值：逐 `(应用, 接入节点)` 到达率乘 `U(0.6, 1.4)`，pattern flow 用浓度 40 的 Dirichlet 抽样，token 均值乘 `U(0.8, 1.2)`，带宽乘 `U(0.7, 1.0)`，CPU/内存可用量乘 `U(0.8, 1.0)`。应用 SLO、拓扑、物理 GPU 库存、模型质量及服务 profile 默认固定。**这些范围和质量/SLO 数值尚未完成数据集标定。**

部署与模型选择各有独立的观测编码器、actor 和价值头，但共享时隙终端的系统目标。主场景一个时隙有 33 个 LLM 候选二元决策、`6 × 12 = 72` 个服务池的 `0…4` 副本数决策，以及 20 个模型选择决策，共 125 个子步。部署从空表逐项构造；上一时隙最终部署进入观测，并用于计算模型加载和副本启动成本。每个动作都受资源可行性掩码限制。

训练阶段由 `--route-updates`、`--deploy-updates` 与剩余的联合更新数指定：

| 阶段 | 可训练参数 | 部署/模型选择如何给出 |
|---|---|---|
| `composition` | 模型选择 PPO | 每时隙用容量规划器的可行部署加可行随机扩充；该时隙的部署在所有 episode 固定，部署 actor 冻结 |
| `deployment` | 部署 PPO | PPO 顺序构造部署；模型选择 actor 冻结 |
| `joint` | 两个 PPO 分支 | 部署、模型选择依次决策，两个分支共同更新 |

已完成的两次训练**只有 `composition: 120`**，没有加载教师或蒸馏权重。模型选择阶段的参考部署在主场景 8 个训练时隙各不相同、在两个 PPO 种子间相同；这提供了多样化且可复现的部署上下文，但**不代表部署策略已经学好**。贪心首轮基线也只是奖励参考，不是策略初始化。物理实例路由始终为解析计算，不由 PPO 学习。

## 2. 安装与快速检查

在仓库目录执行（Windows PowerShell）：

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e ".[rl,dev,plot]"
pytest -q
```

环境已建立时跳过第一行。Windows 下绘图前务必先 `conda activate agent-orch`；仅直接调用环境中的 `python.exe` 可能缺少 `Library\bin` DLL 路径。

下面是一个**新输出目录**上的 Toy 冒烟，不作为论文结果：

```powershell
python scripts/run_slot_joint.py `
  --scenario configs/toy.yaml --output results/slot_joint_toy_smoke_01 `
  --slots 2 --updates 2 --route-updates 1 --deploy-updates 0 `
  --trajectory-seed 101 --seed 0 --mapping-samples 2 --device cpu
```

入口默认使用 CPU、无 RND/ICM 探索奖励。若没有以 editable 模式安装包，可先设置 `$env:PYTHONPATH='src'`。

## 3. 复现当前模型选择阶段

以下是两次主场景训练的参数。`results/` 被 Git 忽略；重新运行时请使用**未被占用的新目录**，不要覆盖现有结果。单次 120 更新可能耗时数小时。

```powershell
# PPO seed 0；seed 1 只需更换 --seed 与 --output。
python scripts/run_slot_joint.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --output results/slot_joint_v4_route_shared_s0_reproduce `
  --slots 8 --updates 120 --route-updates 120 --deploy-updates 0 `
  --trajectory-seed 101 --seed 0 --mapping-samples 8 `
  --update-epochs 3 --minibatch-size 128 `
  --shared-model-head --exploration none --device cpu
```

`run_slot_joint.py` 没有蒸馏/预训练权重参数：不带 `--resume` 时，它拒绝覆盖已有运行并从随机初始化创建新策略。中断后的 `--resume` 只用于**同一输出目录、完全相同的场景、轨迹、种子和训练参数**；当前实现不能仅把已完成运行的 `--updates 120` 改成 160 来无损延长训练。

若要试跑完整三阶段流程，应另建目录，并明确分配更新数，例如 `--updates 12 --route-updates 4 --deploy-updates 4`（剩余 4 次为联合更新）。这是功能试跑示例，**不是已有的部署或联合阶段成绩**。

### 留出评估与绘图

训练入口会自动评估种子 `trajectory-seed + 1` 的留出轨迹，并保存 `evaluation.json`、`baselines.json`；模型选择单阶段还保存**同部署**的 `routing_baselines_same_placement.json`。对保存的检查点进行额外留出评估：

```powershell
python scripts/evaluate_slot_joint_snapshot.py `
  --run results/<run-directory> --evaluation-seed 103 `
  --mapping-samples 128 --baselines

python scripts/evaluate_slot_joint_routing_baselines.py `
  --run results/<run-directory> --evaluation-seed 104 --mapping-samples 128
```

这些脚本只读取训练检查点或 manifest；输出文件已存在时会报错，不会静默覆盖。比较策略与基线前，须核对 `scenario_hash`、留出 `trajectory_digest`、映射采样数、部署条件和检查点更新数。完整评估应使用未见轨迹和多个 PPO 种子，不能只看训练回报或总体平均达标率。

当前两次 120 更新运行的图由下列命令生成（默认读取 `results/slot_joint_v4_route_shared_dual_s0_2026_09_24/` 与 `results/slot_joint_v4_route_shared_s1_2026_09_24/`）：

```powershell
python scripts/plot_slot_joint_route_long.py
```

脚本检查两次运行的协议、场景、训练轨迹、训练配置，以及策略/基线的留出轨迹摘要一致性。它生成训练曲线、五条留出轨迹对比图和逐槽 SLO 热图；图像与原始 `results/` 一样不提交到 Git。

## 4. 当前观察与限制

固定第 120 更新权重，用五条未见轨迹（种子 102–106；各 8 个时隙；每槽 128 个映射样本）评估：

| PPO 种子 | 五轨迹平均效用 | 平均 SLO 达标率 | 低于 90% 的时隙 |
|---:|---:|---:|---:|
| 0 | 0.020982 | 93.42% | 0/40 |
| 1 | 0.019298 | 92.54% | 9/40 |

两个策略在这些轨迹上的效用和**平均**达标率均优于同部署的 greedy、equal、least-load、random 模型选择启发式；但 seed 1 仍有逐时隙 SLO 违约，seed 0 的最小达标余量也很小。详细逐轨迹记录与评估边界见 [`docs/slot_joint_route_long_2026-09-24.md`](docs/slot_joint_route_long_2026-09-24.md)。这些是**两次随机初始化的描述统计**，不是显著性证明，更不是标定后的论文结论。

目前另有一项必须在正式实验前处理的量纲问题：主场景仍设置 `slot_seconds: 1.0`，仿真会将其用于运行成本等换算；它只是占位值，**不能解释成实际部署只需 1 秒**。场景中遗留的 `orchestration_period_s` 由加载器丢弃，不构成第二个时间轴。确定真实时隙时长、校准质量/SLO 与外生变化范围后，应使用新输出目录重新训练和评估。

## 5. 文件与可复现产物

| 位置 | 当前用途 |
|---|---|
| `configs/toy.yaml`、`configs/benchmarks/` | Toy 冒烟与主/规模/压力场景 |
| `src/agent_orch/workload/slot_trajectory.py` | 固定、可序列化的预采样时隙轨迹 |
| `src/agent_orch/envs/slot_joint_env.py` | 时隙内顺序部署、逐组模型选择、同槽差分奖励 |
| `src/agent_orch/agents/` | 两个 actor/critic 分支、PPO 更新与约束乘子 |
| `src/agent_orch/routing/physical.py` | 解析实例与服务路由 |
| `src/agent_orch/simulator/`、`performance/` | LLM、无状态服务、网络及工作流指标 |
| `tests/test_slot_joint_protocol.py` | 新协议的轨迹、动作、奖励和训练阶段回归测试 |
| `docs/slot_joint_protocol_2026-09-23.md` | 当前协议的公式与边界；README 的状态更新优先 |
| `old_exp/`（仓库外） | 实验室旧微服务代码，只作只读参考，本包不导入 |

每个 `run_slot_joint.py` 输出目录中的 `manifest.json` 记录协议、场景哈希、轨迹摘要、PPO 种子和训练配置；`train_trajectory.json` 与 `evaluation_trajectory.json` 保存外生轨迹；`checkpoint.pt` 用于同配置恢复；`policy.pt` 是最终权重；`history.json`、`fixed_baseline.json`、`evaluation.json` 及基线文件记录训练和评估。**仅看到 `checkpoint.pt` 不等于训练完成**：还应核对 `history.json` 的更新数、`policy.pt` 和最终评估文件。

旧 `run_rl_matrix.py`、部署库、内层求解器与蒸馏脚本仍保留供历史复核，不是本 README 所述 `slot_joint_v4` 实验的前置步骤或结果来源。论文正文、实验协议、场景参数和代码有冲突时，请先检查当前源码与运行 manifest，并明确记录差异，不要把旧结论补写进新实验。
