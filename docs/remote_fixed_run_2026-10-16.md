## Material Passport

- ARS academic-research-suite / experiment-agent；用户指定服务器上的实验部署与启动。
- 输入为原冻结场景及部署布局，不重新生成不同应用、不载入训练权重。
- 状态：最初从零启动；已在第54轮检查点暂停，仅更新进度显示后，从第55轮续跑并确认新检查点保存。尚未完成1000轮，不声明收敛。

# 远端固定条件模型选择实验

## 1. 实际运行位置

| 项目 | 值 |
|---|---|
| SSH | `zf@115.156.159.183` |
| 仓库绝对目录 | `/home/zf/zf/agent_orchestration`（即`~/zf/agent_orchestration`） |
| Conda环境 | `/home/zf/miniconda3/envs/agent_orchestration`，按用户确认使用既有环境 |
| 运行源码提交 | `ff60c88652055e63ab2e051c31f44570e1f7d116` |
| tmux会话 | `agent_fixed_n20a20_s0` |
| 启动PID | `2770993`；后续以进程及tmux状态核对，不能只凭旧PID判断 |
| 启动时间 | 2026-10-16 12:35:44，北京时间 |
| 输出目录 | `results/composition_n20_a20_32slots_fixed_base_s0` |
| 控制台日志 | `results/remote_logs/composition_fixed_n20_a20_s0_20261016.log` |
| 启动记录 | `results/remote_preflight_20261016/launch.json` |

按1000轮模型选择训练预算运行，每轮32槽，部署分支冻结；CPU、array引擎、单worker。Python/BLAS线程设为1，避免与服务器其他任务争抢线程。不使用教师、蒸馏、RND或ICM，不载入旧策略、Adam状态或奖励历史。

采用tmux持久运行，SSH断开不终止任务；不会因当前聊天结束而停止。未配置自动重启、自动提前停止、未来通知或定时监控。

## 2. 迁移了什么

Git克隆没有`results/`。为保持同一应用池子集，没有在Linux上重新随机抽取应用，而是迁移：

- `results/slo_calibrated_main_20261014/servers_20_applications_20.yaml`：包含完整20应用图、pattern、token、工具和SLO，文件SHA256仍为`037b24e08711b8feab4675767bfce2c50c4106c4c59ce36347c4540ad5f881ce`。
- `results/composition_n20_a20_32slots_s0/train_trajectory.json`和`warmup_deployments.json`：仅作为原布局来源。远端该目录**没有**Windows训练权重、优化器或训练历史，不是可续训的旧运行。
- 已跟踪SLO合同的原始字节，保持`bff21619491b5e0ffc9e9a8eec9da4e0fd1b920a507a3841093cdcd7e034c154`。Windows原文件为CRLF，Linux克隆原为LF，JSON合同摘要相同；使用当前仓库的`.git/info/attributes`保留精确字节，不重新标定SLO或改变合同内容。

随后在服务器重建`results/warmup_n20_a20_32slots_fixed_base_20261016/`。所有外生设置重复场景基准，32份物理部署与源布局逐项一致，仍含47工具和129个正入口组。重建未按策略性能筛选，也未修复/替换布局。

## 3. 跨平台校验与兼容修复

原来源检查把`settings_path`的机器绝对路径也用于相等判断，导致完全相同的参数文件在Windows/Linux迁移后被拒绝。提交`ff60c88`只将机器路径视为来源说明，**仍严格比较全部5项文件哈希**，缺失或未知哈希字段仍拒绝。实验模型、PPO及奖励计算未改。

修复通过Git bundle快进同步至远端，未强制覆盖工作区。本任务没有再次推送GitHub；远端当前分支相对原`origin/main`领先此修复提交。运行期间不要拉取或修改训练源码。

已核验：

- 本机全量420项测试通过；远端65项固定条件、训练恢复、参数及部署测试通过。
- 服务器重建可以确定性复现；六类外生输入的不同取值数均为1，32/32物理布局不同，所有布局原样保留。
- 固定轨迹摘要两端相同：`4a5ef738795383d5d4be03b499c95eabc71e68d2b4829f896cd52dcbcbf8c5c7`。
- 远端部署库摘要：`0322b649c2a53ce5eb028b1de0f536cf76c795372ddce0ceb7d1b9adb6af26ca`。它与Windows库的摘要不同，仅源于重算诊断中的微小浮点差异：逐字段比较最大绝对差约`4.44e-15`，全部满足`atol=rtol=1e-12`；部署、轨迹和探针离散标志均相同。
- 远端原有可编辑包指向另一份checkout。运行显式设置`PYTHONPATH=src`，已确认导入当前`/home/zf/zf/agent_orchestration/src/agent_orch`，未修改另一份项目或已有Conda环境。

远端使用Python3.12.15、PyTorch2.14.0+cu126、pandas3.0.6；原本机为Python3.11.16、PyTorch2.13.0、pandas3.0.5。numpy2.4.6、scipy1.17.1、networkx3.6.1、PyYAML6.0.3和gymnasium1.3.0相同。用户确认使用既有环境，因此应记录版本差异，**不能把与Windows曲线的差异全部归因于固定外生条件**，也不承诺长期随机训练逐位一致。

## 4. 查看进度

登录服务器后：

```bash
cd ~/zf/agent_orchestration
tmux attach -t agent_fixed_n20a20_s0
# Ctrl+B，再按D：只脱离会话，不中断训练。

# 或直接追踪续跑日志（含进度条的回车刷新；查看动态条优先使用tmux）：
tail -f results/remote_logs/composition_fixed_n20_a20_s0_progress_resume_20261016T0500.log

# 结构化状态与最近训练记录：
cat results/composition_n20_a20_32slots_fixed_base_s0/training_status.json
tail -n 1 results/composition_n20_a20_32slots_fixed_base_s0/training_history.jsonl
```

前5轮已确认写出约25MB检查点，阶段为composition，部署动作数为0；每轮4,128个模型选择子动作。初步每轮约26秒，只是启动阶段观测，不是整体耗时或收敛保证。

## 5. 中断恢复

先确认原进程已结束，不能对同一输出目录并发启动。然后保持以下全部参数，添加`--resume`：

```bash
cd ~/zf/agent_orchestration
conda activate agent_orchestration
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u scripts/run_slot_joint.py \
  --scenario results/slo_calibrated_main_20261014/servers_20_applications_20.yaml \
  --warmup-bundle results/warmup_n20_a20_32slots_fixed_base_20261016 \
  --output results/composition_n20_a20_32slots_fixed_base_s0 \
  --fixed-exogenous --slots 32 \
  --updates 1000 --route-updates 1000 --deploy-updates 0 \
  --mapping-samples 128 --seed 0 --trajectory-seed 101 --warmup-seed 20261014 \
  --device cpu --workflow-engine array --workflow-workers 1 --resume
```

恢复前可再加`--check-resume`做只读核验。仅完成PPO更新后，程序还会进行原训练轨迹上的最终评价和基线计算；不要只看到`training_status.json`训练更新完成，就认定整个脚本已退出。以末尾`Completed ...`及进程退出状态共同确认。

## 6. 进度条修改与已核验续跑（2026-10-16 13:00，北京时间）

按用户要求，不再逐槽打印`physical_routing`、`analytical`、`workflow`阶段行。源码提交`166330c`改为默认显示单行进度、轮次、当轮完成槽数与ETA，保留每轮一次指标和分项计时汇总；阶段详情仍在结构化状态文件中。续跑ETA使用检查点最近十轮采集及更新耗时初始化，再随实际耗时调整，不把已完成轮数当作新进程的速度。

本轮操作及核验记录：

- 第54轮检查点完整保存后发送SIGINT，未丢弃已完成轮次；刚开始的第55轮从保存的随机数状态重放。
- 原检查点、manifest、两种历史、状态及旧日志备份到运行目录下`reporting_restart_20261016T045947Z/`。原日志未覆盖。
- `--resume --check-resume`通过，且核验前后运行文件哈希不变；之后使用原完整指令，仅添加`--resume --progress-bar`，仍为1000轮模型选择、32槽、array单进程。
- 续跑PID为`2777718`，会话仍为`agent_fixed_n20a20_s0`；PID仅是此次启动记录，当前状态须重新查询。
- 核验时新检查点已保存55轮、正在第56轮；前54轮历史逐项相同，JSONL旧前缀逐字节保留，无重复轮次，manifest逐字节相同。Adam步数从20,898递增到21,285，未重置优化器；输入摘要、PPO配置、阶段计划、固定部署库与参数来源均与原检查点一致。
- 新日志已显示ETA且没有逐阶段刷屏。当时估计剩余约6小时57分，仅为动态估计，不含训练后评价。
- 本机428项已跟踪测试通过；远端相关测试63通过、1项因CUDA可用而跳过。原有未跟踪`test_physical_routing_probe.py`依赖已不存在的旧模块，导致不加排除的全目录收集失败；本轮未修改该旧探针。测试中另有既有NetworkX提示及Windows SwanLab退出编码警告，不影响测试退出码。

完整暂停、续跑与核验收据在上述备份子目录的`pause_receipt.json`、`resume_check.txt`、`resume_receipt.json`、`verification.json`。本轮不修改算法或科学参数，不引入教师/蒸馏，不自动推送GitHub。
