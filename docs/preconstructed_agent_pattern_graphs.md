# 四类 Agent 应用概率图与负载预构建

## 1. 构建口径

主实验采用四类应用：检索增强交互、事务型工具 Agent、深度研究和编码 Agent。每类应用由一个应用级概率有向无环图描述。图中的概率调用表示 Agent LLM 根据当前执行结果选择后续处理路径；同一并行组中的分支在被选择后同时执行，并在后继 LLM 节点接收各分支结果。循环式推理和重试按照有限轮次展开为有向无环执行结构。

图中 LLM 节点标注条件中位输入和输出 token 数 `P50(P, O)`，无状态服务节点标注低负载平均处理时间。每个 LLM 节点的完整输入、输出长度分布由 P50 和 P95 共同确定，并记录在工作负载配置文件中。所有数值均为理论仿真的参考配置，用于覆盖短请求、工具密集请求、并行研究请求和长上下文编码请求，不作为生产系统的逐请求测量结果。

## 2. 长度分布与上下文累积

输入长度、输出长度和工具返回长度采用截断对数正态分布。给定中位数 `x50` 和第 95 百分位数 `x95`，对数尺度参数为

\[
\mu=\ln x_{50},\qquad
\sigma=\frac{\ln(x_{95}/x_{50})}{1.645}.
\]

截断前的分布均值为 $\exp(\{\mu+\sigma^2/2\})$；实际平均 token 数在上下文窗口截断后由该分布计算。

采样结果限制在模型上下文窗口内。第一个 LLM 节点的输入由用户请求和固定系统指令组成。后续 LLM 节点输入按照

\[
P_{a,i+1}=
\min\!\left\{
C_m,
P_{a,i}+O_{a,i}+D_{a,i}+A_{a,i+1}
\right\}
\]

递增，其中 `D` 为当前阶段各无状态服务返回内容折算的 token 数，`A` 为下一阶段新增的角色指令、工具描述或格式化开销。并行分支汇聚时，`D` 和分支 LLM 输出取所有已完成分支之和。首个 LLM 输入、各 LLM 输出和无状态服务返回量按照相应分布采样，后续 LLM 输入由上述递推关系计算；节点级输入 P50/P95 用于校准递推结果。

## 3. 应用级负载锚点

| 应用 | 初始请求或请求级统计锚点 | 图内 LLM 调用特征 | 主要无状态服务返回规模 |
|---|---|---|---|
| 检索增强交互 | 用户请求采用 Chatbot-Single 输入分布：P50=27、P95=391；最终输出采用 P50=225、P95=1024 | 1--3 次 LLM 调用，检索后上下文约 1--2k tokens | Dense retrieval：P50=700、P95=1800 tokens；hybrid search：P50=1000、P95=3000 tokens |
| 事务型工具 Agent | 单次工具决策调用以 BFCL 的平均 735 个输入和 34 个输出 tokens 为中心；采用输入 P50=512、P95=2048，输出 P50=24、P95=80 | 3 个主要 LLM 阶段，输出短、输入随 API 结果逐步增长 | DB/API 返回：P50=160、P95=700 tokens |
| 深度研究 | 请求级累计输入以 Deep-Research-Compound 的 P50=10807、P95=29282 为校准范围；累计输出以 P50=3148、P95=7525 为校准范围 | 规划、并行检索与撰写、反思、可选细化和总结；后期上下文约 10--30k tokens | 每个搜索分支：P50=900、P95=2800 tokens |
| 编码 Agent | 初始任务输入以 Compound 请求 P50=1097、P95=2767 为起点；累计输出以 P50=4417、P95=6452 为参考；后期上下文扩展至约 15--32k tokens | 代码读取与符号搜索并行，随后编辑、测试、审查，并以 0.40 概率执行一次展开后的调试重试 | 文件内容：P50=1200、P95=5000；测试日志：P50=400、P95=1600 tokens |

四类应用的图结构、概率选择、并行分支和节点级 token 特征统一由 `data/preconstructed_agent_workloads.yaml` 定义；Agentix 的 BFCL 工作负载、Agentic AI Workload Characteristics 与 TraceLab 作为该预构建配置的设计依据。

## 4. 四类概率图

### 4.1 检索增强交互

入口 LLM 在直接回答、稠密检索和混合搜索之间进行概率选择，概率分别为 0.15、0.55 和 0.30。直接回答分支的输入、输出 token 中位数为 $(140,320)$，相应 P95 为 $(650,1024)$，用于表示短输入、长输出请求；两类检索结果由证据整合 LLM 处理，并进入检索增强生成 LLM。该图形成直接生成、稠密检索增强和混合检索增强三种 pattern flows。

### 4.2 事务型工具 Agent

入口 LLM 在数据库查询、外部 API 调用和数据库到 API 的工具链之间进行概率选择，概率分别为 0.55、0.30 和 0.15。验证 LLM 根据返回状态以 0.25 的概率调用补偿服务，随后由最终 LLM 生成结果。该图覆盖单工具调用、工具链和执行后补偿。

### 4.3 深度研究

规划 LLM 同时发起两个搜索分支，各分支由独立撰写 LLM 形成阶段结果，并在反思 LLM 处汇聚。反思 LLM 以 0.65 的概率直接进入总结，以 0.35 的概率同时执行两个定向检索分支并进行细化，最后进入总结 LLM。

### 4.4 编码 Agent

分析 LLM 同时调用代码读取和符号检索服务，诊断 LLM 根据结果生成修改方案，随后依次执行编辑和测试。审查 LLM 以 0.60 的概率直接完成请求，以 0.40 的概率执行一次日志读取、调试、再次编辑和测试，最后由完成 LLM 返回结果。

## 5. 图文件

- 可编辑源文件：`latex/figures/agent_pattern_graphs_preconstructed.drawio`
- 四幅预览图：`latex/figures/agent_pattern_rag.png`、`agent_pattern_transactional.png`、`agent_pattern_deep_research.png`、`agent_pattern_coding.png`
- 可缩放版本：对应的 `.svg` 文件

## 6. 参考来源

- Agentix: https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_luo_prepub.pdf
- Agentic AI Workload Characteristics: https://arxiv.org/pdf/2605.26297
- AgentSysBench: https://arxiv.org/pdf/2608.15127
- TraceLab: https://tracelab.cs.washington.edu/
