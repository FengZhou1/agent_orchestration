# 概率扩展式Agent应用图生成器

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: generator implementation and bounded validation
- Origin Date: 2026-10-13
- Version: probabilistic-agent-pool-v1
- Scope: 合成应用池、结构/概率/长度及接口验证；不代表PPO或真实服务性能结果

## 生成方法

入口为一个LLM请求。节点预算从声明的右偏分布抽取，在预算内递归组合继续推理、串行/并行工具、互斥片段、有限重复和提前响应。后续推理也可以分支、汇聚和提前结束，工具之前可继续推理；每次重复创建新节点，不产生环。入口及前两次推理后的终止概率偏向短执行，使应用等权、pattern概率加权的执行LLM数量中位数为2。概率范围是合成设计，不是经验观测或四张模板。

含工具图为2–15个LLM、1–30个工具、深度2–30；用户确认少量纯推理图可为1个LLM、0工具、深度1，示例比例5%。深度按路径上的实际请求节点计。局部并行上限4，不限制整图层宽。图大小是联合约束，不独立强行补边。全图节点和每个执行节点分别统计。

串行/并行控制取完整组合，互斥取一个选项；只有互斥概率归一化。构建子片段前分配pattern预算，示例每应用最多32个，生成后不删pattern。所有节点都必须在正概率pattern内。控制树会独立重放并与实际pattern逐项比较，检查并行依赖及概率乘积。直接响应不执行后继部分。

图以节点—边紧凑存储，避免枚举嵌套并行的指数数量路径。旧链列表仍可读取；关键路径按拓扑动态规划求取，先对每个具体执行映射取最长依赖，再对映射和pattern加权。

## 联合token与通信

输入目标P50/P90为2,048/8,192，范围256–16,384；输出为200/1,200，范围16–2,048；工具文本范围48–5,000，合成目标256/1,500。使用截断右偏提案分布，实际分布以合法生成后的统计为准。

输出和工具文本先抽样。沿依赖用消息标识累计历史，汇聚时共享消息只计一次。生成某LLM的新增内容之前，按它将进入的所有后续LLM输入计算重复prefill系数，为所有pattern保留窗口和剩余累计预算；然后条件抽样非负新增输入。必要历史不可容纳时重采样整组输出/工具候选，但保持图、概率和所有pattern不变，不在评价时缩短输出。

每个pattern逐一检查输入加输出≤32,768、完整执行累计≤128,000；逻辑预算不扣前缀缓存。根输入和每个节点的新增消息在相同上游历史中保持一致。详细pattern长度保留，性能模型使用按经过节点的pattern概率条件加权的输入代表值。模型间不添加无来源的长度倍率；本例是抽象token长度，不调用真实tokenizer。

提案参数可在生成阶段按最终长度分位数预校准，使用固定结构和共同随机数。该过程只调整合成长度分布，不训练策略、不使用教师、不计算PPO成绩或SLO成功率。最终参数、校准步骤、拒绝数和实际分位数一并保存。

每个工具请求保存可见文本长度与额外非文本MB；网络传输量与逻辑上下文分开。中间边采用源节点输出文本及可选非文本载荷；末级返回按对应pattern最终LLM输出长度保存。该通信设置不宣称复现真实消息协议，网络时隙单位问题仍是独立待办。

## 工具库、应用属性与去重

工具类型为可部署服务，节点为一次请求。同工具可重复出现在多个节点。共享候选以Zipf权重选择，私有候选只允许所属应用使用；不强制每个应用具有共享工具。所有库中类型至少被一个正概率pattern使用。共享候选不保证在每个子集都被多个应用使用，因此另报告实际跨应用复用类型与未复用类型。

示例工具每副本1CPU核，主存0.5–3GB，单核率2–8请求/s；范围借鉴前作，但秒单位与分布是明确的合成假设。同类工具在各服务器处理率相同。SLO、质量、接入支持及基准率由显式配置独立生成，质量仅为代理分数。硬件、模型与价格从指定基础场景读取；基础场景原应用和工具被替换，不把遗留profile当作新测量。

应用身份按无节点/应用编号的属性图与全局参数比较，包含结构、工具绑定、pattern概率、长度和通信量。发生相同配置时重新生成长度，而非只换名字；不按拓扑去重。拓扑数忽略资源/绑定/token，按节点类型和依赖用图同构检查，哈希只用于候选分桶。

应用池只生成一次，按结构规模分层、按原层比例打散；应用数场景取固定排列的嵌套子集。工具数实验读取已保存的池，仅改变绑定，保留图、概率、token、SLO、质量和到达率；默认使用统一工具资源配置以隔离种类/复用作用。报告实际活跃工具数，不计闲置目录项。接入分布属于基础网络；更换服务器集合时必须显式重新映射接入，不能直接将旧入口ID用于新拓扑。

## 运行

```powershell
conda activate agent-orch
$env:PYTHONPATH='src'
python scripts/generate_agent_pool.py `
  --infrastructure configs/benchmarks/main_abilene.yaml `
  --config configs/agent_generator.yaml `
  --output results/generated_agent_pool --subsets 20,50,100,200

python scripts/rebind_agent_tools.py `
  --pool results/generated_agent_pool/pool.json `
  --infrastructure configs/benchmarks/main_abilene.yaml `
  --tool-counts 20,50,100,200 --output results/generated_tool_scale
```

输出目录必须不存在。pool.json保存控制树、详细pattern消息长度、固定属性和工具库；manifest保存配置、摘要、各子集ID与最终统计；applications_N.yaml/tools_H.yaml可由生产ScenarioLoader读取。输出是固定工作负载，不保存部署或教师动作，也不另外生成评价轨迹。

生成器不调用CapacityPlanner，不以贪心满足全部SLO为筛选条件。时隙模块另处理到达率与可用资源，token与概率跨槽固定；各场景独立训练，仍采用既定三阶段PPO和容量比例路由。当前配置是生成/接口验收示例，正式硬件、费用、SLO、质量和负载需另冻结；没有正式算法效果声明。

## 来源边界

完整结果、命令和适用边界见[2026-10-13验收记录](agent_generator_validation_2026-10-13.md)。

- inference-perf的tool-loop、并行、上下文与固定种子思路：[固定文档commit](https://github.com/kubernetes-sigs/inference-perf/blob/af6a7c2cba7b34f2567d1bbc3f3cfc514cb8f96b/docs/synthetic_agentic.md)。本文不导入它的程序、内容主题、调用工具或真实LLM。
- 博士第四章：概率调用图的逐步扩展、服务类型绑定与复用。该图范围不作为本文Agent图分布的实测依据。
- CPU/内存/服务率范围参考Energy；概率、计数分布、pattern上限、校准次数、SLO/质量示例和非文本载荷均是自设合成参数，完整值在configs/agent_generator.yaml。
