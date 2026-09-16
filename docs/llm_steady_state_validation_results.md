# LLM 稳态服务模型：修订与验证结果

本文记录当前宏观稳态 LLM 服务模型的验证口径。主模型使用 mixed/decode iteration 服务曲线、KV 与序列驻留上限、稳态并发固定点以及首次准入等待近似。

## 当前模型

对实例 \(\ell\)，设 \(\nu\) 为稳态 decode 并发度，\(C^{\rm ch}\) 为 prefill chunk 长度。一次调用的服务需求为

\[
D_{a,i,\ell}^{\rm svc}(\nu)=D_{a,i,\ell}^{\rm pre}(\nu)+D_{a,i,\ell}^{\rm dec}(\nu),
\]

其中 prefill 由 mixed iterations 推进，后续输出 token 由 decode iterations 推进。实例的稳态容量由服务曲线和可驻留运行集共同确定：

\[
\mu_{\ell}^{\rm ana}=
\max_{1\leq \nu\leq C_{\ell}^{\rm run}}
\frac{\nu}{\bar D_{\ell}(\nu)}.
\]

稳态并发度由 Little 定律的一维固定点得到。首次准入等待由 prefill chunk 流的平均排队负载给出，LLM 调用指标为

\[
\mathrm{TTFT}_{a,i,\ell}=W_{\ell}^{\rm adm}+D_{a,i,\ell}^{\rm pre}(\bar B_{\ell}),
\]

\[
\mathrm{TBT}_{a,i,\ell}=
\frac{D_{a,i,\ell}^{\rm dec}(\bar B_{\ell})}{O_{a,i}-1},
\qquad
T_{a,i,\ell}^{\rm LLM}=\mathrm{TTFT}_{a,i,\ell}+(O_{a,i}-1)\mathrm{TBT}_{a,i,\ell}.
\]

\(C_{\ell}^{\rm run}\) 表示 KV cache 和最大序列数共同确定的常驻运行集上限，不作为传统独立服务台数量。实例利用率、KV 容量和固定点稳定性共同决定过载约束。

## 验证口径

LLMServingSim 2.0 用于独立验证服务曲线的负载趋势、配置排序和稳定容量边界；主编排仿真使用解析模型。验证覆盖不同输入输出长度、不同 workload composition、不同负载水平以及单 GPU 模型配置。

解析模型的验证重点为：

- 负载提高时 TTFT、TBT 和完整响应时延的单调性；
- 不同模型—GPU 配置的相对容量和相对时延排序；
- KV 驻留上限与最大常驻序列数的数量级一致性；
- 低负载稳定、接近容量时延和 SLO 违约率上升、过载时容量约束生效。

绝对时延误差用于界定解析模型的适用范围，不将解析模型表述为请求级调度器或精确事件仿真器。

## 当前正式输入

正式场景为 `configs/benchmarks/main_abilene.yaml`，负载档位为 `data/processed/load_levels.json`。负载档位由当前场景、当前宏观服务模型和固定 Greedy 参考部署重新生成，两个文件保存同一场景路径和 SHA256。

历史 LLMServingSim 验证脚本和旧模型结果保留在验证目录中，仅用于复现历史对照，不进入当前主实验入口。
