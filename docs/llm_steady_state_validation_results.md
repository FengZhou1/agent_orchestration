# LLM 稳态服务模型：修订与验证结果

本文记录 2026-09-14 对 LLM 稳态服务模型的四项修订、逐实例验证，以及修订向宏观编排模型的传播结果。逐实例验证的完整数值产物在 `results/llm_priority_fixes_validation/`，宏观重标定的产物在 `results/baseline_revised/`。

## 一、当前模型

对固定部署配置的实例 \(\ell\)，一次 iteration 同时承载常驻 decode 序列与至多一个 prefill chunk。设 \(\nu\) 为常驻序列数，\(C\) 为 chunk 长度，则一条 \((P,O)\) 调用在该实例上的服务曲线为

\[
\bar D_{\ell}(\nu) = D^{\rm pre}_{\ell}(\nu) + D^{\rm dec}_{\ell}(\nu),
\qquad
{\rm TTFT} = D^{\rm pre}_{\ell}(\nu),\;
{\rm TBT} = \frac{D^{\rm dec}_{\ell}(\nu)}{O-1}.
\]

驻留上限只表示 KV 与序列槽位对常驻集合的约束，不作为排队服务台数：

\[
C_{\ell}^{\rm run} = \min\left\{ N^{\max},\;
\frac{(1-\delta_{\ell}^{\rm KV}) K_{\ell}^{\rm KV}}{\bar g_{\ell}} \right\}.
\]

由服务曲线给出实例的宏观调用容量与利用率：

\[
\mu_{\ell}^{\rm ana} = \max_{1 \le \nu \le C_{\ell}^{\rm run}}
\frac{\nu}{\bar D_{\ell}(\nu)},
\qquad
\rho_{\ell}^{\rm LLM} = \frac{\Lambda_{\ell}^{\rm LLM}}{\mu_{\ell}^{\rm ana}}.
\]

稳态常驻序列数由 Little 定律的一维不动点给出：

\[
\bar B_{\ell} = \Lambda_{\ell}^{\rm LLM}\,\bar D_{\ell}(\bar B_{\ell}).
\]

稳态时延只由 prefill 与 decode 的处理需求决定，不再引入准入排队等待项：

\[
{\rm TTFT}_{a,i,\ell} = D^{\rm pre}_{a,i,\ell}(\bar B_{\ell}),\qquad
{\rm TBT}_{a,i,\ell} = \frac{D^{\rm dec}_{a,i,\ell}(\bar B_{\ell})}{O_{a,i}-1},
\qquad
T_{a,i,\ell} = {\rm TTFT}_{a,i,\ell} + (O_{a,i}-1)\,{\rm TBT}_{a,i,\ell}.
\]

当 \(\rho_{\ell}^{\rm LLM} \ge 1\)、不动点不收敛或 \(\bar B_{\ell} \ge C_{\ell}^{\rm run}\) 时，实例记为过载，写入容量约束代价，而不是返回一个发散的平均等待时间。

## 二、逐实例独立验证

验证数据为 LLMServingSim 2.0 的请求级轨迹，覆盖 8 个模型—GPU 配置、36 个稳态负载点和 63 个负载组成点。所有观测值由原始请求记录重算（去掉前 20% 与后 20%），不复用仿真器汇总列。`legacy` 变体复现修订前的公式，仅作为消融基线。

| 数据集 | 指标 | 修订前 WAPE | 修订后 WAPE |
|---|---|---|---|
| 稳态混合 | TTFT | 53.87 | 1.06–1.23 |
| 稳态混合 | 完整响应 | 2.47 | 0.41 |
| 负载组成 | TTFT | 6.94 | 0.62–0.77 |
| 负载组成 | 完整响应 | 0.79 | 0.20–0.24 |

随负载变化的单调一致率由 0.875 提高到 1.000，每个配置的预测—观测秩相关由 0.925 提高到 1.000。按饱和速率排序配置时，共享硬件效率（`revised_global`，\(\eta^{\rm cmp}=0.41\)、\(\eta^{\rm bw}=1.0\)）的秩相关为 1.000，峰值速率与逐配置效率为 0.738。

## 三、驻留上限验证

饱和试验本身就能判定 \(C_{\ell}^{\rm run}\)。同一条轨迹上，常驻集合是“首次准入至完成”的区间（到达至完成还包含排队，不能作为对照），据此重算峰值与均值常驻序列数，并在该轨迹的经验 \((P,O)\) 组成上求驻留上限。

| 配置 | 观测峰值常驻 | 观测均值常驻 | 模型 \(C_{\ell}^{\rm run}\) | 峰值/上限 |
|---|---|---|---|---|
| qwen3-14b-h20 | 132 | 127.99 | 128 | 1.03 |
| qwen3-14b-l20 | 132 | 127.81 | 122 | 1.08 |
| qwen3-32b-h20 | 133 | 127.83 | 128 | 1.04 |
| qwen3-32b-2xl20 | 133 | 127.84 | 119 | 1.12 |
| qwen3-4b-a10 | 132 | 127.20 | 120 | 1.10 |
| qwen3-4b-l20 | 133 | 127.99 | 128 | 1.04 |
| qwen3-8b-h20 | 132 | 127.99 | 128 | 1.03 |
| qwen3-8b-l20 | 133 | 127.99 | 128 | 1.04 |

驻留上限比观测峰值低 3%–12%，方向保守，量级一致。

## 四、宏观模型的传播

驻留上限与调用容量的分离改变了宏观容量，因此以旧模型标定的实验输入全部需要重标定。以 `main_abilene_revised.yaml` 的参考部署（greedy）重新求参考容量：

| 量 | 修订前 | 修订后 |
|---|---|---|
| 参考稳定容量 | 0.0265 req/s | 0.0496 req/s |
| 参考到达缩放 | 6.61 | 12.39 |
| 限制资源 | KV：qwen3-32b-h20 | LLM 调用率：qwen3-32b-h20 |

SLO 阈值按低负载参考部署的 P95 重新标定，得到 `configs/benchmarks/main_abilene_revised.yaml`：TTFT 阈值下降约 12%–26%，TBT 阈值下降约 12%，deadline 下降约 13%–27%。负载档位按新参考容量重算为 `data/processed/load_levels_revised.json`。

在该配置与档位下重跑多负载强度 baseline（`results/baseline_revised/`）：

| 策略 | 低负载时延 | 过载时延 | 低→过载 SLO 满足率 |
|---|---|---|---|
| greedy | 69.5 s | 77.3 s | 1.00 → 0.70 |
| static | 56.1 s | 89.4 s | 1.00 → 0.50 |
| equal | 78.4 s | 80.0 s | 0.66 → 0.66 |
| least_load | 77.4 s | 77.7 s | 0.67 → 0.66 |

所有策略的时延随负载单调上升；greedy 的 SLO 满足率随负载下降、goodput 在过载点回落，符合 SLO 加权 goodput 的预期形状。修订前的 `results/baseline_mgc/` 在 `static` 与 `greedy` 上出现 2021 s 与 168–280 s 的伪过载坍塌，该现象由已删除的准入等待项产生，不再复现。

## 五、局限与待决项

1. 参考试验使用的是 A10、L20、H20 的 roofline 合成 profile（`results/roofline_qwen_matrix/profiles`），不是实测内核。因此第二节的绝对时延比较的是两套 roofline 记账，验证侧重于消融、负载响应方向、配置排序与稳定边界，这些量对共同的效率因子不敏感。
2. 饱和容量仍被高估 1.3–4 倍。服务时延希望更快的有效硬件，饱和容量希望更慢的有效硬件，单一效率常数无法同时匹配两端；残差来自服务曲线随 \(\nu\) 增长的斜率，而不是驻留上限或稳定性判据。
3. `data/processed/load_levels_revised.json` 与 `configs/benchmarks/main_abilene_revised.yaml` 是当前正式实验输入。
4. 旧的准入等待拟合脚本与旧标定文件已从项目中移除。

## 六、复现命令

```powershell
conda activate agent-orch
cd C:/Users/01/Desktop/agent/experiments/agent_orchestration

python scripts/validate_llm_priority_fixes.py --output results/llm_priority_fixes_validation

python scripts/calibrate_load_levels.py `
  --scenario configs/benchmarks/main_abilene_revised.yaml `
  --output data/processed/load_levels_revised.json

python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene_revised.yaml `
  --load-levels data/processed/load_levels_revised.json `
  --slots 10 --seeds 0,1,2 `
  --policies static,equal,least_load,random,greedy `
  --output results/baseline_revised
```
