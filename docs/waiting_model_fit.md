# 等待时延模型拟合结果

本实验专门拟合 LLMServingSim 2.0 中的首次准入等待时延。`waiting_s` 表示请求从到达到首次进入可执行集合之间的等待时间；它不包含已经进入连续批处理后的 prefill 和 decode 执行时间。负载点由到达率除以该 workload composition 的饱和处理率得到，记为

\[
\rho=\frac{\Lambda}{\widehat\mu^{\rm sat}}.
\]

当 \(\rho\geq 1\) 时，系统不处于有限稳态，因而不将该点用于有限等待时延的拟合。

## 候选模型

原有模型采用多类别 M/G/1 的 Pollaczek--Khinchine 近似。它使用 prefill 工作量估计利用率，但没有反映 continuous batching 下的 decode 竞争，因此在本实验中作为基线。

全局拥塞曲线采用

\[
W(\rho)=w_0+\alpha\,\bar t^{\rm iter}\frac{\rho^\nu}{1-\rho}.
\]

最终模型为

\[
\boxed{
W_c(\rho)=w_{0,c}+\alpha_c\,\bar t^{\rm iter}
\left(\frac{[\rho-\rho_{0,c}]^+}{1-\rho}\right)^\nu,
}
\]

其中，\(c\) 表示当前工作负载组成；\(w_{0,c}\) 表示低负载下由下一 iteration 调度边界产生的基础等待；\(\rho_{0,c}\) 表示该组成开始出现明显拥塞增长的负载水平；\(\alpha_c\) 为无量纲拥塞尺度；\(\nu\) 为所有组成共享的增长指数；\(\bar t^{\rm iter}\) 为对应实例的平均 iteration 时长。训练时按组成和负载点的多种子均值拟合参数，种子级结果只用于评估随机波动。

## 结果

| 模型 | 原始运行 WAPE | 原始运行 median APE | 组成–负载均值 WAPE | 组成–负载均值 median APE | 组成–负载均值 P95 APE |
|---|---:|---:|---:|---:|---:|
| 原 PK 近似 | 80.36% | 58.52% | 78.49% | 53.44% | 94.32% |
| 剩余 iteration + PK | 87.74% | 78.18% | 227.87% | 82.18% | 72.99% | 175.86% |
| 仅组成基线 | 78.88% | 29.82% | 83.01% | 31.72% | 95.19% |
| 全局拥塞曲线 | 79.65% | 17.01% | 79.18% | 14.20% | 111.71% |
| 组成感知拐点模型 | 58.95% | 12.91% | **2.61%** | **7.05%** | **25.51%** |

原始运行指标包含单次 200 请求窗口的随机到达波动，特别是 AGENT 在 \(\rho=0.85\) 附近已经接近稳定边界，三个种子之间出现了显著差异。因此，论文中的解析式应解释为稳态平均等待时延模型，验证应以多个种子按组成和负载聚合后的均值为主；单次短窗口不能作为确定性逐请求预测目标。

拟合参数保存在 `results/llm_queue_validation_v2/waiting_fit/waiting_fit.json`，其中：

- AGENT: \(w_0=5.943\) ms，\(\alpha=11.997\)，\(\rho_0=0.582\)；
- CS: \(w_0=10.486\) ms，\(\alpha=0.00245\)，\(\rho_0\approx0\)；
- DS: \(w_0=9.027\) ms，\(\alpha=0.3445\)，\(\rho_0=0.473\)；
- MIX: \(w_0=10.875\) ms，\(\alpha=0.01945\)，\(\rho_0\approx0\)；
- 共享指数：\(\nu=1.278\)。

由 iteration 剩余时间和 PK 项直接相加得到的理论候选式在组成–负载均值上的 WAPE 为 82.18%，因此不采用。它将 iteration 边界等待与 prefill 队列等待简单相加，未能描述 continuous batching 中两者共享服务状态的耦合。

## 使用边界

该式适合宏观优化中根据实例的当前组成和归一化负载估计平均准入等待。它不模拟 vLLM 的每一次调度，也不替代 LLMServingSim 的请求级回放。\(\widehat\mu^{\rm sat}\) 可由公开 simulator profile 和固定配置的容量校准得到；在尚未完成容量校准的训练阶段，应将该等待项限制在 \(\rho<1\) 的稳定区域。

## 复现实验

```powershell
conda activate agent-orch
$env:PYTHONPATH="src"
python scripts/fit_waiting_model.py `
  --v2 results/llm_queue_validation_v2 `
  --output results/llm_queue_validation_v2/waiting_fit
```

主要产物包括 `waiting_curve_means.csv`、`waiting_predictions.csv`、`waiting_cv.csv`、`waiting_fit_curves.pdf` 和 `waiting_fit_report.md`。

该建模取 continuous batching 的状态相关服务率作为背景：异构 LLM 工作负载研究将 prefill--decode contention 建模为状态相关的多类排队网络；LLM agent 排队研究则强调 work-conserving 调度对吞吐稳定性的作用。等待项采用了这两类工作的宏观思想，但参数由当前公开 LLMServingSim profile 的实验曲线校准。
