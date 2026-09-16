# LLM 宏观稳态服务模型验证

本实验使用 LLMServingSim 2.0 检验论文中的 Roofline 工作量、稳态有效并发度、KV 驻留边界和宏观请求时延趋势。参考配置为单实例 Llama-3.1-8B/RTX4090，采用 BF16、2048 token budget、128 条最大序列、512 token prefill chunk，并关闭 prefix caching。解析模型用于描述长期稳态下的平均服务特征，仿真器用于回放请求流并比较负载变化趋势。

## 运行环境

本地激活实验环境并安装项目：

```powershell
conda activate agent-orch
python -m pip install -e ".[plot]"
$env:PYTHONPATH="src"
```

远端默认使用 `zf@192.168.234.128` 和容器 `servingsim_docker`。SSH 必须已经配置免密登录。

## 分阶段运行

```powershell
python scripts/run_llmservingsim_validation.py prepare
python scripts/run_llmservingsim_validation.py sanity
python scripts/run_llmservingsim_validation.py instrument-check
python scripts/run_llmservingsim_validation.py run-calibration --workers 1
python scripts/run_llmservingsim_validation.py calibrate
python scripts/run_llmservingsim_validation.py run-queue --workers 1
python scripts/run_llmservingsim_validation.py analyze
```

已有成功结果默认跳过，可安全地从中断处继续。使用 `--force` 才会重新运行已有任务。虚拟机只有约 4 GB 内存，`--workers` 即使设置得更大也会被限制为 2；正式运行推荐保持为 1。

也可以顺序执行全部阶段：

```powershell
python scripts/run_llmservingsim_validation.py all --workers 1
```

## 输出

本地输出位于 `results/llm_queue_validation/`：

- `provenance.json`：模拟器提交、配置与插桩哈希；
- `service_predictions.csv`：处理时延校准和留出测试；
- `capacity_observations.csv`：各工作负载组成的饱和吞吐；
- `capacity_predictions.csv`：仿真容量与三种诊断模型的容量对比；
- `calibration.json`：有效计算速率、带宽和固定并发度；
- `queue_predictions.csv`：解析稳态模型与仿真的逐次运行结果；
- `queue_error_summary.csv` 和 `validity.json`：误差与有效性判定；
- `instrumentation_check.json`：排队指标插桩的无扰动验证；
- `validation_report.md`：实验结论及论文模型修改建议；
- `figures/`：PNG 和 PDF 图。

远端临时文件位于 LLMServingSim 的 `outputs/queue_validation/`，该目录已被其 `.gitignore` 忽略。排队时延插桩只改变 CSV 指标的赋值时机，不改变请求调度、批处理或完成时间。
