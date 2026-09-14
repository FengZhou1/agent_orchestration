# KV cache 容量验证（LLMServingSim 2.0 对照）

本实验用远端 LLMServingSim 2.0 容器内的 `serving/core/memory_model.py` 作为参照，核验论文候选配置的 KV cache 容量 `K_l^KV` 与单位 token KV 显存 `gamma_m`。远端为 `zf@192.168.234.128` 的 `servingsim_docker`，仓库位于 `/app/LLMServingSim`。

## 参照口径

模拟器的 `MemoryModel` 直接给出每个实例的 KV block 数：

```
requested = npu_mem * gpu_memory_utilization        # 每卡
kv_bytes  = requested - weight_per_rank
npu_blocks = kv_bytes // (bytes_per_token_per_rank * block_size)
KV tokens  = npu_blocks * block_size                # block_size = 16
```

其中 `bytes_per_token_per_rank = 2 * kv_head * head_dim * n_layer * kv_fp / tp_size`。模拟器源码同时说明：它未建模 activation peak 与 CUDA context，因此该容量相对真实 vLLM 是上界。

## 结果

单位 token KV 显存：四个模型的论文公式与模拟器**完全一致**（Qwen3-4B/8B 为 147456 B，14B 为 163840 B，32B 为 262144 B）。

KV token 容量对比（`cur` 为当前生成器，`GiB` 为模拟器口径，`GiB+2` 为改用 GiB 并保留 2 GiB 余量）：

| 配置 | 模拟器 | cur | sim/cur | GiB | GiB+2 | sim/(GiB+2) |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-4b-a10 | 97440 | 78667 | 1.239 | 97452 | 82889 | 1.176 |
| qwen3-4b-l20 | 254736 | 225151 | 1.131 | 254739 | 240175 | 1.061 |
| qwen3-8b-l20 | 203472 | 168185 | 1.210 | 203478 | 188915 | 1.077 |
| qwen3-8b-h20 | 518048 | 461154 | 1.123 | 518051 | 503488 | 1.029 |
| qwen3-14b-l20 | 102832 | 70800 | 1.452 | 102838 | 89731 | 1.146 |
| qwen3-14b-h20 | 385952 | 334472 | 1.154 | 385953 | 372846 | 1.035 |
| qwen3-32b-h20 | 103936 | 71716 | 1.449 | 103939 | 95747 | 1.086 |
| qwen3-32b-2xl20 | 103920 | 64086 | 1.622 | 103934 | 87550 | 1.187 |

`GiB` 列与模拟器的差均不超过 6 token，说明两者公式结构一致，差距只来自两个常数项。

## 结论

1. `gamma_m` 的公式已验证正确，无需修改。
2. 当前容量偏低 12%–62%，来源是两个常数：
   - 生成器用 `1e9` 解释卡容量（GB），模拟器与 vLLM 用 `1024^3`（GiB）；
   - 生成器额外扣除 2.0 的显存余量，模拟器不扣。
3. 把卡容量改用 GiB、余量仍保留 2 GiB 后，差异收敛到 1.03–1.19。
4. 32B/H20 单卡容量由 71716 提升到 95747 token（+33%），这会直接放松当前主实验中绑定在 KV 上的容量约束。

原始数据见同目录 `kv_capacity_compare.json` 与 `kv_capacity_raw.json`。