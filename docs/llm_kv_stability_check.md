# KV 稳定供给约束的一阶核验

上一次只验证了 KV 容量 `K_l^KV`（分母），本文核验稳定供给条件本身。数据取自既有 v1 试验（Llama-3.1-8B / RTX4090，prefix caching 关闭），不是专门为本条件设计的实验，因此只能算一阶核验。

## 被核验的条件

论文写法：

```
sum_{a,i} lambda_{a,i,l} G_{a,i}^m  <  (1 - delta_l) K_l^KV / tbar_l^iter
G_{a,i}^m = (1 + P/C) P / 2 + P O + (1 + O) O / 2
delta_l   = max_k (P_k + O_k) / K_l^KV
```

`G` 的单位是 token·iteration，乘 `tbar^iter` 后是 token·秒，再乘到达率即得稳态平均 KV 占用（Little 定律），因此该条件等价于"平均 KV 占用 < 可用 KV 容量"。

## 输入

- RTX4090 / Llama-3.1-8B，sim 口径 KV 容量 K = 54400 token；
- 实测平均服务时间取自 `results/llm_queue_validation/calibration.json`；
- 负载类别取 JITServe 四类：CS(93,318)、CC(1300,4458)、DS(1911,534)、DC(12223,3541)。

## 结果

| 组成 | sum r*G | tbar^iter (s) | delta | 预测容量 (rps) | 实测容量 (rps) | 预测/实测 |
|---|---:|---:|---:|---:|---:|---:|
| CS | 8.035e4 | 0.01677 | 0.0076 | 40.07 | 12.902 | 3.11 |
| DS | 1.168e6 | 0.01722 | 0.0449 | 2.584 | 1.017 | 2.54 |
| MIX | 6.241e5 | 0.01699 | 0.0449 | 4.899 | 1.793 | 2.73 |
| AGENT | 6.717e6 | 0.01700 | 0.2898 | 0.338 | 0.253 | 1.34 |

与既有三种容量诊断模型的对比（对四个组成做 log-log 拟合）：

| 模型 | 斜率 | R2 | 平均预测/实测 |
|---|---:|---:|---:|
| KV 稳定条件（本次核验） | 0.825 | 0.9912 | 2.43 |
| Current（compute-only，固定 B_eff=13） | 1.905 | 0.9495 | 1.07 |
| Empirical-Service | 3.617 | 0.9471 | 1.49 |
| Capacity-Matched（按组成重拟 B_eff） | 1.007 | 0.9995 | 1.00 |

## 判读

1. KV 稳定条件的**动态范围正确**（斜率 0.83），而 compute-only 模型把容量差异压缩了（斜率 1.91）。既有实验里 "Capacity-Matched" 需要为 CS 拟合出 B_eff=69、为 AGENT 拟合出 B_eff=4，这个巨大差异正是 KV 效应被吸收进单一拟合参数的结果。
2. KV 条件**系统性偏松约 2.4 倍**。这与它是必要条件而非充分条件一致：compute 等其它瓶颈会先绑定，真实容量低于 KV 上界。
3. 该条件目前**尚未被专门实验验证**。要成为可信结论，需直接测量 KV 边界（抢占起点与占用率），见下节设计。

## 待做的专门实验

sim 已具备所需可观测性：

- `serving/core/block_pool.py`：`used_blocks`、`get_num_free_blocks()`、`used_bytes()`、`usage()`；
- `serving/core/kv_cache_manager.py`：`npu_used_bytes()`；
- `serving/core/scheduler.py`：`num_preemptions`、`recompute_tokens`；
- `serving/__main__.py` 运行摘要已打印 Preemptions 与 Recomputed prompt tokens。

实验步骤：

1. 固定单实例（RTX4090/Llama-3.1-8B 与 Qwen3-14B/L20 各一组），prefix caching 关闭。
2. 按组成的固定 (P,O) 生成泊松到达 workload，到达率从远低于到远高于解析阈值扫若干档。
3. 每次运行记录：抢占次数、重算 token 数、吞吐、TTFT、端到端时延，并插桩逐 iteration 记录 `used_blocks`。
4. 定义 KV 边界的观测量：抢占率相对低负载基线的抬升点。
5. 对比三项：解析阈值 `(1-delta)K/(tbar*E[G])`、实测抢占起点、逐 iteration 平均占用 `used_blocks*block_size` 与解析值 `Lambda*E[G]*tbar`。
6. 判据：解析占用与实测占用的比值在稳定区应接近常数；抢占起点应不低于解析阈值，且两者比值在不同组成间保持一致。