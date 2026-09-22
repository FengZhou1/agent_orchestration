# 时间模型改造：从"压平的时间轴"到博士第四章式的时间轴

> 2026-09-21 起执行。**语义已锁定**，下面阶段 0 的三条决策不再讨论。

---

## 阶段 0：语义（已锁定）

| # | 决策 | 取值 | 含义 |
|---|---|---|---|
| **0.1** | period 与时隙的关系 | **① period ≡ slot** | 时隙是唯一时钟，也是唯一决策单元。`SimulationSpec.orchestration_period_s` **已从代码中删除**；场景 yaml 里遗留的这个键由加载器丢弃，不再生效 |
| **0.2** | 目标口径 | **累积** $J=\mathbb{E}[\sum_t u(t)]$ | 单周期绝对值在恒定 λ 下必然退化成静态优化；只有累积目标才让"适应时变"有适应对象 |
| **0.3** | episode 语义 | **一条时间轴**（长度 T 的决策步轨迹） | reset 回 $t=0$（或轨迹上的一个相位偏移）；训练/验证/评估用**同一条进程的不同实现**（`realization`），而不是留出不同的部署上下文 |

**由此推出的三条硬约束**（改动时不要违反）：

1. **不存在单独的编排周期符号**：$\Delta^{\rm org}$、索引 $k$ 在代码里已不存在；论文侧待删（阶段 5）。
2. **"累积目标"与"时变 λ"必须同时存在**：缺任何一个，部署子问题都退化成静态优化。
3. **特征宽度已变化**（状态新增 next-λ 与剩余冻结）：**所有旧 checkpoint 失效**，新实验必须新目录、新 seed 集。

---

## 阶段 1：时间模型 λ(t)

| # | 项 | 状态 |
|---|---|---|
| 1.1 | `ArrivalTrace` 支持时变轨迹 | ✅ `gaussian_burst_intensity(base_scale, burst_scale, period, sigma, phase, seed, jitter)`；`stationary_poisson_intensity` 保留 |
| 1.2 | 轨迹生成脚本 `scripts/build_arrival_traces.py` + manifest | ⬜ |
| 1.3 | 补 `stress_arrival_burst.yaml`（协议列了但未生成） | ⬜ |
| 1.4 | 落实 0.1 | ✅ 走 ①：成本一律按 `slot_seconds` 计价；`cost_per_period` → `cost_per_slot`（库 schema v2）；`orchestration_period_s` 移除 |

**验收**：同一条轨迹上 `simulator.current_arrival_rates()` 在不同 slot 返回不同强度 ✅（实测 8 个 episode 起始 λ 覆盖 0.0115–0.0268，轨迹范围 0.0109–0.0301）；平稳轨迹下与旧结果一致 ⬜（待回归）。

---

## 阶段 2：目标与奖励

| # | 项 | 状态 |
|---|---|---|
| 2.1 | 累积目标：env 记录轨迹累积量，telemetry 记录累计效用/时延/成本 | ⬜ |
| 2.2 | 决策步级奖励：势函数差分**默认开启**且含成本项 | ✅ `--potential-cost-weight` 默认 0.05；`_variant_config` 全部变体返回 `potential_shaping=True`（此前 `rnd` 变体下 100 个 update 的 `mean_deployment_shaping_reward` 恒为 0） |
| 2.3 | γ 不再为 0（组成与部署共用一条轨迹） | ⬜ |
| 2.4 | 验证换成留出的 λ(t) **轨迹** | ✅ 训练/验证/评估分别用 `realization=0/1/2`；⬜ 待验证"验证指标与测试指标相关性显著为正" |

---

## 阶段 3：episode 语义

| # | 项 | 状态 |
|---|---|---|
| 3.1 | episode = 一条时间轴；route/joint 不再差 450 倍 | ⬜ |
| 3.2 | "多部署上下文"移到轨迹之间（泛化轴），不再是时间轴内部的事 | ⬜ |
| 3.3 | telemetry 记录 episode 长度、轨迹 id、λ(t) 摘要 | ⬜ |

**附加的必要修复**（清单未列，但不做则 λ(t) 无效）：**episode 必须从轨迹的不同相位开始**。原来 `simulator.reset` 把 slot 归零、而一个 episode 只有 8 个 slot，策略永远只看到轨迹的前 8 步。已实现 `trace_offset_span`，reset 时按 episode 种子在轨迹内取偏移。

---

## 阶段 4：状态

| # | 项 | 状态 |
|---|---|---|
| 4.1 | 状态加入时变信息 | ✅ 已加 **下一 slot 的总 λ**（归一化）与 **剩余冻结时长**；λ(t) 本身原本就在（按 (app,ingress) 的到达强度）。⬜ `t/T` 轨迹内进度待加（会破坏 `test_feature_vector_excludes_horizon_progress_and_duplicate_phase`，该测试基于旧前提，需同步更新） |

---

## 阶段 5：论文同步（`latex/`，待做）

5.1 删 $\Delta^{\rm org}$/`k`（L130 符号表、L747、L798、L803、L961） 5.2 目标口径统一为累积，删或实现 $\varepsilon_r$ 持平区间 5.3 补 λ(t)/突发/轨迹定义与划分 5.4 统一 admission-wait 口径 5.5 补符号表缺失项、清理 L766–794 空白 5.6 若走 ② 需实现 LLM `start_cost`

---

## 阶段 6：Stage B 验收

| # | 项 | 状态 |
|---|---|---|
| 6.1 | **量化静态门槛**：平稳轨迹上扫部署库，比较"最好固定部署" vs "最好反应式策略"的累积效用。差距≈0 ⇒ 平稳下 Stage B 无可学 | ⬜ 进行中 |
| 6.2 | 突发轨迹上重做 6.1，差距应显著 > 0 | ⬜ |
| 6.3 | Stage B 训练（组成策略用蒸馏版冻结，门禁 ρ=0.863/捕获 136.1%） | ⬜ |
| 6.4 | 在留出突发轨迹上，累积效用显著优于最好固定部署（配对检验，≥5 seeds） | ⬜ |

**评测仪器**：`scripts/evaluate_deployment_policy.py` —— 固定组成策略、给定部署、同一轨迹，输出累积与均值效用。已具备 `--library-scan`（扫全库）与 `--arrival-pattern bursty`。

**已有的关键数据**（平稳轨迹、蒸馏组成策略、3 seeds、25 周期）：

| 部署 | LLM 实例 | 模型 | 稳态成本(每 slot) | 均值效用 | SLO 达成 |
|---|---|---|---|---|---|
| `initial_deployment`（= equal/least_load/random/greedy 的部署） | 4 | 4 | 0.1600 | **+0.25129** | 0.9354 |
| `static`（最便宜） | 1 | 1 | 0.0267 | +0.14218 | 1.0000 |

⚠ **五个基线里只有 2 个不同的部署**（4 个都用 `initial_deployment`）。所以 Stage B 的对比必须用**部署库的 128 套**来定门槛，而不是这 2 个退化基线。

---

## 阶段 7：工程清理

`configs/generated/` 不存在；`aggregate_llm_steady_state_validation.py` 的输入/输出目录不存在；质量分与 SLO 是占位值；BurstGPT v2 零引用；`latex/` 有未提交改动。

---

## 依赖顺序

```
阶段0（已锁定）──┬─→ 阶段1（λ(t)）──→ 阶段2（目标/奖励）──→ 阶段3（episode）──→ 阶段4（状态）──→ 阶段6（验收）
                 └─→ 阶段5（论文）可并行，5.1/5.2 依赖 0.1/0.2（已定）
```

**最省力的三件事**：0.1+0.2 拍板（✅）→ 6.1 静态门槛量化（进行中）→ 1.1–1.3 引入 λ(t) 与突发场景（部分✅）。
