# Experiment protocol

## Research questions

1. Compare joint LLM/tool deployment and routing with static and greedy policies.
2. Isolate PPO-deployment, PPO-routing, and joint structured PPO.
3. Compare vanilla PPO, PPO+ICM, and objective-consistent potential shaping.
4. Sweep prompt/output composition at fixed aggregate arrival rate.
5. Sweep workflow depth, parallel width, tool intensity, topology, and failures.
6. Validate the analytical backend against offline profiles and selected
   LLMServingSim 2.0 replays.

## Primary comparisons

- Static-Homogeneous and Static-MultiModel;
- EqualSplit, LeastLoad, and Greedy-SLO;
- PPO-Deploy with heuristic routing;
- PPO-Route with heuristic deployment;
- Joint-PPO;
- Joint-PPO+ICM;
- Joint-PPO+Potential;
- exhaustive enumeration on toy cases only.

Every algorithm reports the original four objectives independently: cost,
request-weighted end-to-end latency, request goodput, and task quality. Additional
metrics include P50/P95/P99 latency, SLO attainment, TTFT/TBT, GPU/KV/tool/link
utilization, reconfiguration count, decision latency, and training wall time.

## Scenarios and splits

- toy: 4 servers, 2 models, 3 tools, 2 applications;
- small: 8 servers, 5 applications, 3 models, 5 tools;
- medium: Abilene, 20 applications, 3 models, 6 tools;
- large: GEANT, 50 applications, 3 models, 8 tools.

Trace windows are split chronologically into 60% training, 20% validation, and
20% test. Workflows are split by session/workflow ID. Hyperparameters and reward
normalizers are frozen before the test split is opened.

## Statistical reporting

- 3 seeds for pilot experiments;
- at least 5 independent seeds for full experiments;
- 10 seeds for the main algorithm comparison when resources permit;
- common random numbers and identical trace windows across algorithms;
- mean, median, effect size, and 95% bootstrap confidence intervals;
- paired bootstrap or Wilcoxon signed-rank tests with Holm correction.

Raw per-slot and per-request outputs are retained in Parquet or JSON Lines.
Figures and tables are regenerated only from those raw records.

## Fidelity rule

On held-out profile points, report error for TTFT, TBT, response time, and the
capacity boundary. If analytical median error exceeds 10% or P95 error exceeds
20% in a load region, final evaluation in that region uses the profile backend
or explicitly narrows the analytical model's stated validity range.

