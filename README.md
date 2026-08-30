# Agent Orchestration Simulator

This project is the independent experiment implementation for the system model in
`latex/agent_system_model_service_mesh_style.tex`. The legacy code under `old_exp/`
is reference-only and is not imported.

The first milestone provides:

- typed scenario and decision schemas;
- an analytical LLM/tool/network performance backend;
- probabilistic model, instance, and tool routing;
- chain and fork--join critical-path evaluation;
- static-homogeneous, equal-split, least-load, random, and greedy policies;
- deterministic, seed-controlled command-line runs.

## Environment

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e .
```

## Run the toy scenario

```powershell
conda activate agent-orch
agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 10 --seed 7
```

Synthetic burst trace:

```powershell
agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 100 --seed 7 --synthetic-bursty
```

Structured PPO smoke training:

```powershell
agent-orch-sim train --scenario configs/toy.yaml --updates 10 --rollout-steps 256 --potential-shaping
```

The optimization split is selected with `--mode joint`, `--mode deploy`, or
`--mode route`. Reward-component comparisons use the default PPO, potential
shaping, or ICM:

```powershell
agent-orch-sim train --scenario configs/toy.yaml --mode joint --icm
```

Multi-seed baseline matrix:

```powershell
python scripts/run_baseline_matrix.py --scenario configs/toy.yaml --slots 600 --bursty
```

Multi-seed PPO structural and component matrix:

```powershell
python scripts/run_rl_matrix.py --scenario configs/toy.yaml --updates 100 --rollout-steps 1024
```

The matrix writes per-slot metrics, one-row-per-run summaries, checkpoints,
training histories, and a software/scenario manifest. A quick integration check
can use one seed, one update, and one mode before launching the full matrix.
Its default `auto` matrix evaluates joint PPO with vanilla, potential-shaping,
and ICM rewards, while the deployment-only and routing-only comparisons use
vanilla PPO. Explicit `--variants` values request a full Cartesian sweep.

Generate reproducible confidence intervals and paired significance tests from a
multi-seed run summary:

```powershell
python scripts/summarize_results.py `
  --input results/baseline_matrix.summary.parquet `
  --group-columns policy `
  --metrics mean_cost,mean_latency_s,mean_goodput_rps,mean_quality,mean_slo_attainment `
  --baseline static
```

Generate a workload-composition sweep while holding the combined request rate
constant:

```powershell
python scripts/generate_composition_sweep.py `
  --scenario configs/toy.yaml `
  --short-app chat-search `
  --long-app research-agent `
  --long-fractions 0,0.25,0.5,0.75,1
```

Validate the analytical LLM approximation against a held-out vLLM or
LLMServingSim profile table:

```powershell
python scripts/validate_llm_model.py `
  --scenario configs/toy.yaml `
  --profile data/processed/llm_profile_test.csv
```

The validator reports TTFT, TBT, response-time, and stable-capacity errors. Its
validity flags implement the thresholds in `docs/experiment_protocol.md`.

Results are written to `results/` as JSON Lines plus a run manifest. All time
values use seconds, rates use requests/second, data sizes use MB, link capacities
use Mbit/second, FLOPs use operations, and memory traffic uses bytes.

The structured PPO has one binary deployment head and three grouped Dirichlet
heads for model selection, LLM-instance allocation, and tool routing. Heuristics
are baselines only; they do not decode the proposed policy's routing actions.
