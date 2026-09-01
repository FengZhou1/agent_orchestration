# Agent Orchestration Simulator

This project is the independent experiment implementation for the system model in
`latex/agent_system_model_service_mesh_style.tex`. The legacy code under `old_exp/`
is reference-only and is not imported.

The first milestone provides:

- typed scenario and decision schemas;
- analytical and profile-driven LLM/stateless-service/network backends;
- probabilistic model, instance, and stateless-service routing;
- chain and fork--join critical-path evaluation;
- static-homogeneous, equal-split, least-load, random, and greedy policies;
- deterministic, seed-controlled command-line runs.

## Environment

```powershell
conda env create -f environment.yml
conda activate agent-orch
python -m pip install -e .
```

## Build benchmark scenarios

`toy.yaml` is a smoke-test scenario only. Paper experiments use the generated
Abilene Main, GEANT Scale, and explicit Stress scenarios.

```powershell
conda activate agent-orch
$env:PYTHONPATH='src'
python scripts/build_benchmark_scenarios.py
python scripts/export_reference_catalogs.py
```

After pinning the public traces and controlled service measurements, rebuild
the scenarios from the processed joint server rows and service profiles:

```powershell
python scripts/build_benchmark_scenarios.py `
  --infrastructure-catalog data/processed/alibaba_gpu_server_catalog.yaml `
  --service-profile data/processed/stateless_service_profile.csv
```

The generated scenarios pin Qwen2.5-7B/14B/32B BF16 configurations, vLLM
engine settings, six/eight stateless-service types, SNDlib topology, normalized
GPU costs, parameter units, and source provenance. Reference quality and SLO
values are visibly marked and must be replaced by pinned benchmark/calibration
results before final paper runs.

## Run the toy scenario

```powershell
conda activate agent-orch
agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 10 --seed 7
```

The legacy synthetic burst is an explicitly named stress case:

```powershell
agent-orch-sim run --scenario configs/toy.yaml --policy greedy --slots 100 --seed 7 --synthetic-bursty
```

Structured PPO smoke training:

```powershell
agent-orch-sim train --scenario configs/toy.yaml --updates 10 --rollout-steps 256
```

The optimization split is selected with `--mode joint`, `--mode deploy`, or
`--mode route`. The default is constrained PPO; `--unconstrained` is retained
as an ablation, while potential shaping and ICM are optional comparisons:

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
Its default `auto` matrix evaluates constrained and unconstrained joint PPO,
plus the potential-shaping and ICM ablations. Deployment-only and routing-only
runs use constrained PPO. Explicit `--variants` values request a full Cartesian
sweep.

Generate reproducible confidence intervals and paired significance tests from a
multi-seed run summary:

```powershell
python scripts/summarize_results.py `
  --input results/baseline_matrix.summary.parquet `
  --group-columns policy `
  --metrics mean_cost,mean_latency_s,mean_goodput_rps,mean_quality,mean_slo_attainment `
  --baseline static
```

Generate the unified fixed-load statistics, IEEE-style PDF/PNG figures, RL
ablation plots, convergence curves, overhead plots, and bursty-workload traces:

```powershell
python scripts/plot_results.py `
  --results results `
  --output results/figures `
  --analysis-output results/analysis
```

The plotting command verifies that baseline and RL runs use the same scenario
hash, seed set, and evaluation horizon before combining them.

Generate a workload-composition sweep while holding the combined request rate
constant:

```powershell
python scripts/generate_composition_sweep.py `
  --scenario configs/toy.yaml `
  --short-app chat-search `
  --long-app research-agent `
  --long-fractions 0,0.25,0.5,0.75,1
```

The four-family 60/20/10/10 sweeps used by the paper are generated with:

```powershell
python scripts/generate_composition_sweep.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --family-sweep --output configs/generated/family_composition
```

## Prepare trace-driven arrivals

Raw datasets stay outside the repository. Convert a pinned BurstGPT release
into trace, minute-level NHPP, and homogeneous-Poisson train/validation/test
bundles as follows:

```powershell
python scripts/prepare_arrival_traces.py `
  --input <BurstGPT.csv> --source burstgpt --source-version <release> `
  --scenario configs/benchmarks/main_abilene.yaml `
  --reference-capacity-rps <pinned-capacity>
```

The output includes the paired input/output token audit table and a manifest
with source checksum, command, family mapping, seed, global timestamp scaling,
and split boundaries. Compute the pinned reference-deployment capacity before
creating the 0.40/0.65/0.85/1.05 bundles:

```powershell
python scripts/estimate_reference_capacity.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --profile data/processed/llm_profile.csv
```

Freeze the default SLOs from the low-load validation window before running the
test split:

```powershell
python scripts/calibrate_slos.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --profile data/processed/llm_profile.csv `
  --trace data/processed/arrivals/load_0p40/trace/validation.csv `
  --output configs/generated/main_abilene_slo.yaml
```

Run a trace-driven baseline matrix with:

```powershell
python scripts/run_baseline_matrix.py `
  --scenario configs/benchmarks/main_abilene.yaml --slots 3600 `
  --trace data/processed/arrivals/load_0p65/trace/test.csv `
  --profile data/processed/llm_profile.csv `
  --arrival-mode trace
```

Use `--arrival-mode nhpp` or `--arrival-mode poisson` with the same source trace
for controlled arrival-process comparisons. RL training accepts separate
`--train-trace` and `--eval-trace` inputs.

Validate the analytical LLM approximation against a held-out vLLM or
LLMServingSim profile table:

```powershell
python scripts/prepare_llm_profiles.py `
  --input <normalized-LLMServingSim-output.csv> `
  --source-version <pinned-commit> `
  --output data/processed/llm_profile.csv
python scripts/validate_llm_model.py `
  --scenario configs/benchmarks/main_abilene.yaml `
  --profile data/processed/llm_profile.csv
```

The validator reports TTFT, TBT, response-time, and stable-capacity errors. Its
validity flags implement the thresholds in `docs/experiment_protocol.md`.

Results are written to `results/` as JSON Lines plus a run manifest. All time
values use seconds, rates use requests/second, per-request data sizes use MB,
offered link load and link capacity use Mbit/second, FLOPs use operations, and
memory traffic uses bytes. Control-slot duration does not change Mbps load or
per-request serialization delay.

The joint controller operates at two time scales. At every deployment epoch, a
masked categorical head produces one complete deployment plan: a binary choice
for each LLM candidate and a replica count for each tool--server pool. During
the following physical slots, a grouped Dirichlet head controls application
model shares. LLM-instance allocation and tool routing are then derived from
deployed capacity, current utilization, analytical service demand, and network
propagation delay.

The slot utility combines fixed-scale normalized cost and latency with SLO
attainment and quality. Physical-resource, queue, KV-cache, and link-capacity
excesses are returned separately as constraint costs. PPO applies an adaptive
Lagrange multiplier, so objective weights express operational preferences and
do not also serve as arbitrary feasibility penalties.
