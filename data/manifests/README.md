# Data manifests

Raw public datasets are not committed to the experiment source tree. Every
processed artifact must have a JSON manifest recording the source URL, pinned
release or commit, license, SHA-256 checksum, exact preprocessing command,
random seed, transform parameters, and chronological split boundaries. The
manifest is validated by `agent_orch.data.DatasetManifest`.

The benchmark uses separate sources for separate model parameters:

| Artifact | Primary evidence | Explicitly not inferred |
|---|---|---|
| arrivals | BurstGPT v2; Azure 2024 for external validation | workflow topology and GPU capacity |
| workflows | TraceLab v2; BFCL V3/V4 | production arrival intensity |
| stateless-service profile | DeathStarBench or local low-load measurements | LLM inference time |
| infrastructure | Alibaba GPU Trace v2026 | request tokens and LLM latency |
| network | SNDlib Abilene/GEANT | Agent demand semantics |

The common arrival-trace schema is:

```text
slot,application,ingress,rate_rps
```

Rates use request/s. Missing application rows inside a covered trace slot are
interpreted as zero, not as the scenario default.

The normalized assigned-request audit table is:

```text
timestamp_s,session_id,prompt_tokens,output_tokens,length_class,
family,application,ingress
```

Input and output token values remain paired from the same source request.
Offered-load levels use one global timestamp scale rather than independently
resampling requests or multiplying selected bursts.

The common LLM profile schema is:

```text
model,config,prompt_tokens,output_tokens,arrival_rate_rps,
long_request_fraction,interactive_retrieval_fraction,
transactional_tool_fraction,deep_research_fraction,coding_agent_fraction,
ttft_s,tbt_s,response_s,stable_capacity_rps,kv_tokens
```

The four family-fraction columns are optional as a group. Profiles without
them remain valid and represent a workload-composition-agnostic backend.

The stateless-service measurement input schema is:

```text
service,server,vcpu,arrival_rate_rps,latency_ms,error,
request_bytes,response_bytes
```

`prepare_service_profiles.py` computes the low-load processing mean and
service-time SCV; arrival SCV remains a separate scenario parameter.
The stable rate is the highest measured rate with error rate below 1% and P95
latency no greater than twice the low-load P95. End-to-end production RT is not
silently reused as processing time.
