# Data manifests

Raw public datasets are not committed to the experiment source tree. Every
processed trace must have a manifest recording source URL, release or commit,
download date, checksum, preprocessing command, split boundaries, and license.

The common arrival-trace schema is:

```text
slot,application,ingress,rate_rps
```

The common LLM profile schema is:

```text
model,config,prompt_tokens,output_tokens,arrival_rate_rps,
long_request_fraction,ttft_s,tbt_s,response_s,stable_capacity_rps,kv_tokens
```

