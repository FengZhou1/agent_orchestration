from __future__ import annotations

import math
from dataclasses import dataclass

from agent_orch.schema.models import LLMConfigSpec, ModelSpec


@dataclass(frozen=True)
class ServiceDemand:
    prefill_s: float
    decode_s: float
    service_s: float
    mean_iteration_s: float
    kv_work_tokens: float


def roofline_time(flops: float, memory_bytes: float, config: LLMConfigSpec) -> float:
    return max(
        flops / config.effective_flops,
        memory_bytes / config.effective_bandwidth_bytes_s,
    )


def prefill_work(
    model: ModelSpec,
    new_tokens: float,
    context_tokens: float,
    concurrency: int,
) -> tuple[float, float]:
    u = new_tokens
    chi = context_tokens
    nu = concurrency
    flops = nu * (
        2.0 * model.parameter_count * u
        + 4.0 * model.layers * model.hidden_size * u * (chi + (u + 1.0) / 2.0)
    )
    memory = model.weight_bytes + nu * model.kv_bytes_per_token * (
        u * (chi + (u + 1.0) / 2.0) + u
    )
    return flops, memory


def decode_work(
    model: ModelSpec,
    context_tokens: float,
    concurrency: int,
) -> tuple[float, float]:
    chi = context_tokens
    nu = concurrency
    flops = nu * (
        2.0 * model.parameter_count
        + 4.0 * model.layers * model.hidden_size * (chi + 1.0)
    )
    memory = model.weight_bytes + nu * model.kv_bytes_per_token * (chi + 1.0)
    return flops, memory


def service_demand(
    model: ModelSpec,
    config: LLMConfigSpec,
    prompt_tokens: float,
    output_tokens: float,
    chunk_tokens: int,
) -> ServiceDemand:
    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    concurrency = max(1, int(config.effective_concurrency))

    prefill = 0.0
    chunks = math.ceil(prompt / chunk_tokens)
    for q in range(chunks):
        context = q * chunk_tokens
        new_tokens = min(chunk_tokens, prompt - context)
        flops, memory = prefill_work(model, new_tokens, context, concurrency)
        prefill += roofline_time(flops, memory, config)

    decode = 0.0
    for token_index in range(1, output):
        context = prompt + token_index - 1
        flops, memory = decode_work(model, context, concurrency)
        decode += roofline_time(flops, memory, config)

    iterations = chunks + max(0, output - 1)
    service = prefill + decode
    mean_iteration = service / max(1, iterations)
    kv_work = (
        (1.0 + prompt / chunk_tokens) * prompt / 2.0
        + prompt * output
        + (1.0 + output) * output / 2.0
    )
    return ServiceDemand(prefill, decode, service, mean_iteration, kv_work)

