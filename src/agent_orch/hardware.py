"""Pinned vLLM v0.31.0 CUDA online batch defaults, without importing vLLM.

Source: EngineArgs.get_batch_defaults and its performance-mode adjustment at
https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py
Memory is per GPU, never the sum over a tensor-parallel group. These limits
are configuration ceilings, not the number of requests KV memory can sustain.
"""
from __future__ import annotations


def vllm_online_batch_defaults(memory_gib: float, gpu_name: str, engine: dict) -> dict:
    if engine["version"] != "v0.31.0" or engine["entrypoint"] != "vllm serve":
        raise ValueError("Batch defaults are pinned to vLLM v0.31.0 CUDA online serving")
    if engine.get("batch_defaults") != "per_gpu":
        raise ValueError("The registered batch-default policy must be per_gpu")
    if memory_gib <= 0:
        raise ValueError("Per-GPU memory must be positive")
    if memory_gib >= 160:
        tokens, sequences = 16384, 1024
    elif memory_gib >= 70 and "a100" not in gpu_name.lower():
        tokens, sequences = 8192, 1024
    else:
        tokens, sequences = 2048, 256
    mode = engine["performance_mode"]
    if mode == "throughput":
        tokens, sequences = tokens * 2, sequences * 2
    elif mode != "balanced":
        raise ValueError("Unsupported pinned performance mode")
    return {"max_num_batched_tokens": tokens, "max_num_seqs": sequences}
