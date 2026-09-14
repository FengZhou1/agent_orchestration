"""Synthesize LLMServingSim profiler bundles from a per-kernel Roofline model.

The script is intentionally independent of vLLM.  It reads the Qwen model JSON
files used by the simulator, evaluates a compulsory-memory/peak-BF16 Roofline
for every canonical kernel, and writes the CSV contract consumed by
LLMServingSim 2.0.  The generated profiles are analytical lower-bound
profiles, not calibrated replacements for measured vLLM profiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DTYPE_BYTES = 2.0  # BF16
MODEL_DIR_DEFAULT = Path("configs/model/Qwen")
OUTPUT_DIR_DEFAULT = Path("roofline_profiles")

# Public peak values without sparsity.  Memory capacity is used only for the
# feasibility report; bandwidth and BF16 peak are used by the Roofline.
# A10 values are from NVIDIA's A10 product page. L20/H20 values follow the
# published accelerator specification used by the experiment configuration.
GPU_SPECS: dict[str, dict[str, Any]] = {
    "A10": {
        "memory_gb": 24.0,
        "bf16_tflops": 125.0,
        "bandwidth_gbs": 600.0,
        "tdp_w": 150,
        "architecture": "Ampere",
        "source": "https://www.nvidia.com/en-in/data-center/products/a10-gpu/",
    },
    "L20": {
        "memory_gb": 48.0,
        "bf16_tflops": 119.5,
        "bandwidth_gbs": 864.0,
        "tdp_w": 300,
        "architecture": "Ada Lovelace",
        "source": "https://www.nvidia.com/en-us/data-center/",
    },
    "H20": {
        "memory_gb": 96.0,
        "bf16_tflops": 148.0,
        "bandwidth_gbs": 4000.0,
        "tdp_w": 300,
        "architecture": "Hopper",
        "source": "https://docs.nvidia.com/ai-enterprise/release-4/4.9/infra-software/vgpu/reference/hopper.html",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dense_grid(max_tokens: int) -> list[int]:
    points = list(range(1, min(16, max_tokens + 1)))
    points.extend(range(16, min(64, max_tokens + 1), 4))
    points.extend(range(64, max_tokens + 1, 16))
    if points and points[-1] != max_tokens:
        points.append(max_tokens)
    return sorted(set(points))


def power_grid(max_value: int, start: int = 16) -> list[int]:
    values = [0]
    value = start
    while value < max_value:
        values.append(value)
        value *= 2
    if max_value > 0:
        values.append(max_value)
    return sorted(set(values))


def qwen_models(model_dir: Path) -> list[tuple[str, dict[str, Any], Path]]:
    rows = []
    for path in sorted(model_dir.glob("Qwen3-*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        model_id = f"Qwen/{path.stem}"
        rows.append((model_id, cfg, path))
    if not rows:
        raise FileNotFoundError(f"No Qwen3 JSON files found under {model_dir}")
    return rows


def model_type(cfg: dict[str, Any]) -> str:
    return "qwen3_moe" if "Moe" in cfg["architectures"][0] else "qwen3"


def peak_roofline(flops: float, bytes_: float, gpu: dict[str, Any]) -> float:
    compute_s = flops / (gpu["bf16_tflops"] * 1e12)
    memory_s = bytes_ / (gpu["bandwidth_gbs"] * 1e9)
    return max(compute_s, memory_s)


def gemm_cost(tokens: float, k: float, n: float) -> tuple[float, float]:
    flops = 2.0 * tokens * k * n
    bytes_ = DTYPE_BYTES * (k * n + tokens * (k + n))
    return flops, bytes_


def dense_cost(layer: str, tokens: int, cfg: dict[str, Any], tp: int) -> tuple[float, float]:
    t = float(tokens)
    hidden = float(cfg["hidden_size"])
    heads = float(cfg["num_attention_heads"])
    kv_heads = float(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    head_dim = float(cfg.get("head_dim", int(hidden // heads)))
    intermediate = float(cfg.get("intermediate_size", 4 * hidden))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    if layer == "embedding":
        flops, bytes_ = 0.0, DTYPE_BYTES * t * hidden * 2.0
    elif layer in {"layernorm", "final_layernorm"}:
        flops, bytes_ = 5.0 * t * hidden, DTYPE_BYTES * t * hidden * 2.0
    elif layer == "qk_norm":
        elements = t * (q_dim + kv_dim)
        flops, bytes_ = 5.0 * elements, DTYPE_BYTES * elements * 2.0
    elif layer == "qkv_proj":
        flops, bytes_ = gemm_cost(t, hidden, q_dim + 2.0 * kv_dim)
    elif layer == "rotary_emb":
        elements = t * (q_dim + kv_dim)
        flops, bytes_ = 6.0 * elements, DTYPE_BYTES * elements * 2.0
    elif layer == "o_proj":
        flops, bytes_ = gemm_cost(t, q_dim, hidden)
    elif layer == "gate_up_proj":
        flops, bytes_ = gemm_cost(t, hidden, 2.0 * intermediate)
    elif layer == "act_fn":
        flops, bytes_ = 8.0 * t * intermediate, DTYPE_BYTES * t * 3.0 * intermediate
    elif layer == "down_proj":
        flops, bytes_ = gemm_cost(t, intermediate, hidden)
    else:
        raise KeyError(f"Unsupported dense layer: {layer}")
    return flops / tp, bytes_ / tp


def sequence_cost(layer: str, sequences: int, cfg: dict[str, Any], tp: int) -> tuple[float, float]:
    n = float(sequences)
    hidden = float(cfg["hidden_size"])
    vocab = float(cfg["vocab_size"])
    if layer == "lm_head":
        flops, bytes_ = gemm_cost(n, hidden, vocab)
        return flops / tp, bytes_ / tp
    if layer == "sampler":
        return 5.0 * n * vocab, 4.0 * n * vocab
    raise KeyError(f"Unsupported sequence layer: {layer}")


def attention_cost(pc: int, kp: int, nd: int, kd: int, cfg: dict[str, Any], tp: int) -> tuple[float, float]:
    heads = float(cfg["num_attention_heads"])
    kv_heads = float(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    head_dim = float(cfg.get("head_dim", int(cfg["hidden_size"] // cfg["num_attention_heads"])))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    prefill_pairs = float(pc) * (float(kp) + (float(pc) + 1.0) / 2.0)
    decode_pairs = float(nd) * (float(kd) + 1.0)
    flops = 4.0 * q_dim * (prefill_pairs + decode_pairs)
    kv_pair_bytes = 2.0 * kv_dim * DTYPE_BYTES
    prefill_bytes = 2.0 * float(pc) * q_dim * DTYPE_BYTES + (float(kp) + pc) * kv_pair_bytes
    decode_bytes = 2.0 * float(nd) * q_dim * DTYPE_BYTES + float(nd) * (float(kd) + 1.0) * kv_pair_bytes
    return flops / tp, (prefill_bytes + decode_bytes) / tp


def moe_cost(tokens: int, active_experts: int, cfg: dict[str, Any], tp: int) -> tuple[float, float]:
    t = float(tokens)
    hidden = float(cfg["hidden_size"])
    experts = float(cfg["num_experts"])
    top_k = float(cfg["num_experts_per_tok"])
    expert_hidden = float(cfg["moe_intermediate_size"])
    # Router plus the two projections and output projection for top-k experts.
    flops = 2.0 * t * hidden * experts + 4.0 * t * hidden * expert_hidden * top_k
    weight_bytes = DTYPE_BYTES * (experts * hidden * expert_hidden * 2.0) * min(active_experts, top_k) / max(experts, 1.0)
    activation_bytes = DTYPE_BYTES * t * (hidden + top_k * expert_hidden)
    return flops / tp, (weight_bytes + activation_bytes) / tp


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bundle(
    root: Path,
    hardware: str,
    model_id: str,
    cfg: dict[str, Any],
    model_path: Path,
    max_tokens: int,
    max_seqs: int,
    max_kv: int,
    tps: list[int],
) -> dict[str, Any]:
    gpu = GPU_SPECS[hardware]
    arch = model_type(cfg)
    variant_root = root / hardware / model_id / "bf16"
    for tp in tps:
        out = variant_root / f"tp{tp}"
        dense_layers = ["embedding", "layernorm", "qkv_proj", "qk_norm", "rotary_emb", "o_proj", "final_layernorm"]
        if arch == "qwen3":
            dense_layers += ["gate_up_proj", "act_fn", "down_proj"]
        dense_rows = []
        for layer in dense_layers:
            for tokens in dense_grid(max_tokens):
                flops, bytes_ = dense_cost(layer, tokens, cfg, tp)
                dense_rows.append({"layer": layer, "tokens": tokens, "time_us": 1e6 * peak_roofline(flops, bytes_, gpu)})
        write_csv(out / "dense.csv", ["layer", "tokens", "time_us"], dense_rows)

        seq_rows = []
        for layer in ("lm_head", "sampler"):
            for seq in dense_grid(max_seqs):
                flops, bytes_ = sequence_cost(layer, seq, cfg, tp)
                seq_rows.append({"layer": layer, "sequences": seq, "time_us": 1e6 * peak_roofline(flops, bytes_, gpu)})
        write_csv(out / "per_sequence.csv", ["layer", "sequences", "time_us"], seq_rows)

        attn_rows = []
        for pc in power_grid(max_tokens):
            for kp in power_grid(max_kv):
                for nd in power_grid(max_seqs, start=1):
                    for kd in power_grid(max_kv):
                        flops, bytes_ = attention_cost(pc, kp, nd, kd, cfg, tp)
                        attn_rows.append({
                            "prefill_chunk": pc,
                            "kv_prefill": kp,
                            "n_decode": nd,
                            "kv_decode": kd,
                            "time_us": 1e6 * peak_roofline(flops, bytes_, gpu),
                        })
        write_csv(out / "attention.csv", ["prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"], attn_rows)

        if arch == "qwen3_moe":
            moe_rows = []
            for active in power_grid(int(cfg["num_experts"]), start=1):
                active = max(active, 1)
                for tokens in dense_grid(max_tokens):
                    flops, bytes_ = moe_cost(tokens, active, cfg, tp)
                    moe_rows.append({"tokens": tokens, "activated_experts": active, "time_us": 1e6 * peak_roofline(flops, bytes_, gpu)})
            write_csv(out / "moe.csv", ["tokens", "activated_experts", "time_us"], moe_rows)

    meta = {
        "profiler_version": "synthetic-roofline-v2",
        "vllm_version": "n/a",
        "gpu": f"NVIDIA {hardware} (datasheet Roofline)",
        "hardware": hardware,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "architecture": arch,
        "architecture_sha256": sha256(Path("profiler/models") / f"{arch}.yaml") if Path("profiler/models") .exists() else "n/a",
        "model": model_id,
        "model_config_sha256": sha256(model_path),
        "variant": "bf16",
        "tp_degrees": tps,
        "engine_effective": {
            "max_num_batched_tokens": max_tokens,
            "max_num_seqs": max_seqs,
            "block_size": 16,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": False,
        },
        "roofline": {
            "bf16_tflops_peak": gpu["bf16_tflops"],
            "memory_bandwidth_gbs_peak": gpu["bandwidth_gbs"],
            "dtype_bytes": DTYPE_BYTES,
            "assumption": "peak Roofline lower bound; no kernel launch, scheduler, communication, or contention overhead",
            "gpu_memory_gb": gpu["memory_gb"],
            "tdp_w": gpu["tdp_w"],
            "architecture": gpu["architecture"],
            "source": gpu["source"],
        },
        "attention_grid": {"max_kv": max_kv, "chunk_factor": 2.0, "kv_factor": 2.0},
        "skew_fit": {"enabled": False},
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in meta.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child, child_value in value.items():
                if isinstance(child_value, dict):
                    lines.append(f"  {child}:")
                    for grandchild, grandchild_value in child_value.items():
                        lines.append(f"    {grandchild}: {yaml_scalar(grandchild_value)}")
                elif isinstance(child_value, list):
                    lines.append(f"  {child}: [{', '.join(yaml_scalar(x) for x in child_value)}]")
                else:
                    lines.append(f"  {child}: {yaml_scalar(child_value)}")
        elif isinstance(value, list):
            lines.append(f"{key}: [{', '.join(yaml_scalar(x) for x in value)}]")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    (variant_root / "meta.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"hardware": hardware, "model": model_id, "architecture": arch, "tp_degrees": tps, "path": str(variant_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--hardware", default="A10,L20,H20")
    parser.add_argument("--tp", default="1,2")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-kv", type=int, default=32768)
    args = parser.parse_args()
    hardware = [x.strip() for x in args.hardware.split(",") if x.strip()]
    tps = [int(x) for x in args.tp.split(",") if x.strip()]
    unknown = set(hardware) - set(GPU_SPECS)
    if unknown:
        raise SystemExit(f"Unknown hardware: {sorted(unknown)}")

    bundles = []
    for model_id, cfg, model_path in qwen_models(args.model_dir):
        for gpu in hardware:
            bundles.append(write_bundle(args.output, gpu, model_id, cfg, model_path, args.max_num_batched_tokens, args.max_num_seqs, args.max_kv, tps))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": [model_id for model_id, _, _ in qwen_models(args.model_dir)],
        "hardware": hardware,
        "tp_degrees": tps,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_kv": args.max_kv,
        "bundles": bundles,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
