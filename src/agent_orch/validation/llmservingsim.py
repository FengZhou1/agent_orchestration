from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from agent_orch.performance.queueing import erlang_c


@dataclass(frozen=True)
class WorkloadClass:
    id: str
    prompt_tokens: int
    output_tokens: int


JITSERVE_CLASSES: dict[str, WorkloadClass] = {
    "CS": WorkloadClass("CS", 93, 318),
    "CC": WorkloadClass("CC", 1300, 4458),
    "DS": WorkloadClass("DS", 1911, 534),
    "DC": WorkloadClass("DC", 12223, 3541),
}

CALIBRATION_CLASSES: dict[str, WorkloadClass] = {
    f"CAL_{p}_{o}": WorkloadClass(f"CAL_{p}_{o}", p, o)
    for p, o in (
        (128, 64),
        (128, 512),
        (512, 128),
        (512, 1024),
        (2048, 128),
        (2048, 512),
        (8192, 256),
        (8192, 1024),
    )
}

COMPOSITIONS: dict[str, dict[str, float]] = {
    "CS": {"CS": 1.0},
    "DS": {"DS": 1.0},
    "MIX": {"CS": 0.5, "DS": 0.5},
    "AGENT": {"CS": 0.7, "DS": 0.1, "CC": 0.1, "DC": 0.1},
}


@dataclass(frozen=True)
class AnalyticalParameters:
    parameter_count: float = 8.03e9
    layers: int = 32
    hidden_size: int = 4096
    weight_bytes: float = 16.06e9
    kv_bytes_per_token: float = 131072.0
    chunk_tokens: int = 512
    effective_flops: float = 82.6e12
    effective_bandwidth_bytes_s: float = 1008e9


@dataclass(frozen=True)
class ServiceMetrics:
    prefill_s: float
    decode_s: float
    service_s: float
    tbt_s: float


@dataclass(frozen=True)
class QueuePrediction:
    waiting_s: float
    utilization: float
    ttft_s: float
    tbt_s: float
    response_s: float
    overloaded: bool


def _roofline_time(flops: float, memory_bytes: float, params: AnalyticalParameters) -> float:
    return max(
        flops / params.effective_flops,
        memory_bytes / params.effective_bandwidth_bytes_s,
    )


def roofline_service(
    workload: WorkloadClass,
    params: AnalyticalParameters,
    concurrency: int = 1,
) -> ServiceMetrics:
    """Evaluate the paper's prefill/decode Roofline equations."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    prompt = workload.prompt_tokens
    output = workload.output_tokens
    nu = concurrency
    prefill = 0.0
    chunks = math.ceil(prompt / params.chunk_tokens)
    for q in range(chunks):
        context = q * params.chunk_tokens
        new_tokens = min(params.chunk_tokens, prompt - context)
        flops = nu * (
            2.0 * params.parameter_count * new_tokens
            + 4.0
            * params.layers
            * params.hidden_size
            * new_tokens
            * (context + (new_tokens + 1.0) / 2.0)
        )
        memory = params.weight_bytes + nu * params.kv_bytes_per_token * (
            new_tokens * (context + (new_tokens + 1.0) / 2.0) + new_tokens
        )
        prefill += _roofline_time(flops, memory, params)

    decode = 0.0
    for token_index in range(1, output):
        context = prompt + token_index - 1
        flops = nu * (
            2.0 * params.parameter_count
            + 4.0 * params.layers * params.hidden_size * (context + 1.0)
        )
        memory = params.weight_bytes + nu * params.kv_bytes_per_token * (context + 1.0)
        decode += _roofline_time(flops, memory, params)
    service = prefill + decode
    return ServiceMetrics(
        prefill_s=prefill,
        decode_s=decode,
        service_s=service,
        tbt_s=decode / (output - 1) if output > 1 else 0.0,
    )


def fit_effective_rates(
    observations: pd.DataFrame,
    params: AnalyticalParameters | None = None,
) -> tuple[AnalyticalParameters, dict[str, float]]:
    """Fit the two effective Roofline rates on isolated-request observations."""
    base = params or AnalyticalParameters()
    required = {"prompt_tokens", "output_tokens", "prefill_s", "decode_s"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")

    def residual(log_rates: np.ndarray) -> np.ndarray:
        trial = AnalyticalParameters(
            **{
                **asdict(base),
                "effective_flops": float(np.exp(log_rates[0])),
                "effective_bandwidth_bytes_s": float(np.exp(log_rates[1])),
            }
        )
        errors: list[float] = []
        for row in observations.itertuples(index=False):
            workload = WorkloadClass("fit", int(row.prompt_tokens), int(row.output_tokens))
            predicted = roofline_service(workload, trial, concurrency=1)
            errors.append(math.log(max(predicted.prefill_s, 1e-12) / max(row.prefill_s, 1e-12)))
            errors.append(math.log(max(predicted.decode_s, 1e-12) / max(row.decode_s, 1e-12)))
        return np.asarray(errors)

    result = least_squares(
        residual,
        x0=np.log([base.effective_flops, base.effective_bandwidth_bytes_s]),
        bounds=(np.log([1e12, 1e9]), np.log([1e15, 1e13])),
    )
    fitted = AnalyticalParameters(
        **{
            **asdict(base),
            "effective_flops": float(np.exp(result.x[0])),
            "effective_bandwidth_bytes_s": float(np.exp(result.x[1])),
        }
    )
    return fitted, {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "success": bool(result.success),
    }


def normalize_composition(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("composition must have positive mass")
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("composition weights must be non-negative")
    return {key: float(value) / total for key, value in weights.items() if value > 0.0}


def choose_classes(
    weights: Mapping[str, float],
    num_requests: int,
    seed: int,
) -> list[str]:
    normalized = normalize_composition(weights)
    names = list(normalized)
    probabilities = np.asarray([normalized[name] for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    return list(rng.choice(names, size=num_requests, p=probabilities))


def generate_poisson_trace(
    output_jsonl: str | Path,
    manifest_csv: str | Path,
    classes: Mapping[str, WorkloadClass],
    composition: Mapping[str, float],
    num_requests: int,
    arrival_rate_rps: float | None,
    seed: int,
    simultaneous: bool = False,
) -> pd.DataFrame:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if not simultaneous and (arrival_rate_rps is None or arrival_rate_rps <= 0.0):
        raise ValueError("arrival_rate_rps must be positive for a Poisson trace")
    selected = choose_classes(composition, num_requests, seed)
    unknown = set(selected) - set(classes)
    if unknown:
        raise ValueError(f"unknown workload classes: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    if simultaneous:
        arrivals = np.zeros(num_requests, dtype=np.int64)
    else:
        gaps = rng.exponential(1.0 / float(arrival_rate_rps), size=num_requests)
        arrivals = np.rint(np.cumsum(gaps) * 1e9).astype(np.int64)

    output_path = Path(output_jsonl)
    manifest_path = Path(manifest_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, int | str]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for request_id, (class_id, arrival_ns) in enumerate(zip(selected, arrivals, strict=True)):
            workload = classes[class_id]
            row = {
                "input_toks": workload.prompt_tokens,
                "output_toks": workload.output_tokens,
                "arrival_time_ns": int(arrival_ns),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            records.append(
                {
                    "request_id": request_id,
                    "class_id": class_id,
                    "prompt_tokens": workload.prompt_tokens,
                    "output_tokens": workload.output_tokens,
                    "arrival_time_ns": int(arrival_ns),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(manifest_path, index=False)
    return frame


def read_simulator_output(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "request id",
        "arrival",
        "latency",
        "queuing_delay",
        "TTFT",
        "TPOT",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"simulator output is missing columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "request_id": frame["request id"].astype(int),
            "arrival_ns": frame["arrival"].astype(float),
            "waiting_s": frame["queuing_delay"].astype(float) / 1e9,
            "ttft_s": frame["TTFT"].astype(float) / 1e9,
            "tbt_s": frame["TPOT"].astype(float) / 1e9,
            "response_s": frame["latency"].astype(float) / 1e9,
            "end_s": frame["end_time"].astype(float) / 1e9,
        }
    )
    result["prefill_s"] = result["ttft_s"] - result["waiting_s"]
    result["decode_s"] = result["response_s"] - result["ttft_s"]
    result["service_s"] = result["response_s"] - result["waiting_s"]
    return result


def service_moments(
    composition: Mapping[str, float],
    services: Mapping[str, ServiceMetrics],
) -> tuple[float, float]:
    weights = normalize_composition(composition)
    mean = sum(weights[key] * services[key].service_s for key in weights)
    second = sum(weights[key] * services[key].service_s**2 for key in weights)
    return mean, second


def allen_cunneen_wait(
    arrival_rate_rps: float,
    mean_service_s: float,
    second_moment_s2: float,
    concurrency: int,
) -> tuple[float, float, bool]:
    utilization = arrival_rate_rps * mean_service_s / concurrency
    if utilization >= 1.0:
        return math.inf, utilization, True
    if arrival_rate_rps <= 0.0:
        return 0.0, 0.0, False
    variability = second_moment_s2 / (2.0 * mean_service_s**2)
    denominator = concurrency / mean_service_s - arrival_rate_rps
    waiting = variability * erlang_c(concurrency, utilization) / denominator
    return waiting, utilization, False


def allen_cunneen_prediction(
    arrival_rate_rps: float,
    composition: Mapping[str, float],
    services: Mapping[str, ServiceMetrics],
    concurrency: int,
) -> QueuePrediction:
    weights = normalize_composition(composition)
    mean_service, second_moment = service_moments(weights, services)
    waiting, utilization, overloaded = allen_cunneen_wait(
        arrival_rate_rps, mean_service, second_moment, concurrency
    )
    mean_prefill = sum(weights[key] * services[key].prefill_s for key in weights)
    mean_tbt = sum(weights[key] * services[key].tbt_s for key in weights)
    return QueuePrediction(
        waiting_s=waiting,
        utilization=utilization,
        ttft_s=waiting + mean_prefill,
        tbt_s=mean_tbt,
        response_s=waiting + mean_service,
        overloaded=overloaded,
    )


def saturated_throughput(frame: pd.DataFrame) -> float:
    if len(frame) < 10:
        raise ValueError("at least ten completed requests are required")
    ends = np.sort(frame["end_s"].to_numpy(dtype=float))
    lo = int(math.floor(0.1 * len(ends)))
    hi = int(math.ceil(0.9 * len(ends))) - 1
    elapsed = ends[hi] - ends[lo]
    completed = hi - lo
    if elapsed <= 0.0 or completed <= 0:
        raise ValueError("invalid completion interval")
    return completed / elapsed


def calibrate_effective_concurrency(
    saturated_capacity_rps: float,
    composition: Mapping[str, float],
    classes: Mapping[str, WorkloadClass],
    params: AnalyticalParameters,
    maximum: int = 128,
) -> tuple[int, pd.DataFrame]:
    weights = normalize_composition(composition)
    records = []
    for concurrency in range(1, maximum + 1):
        services = {
            key: roofline_service(classes[key], params, concurrency) for key in weights
        }
        mean_service, _ = service_moments(weights, services)
        capacity = concurrency / mean_service
        records.append(
            {
                "effective_concurrency": concurrency,
                "predicted_capacity_rps": capacity,
                "relative_error": abs(capacity - saturated_capacity_rps)
                / saturated_capacity_rps,
            }
        )
    frame = pd.DataFrame.from_records(records)
    best = int(frame.loc[frame["relative_error"].idxmin(), "effective_concurrency"])
    return best, frame


def write_run_specs(path: str | Path, specs: Iterable[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(dict(spec), sort_keys=True) + "\n")


def read_run_specs(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
