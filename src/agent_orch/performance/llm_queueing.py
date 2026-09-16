"""Low-dimensional steady-state model for colocated LLM serving.

The module separates three quantities that are easy to mix up in a
continuous-batching model:

* iteration execution time, obtained from structured FLOP/HBM work;
* first admission waiting, obtained from an equivalent admission-opportunity
  queue;
* time spent after admission, obtained from the resident service curve.

It is deliberately independent of the request scheduler.  The scheduler is
used by the validation simulator, while this module only exposes the
long-run approximation used by the orchestration model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from agent_orch.schema.models import LLMConfigSpec, ModelSpec


@dataclass(frozen=True)
class IterationCalibration:
    """Fixed execution parameters for one model--hardware configuration."""

    compute_rate: float
    bandwidth_rate: float
    overhead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.compute_rate <= 0 or self.bandwidth_rate <= 0:
            raise ValueError("effective rates must be positive")
        if self.overhead_s < 0:
            raise ValueError("iteration overhead must be non-negative")


@dataclass(frozen=True)
class StructuralIterationCalibration:
    """Fixed additive timing coefficients for one execution mode.

    Dense projections, attention arithmetic, and KV-cache traffic are kept as
    separate Transformer work terms.  The coefficients are configuration
    parameters estimated once from operator timings; they do not depend on
    request rate or workload composition.
    """

    dense_s_per_flop: float
    attention_s_per_flop: float
    kv_s_per_byte: float
    overhead_s: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.dense_s_per_flop,
            self.attention_s_per_flop,
            self.kv_s_per_byte,
        ) < 0.0:
            raise ValueError("structural timing coefficients must be non-negative")
        if self.overhead_s < 0.0:
            raise ValueError("iteration overhead must be non-negative")


@dataclass(frozen=True)
class QueuePrediction:
    waiting_s: float
    utilization: float
    ttft_s: float
    tbt_s: float
    response_s: float
    overloaded: bool


@dataclass(frozen=True)
class TwoModeSteadyState:
    """Macro state of a colocated continuous-batching instance."""

    decode_concurrency: float
    prefill_tokens_per_mixed_iteration: float
    mixed_iteration_s: float
    decode_iteration_s: float
    mean_iteration_s: float
    mixed_fraction: float
    utilization: float
    capacity_rps: float
    overloaded: bool


@dataclass(frozen=True)
class TwoModeCalibration:
    """Execution calibration for decode-only and mixed iterations."""

    decode: IterationCalibration | StructuralIterationCalibration
    mix: IterationCalibration | StructuralIterationCalibration

    def for_iteration(self, has_prefill: bool) -> IterationCalibration | StructuralIterationCalibration:
        return self.mix if has_prefill else self.decode


def aggregate_iteration_work(
    model: ModelSpec,
    new_tokens: float,
    context_tokens: float,
    decode_sequences: float,
) -> tuple[float, float]:
    """Return mixed-iteration FLOPs and HBM bytes.

    Model weights are read once by the iteration and are therefore not
    multiplied by the number of decode sequences.  KV traffic is additive
    across the prefill chunk and decode sequences.
    """

    u = max(0.0, float(new_tokens))
    return aggregate_packed_iteration_work(
        model,
        [] if u == 0.0 else [(u, max(0.0, float(context_tokens)))],
        decode_sequences,
        max(0.0, float(context_tokens)),
    )


def aggregate_packed_iteration_work(
    model: ModelSpec,
    prefill_chunks: Sequence[tuple[float, float]],
    decode_sequences: float,
    decode_context: float,
) -> tuple[float, float]:
    """Return one iteration's work for a packed set of prefill chunks.

    Each tuple contains the number of newly processed tokens and the existing
    context of one prefill sequence.  Weight tensors are read once by the
    fused iteration.  Attention and KV work are additive over independent
    sequences, so packing several chunks must not be approximated as one long
    sequence with their token counts summed.
    """

    nu = max(0.0, float(decode_sequences))
    flops = 0.0
    non_weight_bytes = 0.0
    for new_tokens, context_tokens in prefill_chunks:
        u = max(0.0, float(new_tokens))
        if u == 0.0:
            continue
        pre_flops, pre_bytes = _prefill_work(model, u, max(0.0, float(context_tokens)))
        flops += pre_flops
        non_weight_bytes += pre_bytes - model.weight_bytes
    dec_flops, dec_bytes = _decode_work(model, max(0.0, float(decode_context)))
    flops += nu * dec_flops
    non_weight_bytes += nu * (dec_bytes - model.weight_bytes)
    return float(flops), float(model.weight_bytes + non_weight_bytes)


def _prefill_work(model: ModelSpec, new_tokens: float, context_tokens: float) -> tuple[float, float]:
    u = max(0.0, float(new_tokens))
    chi = max(0.0, float(context_tokens))
    flops = 2.0 * model.parameter_count * u + 4.0 * model.layers * model.hidden_size * u * (
        chi + (u + 1.0) / 2.0
    )
    return flops, model.weight_bytes + model.kv_bytes_per_token * (chi + u)


def _decode_work(model: ModelSpec, context_tokens: float) -> tuple[float, float]:
    chi = max(0.0, float(context_tokens))
    flops = 2.0 * model.parameter_count + 4.0 * model.layers * model.hidden_size * (chi + 1.0)
    return flops, model.weight_bytes + model.kv_bytes_per_token * (chi + 1.0)


def calibrated_iteration_time(
    flops: float,
    memory_bytes: float,
    calibration: IterationCalibration,
) -> float:
    """Roofline execution time with a fixed runtime overhead."""

    return calibration.overhead_s + max(
        float(flops) / calibration.compute_rate,
        float(memory_bytes) / calibration.bandwidth_rate,
    )


def structural_iteration_time(
    model: ModelSpec,
    prefill_chunks: Sequence[tuple[float, float]],
    decode_sequences: float,
    decode_context: float,
    calibration: StructuralIterationCalibration,
) -> float:
    """Return an iteration time from dense, attention, and KV work terms."""

    dense = attention = kv = 0.0
    for new_tokens, context_tokens in prefill_chunks:
        u = max(0.0, float(new_tokens))
        chi = max(0.0, float(context_tokens))
        dense += 2.0 * model.parameter_count * u
        attention += 4.0 * model.layers * model.hidden_size * u * (
            chi + (u + 1.0) / 2.0
        )
        kv += model.kv_bytes_per_token * (chi + u)
    nu = max(0.0, float(decode_sequences))
    chi_dec = max(0.0, float(decode_context))
    dense += nu * 2.0 * model.parameter_count
    attention += nu * 4.0 * model.layers * model.hidden_size * (chi_dec + 1.0)
    kv += nu * model.kv_bytes_per_token * (chi_dec + 1.0)
    return float(
        calibration.overhead_s
        + calibration.dense_s_per_flop * dense
        + calibration.attention_s_per_flop * attention
        + calibration.kv_s_per_byte * kv
    )


def _calibrated_packed_iteration_time(
    model: ModelSpec,
    prefill_chunks: Sequence[tuple[float, float]],
    decode_sequences: float,
    decode_context: float,
    calibration: IterationCalibration | StructuralIterationCalibration,
) -> float:
    if isinstance(calibration, StructuralIterationCalibration):
        return structural_iteration_time(
            model, prefill_chunks, decode_sequences, decode_context, calibration
        )
    flops, bytes_ = aggregate_packed_iteration_work(
        model, prefill_chunks, decode_sequences, decode_context
    )
    return calibrated_iteration_time(flops, bytes_, calibration)


def class_service_times(
    model: ModelSpec,
    calibration: IterationCalibration | StructuralIterationCalibration | TwoModeCalibration,
    prompt_tokens: float,
    output_tokens: float,
    chunk_tokens: int,
    decode_concurrency: float,
) -> tuple[float, float, float]:
    """Return resident prefill, decode, and service times for one class.

    The prefill chunks execute with the resident decode population.  Decode
    iterations use the context trajectory of the request and the same
    resident population.  Each iteration's time is a wall-clock time shared
    by the active sequences; it is not divided by the batch size.
    """

    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    nu = max(1.0, float(decode_concurrency))
    prefill = 0.0
    chunks = math.ceil(prompt / max(1, int(chunk_tokens)))
    for q in range(chunks):
        context = q * max(1, int(chunk_tokens))
        new_tokens = min(max(1, int(chunk_tokens)), prompt - context)
        iteration_calibration = (
            calibration.for_iteration(True)
            if isinstance(calibration, TwoModeCalibration)
            else calibration
        )
        prefill += _calibrated_packed_iteration_time(
            model, [(new_tokens, context)], nu, context, iteration_calibration
        )
    decode = 0.0
    for token in range(max(0, output - 1)):
        context = prompt + token
        iteration_calibration = (
            calibration.for_iteration(False)
            if isinstance(calibration, TwoModeCalibration)
            else calibration
        )
        decode += _calibrated_packed_iteration_time(
            model, [], nu, context, iteration_calibration
        )
    return prefill, decode, prefill + decode


def operating_point(
    model: ModelSpec,
    calibration: IterationCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    arrival_rate: float,
    chunk_tokens: int,
    resident_limit: int,
    tolerance: float = 1.0e-7,
    max_iterations: int = 200,
) -> tuple[float, float, list[tuple[float, float, float]], bool]:
    """Solve the resident Little fixed point without queueing time."""

    normalized = _normalize(weights)
    nu = 1.0
    converged = False
    for _ in range(max_iterations):
        services = [
            class_service_times(model, calibration, p, o, chunk_tokens, nu)
            for p, o in classes
        ]
        mean_service = _weighted_mean([item[2] for item in services], normalized)
        next_nu = max(0.0, arrival_rate * mean_service)
        next_nu = min(float(resident_limit + 1), next_nu)
        if abs(next_nu - nu) <= tolerance * max(1.0, next_nu):
            nu = next_nu
            converged = True
            break
        nu = next_nu
    services = [
        class_service_times(model, calibration, p, o, chunk_tokens, max(1.0, nu))
        for p, o in classes
    ]
    mean_service = _weighted_mean([item[2] for item in services], normalized)
    return nu, mean_service, services, converged


def capacity(
    model: ModelSpec,
    calibration: IterationCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, float]:
    """Return the analytical long-run capacity and its operating concurrency."""

    normalized = _normalize(weights)
    best_rate = 0.0
    best_nu = 1.0
    for nu in range(1, max(1, int(resident_limit)) + 1):
        services = [
            class_service_times(model, calibration, p, o, chunk_tokens, float(nu))
            for p, o in classes
        ]
        mean_service = _weighted_mean([item[2] for item in services], normalized)
        if mean_service > 0.0 and nu / mean_service > best_rate:
            best_rate = nu / mean_service
            best_nu = float(nu)
    return float(best_rate), float(best_nu)


def admission_queue_wait(
    arrival_rate: float,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    service_times: Sequence[tuple[float, float, float]],
    capacity_rate: float,
    capacity_concurrency: float,
) -> tuple[float, float, bool]:
    """M/G/1 first-admission waiting from an equivalent request workload.

    ``service_times`` are resident wall-clock service times at the selected
    capacity concurrency.  Dividing by that concurrency produces the
    request-equivalent workload seen by the admission queue.  This is a
    steady-state capacity abstraction, not a claim that GPU time is divided
    among requests by the runtime scheduler.
    """

    del classes  # retained in the signature to keep the workload interface explicit
    normalized = _normalize(weights)
    if arrival_rate <= 0.0:
        return 0.0, 0.0, False
    if capacity_rate <= 0.0:
        return math.inf, math.inf, True
    concurrency = max(1.0, float(capacity_concurrency))
    equivalent = [max(0.0, float(item[2])) / concurrency for item in service_times]
    # Keep utilization tied to the analytical capacity.  At the selected
    # capacity concurrency this equals the weighted first moment of the
    # equivalent request workload up to numerical rounding.
    mean_service = 1.0 / capacity_rate
    second_moment = _weighted_mean(
        [value**2 for value in equivalent], normalized
    )
    utilization = arrival_rate * mean_service
    if utilization >= 1.0:
        return math.inf, utilization, True
    waiting = arrival_rate * second_moment / (2.0 * max(1.0 - utilization, 1.0e-12))
    return float(waiting), float(utilization), False


def predict(
    arrival_rate: float,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    services: Sequence[tuple[float, float, float]],
    capacity_rate: float,
    capacity_concurrency: float,
    queue_service_times: Sequence[tuple[float, float, float]] | None = None,
) -> list[QueuePrediction]:
    """Add first-admission waiting to resident class performance."""

    waiting, utilization, overloaded = admission_queue_wait(
        arrival_rate,
        classes,
        weights,
        queue_service_times if queue_service_times is not None else services,
        capacity_rate,
        capacity_concurrency,
    )
    predictions: list[QueuePrediction] = []
    for (_, output), (prefill, decode, service) in zip(classes, services):
        tbt = decode / max(1, int(round(output)) - 1) if output > 1 else 0.0
        predictions.append(
            QueuePrediction(
                waiting_s=waiting,
                utilization=utilization,
                ttft_s=waiting + prefill,
                tbt_s=tbt,
                response_s=waiting + service,
                overloaded=overloaded,
            )
        )
    return predictions


def _normalize(weights: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(weights), dtype=float)
    if values.size == 0 or np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with positive mass")
    return values / float(values.sum())


def _weighted_mean(values: Sequence[float], weights: np.ndarray) -> float:
    return float(np.dot(np.asarray(values, dtype=float), weights))


def _decode_residency_weights(
    classes: Sequence[tuple[float, float]],
    arrival_weights: np.ndarray,
) -> np.ndarray:
    """Return active-decode class shares implied by Little's law.

    The prefill stream follows arrival shares.  A class remains in the decode
    population for one iteration per generated token, so its active-set mass
    is proportional to its arrival share and decode-token demand.
    """

    decode_demand = np.asarray(
        [max(0.0, float(output) - 1.0) for _, output in classes], dtype=float
    )
    resident = np.asarray(arrival_weights, dtype=float) * decode_demand
    if float(resident.sum()) <= 0.0:
        return np.asarray(arrival_weights, dtype=float)
    return resident / float(resident.sum())


def _workload_statistics(
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
) -> tuple[float, float, float, float, float]:
    """Return mean prompt, decode-token, chunk, context, and chunk-count demand."""

    normalized = _normalize(weights)
    prompts = np.asarray([max(1.0, float(p)) for p, _ in classes], dtype=float)
    decode = np.asarray([max(0.0, float(o) - 1.0) for _, o in classes], dtype=float)
    chunks = np.ceil(prompts / max(1, int(chunk_tokens)))
    chunk_mean = prompts / chunks
    # A chunk's existing context is its position in the prompt.  Weighting by
    # chunk count gives the context seen by a representative prefill chunk.
    context_mean = np.asarray(
        [
            float(np.mean(np.arange(int(c)) * max(1, int(chunk_tokens))))
            if int(c) > 0
            else 0.0
            for c in chunks
        ],
        dtype=float,
    )
    return (
        _weighted_mean(prompts, normalized),
        _weighted_mean(decode, normalized),
        _weighted_mean(chunk_mean, normalized),
        _weighted_mean(context_mean, normalized),
        _weighted_mean(chunks, normalized),
    )


def _mean_prefill_chunk_time(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    decode_concurrency: float,
    decode_context: float,
) -> float:
    """Mean mixed-iteration time for one arriving prefill chunk."""

    normalized = _normalize(weights)
    total_chunks = 0.0
    total_time = 0.0
    for (prompt_tokens, _), weight in zip(classes, normalized, strict=True):
        prompt = max(1, int(round(prompt_tokens)))
        chunks = math.ceil(prompt / max(1, int(chunk_tokens)))
        for q in range(chunks):
            context = float(q * max(1, int(chunk_tokens)))
            new_tokens = float(
                min(max(1, int(chunk_tokens)), prompt - int(context))
            )
            mixed_s, _ = _two_mode_iteration_times(
                model,
                calibration,
                new_tokens,
                context,
                decode_concurrency,
                decode_context,
            )
            total_time += float(weight) * mixed_s
            total_chunks += float(weight)
    return total_time / max(total_chunks, 1.0e-12)


def _packed_prefill_chunks(
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    token_budget: float,
) -> list[tuple[float, float]]:
    """Construct a deterministic representative packed prefill batch.

    A random prefill chunk belongs to class ``k`` with probability
    proportional to its request share times the number of chunks in that
    request.  The routine uses a deficit round-robin construction with this
    distribution, preserving each chunk's own context while filling one
    iteration token budget.
    """

    unit = max(1, int(chunk_tokens))
    budget = max(1.0, float(token_budget))
    normalized = _normalize(weights)
    candidates: list[tuple[float, float, float]] = []
    total_chunk_mass = 0.0
    for (prompt_tokens, _), weight in zip(classes, normalized, strict=True):
        prompt = max(1, int(round(prompt_tokens)))
        chunks = math.ceil(prompt / unit)
        for q in range(chunks):
            u = float(min(unit, prompt - q * unit))
            candidates.append((u, float(q * unit), float(weight)))
            total_chunk_mass += float(weight)
    if not candidates:
        return [(budget, 0.0)]

    # The accumulated deficit is proportional to the chunk-arrival mass.  It
    # gives an order-independent deterministic realization of the mixture.
    deficits = np.zeros(len(candidates), dtype=float)
    selected: list[tuple[float, float]] = []
    remaining = budget
    while remaining > 1.0e-9:
        deficits += np.asarray([item[2] for item in candidates], dtype=float)
        index = int(np.argmax(deficits))
        u, context, _ = candidates[index]
        take = min(u, remaining)
        selected.append((float(take), context))
        deficits[index] -= total_chunk_mass
        remaining -= take
    return selected


def _packed_mixed_iteration_times(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    prefill_token_budget: float,
    decode_concurrency: float,
    decode_context: float,
) -> tuple[float, float]:
    """Return mixed/decode iteration times for a packed prefill batch."""

    chunks = _packed_prefill_chunks(classes, weights, chunk_tokens, prefill_token_budget)
    return (
        _calibrated_packed_iteration_time(
            model, chunks, decode_concurrency, decode_context, calibration.mix
        ),
        _calibrated_packed_iteration_time(
            model, [], decode_concurrency, decode_context, calibration.decode
        ),
    )


def _two_mode_iteration_times(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    prefill_tokens: float,
    prefill_context: float,
    decode_concurrency: float,
    decode_context: float,
) -> tuple[float, float]:
    """Return ``(mixed, decode-only)`` iteration times."""

    nu = max(1.0, float(decode_concurrency))
    return (
        _calibrated_packed_iteration_time(
            model,
            [] if prefill_tokens <= 0.0 else [(prefill_tokens, prefill_context)],
            nu,
            decode_context,
            calibration.mix,
        ),
        _calibrated_packed_iteration_time(
            model, [], nu, decode_context, calibration.decode
        ),
    )


def two_mode_capacity(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    token_budget: int,
    resident_limit: int,
) -> tuple[float, float]:
    """Estimate request capacity from prefill/decode token-flow balance.

    At a candidate resident decode population ``nu``, a saturated scheduler
    can use the remaining token budget for prefill work.  One request then
    contributes ``P/u`` mixed iterations and ``D/nu`` decode iterations, where
    ``P`` and ``D`` are its mean prompt and decode-token demands.  The mixed
    iterations are counted once and the remaining decode iterations use the
    decode-only service time.  This is a fluid approximation of the
    iteration server, not a collection of independent service positions.
    """

    best_rate = 0.0
    best_nu = 1.0
    for nu_i in range(1, max(1, int(resident_limit)) + 1):
        nu = float(nu_i)
        candidate = two_mode_capacity_at_concurrency(
            model, calibration, classes, weights, chunk_tokens, token_budget, nu
        )
        if candidate > best_rate:
            best_rate = candidate
            best_nu = nu
    return float(best_rate), float(best_nu)


def two_mode_capacity_at_concurrency(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    chunk_tokens: int,
    token_budget: int,
    decode_concurrency: float,
) -> float:
    """Return the token-flow request capacity at one resident population."""

    p_mean, d_mean, _, _, _ = _workload_statistics(classes, weights, chunk_tokens)
    arrival_weights = _normalize(weights)
    decode_weights = _decode_residency_weights(classes, arrival_weights)
    decode_context = _weighted_mean(
        [max(1.0, float(p) + max(0.0, float(o) - 1.0) / 2.0) for p, o in classes],
        decode_weights,
    )
    nu = max(1.0, float(decode_concurrency))
    available = max(1.0, float(token_budget) - nu)
    mixed_s, decode_s = _packed_mixed_iteration_times(
        model,
        calibration,
        classes,
        weights,
        chunk_tokens,
        available,
        nu,
        decode_context,
    )
    mixed_iterations_per_request = p_mean / available
    decode_iterations_per_request = d_mean / nu
    work_s = mixed_iterations_per_request * mixed_s
    work_s += max(0.0, decode_iterations_per_request - mixed_iterations_per_request) * decode_s
    return 0.0 if work_s <= 0.0 else float(1.0 / work_s)


def two_mode_operating_point(
    model: ModelSpec,
    calibration: TwoModeCalibration,
    classes: Sequence[tuple[float, float]],
    weights: Sequence[float],
    arrival_rate: float,
    chunk_tokens: int,
    token_budget: int,
    resident_limit: int,
    initial_concurrency: float | None = None,
    tolerance: float = 1.0e-6,
    max_iterations: int = 200,
) -> TwoModeSteadyState:
    """Solve a scalar fluid fixed point for the mixed/decode iteration server."""

    p_mean, d_mean, chunk_mean, _, chunk_count = _workload_statistics(
        classes, weights, chunk_tokens
    )
    normalized = _normalize(weights)
    decode_weights = _decode_residency_weights(classes, normalized)
    decode_context = _weighted_mean(
        [max(1.0, float(p) + max(0.0, float(o) - 1.0) / 2.0) for p, o in classes],
        decode_weights,
    )
    capacity_rate, capacity_nu = two_mode_capacity(
        model, calibration, classes, weights, chunk_tokens, token_budget, resident_limit
    )
    # The coupled workload equations can have a low-concurrency and a
    # high-concurrency fixed point near saturation.  The long-run operating
    # branch is initialized from the offered load relative to the analytical
    # capacity; this avoids always selecting the empty-system branch.
    initial = (
        float(capacity_nu) * arrival_rate / max(capacity_rate, 1.0e-12)
        if initial_concurrency is None
        else float(initial_concurrency)
    )
    nu = max(1.0, min(float(resident_limit), initial))
    u = max(1.0, min(chunk_mean, float(token_budget) - nu))
    mixed_s = decode_s = mean_s = 0.0
    mixed_fraction = utilization = 0.0
    converged = False
    for _ in range(max_iterations):
        nu = min(float(resident_limit), max(1.0, nu))
        decode_weights = _decode_residency_weights(classes, normalized)
        decode_context = _weighted_mean(
            [max(1.0, float(p) + max(0.0, float(o) - 1.0) / 2.0) for p, o in classes],
            decode_weights,
        )
        u_max = max(1.0, float(token_budget) - nu)
        # A waiting request is admitted in chunks.  If the mean chunk stream
        # would occupy the iteration server beyond one, waiting chunks build
        # up and the scheduler fills the remaining token budget.  This
        # switch follows the queue balance and does not introduce a fitted
        # composition-dependent threshold.
        single_mixed_s = _mean_prefill_chunk_time(
            model, calibration, classes, weights, chunk_tokens, nu, decode_context
        )
        single_prefill_load = arrival_rate * chunk_count * single_mixed_s
        if single_prefill_load < 1.0:
            u = max(1.0, min(chunk_mean, u_max))
            mixed_s = single_mixed_s
            mixed_rate = arrival_rate * chunk_count
        else:
            u = u_max
            mixed_s, _ = _packed_mixed_iteration_times(
                model,
                calibration,
                classes,
                weights,
                chunk_tokens,
                u,
                nu,
                decode_context,
            )
            mixed_rate = arrival_rate * p_mean / max(u, 1.0)
        _, decode_s = _two_mode_iteration_times(
            model, calibration, 0.0, decode_context, nu, decode_context
        )
        total_rate = arrival_rate * d_mean / max(nu, 1.0)
        decode_only_rate = max(0.0, total_rate - mixed_rate)
        busy = mixed_rate * mixed_s + decode_only_rate * decode_s
        denominator = max(total_rate, mixed_rate, 1.0e-12)
        mean_s = busy / denominator
        next_nu = min(float(resident_limit), max(1.0, arrival_rate * d_mean * mean_s))
        if abs(next_nu - nu) <= tolerance * max(1.0, next_nu):
            nu = next_nu
            converged = True
            break
        nu = 0.5 * nu + 0.5 * next_nu

    utilization = float(busy)
    # The active iteration stream is work-conserving, so its busy fraction is
    # close to one even for a stable long-lived workload.  Stability is
    # determined by the request-rate boundary, not by treating that active
    # stream as an independent utilization measurement.
    overloaded = bool(arrival_rate > capacity_rate * (1.0 + 1.0e-9))
    mixed_fraction = float(min(1.0, mixed_rate / max(denominator, 1.0e-12)))
    if not converged and arrival_rate > capacity_rate:
        overloaded = True
    return TwoModeSteadyState(
        decode_concurrency=float(nu),
        prefill_tokens_per_mixed_iteration=float(u),
        mixed_iteration_s=float(mixed_s),
        decode_iteration_s=float(decode_s),
        mean_iteration_s=float(mean_s),
        mixed_fraction=mixed_fraction,
        utilization=utilization,
        capacity_rps=float(capacity_rate),
        overloaded=overloaded,
    )


def two_mode_first_admission_wait(
    state: TwoModeSteadyState,
    arrival_rate: float,
    prefill_chunk_mean: float,
    prefill_chunks_per_request: float = 1.0,
    capacity_rate: float | None = None,
) -> tuple[float, float, bool]:
    """Estimate the first running-set admission wait.

    The queue contains prefill chunks awaiting a first opportunity to enter
    the running set.  A mixed iteration can admit several representative
    chunks.  If the call stream has mean chunk count ``h``, the chunk arrival
    rate is ``arrival_rate * h`` and the token packing factor determines the
    bulk-service utilization.  The M/D/1 residual-service term uses the
    duration of a complete mixed iteration, because a queued chunk observes
    the residual wall-clock time of that iteration before it can be admitted.
    Post-admission prefill and decode execution remain in the iteration
    service terms.
    """

    del capacity_rate  # request-level stability is checked by the caller.
    if arrival_rate <= 0.0:
        return 0.0, 0.0, False
    opportunity = state.mixed_iteration_s * max(1.0, prefill_chunk_mean) / max(
        state.prefill_tokens_per_mixed_iteration, 1.0
    )
    rho = (
        arrival_rate
        * max(1.0, float(prefill_chunks_per_request))
        * opportunity
    )
    if rho >= 1.0:
        return math.inf, rho, True
    wait = 0.5 * state.mixed_iteration_s * rho / max(1.0 - rho, 1.0e-12)
    return float(wait), float(rho), False
