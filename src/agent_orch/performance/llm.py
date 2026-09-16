from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agent_orch.schema.models import LLMConfigSpec, ModelSpec


@dataclass(frozen=True)
class ServiceDemand:
    """Processing demand of one class of LLM calls on one instance."""

    prefill_s: float
    decode_s: float
    service_s: float
    mean_iteration_s: float
    kv_work_tokens: float
    decode_concurrency: float


def roofline_time(flops: float, memory_bytes: float, config: LLMConfigSpec) -> float:
    return max(
        flops / config.effective_flops,
        memory_bytes / config.effective_bandwidth_bytes_s,
    )


def prefill_work(
    model: ModelSpec,
    new_tokens: float,
    context_tokens: float,
    concurrency: float,
) -> tuple[float, float]:
    u = new_tokens
    chi = context_tokens
    nu = concurrency
    flops = nu * (
        2.0 * model.parameter_count * u
        + 4.0 * model.layers * model.hidden_size * u * (chi + (u + 1.0) / 2.0)
    )
    # Minimum HBM traffic of one prefill chunk: the chunk reads the cached
    # context once and writes its own KV.  Charging one KV read per query token
    # would describe the arithmetic of attention, not the traffic a tiled
    # kernel moves, and would make prefill memory bound on every GPU.
    memory = model.weight_bytes + nu * model.kv_bytes_per_token * (chi + u)
    return flops, memory


def decode_work(
    model: ModelSpec,
    context_tokens: float,
    concurrency: float,
) -> tuple[float, float]:
    chi = context_tokens
    nu = concurrency
    flops = nu * (
        2.0 * model.parameter_count
        + 4.0 * model.layers * model.hidden_size * (chi + 1.0)
    )
    memory = model.weight_bytes + nu * model.kv_bytes_per_token * (chi + 1.0)
    return flops, memory


def iteration_count(prompt_tokens: float, output_tokens: float, chunk_tokens: int) -> int:
    """Iterations one call occupies: one per prefill chunk, one per decode token."""
    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    return math.ceil(prompt / chunk_tokens) + max(0, output - 1)


def kv_lifetime_work(prompt_tokens: float, output_tokens: float, chunk_tokens: int) -> float:
    """Cumulative KV token occupancy of one call over its lifetime."""
    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    return (
        (1.0 + prompt / chunk_tokens) * prompt / 2.0
        + prompt * output
        + (1.0 + output) * output / 2.0
    )


@dataclass(frozen=True)
class InstanceResidency:
    """KV-cache residency of one instance under a mixed call composition."""

    capacity: int
    kv_slack: float
    active_kv_tokens: float


def residency_capacity(
    classes: list[tuple[float, float]],
    weights: list[float],
    kv_token_capacity: float,
    max_num_seqs: int,
    chunk_tokens: int,
) -> InstanceResidency:
    """KV/sequence residency limit of one instance.

    The limit is set by the longest single call that must still fit next to the
    running set (``kv_slack``) and by the mean KV footprint of one resident
    request, which is the lifetime KV work per iteration.  It is a residency
    limit, not a number of independent servers.
    """
    total = sum(weights)
    if total <= 0.0:
        return InstanceResidency(0, 0.0, 0.0)
    kv_slack = max(
        0.0,
        1.0 - max((prompt + output) / kv_token_capacity for prompt, output in classes),
    )
    denominator = sum(
        weight * iteration_count(prompt, output, chunk_tokens)
        for (prompt, output), weight in zip(classes, weights)
    )
    active_kv = (
        sum(
            weight * kv_lifetime_work(prompt, output, chunk_tokens)
            for (prompt, output), weight in zip(classes, weights)
        )
        / max(denominator, 1.0e-12)
    )
    capacity = min(
        int(max_num_seqs),
        int((kv_slack * kv_token_capacity) / max(active_kv, 1.0e-12)),
    )
    return InstanceResidency(capacity, kv_slack, active_kv)


@dataclass(frozen=True)
class ServiceCurve:
    """Piecewise-linear service curve of one call class on one fixed instance.

    One iteration of a continuous-batching instance carries the resident decode
    sequences and, when chunked prefill is enabled, at most one prefill chunk.
    Writing the Roofline maximum of such an iteration as the maximum of two
    affine functions of the decode concurrency ``nu`` makes every stage cost an
    explicit piecewise-linear function of ``nu``:

        ``D(nu) = prefill_base + prefill_slope * nu
                  + sum_q max(0, prefill_delta_q + prefill_epsilon_q * nu)
                  + decode_base + decode_slope * nu
                  + sum_t max(0, decode_slope_t * nu - weight_read_s)``

    The prefill sum has one term per admitted chunk and is evaluated directly.
    The decode sum is evaluated through sorted slopes and a prefix sum, so a
    whole curve costs ``O(chunks * |nu| + log n_decode)``.
    """

    prefill_base: float
    prefill_slope: float
    prefill_delta: np.ndarray
    prefill_epsilon: np.ndarray
    weight_read_s: float
    n_decode: int
    decode_slope_sum: float
    decode_delta_total: float
    decode_delta_cumsum: np.ndarray
    decode_delta_sorted: np.ndarray
    kv_work_tokens: float

    def _prefill_excess(self, nu: np.ndarray) -> np.ndarray:
        if self.prefill_delta.size == 0:
            return np.zeros_like(nu)
        excess = self.prefill_delta[:, None] + self.prefill_epsilon[:, None] * nu[None, :]
        return np.maximum(0.0, excess).sum(axis=0)

    def prefill_on_grid(self, decode_concurrency: np.ndarray) -> np.ndarray:
        nu = np.maximum(1.0, np.asarray(decode_concurrency, dtype=float))
        return self.prefill_base + self.prefill_slope * nu + self._prefill_excess(nu)

    def decode_on_grid(self, decode_concurrency: np.ndarray) -> np.ndarray:
        nu = np.maximum(1.0, np.asarray(decode_concurrency, dtype=float))
        total = self.decode_delta_sorted.size
        counts = total - np.searchsorted(
            self.decode_delta_sorted, self.weight_read_s / nu, side="right"
        )
        tail_sum = self.decode_delta_total - self.decode_delta_cumsum[total - counts]
        return (
            self.weight_read_s * self.n_decode
            + self.decode_slope_sum * nu
            + nu * tail_sum
            - self.weight_read_s * counts
        )

    def service_on_grid(self, decode_concurrency: np.ndarray) -> np.ndarray:
        return self.prefill_on_grid(decode_concurrency) + self.decode_on_grid(
            decode_concurrency
        )

    def service_at(self, decode_concurrency: float) -> float:
        return float(self.service_on_grid(np.asarray([decode_concurrency], dtype=float))[0])

    def prefill_at(self, decode_concurrency: float) -> float:
        return float(
            self.prefill_on_grid(np.asarray([decode_concurrency], dtype=float))[0]
        )

    def decode_at(self, decode_concurrency: float) -> float:
        return float(
            self.decode_on_grid(np.asarray([decode_concurrency], dtype=float))[0]
        )


def mean_decode_context(
    classes: list[tuple[float, float]], weights: list[float]
) -> float:
    """Rate-weighted context length of one resident decode sequence."""
    total = sum(weights)
    if total <= 0.0:
        return 1.0
    return (
        sum(
            weight * (prompt + (max(1.0, output) - 1.0) / 2.0)
            for (prompt, output), weight in zip(classes, weights)
        )
        / total
    )


def resident_decode_context(
    classes: list[tuple[float, float]],
    weights: list[float],
    curves: list[ServiceCurve],
    decode_concurrency: float,
) -> float:
    """Context length of the active decode population.

    An arrival-rate weighted mean describes an arriving request, whereas the
    decode batch is sampled from requests that are currently resident.  The
    latter population is length-biased by its decode residence time.  The
    service curves provide that residence time without introducing request-
    level state into the steady-state model.
    """
    residence = np.asarray(
        [max(curve.decode_at(decode_concurrency), 0.0) for curve in curves],
        dtype=float,
    )
    weighted = np.asarray(weights, dtype=float) * residence
    contexts = np.asarray(
        [prompt + (max(1.0, output) - 1.0) / 2.0 for prompt, output in classes],
        dtype=float,
    )
    total = float(weighted.sum())
    if total <= 1.0e-12:
        return mean_decode_context(classes, weights)
    return float(np.dot(weighted, contexts) / total)


def service_curve(
    model: ModelSpec,
    config: LLMConfigSpec,
    prompt_tokens: float,
    output_tokens: float,
    chunk_tokens: int,
    peer_decode_context: float | None = None,
) -> ServiceCurve:
    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    if peer_decode_context is None:
        peer_decode_context = prompt + (output - 1.0) / 2.0
    peer_context = max(1.0, float(peer_decode_context))

    # Decode work that each resident sequence adds to any iteration.
    decode_flops, decode_memory = decode_work(model, peer_context, 1.0)

    chunks = math.ceil(prompt / chunk_tokens)
    prefill_base = 0.0
    prefill_slope = 0.0
    deltas: list[float] = []
    epsilons: list[float] = []
    for q in range(chunks):
        context = float(q * chunk_tokens)
        new_tokens = float(min(chunk_tokens, prompt - context))
        flops, memory = prefill_work(model, new_tokens, context, 1.0)
        chunk_compute = flops / config.effective_flops
        chunk_memory = memory / config.effective_bandwidth_bytes_s
        decode_compute = decode_flops / config.effective_flops
        decode_memory_rate = decode_memory / config.effective_bandwidth_bytes_s
        # max(chunk_compute + nu * decode_compute, chunk_memory + nu * decode_memory_rate)
        prefill_base += chunk_memory
        prefill_slope += decode_memory_rate
        deltas.append(chunk_compute - chunk_memory)
        epsilons.append(decode_compute - decode_memory_rate)

    contexts = prompt + np.arange(max(0, output - 1), dtype=float)
    flops = (
        2.0 * model.parameter_count
        + 4.0 * model.layers * model.hidden_size * (contexts + 1.0)
    )
    kv_bytes = model.kv_bytes_per_token * (contexts + 1.0)
    alpha = flops / config.effective_flops
    gamma = kv_bytes / config.effective_bandwidth_bytes_s
    weight_read = model.weight_bytes / config.effective_bandwidth_bytes_s
    deltas_sorted = np.sort(np.maximum(0.0, alpha - gamma))
    return ServiceCurve(
        prefill_base=prefill_base,
        prefill_slope=prefill_slope,
        prefill_delta=np.asarray(deltas, dtype=float),
        prefill_epsilon=np.asarray(epsilons, dtype=float),
        weight_read_s=weight_read,
        n_decode=int(max(0, output - 1)),
        decode_slope_sum=float(gamma.sum()),
        decode_delta_total=float(deltas_sorted.sum()),
        decode_delta_cumsum=np.concatenate(([0.0], np.cumsum(deltas_sorted))),
        decode_delta_sorted=deltas_sorted,
        kv_work_tokens=kv_lifetime_work(prompt, output, chunk_tokens),
    )


def service_demand(
    model: ModelSpec,
    config: LLMConfigSpec,
    prompt_tokens: float,
    output_tokens: float,
    chunk_tokens: int,
    decode_concurrency: float = 1.0,
    peer_decode_context: float | None = None,
) -> ServiceDemand:
    """Return the prefill, decode, and total demand of one LLM call.

    ``decode_concurrency`` is the steady number of resident decode sequences
    that share every iteration of the target instance.  The call's prefill
    chunks each occupy one iteration of their own, so its time to first token is
    the sum of those iterations at that concurrency, while each decoded token
    costs one further iteration.
    """
    prompt = max(1, int(round(prompt_tokens)))
    output = max(1, int(round(output_tokens)))
    concurrency = max(1.0, float(decode_concurrency))
    curve = service_curve(
        model, config, prompt, output, chunk_tokens, peer_decode_context
    )
    prefill = curve.prefill_at(concurrency)
    decode = curve.decode_at(concurrency)
    service = prefill + decode
    iterations = iteration_count(prompt, output, chunk_tokens)
    return ServiceDemand(
        prefill_s=prefill,
        decode_s=decode,
        service_s=service,
        mean_iteration_s=service / max(1, iterations),
        kv_work_tokens=curve.kv_work_tokens,
        decode_concurrency=concurrency,
    )


def mean_service_time(
    curves: list[ServiceCurve],
    weights: list[float],
    decode_concurrency: float,
) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return sum(
        weight * curve.service_at(decode_concurrency)
        for weight, curve in zip(weights, curves)
    ) / total


def throughput_capacity(
    curves: list[ServiceCurve],
    weights: list[float],
    residency_capacity: int,
) -> tuple[float, float]:
    """Maximum stable call rate of one instance and the concurrency attaining it.

    Returns ``(mu, nu_star)`` with ``mu = max_{1 <= nu <= C_run} nu / mean_nu``,
    where ``mean_nu`` is the rate-weighted mean service time at decode
    concurrency ``nu``.  ``C_run`` is the KV/sequence residency limit, not a
    number of independent servers.
    """
    total = sum(weights)
    if total <= 0.0 or residency_capacity < 1:
        return 0.0, 1.0
    grid = np.arange(1, int(residency_capacity) + 1, dtype=float)
    mean = np.zeros_like(grid)
    for weight, curve in zip(weights, curves):
        mean += (weight / total) * curve.service_on_grid(grid)
    rate = grid / mean
    index = int(np.argmax(rate))
    return float(rate[index]), float(grid[index])


def steady_active_concurrency(
    curves: list[ServiceCurve],
    weights: list[float],
    total_rate: float,
    residency_capacity: int,
    tolerance: float = 1.0e-6,
    max_iterations: int = 500,
) -> tuple[float, bool, float]:
    """Least fixed point of Little's law under the instance service curve.

    Returns ``(bar_B, converged, residual)``.  The iteration is not clipped to
    the residency limit: crossing that limit is reported as overload instead.
    """
    batch = 0.0
    residual = math.inf
    converged = False
    for _ in range(max_iterations):
        target = total_rate * mean_service_time(curves, weights, max(1.0, batch))
        residual = abs(target - batch)
        if residual <= tolerance * max(1.0, target):
            batch = target
            converged = True
            break
        batch = target
        if batch > residency_capacity:
            break
    return batch, converged, residual
