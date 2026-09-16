from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np

from agent_orch.schema.models import (
    DeploymentDecision,
    LLMClassPerformance,
    LLMConfigSpec,
    ModelSpec,
    NodeType,
    RoutingDecision,
    Scenario,
)

from .llm import (
    mean_decode_context,
    mean_service_time,
    residency_capacity,
    resident_decode_context,
    service_curve,
    steady_active_concurrency,
    throughput_capacity,
)
from .llm_queueing import (
    IterationCalibration,
    TwoModeCalibration,
    two_mode_capacity,
    two_mode_first_admission_wait,
    two_mode_operating_point,
)
from .network import Edge, NetworkBackend
from .queueing import tool_response_time


LLMClass = tuple[str, str, str]
ToolPool = tuple[str, str]

@dataclass
class AnalyticalResult:
    llm_performance: dict[LLMClass, LLMClassPerformance]
    llm_utilization: dict[str, float]
    llm_kv_stable: dict[str, bool]
    tool_delay: dict[ToolPool, float]
    tool_utilization: dict[ToolPool, float]
    link_load_mbps: dict[Edge, float]
    node_server_distribution: dict[tuple[str, str, str, str, str], dict[str, float]]
    violations: list[str] = field(default_factory=list)
    llm_instance_performance: dict[str, "LLMInstancePerformance"] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LLMInstancePerformance:
    arrival_rate_rps: float
    active_concurrency: float
    active_kv_tokens: float
    resident_capacity: int
    kv_slack: float
    mean_service_s: float
    throughput_capacity_rps: float
    capacity_concurrency: float
    utilization: float
    stable: bool
    fixed_point_residual: float


def _build_llm_curves(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    chunk_tokens: int,
    peer_context: float,
) -> list:
    """Build class curves for one common active decode context."""
    return [
        service_curve(
            model,
            config,
            prompt,
            output,
            chunk_tokens,
            peer_decode_context=peer_context,
        )
        for prompt, output in classes
    ]


def _resident_operating_point(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    arrival_rate_rps: float,
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, list, bool, float]:
    """Solve the low-dimensional active-set composition fixed point.

    The old approximation represented one batch with an arrival-rate weighted
    context.  The resident version updates that context using the residence
    time of each class, while retaining a single steady-state concurrency.
    Thus it captures prefill-heavy/decode-heavy composition without adding
    request-level scheduler state.
    """
    peer_context = mean_decode_context(classes, weights)
    batch = 0.0
    residual = math.inf
    converged = False
    for _ in range(100):
        curves = _build_llm_curves(
            model, config, classes, chunk_tokens, peer_context
        )
        next_batch = arrival_rate_rps * mean_service_time(
            curves, weights, max(1.0, batch)
        )
        next_context = resident_decode_context(
            classes, weights, curves, max(1.0, next_batch)
        )
        updated_context = 0.5 * peer_context + 0.5 * next_context
        residual = max(
            abs(next_batch - batch),
            abs(updated_context - peer_context),
        )
        batch = next_batch
        peer_context = updated_context
        if residual <= 1.0e-7 * max(1.0, batch, peer_context):
            converged = True
            break
        if batch > max(1, resident_limit) * 2.0:
            break

    curves = _build_llm_curves(model, config, classes, chunk_tokens, peer_context)
    return batch, curves, converged, residual


def _resident_throughput_capacity(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, float]:
    """Maximize throughput while recomputing the active composition at each
    candidate concurrency.

    This is a one-dimensional steady-state search, not a request-level
    simulation or a fitted workload table.
    """
    if resident_limit < 1:
        return 0.0, 0.0
    best_rate = 0.0
    best_concurrency = 1.0
    initial_context = mean_decode_context(classes, weights)
    for nu in range(1, resident_limit + 1):
        peer_context = initial_context
        for _ in range(100):
            curves = _build_llm_curves(
                model, config, classes, chunk_tokens, peer_context
            )
            next_context = resident_decode_context(
                classes, weights, curves, float(nu)
            )
            updated = 0.5 * peer_context + 0.5 * next_context
            if abs(updated - peer_context) <= 1.0e-7 * max(1.0, updated):
                peer_context = updated
                break
            peer_context = updated
        curves = _build_llm_curves(model, config, classes, chunk_tokens, peer_context)
        mean_time = mean_service_time(curves, weights, float(nu))
        if mean_time > 0.0 and nu / mean_time > best_rate:
            best_rate = nu / mean_time
            best_concurrency = float(nu)
    return best_rate, best_concurrency


def _resident_curves_at_concurrency(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    chunk_tokens: int,
    decode_concurrency: float,
) -> tuple[list, float]:
    """Return curves and the self-consistent active decode context."""
    peer_context = mean_decode_context(classes, weights)
    for _ in range(100):
        curves = _build_llm_curves(
            model, config, classes, chunk_tokens, peer_context
        )
        next_context = resident_decode_context(
            classes, weights, curves, max(1.0, decode_concurrency)
        )
        updated = 0.5 * peer_context + 0.5 * next_context
        if abs(updated - peer_context) <= 1.0e-7 * max(1.0, updated):
            peer_context = updated
            break
        peer_context = updated
    return (
        _build_llm_curves(model, config, classes, chunk_tokens, peer_context),
        peer_context,
    )


def _occupancy_ps_operating_point(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    arrival_rate_rps: float,
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, list, bool, float, float]:
    """Solve a macro steady-state operating point for continuous batching.

    ``batch`` is the mean resident decode population rather than an
    independent service position count.  The Roofline curve supplies the
    composition-dependent work at that resident population, while the
    occupancy factor models the additional residence caused by sharing the
    finite running set.  This keeps the model at one scalar fixed point and
    avoids treating an LLM instance as a pool of independent servers.
    """
    if resident_limit < 1 or sum(weights) <= 0.0:
        return float(resident_limit + 1), [], False, math.inf, math.inf
    # Solve F(nu)=0 by bracketing the first stable root.  A damped fixed-point
    # iteration is not reliable close to C_run because the derivative of the
    # occupancy factor becomes large.  The first sign change is the low-
    # occupancy stable branch; if it does not exist, the offered load is
    # outside the finite steady-state region.
    upper = max(1.0, float(resident_limit) * (1.0 - 1.0e-6))

    def residual_at(nu: float) -> float:
        resident_nu = min(max(0.0, float(nu)), upper)
        eval_nu = min(max(1.0, resident_nu), upper)
        curves_at, _ = _resident_curves_at_concurrency(
            model, config, classes, weights, chunk_tokens, eval_nu
        )
        base_service = mean_service_time(curves_at, weights, eval_nu)
        occupancy = resident_nu / max(float(resident_limit), 1.0)
        return resident_nu - arrival_rate_rps * base_service / max(
            1.0 - occupancy, 1.0e-9
        )

    # A coarse grid is sufficient for this macro model.  The service curve is
    # already piecewise smooth, and using a small fixed grid avoids turning a
    # one-dimensional steady-state calculation into a costly inner solver.
    grid = np.linspace(0.0, upper, max(20, min(32, resident_limit // 4 + 8)))
    values = np.asarray([residual_at(value) for value in grid], dtype=float)
    bracket: tuple[float, float] | None = None
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if f_left <= 0.0 <= f_right:
            bracket = (float(left), float(right))
            break

    if bracket is None:
        # Preserve finite fallback values for the simulator/optimizer, but
        # mark the operating point as non-converged so callers can treat it as
        # unstable rather than plotting it as a physical saturation plateau.
        batch = float(resident_limit + 1)
        eval_batch = upper
        residual = math.inf
        converged = False
    else:
        left, right = bracket
        f_left = residual_at(left)
        f_right = residual_at(right)
        # Linear interpolation on the first stable bracket is consistent with
        # the deliberately coarse macro abstraction and avoids a second inner
        # fixed-point iteration.
        denominator = f_left - f_right
        fraction = f_left / denominator if abs(denominator) > 1.0e-12 else 0.5
        batch = left + min(max(fraction, 0.0), 1.0) * (right - left)
        eval_batch = min(max(1.0, batch), upper)
        residual = abs(residual_at(batch))
        converged = True
    curves, _ = _resident_curves_at_concurrency(
        model, config, classes, weights, chunk_tokens, eval_batch
    )
    occupancy = min(max(0.0, batch), upper) / max(float(resident_limit), 1.0)
    slowdown = 1.0 / max(1.0 - occupancy, 1.0e-3)
    return batch, curves, converged, residual, slowdown


def _occupancy_ps_capacity(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, float]:
    """Return the stability boundary implied by the occupancy fixed point.

    At a resident population ``nu``, the fixed point implies

        Lambda(nu) = nu (1 - nu/C_run) / S(nu).

    The maximum of this expression is a derived stability boundary, not an
    independent-server service rate and is not used to scale each request's
    service time.
    """
    if resident_limit < 2 or sum(weights) <= 0.0:
        return 0.0, 1.0
    best_rate = 0.0
    best_nu = 1.0
    for nu in range(1, resident_limit):
        curves, _ = _resident_curves_at_concurrency(
            model, config, classes, weights, chunk_tokens, float(nu)
        )
        base_service = mean_service_time(curves, weights, float(nu))
        rate = nu * (1.0 - nu / float(resident_limit)) / max(
            base_service, 1.0e-12
        )
        if rate > best_rate:
            best_rate = rate
            best_nu = float(nu)
    return best_rate, best_nu


def _two_mode_statistics(
    classes: list[tuple[float, float]],
    weights: list[float],
    curves: list,
    decode_concurrency: float,
    chunk_tokens: int,
) -> tuple[float, float, float, float]:
    """Aggregate prefill chunks and decode tokens at one concurrency.

    Returns ``(mean_prefill_s, mean_decode_iteration_s, mean_chunks,
    mean_decode_tokens)``.  The prefill stream is weighted by chunks per
    request; the decode stream is length-biased by decode residence time.  The
    two streams are therefore retained separately instead of being represented
    by one request-level batch.
    """
    total = sum(weights)
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    request_weights = np.asarray(weights, dtype=float) / total
    chunks = np.asarray(
        [math.ceil(max(1, int(round(prompt))) / chunk_tokens) for prompt, _ in classes],
        dtype=float,
    )
    decode_tokens = np.asarray(
        [max(0, int(round(output)) - 1) for _, output in classes],
        dtype=float,
    )
    prefill = np.asarray(
        [curve.prefill_at(max(1.0, decode_concurrency)) for curve in curves],
        dtype=float,
    )
    decode = np.asarray(
        [curve.decode_at(max(1.0, decode_concurrency)) for curve in curves],
        dtype=float,
    )
    residence_weights = request_weights * decode
    residence_total = float(residence_weights.sum())
    if residence_total <= 1.0e-12:
        mean_iteration = float(np.dot(request_weights, decode / np.maximum(decode_tokens, 1.0)))
    else:
        mean_iteration = float(
            np.dot(
                residence_weights,
                decode / np.maximum(decode_tokens, 1.0),
            )
            / residence_total
        )
    return (
        float(np.dot(request_weights, prefill)),
        mean_iteration,
        float(np.dot(request_weights, chunks)),
        float(np.dot(request_weights, decode_tokens)),
    )


def _two_mode_operating_point(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    arrival_rate_rps: float,
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, list, bool, float, float]:
    """Solve the steady state of the prefill and decode streams."""
    batch = 0.0
    residual = math.inf
    converged = False
    tau_iter = 0.0
    for _ in range(100):
        curves, _ = _resident_curves_at_concurrency(
            model, config, classes, weights, chunk_tokens, max(1.0, batch)
        )
        mean_prefill, tau_iter, mean_chunks, mean_decode_tokens = _two_mode_statistics(
            classes, weights, curves, max(1.0, batch), chunk_tokens
        )
        # A prefill iteration already contains the decode work of the active
        # set.  Only the excess over a decode-only iteration consumes additional
        # iteration capacity; charging the whole prefill time would count the
        # shared decode baseline twice.
        extra_prefill = max(
            0.0, mean_prefill - mean_chunks * tau_iter
        )
        if arrival_rate_rps * extra_prefill >= 1.0:
            next_tau = math.inf
        else:
            next_tau = tau_iter / max(
                1.0 - arrival_rate_rps * extra_prefill,
                1.0e-12,
            )
        next_batch = (
            arrival_rate_rps * mean_decode_tokens * next_tau
            if math.isfinite(next_tau)
            else math.inf
        )
        residual = abs(next_batch - batch)
        batch = next_batch
        if not math.isfinite(batch):
            batch = float(resident_limit + 1)
            break
        if residual <= 1.0e-7 * max(1.0, batch):
            converged = True
            break
        if batch > max(1, resident_limit) * 2.0:
            break
    curves, _ = _resident_curves_at_concurrency(
        model, config, classes, weights, chunk_tokens, max(1.0, batch)
    )
    return batch, curves, converged, residual, tau_iter


def _two_mode_capacity(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    chunk_tokens: int,
    resident_limit: int,
) -> tuple[float, float]:
    """Capacity from separate prefill and decode stability boundaries."""
    if resident_limit < 1 or sum(weights) <= 0.0:
        return 0.0, 1.0
    best_rate = 0.0
    best_nu = 1.0
    total_output = sum(
        weight * max(0, int(round(output)) - 1)
        for (_, output), weight in zip(classes, weights)
    ) / sum(weights)
    for nu in range(1, resident_limit + 1):
        curves, _ = _resident_curves_at_concurrency(
            model, config, classes, weights, chunk_tokens, float(nu)
        )
        mean_prefill, tau_iter, _, _ = _two_mode_statistics(
            classes, weights, curves, float(nu), chunk_tokens
        )
        mean_chunks = sum(
            weight * math.ceil(max(1, int(round(prompt))) / chunk_tokens)
            for (prompt, _), weight in zip(classes, weights)
        ) / sum(weights)
        extra_prefill = max(0.0, mean_prefill - mean_chunks * tau_iter)
        # At concurrency nu, Little's law gives
        # nu = Lambda * E[O] * tau_iter/(1-Lambda*E_extra).
        # Solving this relation for Lambda avoids introducing a fitted
        # congestion coefficient and separates the two workload streams.
        candidate = nu / max(
            total_output * tau_iter + nu * extra_prefill,
            1.0e-12,
        )
        if candidate > best_rate:
            best_rate = candidate
            best_nu = float(nu)
    return best_rate, best_nu


def evaluate_llm_instance(
    model: ModelSpec,
    config: LLMConfigSpec,
    classes: list[tuple[float, float]],
    weights: list[float],
    arrival_rate_rps: float,
    chunk_tokens: int,
    composition_mode: str = "resident",
) -> tuple[LLMInstancePerformance, list[LLMClassPerformance]]:
    """Steady-state performance of one LLM instance under mixed call classes.

    ``classes`` holds the ``(prompt tokens, output tokens)`` of every call class
    routed to the instance and ``weights`` its share of the call rate.  The
    instance runs the fixed deployment configuration in ``config``; only its
    class composition and total offer rate vary.
    """
    if composition_mode not in {"arrival", "resident", "occupancy_ps", "two_mode", "macro"}:
        raise ValueError(
            "composition_mode must be 'arrival', 'resident', 'occupancy_ps', 'two_mode', or 'macro'"
        )
    residency = residency_capacity(
        classes,
        weights,
        config.kv_token_capacity,
        config.max_num_seqs,
        chunk_tokens,
    )
    if composition_mode == "arrival":
        peer_context = mean_decode_context(classes, weights)
        curves = _build_llm_curves(
            model, config, classes, chunk_tokens, peer_context
        )
        # Baseline: one representative batch formed from arriving requests.
        batch, converged, residual = steady_active_concurrency(
            curves, weights, arrival_rate_rps, residency.capacity
        )
        capacity, capacity_concurrency = throughput_capacity(
            curves, weights, residency.capacity
        )
    elif composition_mode == "resident":
        # Improved one-batch approximation: the batch remains scalar, but its
        # context is formed from the residence-time-weighted active classes.
        batch, curves, converged, residual = _resident_operating_point(
            model,
            config,
            classes,
            weights,
            arrival_rate_rps,
            chunk_tokens,
            residency.capacity,
        )
        capacity, capacity_concurrency = _resident_throughput_capacity(
            model,
            config,
            classes,
            weights,
            chunk_tokens,
            residency.capacity,
        )
    elif composition_mode == "occupancy_ps":
        batch, curves, converged, residual, slowdown = _occupancy_ps_operating_point(
            model,
            config,
            classes,
            weights,
            arrival_rate_rps,
            chunk_tokens,
            residency.capacity,
        )
        capacity, capacity_concurrency = _occupancy_ps_capacity(
            model,
            config,
            classes,
            weights,
            chunk_tokens,
            residency.capacity,
        )
    elif composition_mode == "macro":
        curves = []
        if residency.capacity < 1:
            batch = 1.0
            converged = False
            residual = math.inf
            capacity = 0.0
            capacity_concurrency = 1.0
            macro_state = None
            macro_wait = math.inf
        else:
            execution = IterationCalibration(
                config.effective_flops,
                config.effective_bandwidth_bytes_s,
            )
            calibration = TwoModeCalibration(decode=execution, mix=execution)
            capacity, capacity_concurrency = two_mode_capacity(
                model,
                calibration,
                classes,
                weights,
                chunk_tokens,
                config.max_num_batched_tokens,
                residency.capacity,
            )
            macro_state = two_mode_operating_point(
                model,
                calibration,
                classes,
                weights,
                arrival_rate_rps,
                chunk_tokens,
                config.max_num_batched_tokens,
                residency.capacity,
            )
            macro_wait, _, _ = two_mode_first_admission_wait(
                macro_state,
                arrival_rate_rps,
                sum(
                    weight * max(1.0, float(prompt))
                    / math.ceil(max(1.0, float(prompt)) / chunk_tokens)
                    for (prompt, _), weight in zip(classes, weights)
                ) / max(sum(weights), 1.0e-12),
                sum(
                    weight * math.ceil(max(1.0, float(prompt)) / chunk_tokens)
                    for (prompt, _), weight in zip(classes, weights)
                ) / max(sum(weights), 1.0e-12),
                capacity,
            )
            batch = macro_state.decode_concurrency
            converged = not macro_state.overloaded
            residual = 0.0
    else:
        batch, curves, converged, residual, tau_iter = _two_mode_operating_point(
            model,
            config,
            classes,
            weights,
            arrival_rate_rps,
            chunk_tokens,
            residency.capacity,
        )
        capacity, capacity_concurrency = _two_mode_capacity(
            model,
            config,
            classes,
            weights,
            chunk_tokens,
            residency.capacity,
        )
    utilization = arrival_rate_rps / capacity if capacity > 0.0 else math.inf
    stable = (
        converged
        and residency.capacity >= 1
        and (batch <= residency.capacity if composition_mode == "macro" else batch < residency.capacity)
        and residency.kv_slack > 0.0
        and utilization < 1.0
    )
    if composition_mode == "macro":
        if macro_state is None:
            mean_service = math.inf
        else:
            total = max(sum(weights), 1.0e-12)
            mean_service = sum(
                weight * (
                    math.ceil(max(1.0, float(prompt)) / chunk_tokens)
                    * macro_state.mixed_iteration_s
                    + max(0, int(round(output)) - 1) * macro_state.mean_iteration_s
                )
                for (prompt, output), weight in zip(classes, weights)
            ) / total
    elif composition_mode == "two_mode":
        mean_prefill, tau_iter, _, mean_decode_tokens = _two_mode_statistics(
            classes, weights, curves, max(1.0, batch), chunk_tokens
        )
        mean_service = mean_prefill + mean_decode_tokens * tau_iter
    elif composition_mode == "occupancy_ps":
        eval_batch = min(
            max(1.0, batch),
            max(1.0, float(residency.capacity) * 0.999),
        )
        mean_service = slowdown * mean_service_time(
            curves, weights, eval_batch
        )
    else:
        mean_service = mean_service_time(
            curves, weights, max(1.0, batch)
        )
    instance = LLMInstancePerformance(
        arrival_rate_rps=arrival_rate_rps,
        active_concurrency=batch,
        active_kv_tokens=residency.active_kv_tokens,
        resident_capacity=residency.capacity,
        kv_slack=residency.kv_slack,
        mean_service_s=mean_service,
        throughput_capacity_rps=capacity,
        capacity_concurrency=capacity_concurrency,
        utilization=utilization,
        stable=stable,
        fixed_point_residual=residual,
    )
    per_class: list[LLMClassPerformance] = []
    if composition_mode == "two_mode":
        _, tau_iter, _, _ = _two_mode_statistics(
            classes, weights, curves, max(1.0, batch), chunk_tokens
        )
    for index, (prompt, output) in enumerate(classes):
        if composition_mode == "macro":
            count = max(1, round(output))
            if macro_state is None:
                prefill = decode = math.inf
                tbt = math.inf
            else:
                prefill = math.ceil(max(1.0, float(prompt)) / chunk_tokens) * macro_state.mixed_iteration_s
                decode = (count - 1) * macro_state.mean_iteration_s
                tbt = macro_state.mean_iteration_s if count > 1 else 0.0
            per_class.append(
                LLMClassPerformance(
                    service_s=prefill + decode,
                    prefill_s=prefill,
                    decode_s=decode,
                    ttft_s=macro_wait + prefill,
                    tbt_s=tbt,
                    response_s=macro_wait + prefill + decode,
                )
            )
            continue
        curve = curves[index]
        eval_batch = min(
            max(1.0, batch),
            max(1.0, float(residency.capacity) * 0.999),
        )
        if composition_mode == "occupancy_ps":
            # A newly admitted request contributes its own prefill work;
            # resident decode work is represented by the shared slowdown.
            prefill = slowdown * curve.prefill_at(1.0)
        else:
            prefill = curve.prefill_at(eval_batch)
        count = max(1, round(output))
        decode = (
            (count - 1) * tau_iter
            if composition_mode == "two_mode"
            else slowdown * curve.decode_at(eval_batch)
            if composition_mode == "occupancy_ps"
            else curve.decode_at(eval_batch)
        )
        per_class.append(
            LLMClassPerformance(
                service_s=prefill + decode,
                prefill_s=prefill,
                decode_s=decode,
                ttft_s=prefill,
                tbt_s=tau_iter if composition_mode == "two_mode" and count > 1 else (
                    decode / (count - 1) if count > 1 else 0.0
                ),
                response_s=prefill + decode,
            )
        )
    return instance, per_class


class AnalyticalBackend:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.network = NetworkBackend(
            scenario.links,
            scenario.simulation.slot_seconds,
            scenario.simulation.overload_delay_s,
        )

    def evaluate(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> AnalyticalResult:
        llm_arrivals = self.llm_arrivals(routing, arrival_rates)
        llm_perf, llm_util, kv_stable, violations = self._llm_performance(
            deployment, llm_arrivals
        )
        distributions = self.node_distributions(deployment, routing)
        tool_arrivals = self.tool_arrivals(routing, distributions, arrival_rates)
        tool_delay, tool_util, tool_violations = self._tool_performance(
            deployment, tool_arrivals
        )
        violations.extend(tool_violations)
        link_loads = self.link_loads(routing, distributions, arrival_rates)
        for edge, utilization in self.network.utilization(link_loads).items():
            if utilization >= 1.0:
                violations.append(f"link_overload:{edge}")
        instance_metrics = getattr(self, "_last_llm_instance_performance", {})
        return AnalyticalResult(
            llm_perf,
            llm_util,
            kv_stable,
            tool_delay,
            tool_util,
            link_loads,
            distributions,
            violations,
            instance_metrics,
        )

    def llm_arrivals(
        self,
        routing: RoutingDecision,
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[LLMClass, float]:
        arrivals: dict[LLMClass, float] = defaultdict(float)
        for app in self.scenario.applications.values():
            for node in app.nodes.values():
                if node.type is not NodeType.LLM:
                    continue
                visit = app.visit_probability(node.id)
                for ingress, base_rate in app.ingress_rates.items():
                    rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                    for candidate_id in self.scenario.candidates:
                        share = routing.llm_share.get(
                            (app.id, ingress, node.id, candidate_id), 0.0
                        )
                        arrivals[(app.id, node.id, candidate_id)] += rate * visit * share
        return dict(arrivals)

    def _llm_performance(
        self,
        deployment: DeploymentDecision,
        arrivals: dict[LLMClass, float],
    ) -> tuple[
        dict[LLMClass, LLMClassPerformance],
        dict[str, float],
        dict[str, bool],
        list[str],
    ]:
        classes_by_instance: dict[str, list[LLMClass]] = defaultdict(list)
        for key, rate in arrivals.items():
            if rate <= 0.0:
                continue
            if deployment.llm_active.get(key[2], 0) != 1:
                continue
            classes_by_instance[key[2]].append(key)

        chunk_tokens = self.scenario.simulation.prefill_chunk_tokens
        perf: dict[LLMClass, LLMClassPerformance] = {}
        utilization: dict[str, float] = {}
        kv_stable: dict[str, bool] = {}
        violations: list[str] = []
        instance_metrics: dict[str, LLMInstancePerformance] = {}
        for candidate_id, keys in classes_by_instance.items():
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            model = self.scenario.models[candidate.model]
            rates = [arrivals[key] for key in keys]
            call_classes = [
                (
                    self.scenario.applications[key[0]]
                    .nodes[key[1]]
                    .prompt_tokens[candidate.model],
                    self.scenario.applications[key[0]]
                    .nodes[key[1]]
                    .output_tokens[candidate.model],
                )
                for key in keys
            ]
            instance, class_performance = evaluate_llm_instance(
                model,
                config,
                call_classes,
                rates,
                sum(rates),
                chunk_tokens,
                composition_mode="macro",
            )
            instance_metrics[candidate_id] = instance
            utilization[candidate_id] = instance.utilization
            kv_stable[candidate_id] = instance.stable
            if not instance.stable:
                violations.append(f"llm_queue_overload:{candidate_id}")
            if instance.resident_capacity < 1 or instance.kv_slack <= 0.0:
                violations.append(f"llm_kv_overload:{candidate_id}")
            for key, value in zip(keys, class_performance):
                perf[key] = value
        for key, rate in arrivals.items():
            if rate > 0.0 and key not in perf:
                violations.append(f"llm_unserved:{key[2]}")
        self._last_llm_instance_performance = instance_metrics
        return perf, utilization, kv_stable, violations
    def node_distributions(
        self,
        deployment: DeploymentDecision,
        routing: RoutingDecision,
    ) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
        result: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        for node_id in flow.nodes:
                            node = app.nodes[node_id]
                            key = (app.id, ingress, model, flow.id, node_id)
                            if node.type is NodeType.LLM:
                                dist: dict[str, float] = defaultdict(float)
                                for candidate_id, candidate in self.scenario.candidates.items():
                                    if candidate.model != model:
                                        continue
                                    absolute = routing.llm_share.get(
                                        (app.id, ingress, node_id, candidate_id), 0.0
                                    )
                                    if absolute > 0.0:
                                        dist[candidate.server] += absolute / model_share
                                result[key] = _normalize(dist)

                        unresolved = [
                            edge
                            for edge in flow.edges
                            if app.nodes[edge[1]].type is NodeType.TOOL
                        ]
                        for _ in range(len(flow.nodes)):
                            next_unresolved = []
                            for source, target in unresolved:
                                source_key = (app.id, ingress, model, flow.id, source)
                                if source_key not in result:
                                    next_unresolved.append((source, target))
                                    continue
                                target_key = (app.id, ingress, model, flow.id, target)
                                target_dist: dict[str, float] = defaultdict(float)
                                for u, source_probability in result[source_key].items():
                                    for v in self.scenario.servers:
                                        probability = routing.tool_route.get(
                                            (app.id, source, target, u, v), 0.0
                                        )
                                        target_dist[v] += source_probability * probability
                                if target_key in result:
                                    for server, probability in result[target_key].items():
                                        target_dist[server] += probability
                                result[target_key] = _normalize(target_dist)
                            unresolved = next_unresolved
                            if not unresolved:
                                break
        return result

    def tool_arrivals(
        self,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[ToolPool, float]:
        arrivals: dict[ToolPool, float] = defaultdict(float)
        for app in self.scenario.applications.values():
            for ingress, base_rate in app.ingress_rates.items():
                rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        flow_rate = rate * model_share * flow.probability
                        for node_id in flow.nodes:
                            node = app.nodes[node_id]
                            if node.type is not NodeType.TOOL or node.tool is None:
                                continue
                            key = (app.id, ingress, model, flow.id, node_id)
                            for server, probability in distributions.get(key, {}).items():
                                arrivals[(node.tool, server)] += flow_rate * probability
        return dict(arrivals)

    def _tool_performance(
        self,
        deployment: DeploymentDecision,
        arrivals: dict[ToolPool, float],
    ) -> tuple[dict[ToolPool, float], dict[ToolPool, float], list[str]]:
        delays: dict[ToolPool, float] = {}
        utilizations: dict[ToolPool, float] = {}
        violations: list[str] = []
        for pool, arrival_rate in arrivals.items():
            tool_id, server = pool
            spec = self.scenario.tools[tool_id]
            replicas = deployment.tool_replicas.get(pool, 0)
            wait, process, rho, overloaded = tool_response_time(
                arrival_rate,
                spec.service_rate[server],
                replicas,
                spec.arrival_scv,
                self.scenario.simulation.overload_delay_s,
            )
            delays[pool] = wait + process
            utilizations[pool] = rho
            if overloaded:
                violations.append(f"tool_overload:{tool_id}@{server}")
        return delays, utilizations, violations

    def link_loads(
        self,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> dict[Edge, float]:
        loads: dict[Edge, float] = {}
        for app in self.scenario.applications.values():
            for ingress, base_rate in app.ingress_rates.items():
                arrival_rate = (arrival_rates or {}).get((app.id, ingress), base_rate)
                for model in self.scenario.models:
                    model_share = routing.model_share.get((app.id, ingress, model), 0.0)
                    if model_share <= 0.0:
                        continue
                    for flow in app.pattern_flows:
                        rate = arrival_rate * model_share * flow.probability
                        for source in flow.sources:
                            key = (app.id, ingress, model, flow.id, source)
                            for server, probability in distributions.get(key, {}).items():
                                self.network.add_traffic(
                                    loads,
                                    ingress,
                                    server,
                                    rate * probability,
                                    app.entry_data_mb[model],
                                )
                        final_key = (app.id, ingress, model, flow.id, flow.final_node)
                        for server, probability in distributions.get(final_key, {}).items():
                            self.network.add_traffic(
                                loads,
                                server,
                                ingress,
                                rate * probability,
                                app.exit_data_mb[model],
                            )
                        for source, target in flow.edges:
                            data_mb = app.edge_data_mb.get((model, source, target), 0.0)
                            for u, v, probability in self.edge_pair_distribution(
                                app.id,
                                ingress,
                                model,
                                flow.id,
                                source,
                                target,
                                routing,
                                distributions,
                            ):
                                self.network.add_traffic(
                                    loads, u, v, rate * probability, data_mb
                                )
        return loads

    def edge_pair_distribution(
        self,
        app_id: str,
        ingress: str,
        model: str,
        flow_id: str,
        source: str,
        target: str,
        routing: RoutingDecision,
        distributions: dict[tuple[str, str, str, str, str], dict[str, float]],
    ) -> list[tuple[str, str, float]]:
        app = self.scenario.applications[app_id]
        source_dist = distributions.get((app_id, ingress, model, flow_id, source), {})
        target_dist = distributions.get((app_id, ingress, model, flow_id, target), {})
        pairs: list[tuple[str, str, float]] = []
        if app.nodes[target].type is NodeType.TOOL:
            for u, source_probability in source_dist.items():
                for v in self.scenario.servers:
                    p = routing.tool_route.get((app_id, source, target, u, v), 0.0)
                    if p > 0.0:
                        pairs.append((u, v, source_probability * p))
        else:
            for u, source_probability in source_dist.items():
                for v, target_probability in target_dist.items():
                    pairs.append((u, v, source_probability * target_probability))
        total = sum(item[2] for item in pairs)
        return [(u, v, p / total) for u, v, p in pairs] if total > 0.0 else []


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in values.values() if value > 0.0)
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in values.items() if value > 0.0}
