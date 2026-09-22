"""Inner composition solver: the best model shares for one fixed deployment.

The joint orchestration problem has an outer part (which LLM instances and tool
replicas to run) and an inner part (how to split each ``(application, ingress)``
request stream over the models that deployment activates).  This module solves the
inner part to near-optimality for a *given* deployment, which gives the training
pipeline two things it cannot get from a learned policy alone:

* a reference for the held-out evaluation gate -- the learned composition policy is
  scored by its Spearman rank correlation against this solver's utility;
* an offline teacher -- the solved shares are supervision targets for warm-starting
  the composition policy.

The inner problem is small but not separable: the model choice of one group changes
the load, hence the utilization, hence the service time seen by every other group.
The solver therefore evaluates whole compositions (no per-group surrogate) and
searches with deterministic coordinate ascent over the canonical candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from agent_orch.objective import ObjectiveEvaluator
from agent_orch.performance.analytical import evaluate_llm_instance
from agent_orch.routing.physical import PhysicalRouter
from agent_orch.schema.models import ApplicationSpec, DeploymentDecision, NodeType, Scenario
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace

ModelShare = dict[tuple[str, str, str], float]
ArrivalRates = Mapping[tuple[str, str], float]

SHARE_KEY_SEPARATOR = "|"
QUALITY_SOFTMAX_TEMPERATURE = 0.1
"""Softmax temperature on the per-application quality scores.

Quality scores live in ``[0, 1]`` and neighbouring models differ by ~0.1, so a
temperature of 0.1 keeps the candidate close to uniform when two models are equally
good instead of collapsing onto the single best one.
"""

IMPROVEMENT_EPSILON = 1.0e-12
"""Utility gain a greedy move must beat to be accepted (float-noise guard)."""


def _encode_share_key(key: tuple[str, str, str]) -> str:
    return SHARE_KEY_SEPARATOR.join(key)


def _decode_share_key(key: str) -> tuple[str, str, str]:
    app_id, ingress, model = key.split(SHARE_KEY_SEPARATOR, 2)
    return app_id, ingress, model


@dataclass(frozen=True)
class CompositionCandidate:
    """One canonical composition: a named share vector over the active models."""

    name: str
    share: ModelShare


@dataclass(frozen=True)
class CompositionSolution:
    """The solver's answer for one deployment, with its evaluation bookkeeping."""

    model_share: ModelShare
    utility: float
    evaluations: int
    candidate_utilities: dict[str, float]
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "utility": float(self.utility),
            "evaluations": int(self.evaluations),
            "source": self.source,
            "candidate_utilities": {
                name: float(value) for name, value in sorted(self.candidate_utilities.items())
            },
            "model_share": {
                _encode_share_key(key): float(value)
                for key, value in sorted(self.model_share.items())
            },
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "CompositionSolution":
        return CompositionSolution(
            model_share={
                _decode_share_key(str(key)): float(value)
                for key, value in dict(payload["model_share"]).items()
            },
            utility=float(payload["utility"]),
            evaluations=int(payload["evaluations"]),
            candidate_utilities={
                str(name): float(value)
                for name, value in dict(payload["candidate_utilities"]).items()
            },
            source=str(payload["source"]),
        )


class CompositionSolver:
    """Evaluate and optimise the model composition of one fixed deployment.

    The solver owns one :class:`~agent_orch.simulator.Simulator`, reused across every
    evaluation because building the analytical backend is expensive.
    """

    CANONICAL_NAMES: tuple[str, ...] = (
        "uniform",
        "quality_greedy",
        "cost_greedy",
        "latency_greedy",
        "quality_softmax",
        "quality_uniform_mix",
    )

    # How far toward the uniform split the line search probes.  1.0 is excluded: it
    # is the incumbent, already evaluated.
    BLEND_TEMPERATURES: tuple[float, ...] = (0.75, 0.5, 0.25)

    def __init__(
        self,
        scenario: Scenario,
        evaluator: ObjectiveEvaluator,
        mapping_samples: int = 128,
        seed: int = 0,
        protocol_periods: int = 1,
        protocol_warmup: int = 0,
        scorer: Callable[[DeploymentDecision, Mapping[tuple[str, str, str], float]], float]
        | None = None,
    ) -> None:
        self.scenario = scenario
        self.evaluator = evaluator
        self.mapping_samples = int(mapping_samples)
        self.seed = int(seed)
        self.protocol_periods = max(1, int(protocol_periods))
        self.protocol_warmup = max(0, int(protocol_warmup))
        # When a scorer is supplied the search optimises that function instead of
        # the raw simulator rollout.  A reference has to be the argmax of the
        # metric it will be compared against; scoring it on a second path makes
        # the two disagree on individual deployments and voids the comparison.
        self.scorer = scorer
        self.simulator = Simulator(scenario, max_mapping_samples=self.mapping_samples)
        self.router = PhysicalRouter(scenario)
        self.groups: tuple[tuple[str, str], ...] = tuple(
            (app.id, ingress)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
        )
        self._evaluations = 0
        self._service_time_cache: dict[tuple[str, str, float, float], float] = {}

    # ------------------------------------------------------------------ evaluation

    def evaluate_share(
        self,
        deployment: DeploymentDecision,
        model_share: Mapping[tuple[str, str, str], float],
        arrival_rates: ArrivalRates,
    ) -> float:
        """Utility of one ``(deployment, composition)`` pair.

        Every evaluation starts from a cold, identical physical state: the simulator
        is reset, so no candidate inherits the utilization feedback or the switching
        cost of the composition evaluated before it.  Without that, the comparison
        between two candidates would be polluted by history rather than decided by
        the candidate itself, and the resulting reference would depend on evaluation
        order.

        ``protocol_periods`` controls how many periods the composition is held for
        before it is scored.  With the default of one, the router is called with
        ``previous_metrics=None`` and the score is a pure cold-start single period.
        With a longer protocol the router feeds back utilization as it would during
        training, and the score is the mean of the post-warmup periods -- which is
        the objective a composition policy trained in the environment actually
        optimises.  A reference solved under one protocol is not the optimum of the
        other, so the protocol must match whatever it is compared against.

        ``model_share`` is keyed by ``(app_id, ingress, model)`` and must sum to one
        per group over the models the deployment activates; the router normalises
        nothing, so a share vector that does not sum to one is rejected by the
        routing validator.
        """

        self._evaluations += 1
        if self.scorer is not None:
            return float(self.scorer(deployment, model_share))
        rates = self._effective_arrival_rates(arrival_rates)
        self.simulator.set_arrival_trace(
            ArrivalTrace({slot: dict(rates) for slot in range(self.protocol_periods)})
        )
        self.simulator.reset(self.seed)
        utilities: list[float] = []
        for period in range(self.protocol_periods):
            previous = None if period == 0 else self.simulator.last_metrics
            routing = self.router.route(deployment, model_share, previous, rates)
            metrics = self.simulator.step(deployment, routing).metrics
            if period >= self.protocol_warmup:
                utilities.append(float(self.evaluator.evaluate(metrics, rates).utility))
        return float(sum(utilities) / len(utilities)) if utilities else float("nan")

    @property
    def evaluations(self) -> int:
        """Number of simulator evaluations performed since construction."""

        return self._evaluations

    # ------------------------------------------------------------------ candidates

    def canonical_candidates(self, deployment: DeploymentDecision) -> list[CompositionCandidate]:
        """The canonical compositions, in the order :meth:`solve` evaluates them.

        Shares are allocated only over the models the deployment activates; an
        inactive model always receives ``0.0`` and a group with no active model is
        left out entirely (its share vector is empty, so it stays unrouted).  The six
        candidates are ``uniform``, ``quality_greedy``, ``cost_greedy``,
        ``latency_greedy``, ``quality_softmax`` and ``quality_uniform_mix``.
        """

        active = self._active_models(deployment)
        weights: dict[str, dict[tuple[str, str], dict[str, float]]] = {
            "uniform": self._uniform_weights(active),
            "quality_greedy": self._quality_weights(active),
            "cost_greedy": self._cost_weights(deployment, active),
            "latency_greedy": self._latency_weights(deployment, active),
            "quality_softmax": self._quality_softmax_weights(active),
        }
        shares = {name: self._flatten(per_group, active) for name, per_group in weights.items()}
        weights["quality_uniform_mix"] = self._mix(
            weights["uniform"], weights["quality_greedy"]
        )
        shares["quality_uniform_mix"] = self._flatten(
            weights["quality_uniform_mix"], active
        )
        return [CompositionCandidate(name, shares[name]) for name in self.CANONICAL_NAMES]

    def uniform_share(self, deployment: DeploymentDecision) -> ModelShare:
        """Equal split over the active models -- the control composition."""

        active = self._active_models(deployment)
        return self._flatten(self._uniform_weights(active), active)

    # ------------------------------------------------------------------ solve

    def solve(
        self,
        deployment: DeploymentDecision,
        arrival_rates: ArrivalRates,
        *,
        budget: int = 32,
        sweeps: int = 1,
    ) -> CompositionSolution:
        """Search for the best composition of ``deployment`` within ``budget`` evaluations.

        Step one evaluates every canonical candidate and keeps the best one.  Step two
        runs ``sweeps`` rounds of per-group coordinate ascent: for each
        ``(application, ingress)`` group in a fixed order it tries every single-model
        vertex plus the uniform split over that group alone, keeping the other groups
        fixed, and accepts a move only if the whole-composition utility improves.  A
        round that improves nothing ends the search.

        The search is deterministic -- no unseeded randomness anywhere -- and stops the
        moment ``budget`` evaluations (canonical ones included) are spent, returning
        the best composition found so far.  ``source`` names the canonical candidate
        when the answer is one of them, and ``"greedy"`` when a greedy move produced
        the best composition.
        """

        if budget < 1:
            raise ValueError("budget must be at least one evaluation")
        start = self._evaluations
        active = self._active_models(deployment)
        # A deployment activates models globally, so either every group has active
        # models to split between or none of them has.
        groups = list(self.groups) if active else []
        candidates = self.canonical_candidates(deployment)

        candidate_utilities: dict[str, float] = {}
        best_share: ModelShare | None = None
        best_utility = -math.inf
        best_source = "greedy"
        for candidate in candidates:
            if self._spent(start) >= budget:
                break
            utility = self.evaluate_share(deployment, candidate.share, arrival_rates)
            candidate_utilities[candidate.name] = utility
            if utility > best_utility:
                best_utility = utility
                best_share = dict(candidate.share)
                best_source = candidate.name
        if best_share is None:  # pragma: no cover - budget >= 1 always evaluates one
            raise RuntimeError("The composition solver evaluated no canonical candidate")

        current = dict(best_share)
        # The canonical candidates are vertices, and the per-group moves below are
        # vertices too, so the search as a whole cannot leave the vertex set.  The
        # optimum generally is not one: on agent-abilene-20 deployment 2, blending
        # the whole composition 25% toward uniform beats the best vertex by 0.0054
        # (+0.19207 vs +0.18668) -- a tenth of the whole lift over uniform.  One
        # cheap line search over that direction puts blends back in reach.
        for temperature in self.BLEND_TEMPERATURES:
            if self._spent(start) >= budget:
                break
            trial = self._blend(best_share, groups, active, temperature)
            if trial == current:
                continue
            utility = self.evaluate_share(deployment, trial, arrival_rates)
            if utility > best_utility + IMPROVEMENT_EPSILON:
                best_utility = utility
                best_share = trial
                best_source = "greedy-blend"
                current = trial

        for _ in range(max(0, int(sweeps))):
            improved = False
            for app_id, ingress in groups:
                if self._spent(start) >= budget:
                    break
                for weights in self._group_options((app_id, ingress), current, active):
                    if self._spent(start) >= budget:
                        break
                    trial = dict(current)
                    for model in active:
                        trial[(app_id, ingress, model)] = weights.get(model, 0.0)
                    if trial == current:
                        continue
                    utility = self.evaluate_share(deployment, trial, arrival_rates)
                    if utility > best_utility + IMPROVEMENT_EPSILON:
                        best_utility = utility
                        best_share = trial
                        best_source = "greedy"
                        current = trial
                        improved = True
            if not improved:
                break

        return CompositionSolution(
            model_share=dict(best_share),
            utility=float(best_utility),
            evaluations=self._spent(start),
            candidate_utilities=candidate_utilities,
            source=best_source,
        )

    @staticmethod
    def _blend(
        share: ModelShare,
        groups: tuple[tuple[str, str], ...],
        active: tuple[str, ...],
        temperature: float,
    ) -> ModelShare:
        """``share`` pulled ``1 - temperature`` of the way toward the uniform split.

        ``temperature == 1`` returns the incumbent (so it is never re-evaluated),
        and ``0`` is the uniform composition itself.
        """

        uniform = 1.0 / len(active) if active else 0.0
        blended: ModelShare = {}
        for app_id, ingress in groups:
            for model in active:
                weight = float(share.get((app_id, ingress, model), 0.0))
                blended[(app_id, ingress, model)] = (
                    temperature * weight + (1.0 - temperature) * uniform
                )
        return blended

    def _group_options(
        self,
        group: tuple[str, str],
        current: ModelShare,
        active: tuple[str, ...],
    ) -> list[dict[str, float]]:
        """Single-model vertices and the uniform split for one group.

        Keeping the current weights is not listed: the incumbent composition was
        already evaluated, so re-evaluating it would only spend budget.
        """

        options: list[dict[str, float]] = [
            {model: (1.0 if model == winner else 0.0) for model in active}
            for winner in active
        ]
        if active:
            probability = 1.0 / len(active)
            options.append({model: probability for model in active})
        return [
            weights
            for weights in options
            if weights != self._current_weights(group, current, active)
        ]

    @staticmethod
    def _current_weights(
        group: tuple[str, str], current: ModelShare, active: tuple[str, ...]
    ) -> dict[str, float]:
        return {
            model: float(current.get((group[0], group[1], model), 0.0))
            for model in active
        }

    def _spent(self, start: int) -> int:
        return self._evaluations - start

    # ------------------------------------------------------------------ weights

    def _active_models(self, deployment: DeploymentDecision) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.scenario.candidates[candidate_id].model
                    for candidate_id, active in deployment.llm_active.items()
                    if active and candidate_id in self.scenario.candidates
                }
            )
        )

    def _uniform_weights(
        self, active: tuple[str, ...]
    ) -> dict[tuple[str, str], dict[str, float]]:
        if not active:
            return {group: {} for group in self.groups}
        probability = 1.0 / len(active)
        return {
            group: {model: probability for model in active} for group in self.groups
        }

    def _quality_weights(
        self, active: tuple[str, ...]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """All traffic of a group to its highest-quality active model.

        Quality is per ``(application, model)``, so the winning model can differ
        between applications.
        """

        weights: dict[tuple[str, str], dict[str, float]] = {}
        for group in self.groups:
            app = self.scenario.applications[group[0]]
            if not active:
                weights[group] = {}
                continue
            best = min(
                active,
                key=lambda model: (-float(app.quality.get(model, 0.0)), model),
            )
            weights[group] = {model: (1.0 if model == best else 0.0) for model in active}
        return weights

    def _cost_weights(
        self, deployment: DeploymentDecision, active: tuple[str, ...]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """All traffic of a group to the cheapest active model.

        A model can be reachable through several configs; the model's cost is the
        cheapest config the deployment activates for it, which is the cost the
        physical router can actually realise for that model.
        """

        cost: dict[str, float] = {}
        for candidate_id, is_active in deployment.llm_active.items():
            if not is_active or candidate_id not in self.scenario.candidates:
                continue
            candidate = self.scenario.candidates[candidate_id]
            if candidate.model not in active:
                continue
            running = float(self.scenario.llm_configs[candidate.config].running_cost_per_slot)
            cost[candidate.model] = min(cost.get(candidate.model, math.inf), running)
        if not cost:
            return {group: {} for group in self.groups}
        best = min(cost, key=lambda model: (cost[model], model))
        return {
            group: {model: (1.0 if model == best else 0.0) for model in active}
            for group in self.groups
        }

    def _latency_weights(
        self, deployment: DeploymentDecision, active: tuple[str, ...]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """All traffic of a group to the active model with the shortest service time.

        The service time is the same quantity the physical router scores candidates
        with: the mean service time of one call class on one instance from
        :func:`~agent_orch.performance.analytical.evaluate_llm_instance` in ``macro``
        composition mode, visit-probability averaged over the application's LLM
        nodes.  A model is represented by its fastest activated instance, and the
        group picks the model with the smallest such service time (ties broken by
        model id).  An application without LLM nodes ties every model at zero, so the
        first model id in sorted order wins.
        """

        weights: dict[tuple[str, str], dict[str, float]] = {}
        for group in self.groups:
            app = self.scenario.applications[group[0]]
            if not active:
                weights[group] = {}
                continue
            service: dict[str, float] = {}
            for model in active:
                best = math.inf
                for candidate_id, is_active in deployment.llm_active.items():
                    if not is_active or candidate_id not in self.scenario.candidates:
                        continue
                    candidate = self.scenario.candidates[candidate_id]
                    if candidate.model != model:
                        continue
                    best = min(best, self._service_time(candidate_id, app))
                service[model] = best
            winner = min(service, key=lambda model: (service[model], model))
            weights[group] = {model: (1.0 if model == winner else 0.0) for model in active}
        return weights

    def _quality_softmax_weights(
        self, active: tuple[str, ...]
    ) -> dict[tuple[str, str], dict[str, float]]:
        weights: dict[tuple[str, str], dict[str, float]] = {}
        for group in self.groups:
            app = self.scenario.applications[group[0]]
            if not active:
                weights[group] = {}
                continue
            scores = {
                model: float(app.quality.get(model, 0.0)) for model in active
            }
            peak = max(scores.values())
            exponentials = {
                model: math.exp((score - peak) / QUALITY_SOFTMAX_TEMPERATURE)
                for model, score in scores.items()
            }
            total = sum(exponentials.values())
            weights[group] = {
                model: exponentials[model] / total for model in active
            }
        return weights

    @staticmethod
    def _mix(
        left: dict[tuple[str, str], dict[str, float]],
        right: dict[tuple[str, str], dict[str, float]],
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Half-and-half blend of two per-group weight tables."""

        mixed: dict[tuple[str, str], dict[str, float]] = {}
        for group in set(left) | set(right):
            first = left.get(group, {})
            second = right.get(group, {})
            models = set(first) | set(second)
            mixed[group] = {
                model: 0.5 * first.get(model, 0.0) + 0.5 * second.get(model, 0.0)
                for model in models
            }
        return mixed

    def _flatten(
        self,
        per_group: dict[tuple[str, str], dict[str, float]],
        active: tuple[str, ...],
    ) -> ModelShare:
        """Render per-group weights as the router's ``(app, ingress, model)`` share.

        Every model of the scenario appears in every group, with ``0.0`` for the
        models this deployment does not activate: the routing validator checks the
        shares of a group sum to one whenever any model is active.
        """

        share: ModelShare = {}
        for group in self.groups:
            group_weights = per_group.get(group, {})
            for model in self.scenario.models:
                share[(group[0], group[1], model)] = float(
                    group_weights.get(model, 0.0) if model in active else 0.0
                )
        return share

    def _service_time(self, candidate_id: str, app: ApplicationSpec) -> float:
        """Visit-probability averaged mean service time of one candidate on one app."""

        candidate = self.scenario.candidates[candidate_id]
        nodes = [node for node in app.nodes.values() if node.type is NodeType.LLM]
        total_weight = 0.0
        weighted = 0.0
        for node in nodes:
            weight = float(app.visit_probability(node.id))
            key = (
                candidate.model,
                candidate.config,
                float(node.prompt_tokens[candidate.model]),
                float(node.output_tokens[candidate.model]),
            )
            if key not in self._service_time_cache:
                instance, _ = evaluate_llm_instance(
                    self.scenario.models[candidate.model],
                    self.scenario.llm_configs[candidate.config],
                    [(node.prompt_tokens[candidate.model], node.output_tokens[candidate.model])],
                    [1.0],
                    0.0,
                    self.scenario.simulation.prefill_chunk_tokens,
                    composition_mode="macro",
                )
                self._service_time_cache[key] = float(instance.mean_service_s)
            weighted += weight * self._service_time_cache[key]
            total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return weighted / total_weight

    def _effective_arrival_rates(self, arrival_rates: ArrivalRates) -> dict[tuple[str, str], float]:
        """Fill missing groups with the scenario's own base rate.

        The simulator resolves a missing ``(app, ingress)`` to the scenario base rate
        while the objective evaluator would read it as zero load, so both sides are
        fed the same complete mapping.
        """

        return {
            (app.id, ingress): float(arrival_rates.get((app.id, ingress), base_rate))
            for app in self.scenario.applications.values()
            for ingress, base_rate in app.ingress_rates.items()
        }


__all__ = ["CompositionCandidate", "CompositionSolution", "CompositionSolver"]
