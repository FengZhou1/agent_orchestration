"""Shared machinery for the orchestration environments.

One Gym transition is a deployment substep until every resource pool has been
processed; the final transition of a period samples the model composition,
invokes the deterministic physical router, evaluates one steady-state period and
only then advances the period index.  Subclasses differ only in which pools they
expose as decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlan, CapacityPlanner
from agent_orch.deployment import DeploymentLibrary
from agent_orch.objective import (
    ObjectiveEvaluator,
    ObjectiveSpec,
    ObjectiveValue,
    ReferenceScales,
)
from agent_orch.routing import PhysicalRouter
from agent_orch.schema.models import DeploymentDecision, RoutingDecision, Scenario
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace

from .layout import StructuredActionLayout


def resolve_deployment_library(
    scenario: Scenario,
    library: DeploymentLibrary | None = None,
    library_path: str | Path | None = None,
    auto: bool = True,
) -> DeploymentLibrary | None:
    """Find the feasible-deployment library for a scenario, if one exists."""

    if library is not None:
        return library
    candidate = Path(library_path) if library_path else DeploymentLibrary.default_path(scenario.id)
    if candidate.exists():
        return DeploymentLibrary.load(candidate)
    if library_path is not None and auto:
        raise FileNotFoundError(
            f"Deployment library {candidate} not found; run scripts/build_deployment_library.py"
        )
    return None


class BaseOrchestrationEnv(gym.Env):
    """Macro-period environment for sequential deployment and steady evaluation."""

    metadata = {"render_modes": []}
    LLM_DEPLOYMENT = 0
    SERVICE_DEPLOYMENT = 1
    COMPOSITION = 2
    DEPLOYMENT = LLM_DEPLOYMENT
    ROUTING = COMPOSITION
    CONSTRAINT_NAMES = ("llm", "service")

    def __init__(
        self,
        scenario: Scenario,
        max_slots: int = 600,
        potential_shaping: bool = True,
        seed: int = 0,
        arrival_trace: ArrivalTrace | None = None,
        gamma: float = 0.99,
        mapping_samples: int = 4096,
        objective: ObjectiveSpec | None = None,
        deployment_library: DeploymentLibrary | None = None,
        deployment_library_path: str | Path | None = None,
        potential_cost_weight: float = 0.0,
        deployment_periods: int = 1,
        trace_offset_span: int | None = None,
    ):
        super().__init__()
        self.scenario = scenario
        self.max_periods = max(1, int(max_slots))
        self.potential_shaping = potential_shaping
        self.potential_cost_weight = float(potential_cost_weight)
        # ``T^dep``: the deployment may only change every ``deployment_periods``
        # orchestration periods.  One means it can change every period, which is
        # the historical behaviour.
        self.deployment_periods = max(1, int(deployment_periods))
        self.trace_offset_span = None if trace_offset_span is None else max(0, int(trace_offset_span))
        self.trace_offset = 0
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario, max_mapping_samples=mapping_samples)
        self.simulator.set_arrival_trace(arrival_trace)
        self.physical_router = PhysicalRouter(scenario)
        self.deployment_library = resolve_deployment_library(
            scenario, deployment_library, deployment_library_path
        )
        spec = objective or ObjectiveSpec.legacy()
        self.objective = ObjectiveEvaluator(
            scenario,
            spec,
            ReferenceScales.from_scenario(scenario, spec, self.deployment_library),
        )
        self._seed = seed
        self.gamma = gamma
        self.deployment_gamma = 1.0
        self.phase = self.LLM_DEPLOYMENT
        self.current_deployment = self.planner.initial_deployment()
        self.cost_min = self.objective.cost_min
        self.cost_max = self.objective.cost_max
        self.latency_reference = self.objective.latency_reference
        self.cost_reference = self._deployment_steady_cost(self.current_deployment)
        self.arrival_scale = max(
            1.0,
            sum(
                rate
                for app in scenario.applications.values()
                for rate in app.ingress_rates.values()
            ),
        )
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(scenario, seed).routing(self.current_deployment)
        self._last_slot_components = self._empty_slot_components()
        self._last_constraint_vector = self.objective.zero_constraints()
        self._last_objective: ObjectiveValue | None = None
        self._planning_model_share = self._uniform_model_share()
        self._period_index = 0
        self._capacity_plan: CapacityPlan | None = None
        self._deployment_targets: list[tuple[str, str, str]] = []
        self._deployment_target_index = 0
        self._current_demand: tuple[str, str, str] | None = None
        self._last_placement = -1
        self._deployment_actions_in_period = 0
        self._planning_shortfall = {"llm": 0.0, "tool": 0.0}
        self._period_initial_potential = 0.0
        self._last_potential = 0.0
        self._begin_deployment_cycle()

        feature_size = self._feature_vector().size
        self.observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(-10.0, 10.0, (feature_size,), dtype=np.float32),
                "action_type": gym.spaces.Discrete(3),
                "deploy_mask": gym.spaces.MultiBinary(self.layout.deployment_action_size),
                "model_mask": gym.spaces.MultiBinary(self.layout.model_action_size),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "deploy": gym.spaces.Discrete(self.layout.deployment_action_size),
                "model": gym.spaces.Box(
                    0.0, 1.0, (self.layout.model_action_size,), dtype=np.float32
                ),
            }
        )

    # ------------------------------------------------------------------ objective

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return self.objective.constraint_names

    @property
    def constraint_count(self) -> int:
        return self.objective.constraint_count

    def app_latency_reference(self, app_id: str) -> float:
        return self.objective.app_latency_reference(app_id)

    def _app_latency_reference(self, app_id: str) -> float:
        return self.objective.app_latency_reference(app_id)

    def _normalized_utility(
        self,
        cost: float,
        mean_latency: float,
        attainment: float,
        quality: float,
        app_latency: dict[str, float],
        arrival_rates: dict[tuple[str, str], float],
    ) -> tuple[float, dict[str, float]]:
        value = self.objective.evaluate_arrays(
            cost=cost,
            mean_latency=mean_latency,
            attainment=attainment,
            quality=quality,
            app_latency=app_latency,
            arrival_rates=arrival_rates,
        )
        return value.utility, dict(value.components)

    def _slot_objective(self, metrics, arrival_rates) -> ObjectiveValue:
        return self.objective.evaluate(metrics, arrival_rates)

    def _slot_reward(self, metrics, arrival_rates):
        value = self._slot_objective(metrics, arrival_rates)
        return value.utility, dict(value.components)

    def _slot_constraint_vector(self, metrics) -> np.ndarray:
        value = self.objective.evaluate(metrics, self.simulator.current_arrival_rates())
        return np.asarray(value.constraints, dtype=np.float32)

    def _deployment_steady_cost(self, deployment: DeploymentDecision) -> float:
        """Steady running cost of a deployment, excluding switch-on surcharges."""

        slot_seconds = self.scenario.simulation.slot_seconds
        llm_cost = sum(
            self.scenario.llm_configs[self.scenario.candidates[candidate_id].config].running_cost_per_slot
            * slot_seconds
            for candidate_id, active in deployment.llm_active.items()
            if active
        )
        service_cost = sum(
            replicas
            * self.scenario.tools[tool_id].running_cost_per_slot
            * slot_seconds
            for (tool_id, _), replicas in deployment.tool_replicas.items()
        )
        return max(float(llm_cost + service_cost), 1.0e-12)

    def _draw_trace_offset(self, seed: int | None) -> int:
        """Pick where in the arrival trace this episode starts.

        The episode is a window of ``max_periods`` steps *inside* the trace, so the
        offset is drawn from ``[0, len(trace) - max_periods]``.  Starting any later
        runs the episode past the end of the trace, and :meth:`ArrivalTrace.at`
        answers a slot it does not hold with the scenario's *unscaled* base rate --
        so the load silently drops by the arrival scale for the tail of the
        episode.  That was happening for 33-40% of every episode (offset 243 of a
        600-slot trace), which is what made the per-slot utility look
        non-stationary and made the 6-slot and 8-slot scoring protocols disagree.

        With a stationary trace the offset is irrelevant.  With a time-varying one
        it is essential: every episode would otherwise replay the same first few
        slots, so the policy would never see the rest of the cycle and could not
        learn to react to the load at all.
        """

        trace = self.simulator.arrival_trace
        span = self.trace_offset_span
        if span is None:
            span = len(trace.rates) if trace is not None else 0
        span = int(span) if span else 0
        if trace is not None and span < self.max_periods:
            raise ValueError(
                f"arrival trace covers {span} slots but the episode runs "
                f"{self.max_periods}; slots past the end fall back to the unscaled "
                "base rate, so the load would change silently mid-episode. Pass a "
                "trace at least as long as the episode."
            )
        usable = span - self.max_periods
        if usable <= 0:
            return 0
        rng = np.random.default_rng(abs(int(self._seed if seed is None else seed)) + 7919)
        return int(rng.integers(0, usable + 1))

    # ------------------------------------------------------------------ gym api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.trace_offset = self._draw_trace_offset(seed)
        self.simulator.reset(self._seed, slot=self.trace_offset)
        self._episode_utility_sum = 0.0
        self._episode_cost_sum = 0.0
        self._episode_latency_sum = 0.0
        self._episode_slots = 0
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(
            self.current_deployment
        )
        self._last_slot_components = self._empty_slot_components()
        self._last_constraint_vector = self.objective.zero_constraints()
        self._last_objective = None
        self._planning_model_share = self._uniform_model_share()
        self._period_index = 0
        self._begin_deployment_cycle()
        return self._observation(), {"discount": 1.0, "phase": self._phase_name()}

    def step(self, action: dict[str, Any]):
        if self.phase in (self.LLM_DEPLOYMENT, self.SERVICE_DEPLOYMENT):
            return self._step_deployment(int(np.asarray(action["deploy"]).item()))
        return self._step_composition(action)

    def _step_deployment(self, selected: int):
        if self._current_demand is None:
            raise RuntimeError("No deployment target is active")
        mask = self._deploy_mask()
        if selected < 0 or selected >= len(mask) or not mask[selected]:
            raise ValueError("The sequential deployment action is masked or invalid")
        target_kind, target_id, target_server = self._current_demand
        changed = False
        if target_kind == "llm":
            current = int(self.current_deployment.llm_active.get(target_id, 0))
            desired = current if selected == 0 else int(selected == 1)
            self.current_deployment.llm_active[target_id] = desired
            changed = desired != current
        elif target_kind == "tool":
            key = (target_id, target_server)
            current = int(self.current_deployment.tool_replicas.get(key, 0))
            desired = current + (1 if selected == 1 else -1 if selected == 2 else 0)
            self.current_deployment.tool_replicas[key] = int(desired)
            changed = desired != current
        else:
            raise ValueError("The deployment action does not match the active target")
        self._last_placement = selected
        self._deployment_actions_in_period += int(changed)
        self._advance_deployment_target()
        info = {
            "discount": 1.0,
            "phase": self._phase_name(),
            "deployment_complete": self.phase == self.COMPOSITION,
            "period_complete": False,
            "constraint_cost": 0.0,
            "constraint_vector": [0.0] * self.constraint_count,
            "constraint_steps": 0,
            "reward_components": {"utility": 0.0},
        }
        reward = 0.0
        if self.potential_shaping and changed:
            new_potential = self._deployment_potential()
            reward = float(new_potential - self._last_potential)
            self._last_potential = new_potential
        return self._observation(), reward, False, False, info

    def _step_composition(self, action: dict[str, Any]):
        arrival_rates = self.simulator.current_arrival_rates()
        routing = self.decode_routing(action, arrival_rates)
        transition = self.simulator.step(self.current_deployment, routing)
        self.last_routing = routing
        metrics = transition.metrics
        value = self._slot_objective(metrics, arrival_rates)
        constraint_vector = np.asarray(value.constraints, dtype=np.float32)
        self._last_slot_components = dict(value.components)
        self._last_constraint_vector = constraint_vector
        self._last_objective = value
        # Cumulative accounting over the episode. The paper's objective is the
        # time sum J = E[sum_t u(t)], and a bursty trace only distinguishes a
        # reactive policy from a fixed one through the accumulated total.
        self._episode_utility_sum += float(value.utility)
        self._episode_cost_sum += float(metrics.cost)
        self._episode_latency_sum += float(metrics.mean_latency_s)
        self._episode_slots += 1
        self._planning_model_share = dict(routing.model_share)
        self._period_index += 1
        terminated = self._period_index >= self.max_periods
        info = {
            "discount": self.gamma,
            "phase": "composition",
            "period_complete": True,
            "metrics": metrics,
            "reward_components": dict(value.components),
            "objective_diagnostics": dict(value.diagnostics),
            "app_utility": dict(value.app_utility),
            "utility": float(value.utility),
            "learning_utility": float(
                value.utility - self._composition_baseline_utility
                if getattr(self, "_composition_baseline_utility", None) is not None
                else value.utility
            ),
            "constraint_cost": float(np.sum(constraint_vector)),
            "constraint_vector": constraint_vector.tolist(),
            "constraint_steps": 1,
            "deployment_steps": self._deployment_actions_in_period,
            "episode_slot": self._episode_slots,
            "episode_utility_sum": self._episode_utility_sum,
            "episode_cost_sum": self._episode_cost_sum,
            "episode_latency_sum": self._episode_latency_sum,
            "trace_offset": self.trace_offset,
        }
        reward = float(value.utility)
        if not terminated:
            self._begin_deployment_cycle()
        return self._observation(), reward, terminated, False, info

    # ------------------------------------------------------------------ routing

    def decode_routing(
        self,
        action: dict[str, Any],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> RoutingDecision:
        model_raw = np.asarray(action["model"], dtype=float).reshape(
            len(self.layout.model_groups), len(self.layout.models)
        )
        model_share: dict[tuple[str, str, str], float] = {}
        active_models = self.active_models()
        for group_index, group in enumerate(self.layout.model_groups):
            weights = {
                model: model_raw[group_index, index]
                for index, model in enumerate(self.layout.models)
                if model in active_models
            }
            normalized = _normalized_or_uniform(weights)
            for model in self.layout.models:
                model_share[(*group, model)] = normalized.get(model, 0.0)
        return self.physical_router.route(
            self.current_deployment,
            model_share,
            self.simulator.last_metrics,
            arrival_rates,
        )

    def active_models(self) -> set[str]:
        return {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
        }

    def action_masks(self) -> dict[str, np.ndarray]:
        model_mask = np.zeros(
            (len(self.layout.model_groups), len(self.layout.models)), dtype=np.int8
        )
        active_models = self.active_models()
        for group_index in range(len(self.layout.model_groups)):
            for model_index, model in enumerate(self.layout.models):
                model_mask[group_index, model_index] = int(model in active_models)
        return {"deploy_mask": self._deploy_mask(), "model_mask": model_mask.reshape(-1)}

    def _deploy_mask(self) -> np.ndarray:
        mask = np.zeros(self.layout.deployment_action_size, dtype=np.int8)
        if self._current_demand is None:
            return mask
        target_kind, target_id, target_server = self._current_demand
        if target_kind == "llm":
            current = int(self.current_deployment.llm_active.get(target_id, 0))
            mask[0] = 1
            if not current:
                mask[1] = int(
                    self.planner.feasible_activation(self.current_deployment, target_id)
                )
            if current:
                mask[2] = 1
            return mask

        current = int(
            self.current_deployment.tool_replicas.get((target_id, target_server), 0)
        )
        mask[0] = 1
        mask[1] = int(
            self.planner.feasible_tool_replica(
                self.current_deployment, target_id, target_server
            )
        )
        mask[2] = int(current > 0)
        return mask

    # -------------------------------------------------- deployment target cycle

    def deployment_is_frozen(self) -> bool:
        """Whether ``t`` is outside ``T^dep`` for the period about to start."""

        return self.deployment_periods > 1 and (self._period_index % self.deployment_periods) != 0

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
        if self.deployment_is_frozen():
            self._capacity_plan = self.planner.plan(
                self.simulator.current_arrival_rates(),
                self._planning_model_share,
                self._period_index,
            )
            self._deployment_targets = []
            self._deployment_target_index = 0
            self._current_demand = None
            self._deployment_actions_in_period = 0
            self._period_initial_potential = self._deployment_potential()
            self._last_potential = self._period_initial_potential
            self.phase = self.COMPOSITION
            return
        arrival_rates = self.simulator.current_arrival_rates()
        self._capacity_plan = self.planner.plan(
            arrival_rates, self._planning_model_share, self._period_index
        )
        self._deployment_targets = [
            ("llm", candidate_id, "") for candidate_id in self.planner.candidate_order()
        ] + [
            ("tool", tool, server)
            for tool in self.layout.tools
            for server in self.layout.servers
        ]
        self._deployment_target_index = 0
        self._last_placement = -1
        self._deployment_actions_in_period = 0
        self._planning_shortfall = {"llm": 0.0, "tool": 0.0}
        self._period_initial_potential = self._deployment_potential()
        self._current_demand = None
        self.phase = self.LLM_DEPLOYMENT
        self._advance_deployment_target()

    def _advance_deployment_target(self) -> None:
        if self._current_demand is not None:
            self._deployment_target_index += 1
        if self._deployment_target_index >= len(self._deployment_targets):
            self._current_demand = None
            self.phase = self.COMPOSITION
            return
        self._current_demand = self._deployment_targets[self._deployment_target_index]
        self.phase = (
            self.LLM_DEPLOYMENT
            if self._current_demand[0] == "llm"
            else self.SERVICE_DEPLOYMENT
        )

    def _deployment_potential(self) -> float:
        if self._capacity_plan is None:
            return 0.0
        llm_gap = 0.0
        for model, required in self._capacity_plan.model_required_capacity.items():
            active = sum(
                self._capacity_plan.candidate_capacity.get(candidate_id, 0.0)
                for candidate_id, enabled in self.current_deployment.llm_active.items()
                if enabled and self.scenario.candidates[candidate_id].model == model
            )
            llm_gap += max(0.0, required - active) / max(required, 1.0)
        service_gap = 0.0
        for tool, required in self._capacity_plan.tool_required.items():
            total = sum(
                self.current_deployment.tool_replicas.get((tool, server), 0)
                for server in self.layout.servers
            )
            service_gap += max(0.0, required - total) / max(required, 1)
        potential = -(llm_gap + service_gap)
        if self.potential_cost_weight > 0.0:
            steady = self._deployment_steady_cost(self.current_deployment)
            potential -= self.potential_cost_weight * (
                steady / max(self.cost_reference, 1.0e-12)
            )
        return float(potential)

    # ------------------------------------------------------------------ features

    def _uniform_model_share(self):
        probability = 1.0 / max(1, len(self.scenario.models))
        return {
            (app.id, ingress, model): probability
            for app in self.scenario.applications.values()
            for ingress in app.ingress_rates
            for model in self.scenario.models
        }

    @staticmethod
    def _empty_slot_components():
        return {
            "utility": 0.0,
            "cost_normalized": 0.0,
            "latency_normalized": 0.0,
            "goodput_normalized": 0.0,
            "quality_normalized": 0.0,
        }

    def _phase_name(self) -> str:
        return {
            self.LLM_DEPLOYMENT: "llm_deployment",
            self.SERVICE_DEPLOYMENT: "service_deployment",
            self.COMPOSITION: "composition",
        }[self.phase]

    def _feature_vector(self) -> np.ndarray:
        features: list[float] = []
        features.extend(
            float(self.current_deployment.llm_active.get(cid, 0))
            for cid in self.layout.candidates
        )
        features.extend(
            self.current_deployment.tool_replicas.get((tool, server), 0)
            / max(1, self.scenario.simulation.max_tool_replicas_per_server)
            for tool in self.layout.tools
            for server in self.layout.servers
        )
        features.extend(
            self.simulator.current_arrival_rates().get((app.id, ingress), rate)
            / self.arrival_scale
            for app in self.scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        )
        metrics = self.simulator.last_metrics
        features.extend(
            [
                metrics.cost / self.cost_reference if metrics else 0.0,
                metrics.mean_latency_s / self.latency_reference if metrics else 0.0,
                metrics.slo_attainment if metrics else 0.0,
                metrics.quality if metrics else 0.0,
                *self._last_constraint_vector.tolist(),
            ]
        )
        features.extend(
            min(2.0, metrics.llm_utilization.get(cid, 0.0)) if metrics else 0.0
            for cid in self.layout.candidates
        )
        features.extend(
            min(2.0, metrics.tool_utilization.get(f"{tool}@{server}", 0.0))
            if metrics
            else 0.0
            for tool in self.layout.tools
            for server in self.layout.servers
        )
        features.extend(
            min(2.0, metrics.link_utilization.get(f"{link.source}->{link.target}", 0.0))
            if metrics
            else 0.0
            for link in self.scenario.links
        )
        remaining = self.planner.normalized_remaining_resources(self.current_deployment)
        for server in self.layout.servers:
            features.extend(remaining[server])
        features.extend(
            float(
                self._current_demand is not None
                and self._current_demand[0] == "llm"
                and self._current_demand[1] == candidate
            )
            for candidate in self.layout.candidates
        )
        features.extend(
            float(self._current_demand == ("tool", tool, server))
            for tool in self.layout.tools
            for server in self.layout.servers
        )
        features.append(
            self._deployment_target_index / max(1, len(self._deployment_targets))
        )
        # Time-varying information. The intensity in force next slot is known to
        # the controller (it is a schedule, not a surprise), and a deployment
        # decision is only worth making differently if what is coming differs from
        # what is here. The remaining freeze is what tells the policy whether this
        # is a slot in which it may act at all.
        trace = self.simulator.arrival_trace
        if trace is not None:
            next_total = sum(
                trace.at(self.simulator.slot + 1, self.scenario).values()
            )
        else:
            next_total = sum(self.simulator.current_arrival_rates().values())
        features.append(min(2.0, float(next_total) / self.arrival_scale))
        if self.deployment_periods > 1:
            remaining = (-self._period_index) % self.deployment_periods
            features.append(remaining / self.deployment_periods)
        else:
            features.append(0.0)
        # Progress through the episode. This is a finite-horizon problem with a
        # time-varying intensity, so where the controller is on the timeline is
        # part of the state; excluding it was a choice made when the timeline was
        # flat.
        features.append(self._period_index / max(1, self.max_periods))
        return np.clip(np.asarray(features, dtype=np.float32), -10.0, 10.0)

    def _observation(self):
        masks = self.action_masks()
        return {"features": self._feature_vector(), "action_type": self.phase, **masks}


def _normalized_or_uniform(values: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(positive.values())
    if total <= 1.0e-12:
        return {key: 1.0 / len(positive) for key in positive} if positive else {}
    return {key: value / total for key, value in positive.items()}
