from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlan, CapacityPlanner
from agent_orch.routing import PhysicalRouter
from agent_orch.schema.models import DeploymentDecision, NodeType, RoutingDecision, Scenario
from agent_orch.simulator import Simulator
from agent_orch.workload import ArrivalTrace


@dataclass(frozen=True)
class StructuredActionLayout:
    models: tuple[str, ...]
    candidates: tuple[str, ...]
    servers: tuple[str, ...]
    tools: tuple[str, ...]
    model_groups: tuple[tuple[str, str], ...]
    deployment_targets: tuple[tuple[str, str], ...]

    @staticmethod
    def build(scenario: Scenario) -> "StructuredActionLayout":
        models = tuple(scenario.models)
        candidates = tuple(scenario.candidates)
        servers = tuple(scenario.servers)
        tools = tuple(scenario.tools)
        model_groups = tuple(
            (app.id, ingress)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
        )
        targets = (("keep", "0"), ("add", "1"), ("remove", "2"))
        return StructuredActionLayout(
            models,
            candidates,
            servers,
            tools,
            model_groups,
            targets,
        )

    @property
    def deployment_action_size(self) -> int:
        return len(self.deployment_targets)

    @property
    def model_action_size(self) -> int:
        return len(self.model_groups) * len(self.models)


class AgentOrchestrationEnv(gym.Env):
    """Macro-period environment for sequential deployment and steady evaluation.

    A Gym transition is a deployment substep until all resource pools have been
    processed.  The final transition samples model composition, invokes the
    deterministic physical router, evaluates one steady-state period, and only
    then advances the period index.
    """

    metadata = {"render_modes": []}
    LLM_DEPLOYMENT = 0
    SERVICE_DEPLOYMENT = 1
    COMPOSITION = 2
    DEPLOYMENT = LLM_DEPLOYMENT
    ROUTING = COMPOSITION
    CONSTRAINT_NAMES = ("llm", "service")
    REWARD_TIE_RELATIVE_TOLERANCE = 1.0e-3
    REWARD_TIE_BONUS = 1.0e-3

    def __init__(
        self,
        scenario: Scenario,
        max_slots: int = 600,
        potential_shaping: bool = True,
        seed: int = 0,
        arrival_trace: ArrivalTrace | None = None,
        gamma: float = 0.99,
        mapping_samples: int = 4096,
    ):
        super().__init__()
        self.scenario = scenario
        self.max_periods = max(1, int(max_slots))
        self.potential_shaping = potential_shaping
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario, max_mapping_samples=mapping_samples)
        self.simulator.set_arrival_trace(arrival_trace)
        self.physical_router = PhysicalRouter(scenario)
        self._seed = seed
        self.gamma = gamma
        self.deployment_gamma = 1.0
        self.phase = self.LLM_DEPLOYMENT
        self.current_deployment = self.planner.initial_deployment()
        self.cost_reference = self._default_cost_reference()
        self.latency_reference = self._default_latency_reference()
        self.arrival_scale = max(
            1.0,
            sum(rate for app in scenario.applications.values() for rate in app.ingress_rates.values()),
        )
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(scenario, seed).routing(self.current_deployment)
        self._last_slot_components = self._empty_slot_components()
        self._last_performance_reference: dict[str, float] | None = None
        self._last_constraint_vector = np.zeros(2, dtype=np.float32)
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
                "model": gym.spaces.Box(0.0, 1.0, (self.layout.model_action_size,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.simulator.reset(self._seed)
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(self.current_deployment)
        self._last_slot_components = self._empty_slot_components()
        self._last_performance_reference = None
        self._last_constraint_vector = np.zeros(2, dtype=np.float32)
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
            "constraint_vector": [0.0, 0.0],
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
        utility, components = self._slot_reward(metrics, arrival_rates)
        constraint_vector = self._slot_constraint_vector(metrics)
        self._last_slot_components = components
        self._last_constraint_vector = constraint_vector
        self._planning_model_share = dict(routing.model_share)
        self._period_index += 1
        terminated = self._period_index >= self.max_periods
        info = {
            "discount": self.gamma,
            "phase": "composition",
            "period_complete": True,
            "metrics": metrics,
            "reward_components": components,
            "utility": float(utility),
            "constraint_cost": float(np.sum(constraint_vector)),
            "constraint_vector": constraint_vector.tolist(),
            "constraint_steps": 1,
            "deployment_steps": self._deployment_actions_in_period,
        }
        reward = float(utility)
        if not terminated:
            self._begin_deployment_cycle()
        return self._observation(), reward, terminated, False, info

    def decode_routing(
        self,
        action: dict[str, Any],
        arrival_rates: dict[tuple[str, str], float] | None = None,
    ) -> RoutingDecision:
        model_raw = np.asarray(action["model"], dtype=float).reshape(
            len(self.layout.model_groups), len(self.layout.models)
        )
        model_share: dict[tuple[str, str, str], float] = {}
        active_models = {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
        }
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

    def action_masks(self) -> dict[str, np.ndarray]:
        model_mask = np.zeros(
            (len(self.layout.model_groups), len(self.layout.models)), dtype=np.int8
        )
        active_models = {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
        }
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

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
        arrival_rates = self.simulator.current_arrival_rates()
        self._capacity_plan = self.planner.plan(
            arrival_rates, self._planning_model_share, self._period_index
        )
        model_targets = [
            ("llm", candidate_id, "")
            for candidate_id in self.planner.candidate_order()
        ]
        tool_targets = [
            ("tool", tool, server)
            for tool in self.layout.tools
            for server in self.layout.servers
        ]
        self._deployment_targets = model_targets + tool_targets
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
        return float(-(llm_gap + service_gap))

    def _slot_reward(self, metrics, arrival_rates):
        del arrival_rates
        utility, components, current = self._incremental_utility(
            metrics.cost,
            metrics.mean_latency_s,
            metrics.goodput_rps,
            metrics.quality,
            self._last_performance_reference,
        )
        self._last_performance_reference = current
        return utility, components

    def _incremental_utility(
        self,
        cost: float,
        mean_latency: float,
        goodput: float,
        quality: float,
        previous: dict[str, float] | None,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        current = {
            "cost": float(cost),
            "latency": float(mean_latency),
            "goodput": float(goodput),
            "quality": float(quality),
        }
        if previous is None:
            raw_deltas = {name: 0.0 for name in current}
        else:
            raw_deltas = {
                "cost": float(previous["cost"] - current["cost"]),
                "latency": float(previous["latency"] - current["latency"]),
                "goodput": float(current["goodput"] - previous["goodput"]),
                "quality": float(current["quality"] - previous["quality"]),
            }
        deltas = {
            name: self._tie_adjusted_delta(
                value,
                current[name],
                None if previous is None else previous[name],
            )
            for name, value in raw_deltas.items()
        }
        components = {
            "cost_delta": deltas["cost"],
            "latency_delta": deltas["latency"],
            "goodput_delta": deltas["goodput"],
            "quality_delta": deltas["quality"],
        }
        weights = self.scenario.reward
        utility = (
            weights.cost_weight * components["cost_delta"]
            + weights.latency_weight * components["latency_delta"]
            + weights.goodput_weight * components["goodput_delta"]
            + weights.quality_weight * components["quality_delta"]
        )
        components["utility"] = float(utility)
        return float(utility), components, current

    def _tie_adjusted_delta(
        self,
        delta: float,
        current: float,
        previous: float | None,
    ) -> float:
        if previous is None:
            return self.REWARD_TIE_BONUS
        scale = max(abs(float(current)), abs(float(previous)), 1.0e-12)
        if abs(float(delta)) <= self.REWARD_TIE_RELATIVE_TOLERANCE * scale:
            return self.REWARD_TIE_BONUS
        return float(delta)

    def _slot_constraint_vector(self, metrics) -> np.ndarray:
        llm = max([max(0.0, value - 0.9) for value in metrics.llm_utilization.values()] or [0.0])
        service = max([max(0.0, value - 0.9) for value in metrics.tool_utilization.values()] or [0.0])
        labels = metrics.diagnostics.get("violation_labels", [])
        llm += float(any(label.startswith("llm_") for label in labels))
        service += float(
            any(
                label.startswith(("service_", "tool_", "link_"))
                for label in labels
            )
        )
        return np.asarray([llm, service], dtype=np.float32)

    def _app_latency_reference(self, app_id: str) -> float:
        app = self.scenario.applications[app_id]
        if app.slo.deadline_s is not None:
            return max(app.slo.deadline_s, 1.0e-9)
        ttft = app.slo.ttft_s or self.scenario.simulation.overload_delay_s
        tbt = app.slo.tbt_s or 0.0
        output_tokens = sum(
            flow.probability * float(np.mean(list(app.nodes[flow.final_node].output_tokens.values())))
            for flow in app.pattern_flows
        )
        return max(ttft + max(0.0, output_tokens - 1.0) * tbt, 1.0e-9)

    def _default_cost_reference(self) -> float:
        """Steady running cost of the scenario's reference deployment."""

        period_seconds = self.scenario.simulation.orchestration_period_s
        llm_cost = sum(
            self.scenario.llm_configs[
                self.scenario.candidates[candidate_id].config
            ].running_cost_per_slot
            * period_seconds
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
        )
        service_cost = sum(
            replicas
            * self.scenario.tools[tool_id].running_cost_per_slot
            * period_seconds
            for (tool_id, _), replicas in self.current_deployment.tool_replicas.items()
        )
        return max(float(llm_cost + service_cost), 1.0e-12)

    def _default_latency_reference(self) -> float:
        """Arrival-weighted application SLO reference under the base workload."""

        weighted = sum(
            max(0.0, rate) * self._app_latency_reference(app.id)
            for app in self.scenario.applications.values()
            for rate in app.ingress_rates.values()
        )
        total_rate = sum(
            max(0.0, rate)
            for app in self.scenario.applications.values()
            for rate in app.ingress_rates.values()
        )
        if total_rate > 0.0:
            return max(float(weighted / total_rate), 1.0e-12)
        return max(
            float(
                np.mean(
                    [
                        self._app_latency_reference(app_id)
                        for app_id in self.scenario.applications
                    ]
                )
            ),
            1.0e-12,
        )

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
            "cost_delta": 0.0,
            "latency_delta": 0.0,
            "goodput_delta": 0.0,
            "quality_delta": 0.0,
        }

    def _phase_name(self) -> str:
        return {self.LLM_DEPLOYMENT: "llm_deployment", self.SERVICE_DEPLOYMENT: "service_deployment", self.COMPOSITION: "composition"}[self.phase]

    def _feature_vector(self) -> np.ndarray:
        features: list[float] = []
        features.extend(float(self.current_deployment.llm_active.get(cid, 0)) for cid in self.layout.candidates)
        features.extend(
            self.current_deployment.tool_replicas.get((tool, server), 0) / max(1, self.scenario.simulation.max_tool_replicas_per_server)
            for tool in self.layout.tools for server in self.layout.servers
        )
        features.extend(
            self.simulator.current_arrival_rates().get((app.id, ingress), rate) / self.arrival_scale
            for app in self.scenario.applications.values() for ingress, rate in app.ingress_rates.items()
        )
        metrics = self.simulator.last_metrics
        features.extend([
            metrics.cost / self.cost_reference if metrics else 0.0,
            metrics.mean_latency_s / self.latency_reference if metrics else 0.0,
            metrics.slo_attainment if metrics else 0.0,
            metrics.quality if metrics else 0.0,
            *self._last_constraint_vector.tolist(),
        ])
        features.extend(min(2.0, metrics.llm_utilization.get(cid, 0.0)) if metrics else 0.0 for cid in self.layout.candidates)
        features.extend(min(2.0, metrics.tool_utilization.get(f"{tool}@{server}", 0.0)) if metrics else 0.0 for tool in self.layout.tools for server in self.layout.servers)
        features.extend(min(2.0, metrics.link_utilization.get(f"{link.source}->{link.target}", 0.0)) if metrics else 0.0 for link in self.scenario.links)
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
        return np.clip(np.asarray(features, dtype=np.float32), -10.0, 10.0)

    def _observation(self):
        masks = self.action_masks()
        return {"features": self._feature_vector(), "action_type": self.phase, **masks}


class RoutingOnlyEnv(AgentOrchestrationEnv):
    """Composition training under one fixed deployment context per episode."""

    def __init__(self, *args, fixed_deployment_count: int = 8, **kwargs):
        self.fixed_deployment_count = max(1, int(fixed_deployment_count))
        self._fixed_deployment_catalog: list[DeploymentDecision] = []
        self._fixed_deployment_index = 0
        super().__init__(*args, **kwargs)
        self._fixed_deployment_catalog = self._build_fixed_deployment_catalog()

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, _ = super().reset(seed=seed, options=options)
        if not self._fixed_deployment_catalog:
            self._fixed_deployment_catalog = self._build_fixed_deployment_catalog()
        deployment_rng = np.random.default_rng(abs(int(self._seed)))
        self._fixed_deployment_index = int(
            deployment_rng.integers(len(self._fixed_deployment_catalog))
        )
        self.current_deployment = self._fixed_deployment_catalog[
            self._fixed_deployment_index
        ].copy()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(
            self.current_deployment
        )
        self.phase = self.COMPOSITION
        return self._observation(), {
            "discount": self.gamma,
            "phase": "composition",
            "fixed_deployment_index": self._fixed_deployment_index,
            "fixed_deployment_count": len(self._fixed_deployment_catalog),
        }

    def _build_fixed_deployment_catalog(self) -> list[DeploymentDecision]:
        catalog = [self.planner.initial_deployment()]
        signatures = {self._deployment_signature(catalog[0])}
        models = tuple(self.scenario.models)
        tools = tuple(self.scenario.tools)
        max_attempts = max(
            self.fixed_deployment_count * 4,
            max((len(self.scenario.candidates), len(self.scenario.servers)), default=1),
        )
        for offset in range(1, max_attempts + 1):
            deployment = self.planner.empty_deployment()
            complete = True
            for model_index, model in enumerate(models):
                candidates = sorted(
                    (
                        candidate
                        for candidate in self.scenario.candidates.values()
                        if candidate.model == model
                    ),
                    key=lambda item: (
                        self.scenario.llm_configs[item.config].running_cost_per_slot,
                        item.server,
                        item.config,
                        item.id,
                    ),
                )
                candidates = self._rotate(candidates, offset + model_index)
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if self.planner.feasible_activation(deployment, candidate.id)
                    ),
                    None,
                )
                if selected is None:
                    complete = False
                    break
                deployment.llm_active[selected.id] = 1
            if not complete:
                continue
            for tool_index, tool_id in enumerate(tools):
                servers = sorted(
                    self.scenario.servers,
                    key=lambda server_id: (
                        -self.scenario.tools[tool_id].service_rate[server_id],
                        server_id,
                    ),
                )
                servers = self._rotate(servers, offset + tool_index)
                selected_server = next(
                    (
                        server_id
                        for server_id in servers
                        if self.planner.feasible_tool_replica(
                            deployment, tool_id, server_id
                        )
                    ),
                    None,
                )
                if selected_server is None:
                    complete = False
                    break
                deployment.tool_replicas[(tool_id, selected_server)] = 1
            if not complete or not self.planner.deployment_feasible(deployment):
                continue
            signature = self._deployment_signature(deployment)
            if signature in signatures:
                continue
            signatures.add(signature)
            catalog.append(deployment)
            if len(catalog) >= self.fixed_deployment_count:
                break
        return catalog

    @staticmethod
    def _rotate(values: list[Any], offset: int) -> list[Any]:
        if not values:
            return []
        pivot = offset % len(values)
        return values[pivot:] + values[:pivot]

    @staticmethod
    def _deployment_signature(deployment: DeploymentDecision) -> tuple:
        return (
            tuple(sorted(deployment.llm_active.items())),
            tuple(sorted(deployment.tool_replicas.items())),
        )


class DeploymentOnlyEnv(AgentOrchestrationEnv):
    """Sequential deployment with a uniform composition at evaluation points."""

    def step(self, action: dict[str, Any]):
        policy_action_ignored = False
        if self.phase == self.COMPOSITION:
            action = dict(action)
            action["model"] = np.ones(self.layout.model_action_size, dtype=np.float32)
            policy_action_ignored = True
        observation, reward, terminated, truncated, info = super().step(action)
        if policy_action_ignored:
            info = dict(info)
            info["policy_action_ignored"] = True
        return observation, reward, terminated, truncated, info


def _normalized_or_uniform(values: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(positive.values())
    if total <= 1.0e-12:
        return {key: 1.0 / len(positive) for key in positive} if positive else {}
    return {key: value / total for key, value in positive.items()}
