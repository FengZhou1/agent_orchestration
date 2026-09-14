from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlan, CapacityPlanner
from agent_orch.routing import PhysicalRouter
from agent_orch.schema.models import (
    DeploymentDecision,
    NodeType,
    RoutingDecision,
    Scenario,
)
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
        targets = tuple(("server", server) for server in servers) + tuple(
            ("candidate", candidate) for candidate in candidates
        )
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
    metadata = {"render_modes": []}

    DEPLOYMENT = 0
    ROUTING = 1
    CONSTRAINT_NAMES = ("llm", "kv", "tool", "link")

    def __init__(
        self,
        scenario: Scenario,
        max_slots: int = 600,
        potential_shaping: bool = False,
        seed: int = 0,
        arrival_trace: ArrivalTrace | None = None,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.scenario = scenario
        self.max_slots = max_slots
        self.potential_shaping = potential_shaping
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario)
        self.simulator.set_arrival_trace(arrival_trace)
        self.physical_router = PhysicalRouter(scenario)
        self._seed = seed
        self.gamma = gamma
        self.deployment_gamma = 1.0
        self.phase = self.DEPLOYMENT
        self.cost_min, self.cost_max = self._fixed_cost_bounds()
        self.arrival_scale = max(
            1.0,
            sum(
                rate
                for app in self.scenario.applications.values()
                for rate in app.ingress_rates.values()
            ),
        )

        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(scenario, seed).routing(
            self.current_deployment
        )
        self._last_slot_components = self._empty_slot_components()
        self._last_constraint_vector = np.zeros(4, dtype=np.float32)
        self._planning_model_share = self._uniform_model_share()
        self._period_index = 0
        self._capacity_plan: CapacityPlan | None = None
        self._deployment_queue: list[tuple[str, str]] = []
        self._current_demand: tuple[str, str] | None = None
        self._model_accumulated_capacity: dict[str, float] = {}
        self._last_placement = -1
        self._deployment_actions_in_period = 0
        self._planning_shortfall = {"llm": 0.0, "tool": 0.0}
        self._period_routing_steps = 0
        self._period_utilities: list[float] = []
        self._period_constraints: list[np.ndarray] = []
        self._period_baseline_utilities: list[float] = []
        self._period_baseline_constraints: list[np.ndarray] = []
        self._period_model_shares: list[dict[tuple[str, str, str], float]] = []
        self._begin_deployment_cycle()

        feature_size = self._feature_vector().size
        self.observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(
                    -10.0, 10.0, (feature_size,), dtype=np.float32
                ),
                "action_type": gym.spaces.Discrete(2),
                "deploy_mask": gym.spaces.MultiBinary(
                    self.layout.deployment_action_size
                ),
                "model_mask": gym.spaces.MultiBinary(self.layout.model_action_size),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "deploy": gym.spaces.Discrete(self.layout.deployment_action_size),
                "model": gym.spaces.Box(
                    0.0,
                    1.0,
                    (self.layout.model_action_size,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.simulator.reset(self._seed)
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(
            self.current_deployment
        )
        self._last_slot_components = self._empty_slot_components()
        self._last_constraint_vector = np.zeros(4, dtype=np.float32)
        self._planning_model_share = self._uniform_model_share()
        self._period_index = 0
        self._begin_deployment_cycle()
        return self._observation(), {
            "discount": self.deployment_gamma,
            "phase": "deployment",
        }

    def step(self, action: dict[str, Any]):
        if self.phase == self.DEPLOYMENT:
            return self._step_deployment(int(np.asarray(action["deploy"]).item()))
        return self._step_routing(action)

    def _step_deployment(self, selected: int):
        if self._current_demand is None:
            raise RuntimeError("No pending deployment demand")
        mask = self._deploy_mask()
        if selected < 0 or selected >= len(mask) or not mask[selected]:
            raise ValueError("The sequential deployment action is masked or invalid")

        kind, service_id = self._current_demand
        target_kind, target_id = self.layout.deployment_targets[selected]
        if kind == "tool":
            if target_kind != "server":
                raise ValueError("A tool replica must be placed on a server")
            key = (service_id, target_id)
            self.current_deployment.tool_replicas[key] += 1
            self._deployment_queue.pop(0)
        else:
            if target_kind != "candidate":
                raise ValueError("An LLM model must select a candidate instance")
            candidate = self.scenario.candidates[target_id]
            if candidate.model != service_id:
                raise ValueError("The selected candidate serves another model")
            self.current_deployment.llm_active[target_id] = 1
            assert self._capacity_plan is not None
            self._model_accumulated_capacity[service_id] += (
                self._capacity_plan.candidate_capacity[target_id]
            )
            required = self._capacity_plan.model_required_capacity[service_id]
            if self._model_accumulated_capacity[service_id] >= required - 1.0e-12:
                self._deployment_queue.pop(0)

        self._last_placement = selected
        self._deployment_actions_in_period += 1
        self._advance_deployment_queue()
        complete = self.phase == self.ROUTING
        info = {
            "discount": self.deployment_gamma,
            "phase": "deployment",
            "deployment_complete": complete,
            "constraint_cost": 0.0,
            "constraint_vector": [0.0, 0.0, 0.0, 0.0],
            "constraint_steps": 0,
            "reward_components": {"utility": 0.0},
        }
        return self._observation(), 0.0, False, False, info

    def _step_routing(self, action: dict[str, Any]):
        arrival_rates = self.simulator.current_arrival_rates()
        previous_metrics = self.simulator.last_metrics
        routing = self.decode_routing(action)
        baseline_utility, baseline_constraints = self._counterfactual_performance(
            self._base_deployment,
            routing.model_share,
            arrival_rates,
            previous_metrics,
        )
        transition = self.simulator.step(self.current_deployment, routing)
        self.last_routing = routing
        metrics = transition.metrics
        reward, reward_components = self._slot_reward(metrics, arrival_rates)
        constraint_vector = self._slot_constraint_vector(metrics)
        self._last_slot_components = reward_components
        self._last_constraint_vector = constraint_vector
        self._period_utilities.append(reward)
        self._period_constraints.append(constraint_vector)
        self._period_baseline_utilities.append(baseline_utility)
        self._period_baseline_constraints.append(baseline_constraints)
        self._period_model_shares.append(dict(routing.model_share))
        self._period_routing_steps += 1

        terminated = self.simulator.slot >= self.max_slots
        period_complete = (
            self._period_routing_steps
            >= self.scenario.simulation.deployment_period_slots
        )
        info = {
            "discount": self.gamma,
            "phase": "routing",
            "metrics": metrics,
            "reward_components": reward_components,
            "constraint_cost": float(np.sum(constraint_vector)),
            "constraint_vector": constraint_vector.tolist(),
            "constraint_steps": 1,
        }
        if period_complete or terminated:
            info["deployment_period_summary"] = self._deployment_period_summary()
            self._update_planning_model_share()
            if not terminated:
                self._period_index += 1
                self._begin_deployment_cycle()
        return self._observation(), reward, terminated, False, info

    def decode_routing(self, action: dict[str, Any]) -> RoutingDecision:
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
        return {
            "deploy_mask": self._deploy_mask(),
            "model_mask": model_mask.reshape(-1),
        }

    def _deploy_mask(self) -> np.ndarray:
        mask = np.zeros(self.layout.deployment_action_size, dtype=np.int8)
        if self.phase != self.DEPLOYMENT or self._current_demand is None:
            return mask
        kind, service_id = self._current_demand
        for index, (target_kind, target_id) in enumerate(
            self.layout.deployment_targets
        ):
            if kind == "tool":
                mask[index] = int(
                    target_kind == "server"
                    and self.planner.feasible_tool_replica(
                        self.current_deployment, service_id, target_id
                    )
                )
            else:
                candidate = self.scenario.candidates.get(target_id)
                mask[index] = int(
                    target_kind == "candidate"
                    and candidate is not None
                    and candidate.model == service_id
                    and not self.current_deployment.llm_active[target_id]
                    and self.planner.feasible_activation(
                        self.current_deployment, target_id
                    )
                )
        return mask

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
        arrival_rates = self._period_average_arrivals()
        self._capacity_plan = self.planner.plan(
            arrival_rates,
            self._planning_model_share,
            self._period_index,
        )
        self.current_deployment = self.planner.empty_deployment()
        self._model_accumulated_capacity = {
            model: 0.0 for model in self.scenario.models
        }
        self._deployment_queue = []
        for kind, service_id in self._capacity_plan.service_order:
            if kind == "tool":
                self._deployment_queue.extend(
                    [(kind, service_id)]
                    * self._capacity_plan.tool_required[service_id]
                )
            elif self._capacity_plan.model_required_capacity[service_id] > 0.0:
                self._deployment_queue.append((kind, service_id))
        self._current_demand = None
        self._last_placement = -1
        self._deployment_actions_in_period = 0
        self._planning_shortfall = {"llm": 0.0, "tool": 0.0}
        self._period_routing_steps = 0
        self._period_utilities = []
        self._period_constraints = []
        self._period_baseline_utilities = []
        self._period_baseline_constraints = []
        self._period_model_shares = []
        self.phase = self.DEPLOYMENT
        self._advance_deployment_queue()

    def _advance_deployment_queue(self) -> None:
        while self._deployment_queue:
            self._current_demand = self._deployment_queue[0]
            if np.any(self._deploy_mask()):
                return
            kind, service_id = self._deployment_queue.pop(0)
            assert self._capacity_plan is not None
            if kind == "tool":
                self._planning_shortfall["tool"] += 1.0
            else:
                required = self._capacity_plan.model_required_capacity[service_id]
                accumulated = self._model_accumulated_capacity[service_id]
                self._planning_shortfall["llm"] += max(
                    0.0, 1.0 - accumulated / max(required, 1.0e-12)
                )
        self._current_demand = None
        self.phase = self.ROUTING

    def _period_average_arrivals(self) -> dict[tuple[str, str], float]:
        start = self.simulator.slot
        stop = min(
            self.max_slots,
            start + self.scenario.simulation.deployment_period_slots,
        )
        if stop <= start or self.simulator.arrival_trace is None:
            return self.simulator.current_arrival_rates()
        totals: dict[tuple[str, str], float] = {}
        for slot in range(start, stop):
            for key, value in self.simulator.arrival_trace.at(
                slot, self.scenario
            ).items():
                totals[key] = totals.get(key, 0.0) + value
        count = stop - start
        return {key: value / count for key, value in totals.items()}

    def _deployment_period_summary(self) -> dict[str, Any]:
        return {
            "actual_utility": float(np.mean(self._period_utilities)),
            "actual_constraints": np.mean(
                np.stack(self._period_constraints), axis=0
            ).tolist(),
            "baseline_utility": float(np.mean(self._period_baseline_utilities)),
            "baseline_constraints": np.mean(
                np.stack(self._period_baseline_constraints), axis=0
            ).tolist(),
            "deployment_steps": self._deployment_actions_in_period,
            "routing_steps": self._period_routing_steps,
            "planning_shortfall": dict(self._planning_shortfall),
        }

    def _update_planning_model_share(self) -> None:
        if not self._period_model_shares:
            return
        keys = self._uniform_model_share()
        self._planning_model_share = {
            key: float(
                np.mean([share.get(key, 0.0) for share in self._period_model_shares])
            )
            for key in keys
        }

    def _counterfactual_performance(
        self,
        deployment: DeploymentDecision,
        model_share: dict[tuple[str, str, str], float],
        arrival_rates: dict[tuple[str, str], float],
        previous_metrics,
    ) -> tuple[float, np.ndarray]:
        routing = self.physical_router.route(
            deployment, model_share, previous_metrics
        )
        analytical = self.simulator.backend.evaluate(
            deployment, routing, arrival_rates
        )
        analytical.violations.extend(
            self.simulator._routing_unserved_violations(
                deployment, routing, arrival_rates
            )
        )
        workflow = self.simulator.workflow.evaluate(
            deployment, routing, analytical, arrival_rates
        )
        cost = self._steady_cost(deployment, analytical.link_load_mbps)
        attainment = (
            workflow.goodput_rps / workflow.total_arrival_rps
            if workflow.total_arrival_rps > 0.0
            else 0.0
        )
        utility, _ = self._normalized_utility(
            cost,
            workflow.mean_latency_s,
            attainment,
            workflow.quality,
            workflow.app_latency_s,
            arrival_rates,
        )
        constraints = self._constraint_vector(
            analytical.llm_utilization,
            analytical.llm_kv_stable,
            {
                f"{tool}@{server}": value
                for (tool, server), value in analytical.tool_utilization.items()
            },
            self.simulator.backend.network.utilization(
                analytical.link_load_mbps
            ),
            analytical.violations,
        )
        return utility, constraints

    def _slot_reward(
        self,
        metrics,
        arrival_rates: dict[tuple[str, str], float],
    ) -> tuple[float, dict[str, float]]:
        return self._normalized_utility(
            metrics.cost,
            metrics.mean_latency_s,
            metrics.slo_attainment,
            metrics.quality,
            metrics.app_latency_s,
            arrival_rates,
        )

    def _normalized_utility(
        self,
        cost: float,
        mean_latency: float,
        attainment: float,
        quality: float,
        app_latency: dict[str, float],
        arrival_rates: dict[tuple[str, str], float],
    ) -> tuple[float, dict[str, float]]:
        cost_normalized = float(
            np.clip(
                (cost - self.cost_min)
                / max(self.cost_max - self.cost_min, 1.0e-12),
                0.0,
                1.0,
            )
        )
        total_rate = sum(max(0.0, rate) for rate in arrival_rates.values())
        if total_rate > 0.0 and app_latency:
            latency_normalized = sum(
                sum(
                    max(0.0, arrival_rates.get((app.id, ingress), 0.0))
                    for ingress in app.ingress_rates
                )
                * min(
                    1.0,
                    app_latency.get(app.id, mean_latency)
                    / self._app_latency_reference(app.id),
                )
                for app in self.scenario.applications.values()
            ) / total_rate
        else:
            references = [
                self._app_latency_reference(app_id)
                for app_id in self.scenario.applications
            ]
            latency_normalized = min(
                1.0, mean_latency / max(float(np.mean(references)), 1.0e-12)
            )
        goodput_normalized = float(np.clip(attainment, 0.0, 1.0))
        quality_normalized = float(np.clip(quality, 0.0, 1.0))
        weights = self.scenario.reward
        utility = (
            weights.goodput_weight * goodput_normalized
            + weights.quality_weight * quality_normalized
            - weights.cost_weight * cost_normalized
            - weights.latency_weight * latency_normalized
        )
        return float(utility), {
            "utility": float(utility),
            "cost_normalized": cost_normalized,
            "latency_normalized": float(latency_normalized),
            "goodput_normalized": goodput_normalized,
            "quality_normalized": quality_normalized,
        }

    def _slot_constraint_vector(self, metrics) -> np.ndarray:
        return self._constraint_vector(
            metrics.llm_utilization,
            metrics.diagnostics.get("kv_stable", {}),
            metrics.tool_utilization,
            metrics.link_utilization,
            metrics.diagnostics.get("violation_labels", []),
        )

    @staticmethod
    def _constraint_vector(
        llm_utilization: dict[str, float],
        kv_stable: dict[str, bool],
        tool_utilization: dict[str, float],
        link_utilization: dict[str, float],
        violation_labels: list[str],
    ) -> np.ndarray:
        llm = sum(max(0.0, value - 1.0) for value in llm_utilization.values())
        kv = sum(float(not stable) for stable in kv_stable.values())
        tool = sum(max(0.0, value - 1.0) for value in tool_utilization.values())
        link = sum(max(0.0, value - 1.0) for value in link_utilization.values())
        llm += sum(label.startswith("llm_unserved:") for label in violation_labels)
        tool += sum(
            label.startswith("service_unserved:") for label in violation_labels
        )
        return np.asarray([llm, kv, tool, link], dtype=np.float32)

    def _app_latency_reference(self, app_id: str) -> float:
        app = self.scenario.applications[app_id]
        if app.slo.deadline_s is not None:
            return max(app.slo.deadline_s, 1.0e-9)
        ttft = app.slo.ttft_s or self.scenario.simulation.overload_delay_s
        tbt = app.slo.tbt_s or 0.0
        output_tokens = sum(
            flow.probability
            * float(np.mean(list(app.nodes[flow.final_node].output_tokens.values())))
            for flow in app.pattern_flows
        )
        return max(ttft + max(0.0, output_tokens - 1.0) * tbt, 1.0e-9)

    def _fixed_cost_bounds(self) -> tuple[float, float]:
        min_llm = min(
            self.scenario.llm_configs[candidate.config].running_cost_per_slot
            for candidate in self.scenario.candidates.values()
        )
        min_tools = sum(
            tool.running_cost_per_slot for tool in self.scenario.tools.values()
        )
        maximum = sum(
            self.scenario.llm_configs[candidate.config].running_cost_per_slot
            + self.scenario.llm_configs[candidate.config].load_cost
            for candidate in self.scenario.candidates.values()
        )
        maximum += sum(
            self.scenario.simulation.max_tool_replicas_per_server
            * (tool.running_cost_per_slot + tool.start_cost)
            for tool in self.scenario.tools.values()
            for _ in self.scenario.servers
        )
        return float(min_llm + min_tools), float(
            max(maximum, min_llm + min_tools + 1.0)
        )

    def _steady_cost(
        self,
        deployment: DeploymentDecision,
        link_load_mbps: dict[tuple[str, str], float],
    ) -> float:
        cost = sum(
            self.scenario.llm_configs[
                self.scenario.candidates[candidate_id].config
            ].running_cost_per_slot
            for candidate_id, active in deployment.llm_active.items()
            if active
        )
        cost += sum(
            replicas * self.scenario.tools[tool_id].running_cost_per_slot
            for (tool_id, _), replicas in deployment.tool_replicas.items()
        )
        return cost + self.simulator.backend.network.traffic_cost(link_load_mbps)

    def _uniform_model_share(self) -> dict[tuple[str, str, str], float]:
        probability = 1.0 / max(1, len(self.scenario.models))
        return {
            (app.id, ingress, model): probability
            for app in self.scenario.applications.values()
            for ingress in app.ingress_rates
            for model in self.scenario.models
        }

    @staticmethod
    def _empty_slot_components() -> dict[str, float]:
        return {
            "utility": 0.0,
            "cost_normalized": 0.0,
            "latency_normalized": 0.0,
            "goodput_normalized": 0.0,
            "quality_normalized": 0.0,
        }

    def _feature_vector(self) -> np.ndarray:
        features: list[float] = [
            self.simulator.slot / max(1, self.max_slots),
        ]
        features.extend(
            float(self.current_deployment.llm_active.get(candidate_id, 0))
            for candidate_id in self.layout.candidates
        )
        features.extend(
            self.current_deployment.tool_replicas.get((tool_id, server), 0)
            / self.scenario.simulation.max_tool_replicas_per_server
            for tool_id in self.layout.tools
            for server in self.layout.servers
        )
        features.extend(
            self.simulator.current_arrival_rates().get((app.id, ingress), rate)
            / self.arrival_scale
            for app in self.scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        )
        metrics = self.simulator.last_metrics
        components = self._last_slot_components
        features.extend(
            [
                components.get("cost_normalized", 0.0),
                components.get("latency_normalized", 0.0),
                metrics.slo_attainment if metrics else 0.0,
                metrics.quality if metrics else 0.0,
                *self._last_constraint_vector.tolist(),
            ]
        )
        features.extend(
            min(2.0, metrics.llm_utilization.get(candidate_id, 0.0))
            if metrics
            else 0.0
            for candidate_id in self.layout.candidates
        )
        features.extend(
            min(2.0, metrics.tool_utilization.get(f"{tool}@{server}", 0.0))
            if metrics
            else 0.0
            for tool in self.layout.tools
            for server in self.layout.servers
        )
        features.extend(
            min(
                2.0,
                metrics.link_utilization.get(
                    f"{link.source}->{link.target}", 0.0
                ),
            )
            if metrics
            else 0.0
            for link in self.scenario.links
        )
        remaining = self.planner.normalized_remaining_resources(
            self.current_deployment
        )
        for server in self.layout.servers:
            features.extend(remaining[server])

        plan = self._capacity_plan
        for tool in self.layout.tools:
            pending = sum(
                demand == ("tool", tool) for demand in self._deployment_queue
            )
            required = plan.tool_required[tool] if plan is not None else 0
            features.append(pending / max(1, required))
        for model in self.layout.models:
            required = (
                plan.model_required_capacity[model] if plan is not None else 0.0
            )
            accumulated = self._model_accumulated_capacity.get(model, 0.0)
            features.append(
                max(0.0, required - accumulated) / max(required, 1.0e-12)
                if required > 0.0
                else 0.0
            )

        current = self._current_demand
        features.extend(
            [float(current is not None and current[0] == "tool"),
             float(current is not None and current[0] == "llm")]
        )
        features.extend(
            float(current == ("tool", tool)) for tool in self.layout.tools
        )
        features.extend(
            float(current == ("llm", model)) for model in self.layout.models
        )
        features.extend(
            float(index == self._last_placement)
            for index in range(self.layout.deployment_action_size)
        )
        return np.clip(np.asarray(features, dtype=np.float32), -10.0, 10.0)

    def _observation(self) -> dict[str, Any]:
        masks = self.action_masks()
        return {
            "features": self._feature_vector(),
            "action_type": self.phase,
            **masks,
        }


def _normalized_or_uniform(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    positive = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(positive.values())
    if total <= 1.0e-12:
        uniform = 1.0 / len(positive)
        return {key: uniform for key in positive}
    return {key: value / total for key, value in positive.items()}


class RoutingOnlyEnv(AgentOrchestrationEnv):
    """Fixed capacity-planner deployment with PPO-controlled routing."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.phase = self.ROUTING
        return self._observation(), {"discount": self.gamma, "phase": "routing"}

    def _step_routing(self, action: dict[str, Any]):
        observation, reward, terminated, truncated, info = super()._step_routing(
            action
        )
        if not terminated and self.phase == self.DEPLOYMENT:
            self.current_deployment = self.planner.initial_deployment()
            self._base_deployment = self.current_deployment.copy()
            self.phase = self.ROUTING
            observation = self._observation()
        return observation, reward, terminated, truncated, info


class DeploymentOnlyEnv(AgentOrchestrationEnv):
    """Sequential PPO deployment with analytical routing over each period."""

    def _step_deployment(self, selected: int):
        observation, reward, terminated, truncated, info = super()._step_deployment(
            selected
        )
        if self.phase != self.ROUTING:
            return observation, reward, terminated, truncated, info

        interval_metrics = []
        utilities = []
        last_info = info
        while self.phase == self.ROUTING and self.simulator.slot < self.max_slots:
            greedy = GreedyPolicy(
                self.scenario, self._seed + self.simulator.slot
            ).routing(self.current_deployment, self.simulator.last_metrics)
            model = np.asarray(
                [
                    greedy.model_share.get((*group, model_id), 0.0)
                    for group in self.layout.model_groups
                    for model_id in self.layout.models
                ],
                dtype=np.float32,
            )
            observation, slot_reward, terminated, truncated, last_info = (
                super()._step_routing({"deploy": selected, "model": model})
            )
            utilities.append(slot_reward)
            interval_metrics.append(last_info["metrics"])
            if terminated:
                break

        info = {
            **last_info,
            "phase": "deployment_interval",
            "interval_metrics": interval_metrics,
            "evaluated_slots": len(interval_metrics),
            "reward_components": {
                "utility": float(np.mean(utilities)) if utilities else 0.0
            },
        }
        return observation, 0.0, terminated, truncated, info
