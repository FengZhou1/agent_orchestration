from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlanner
from agent_orch.performance.llm import service_demand
from agent_orch.performance.network import NetworkBackend
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
    model_groups: tuple[tuple[str, str], ...]
    llm_groups: tuple[tuple[str, str, str, str], ...]
    tool_groups: tuple[tuple[str, str, str, str], ...]
    deployment_groups: tuple[tuple[Any, ...], ...]
    deployment_widths: tuple[int, ...]

    @staticmethod
    def build(scenario: Scenario) -> "StructuredActionLayout":
        models = tuple(scenario.models)
        candidates = tuple(scenario.candidates)
        servers = tuple(scenario.servers)
        model_groups = tuple(
            (app.id, ingress)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
        )
        llm_groups = tuple(
            (app.id, ingress, node.id, model)
            for app in scenario.applications.values()
            for ingress in app.ingress_rates
            for node in app.nodes.values()
            if node.type is NodeType.LLM
            for model in models
        )
        tool_groups_list: list[tuple[str, str, str, str]] = []
        for app in scenario.applications.values():
            edges = {
                edge
                for flow in app.pattern_flows
                for edge in flow.edges
                if app.nodes[edge[1]].type is NodeType.TOOL
            }
            for source, target in sorted(edges):
                for source_server in servers:
                    tool_groups_list.append((app.id, source, target, source_server))
        deployment_groups: list[tuple[Any, ...]] = [
            ("llm", candidate_id) for candidate_id in candidates
        ]
        for tool_id in scenario.tools:
            for server in servers:
                deployment_groups.append(("tool", tool_id, server))
        deployment_widths = tuple(
            2 if group[0] == "llm" else scenario.simulation.max_tool_replicas_per_server + 1
            for group in deployment_groups
        )
        return StructuredActionLayout(
            models,
            candidates,
            servers,
            model_groups,
            llm_groups,
            tuple(tool_groups_list),
            tuple(deployment_groups),
            deployment_widths,
        )

    @property
    def deployment_action_size(self) -> int:
        return sum(self.deployment_widths)

    @property
    def model_action_size(self) -> int:
        return len(self.model_groups) * len(self.models)

    @property
    def llm_action_size(self) -> int:
        return len(self.llm_groups) * len(self.candidates)

    @property
    def tool_action_size(self) -> int:
        return len(self.tool_groups) * len(self.servers)


class AgentOrchestrationEnv(gym.Env):
    metadata = {"render_modes": []}

    DEPLOYMENT = 0
    ROUTING = 1

    def __init__(
        self,
        scenario: Scenario,
        max_slots: int = 600,
        potential_shaping: bool = False,
        seed: int = 0,
        arrival_trace: ArrivalTrace | None = None,
        llm_profile_backend: Any | None = None,
    ):
        super().__init__()
        self.scenario = scenario
        self.max_slots = max_slots
        self.potential_shaping = potential_shaping
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario, llm_profile_backend=llm_profile_backend)
        self.simulator.set_arrival_trace(arrival_trace)
        self.network = NetworkBackend(scenario.links)
        self._seed = seed
        self.phase = self.DEPLOYMENT
        self.current_deployment = self.planner.initial_deployment()
        self.last_routing = GreedyPolicy(scenario, seed).routing(self.current_deployment)
        self.cost_min, self.cost_max = self._fixed_cost_bounds()
        self.arrival_scale = max(
            1.0,
            sum(
                rate
                for app in self.scenario.applications.values()
                for rate in app.ingress_rates.values()
            ),
        )

        feature_size = self._feature_vector().size
        self.observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(-10.0, 10.0, (feature_size,), dtype=np.float32),
                "action_type": gym.spaces.Discrete(2),
                "deploy_mask": gym.spaces.MultiBinary(self.layout.deployment_action_size),
                "model_mask": gym.spaces.MultiBinary(self.layout.model_action_size),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "deploy": gym.spaces.MultiDiscrete(
                    np.asarray(self.layout.deployment_widths, dtype=np.int64)
                ),
                "model": gym.spaces.Box(
                    0.0, 1.0, (self.layout.model_action_size,), dtype=np.float32
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.simulator.reset(self._seed)
        self.current_deployment = self.planner.initial_deployment()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(
            self.current_deployment
        )
        self.phase = self.DEPLOYMENT
        return self._observation(), {"discount": 1.0, "phase": "deployment"}

    def step(self, action: dict[str, Any]):
        if self.phase == self.DEPLOYMENT:
            return self._step_deployment(np.asarray(action["deploy"], dtype=np.int64))
        return self._step_routing(action)

    def _step_deployment(self, selected: np.ndarray):
        before = self._potential(self.current_deployment) if self.potential_shaping else 0.0
        proposed = self.decode_deployment(selected)
        constraint_cost = self._deployment_constraint_cost(proposed)
        invalid = constraint_cost > 0.0
        if not invalid:
            self.current_deployment = proposed
        self.phase = self.ROUTING
        after = self._potential(self.current_deployment) if self.potential_shaping else 0.0
        reward = after - before
        info = {
            "discount": 1.0,
            "phase": "deployment",
            "invalid_action": invalid,
            "constraint_cost": constraint_cost,
            "constraint_steps": 1,
            "potential_before": before,
            "potential_after": after,
            "reward_components": {
                "utility": reward,
                "constraint_cost": constraint_cost,
            },
        }
        return self._observation(), reward, False, False, info

    def _step_routing(self, action: dict[str, Any]):
        arrival_rates = self.simulator.current_arrival_rates()
        routing = self.decode_routing(action)
        transition = self.simulator.step(self.current_deployment, routing)
        self.last_routing = routing
        metrics = transition.metrics
        reward, reward_components = self._slot_reward(metrics, arrival_rates)
        constraint_cost = self._slot_constraint_cost(metrics)
        terminated = self.simulator.slot >= self.max_slots
        if not terminated and self.simulator.slot % self.scenario.simulation.deployment_period_slots == 0:
            self.phase = self.DEPLOYMENT
        info = {
            "discount": 0.99,
            "phase": "routing",
            "metrics": metrics,
            "reward_components": reward_components,
            "constraint_cost": constraint_cost,
            "constraint_steps": 1,
        }
        return self._observation(), reward, terminated, False, info

    def decode_routing(self, action: dict[str, Any]) -> RoutingDecision:
        model_raw = np.asarray(action["model"], dtype=float).reshape(
            len(self.layout.model_groups), len(self.layout.models)
        )
        model_share: dict[tuple[str, str, str], float] = {}
        llm_share: dict[tuple[str, str, str, str], float] = {}
        tool_route: dict[tuple[str, str, str, str, str], float] = {}

        active_models = {
            candidate.model
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
            for candidate in [self.scenario.candidates[candidate_id]]
        }
        model_group_values: dict[tuple[str, str], dict[str, float]] = {}
        for group_index, group in enumerate(self.layout.model_groups):
            weights = {
                model: model_raw[group_index, index]
                for index, model in enumerate(self.layout.models)
                if model in active_models
            }
            normalized = _normalized_or_uniform(weights)
            model_group_values[group] = normalized
            for model in self.layout.models:
                model_share[(*group, model)] = normalized.get(model, 0.0)

        for group_index, (app_id, ingress, node_id, model) in enumerate(
            self.layout.llm_groups
        ):
            app = self.scenario.applications[app_id]
            node = app.nodes[node_id]
            candidates: dict[str, float] = {}
            for candidate_id in self.layout.candidates:
                candidate = self.scenario.candidates[candidate_id]
                if (
                    not self.current_deployment.llm_active.get(candidate_id, 0)
                    or candidate.model != model
                ):
                    continue
                config = self.scenario.llm_configs[candidate.config]
                demand = service_demand(
                    self.scenario.models[model],
                    config,
                    node.prompt_tokens[model],
                    node.output_tokens[model],
                    self.scenario.simulation.prefill_chunk_tokens,
                )
                utilization = (
                    self.simulator.last_metrics.llm_utilization.get(candidate_id, 0.0)
                    if self.simulator.last_metrics
                    else 0.0
                )
                propagation = sum(
                    self.network.links[edge].propagation_ms
                    for edge in self.network.path(ingress, candidate.server)
                ) / 1000.0
                response = demand.service_s / max(
                    0.05, 1.0 - min(utilization, 0.95)
                )
                candidates[candidate_id] = 1.0 / max(1e-9, propagation + response)
            conditional = _normalized_or_uniform(candidates)
            x = model_group_values[(app_id, ingress)].get(model, 0.0)
            for candidate_id, probability in conditional.items():
                llm_share[(app_id, ingress, node_id, candidate_id)] = x * probability

        for group_index, (app_id, source, target, source_server) in enumerate(
            self.layout.tool_groups
        ):
            tool_id = self.scenario.applications[app_id].nodes[target].tool or ""
            destinations: dict[str, float] = {}
            for server in self.layout.servers:
                replicas = self.current_deployment.tool_replicas.get((tool_id, server), 0)
                if replicas <= 0:
                    continue
                propagation = sum(
                    self.network.links[edge].propagation_ms
                    for edge in self.network.path(source_server, server)
                ) / 1000.0
                utilization = (
                    self.simulator.last_metrics.tool_utilization.get(
                        f"{tool_id}@{server}", 0.0
                    )
                    if self.simulator.last_metrics
                    else 0.0
                )
                process = 1.0 / (
                    replicas * self.scenario.tools[tool_id].service_rate[server]
                )
                response = process / max(0.05, 1.0 - min(utilization, 0.95))
                destinations[server] = 1.0 / max(1e-9, propagation + response)
            normalized = _normalized_or_uniform(destinations)
            for server, probability in normalized.items():
                tool_route[(app_id, source, target, source_server, server)] = probability
        return RoutingDecision(model_share, llm_share, tool_route)

    def decode_deployment(self, selected: np.ndarray) -> DeploymentDecision:
        if selected.shape != (len(self.layout.deployment_groups),):
            raise ValueError("Deployment action has an incompatible shape")
        deployment = DeploymentDecision(
            llm_active={candidate_id: 0 for candidate_id in self.scenario.candidates},
            tool_replicas={
                (tool_id, server): 0
                for tool_id in self.scenario.tools
                for server in self.scenario.servers
            },
        )
        for choice, group, width in zip(
            selected, self.layout.deployment_groups, self.layout.deployment_widths
        ):
            value = int(choice)
            if value < 0 or value >= width:
                raise ValueError("Deployment action contains an out-of-range choice")
            if group[0] == "llm":
                deployment.llm_active[group[1]] = value
            else:
                deployment.tool_replicas[(group[1], group[2])] = value
        return deployment

    def encode_deployment(self, deployment: DeploymentDecision) -> np.ndarray:
        choices: list[int] = []
        for group in self.layout.deployment_groups:
            if group[0] == "llm":
                choices.append(int(deployment.llm_active.get(group[1], 0)))
            else:
                choices.append(
                    int(deployment.tool_replicas.get((group[1], group[2]), 0))
                )
        return np.asarray(choices, dtype=np.int64)

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
        masks: list[int] = []
        empty = DeploymentDecision(
            llm_active={candidate_id: 0 for candidate_id in self.scenario.candidates},
            tool_replicas={
                (tool_id, server): 0
                for tool_id in self.scenario.tools
                for server in self.scenario.servers
            },
        )
        for group, width in zip(
            self.layout.deployment_groups, self.layout.deployment_widths
        ):
            group_mask = [1] * width
            if group[0] == "llm":
                group_mask[1] = int(self.planner.feasible_activation(empty, group[1]))
            else:
                tool = self.scenario.tools[group[1]]
                server = self.scenario.servers[group[2]]
                for replicas in range(width):
                    group_mask[replicas] = int(
                        replicas * tool.cpu_cores <= server.cpu_cores
                        and replicas * tool.memory_gb <= server.memory_gb
                    )
            masks.extend(group_mask)
        return np.asarray(masks, dtype=np.int8)

    def _deployment_constraint_cost(self, deployment: DeploymentDecision) -> float:
        missing = float(sum(deployment.llm_active.values()) == 0)
        required_tools = {
            node.tool
            for app in self.scenario.applications.values()
            for node in app.nodes.values()
            if node.type is NodeType.TOOL and node.tool is not None
        }
        for tool_id in required_tools:
            missing += float(
                sum(
                    replicas
                    for (deployed_tool, _), replicas in deployment.tool_replicas.items()
                    if deployed_tool == tool_id
                )
                == 0
            )
        return missing + self.planner.resource_excess(deployment)

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
                (cost - self.cost_min) / max(self.cost_max - self.cost_min, 1e-12),
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
                1.0, mean_latency / max(float(np.mean(references)), 1e-12)
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
        components = {
            "utility": float(utility),
            "cost_normalized": cost_normalized,
            "latency_normalized": float(latency_normalized),
            "goodput_normalized": goodput_normalized,
            "quality_normalized": quality_normalized,
        }
        return float(utility), components

    def _app_latency_reference(self, app_id: str) -> float:
        app = self.scenario.applications[app_id]
        if app.slo.deadline_s is not None:
            return max(app.slo.deadline_s, 1e-9)
        ttft = app.slo.ttft_s or self.scenario.simulation.overload_delay_s
        tbt = app.slo.tbt_s or 0.0
        output_tokens = sum(
            flow.probability
            * float(np.mean(list(app.nodes[flow.final_node].output_tokens.values())))
            for flow in app.pattern_flows
        )
        return max(ttft + max(0.0, output_tokens - 1.0) * tbt, 1e-9)

    def _slot_constraint_cost(self, metrics) -> float:
        excess = sum(max(0.0, value - 1.0) for value in metrics.llm_utilization.values())
        excess += sum(
            max(0.0, value - 1.0) for value in metrics.tool_utilization.values()
        )
        excess += sum(
            max(0.0, value - 1.0) for value in metrics.link_utilization.values()
        )
        excess += sum(
            float(not stable)
            for stable in metrics.diagnostics.get("kv_stable", {}).values()
        )
        excess += sum(
            float("unserved" in label)
            for label in metrics.diagnostics.get("violation_labels", [])
        )
        return float(excess)

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
        return float(min_llm + min_tools), float(max(maximum, min_llm + min_tools + 1.0))

    def _potential(self, deployment: DeploymentDecision) -> float:
        active_models = {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in deployment.llm_active.items()
            if active
        }
        required_tools = {
            node.tool
            for app in self.scenario.applications.values()
            for node in app.nodes.values()
            if node.type is NodeType.TOOL
        }
        available_tools = {
            tool_id
            for (tool_id, _), replicas in deployment.tool_replicas.items()
            if replicas > 0
        }
        missing = int(not active_models) + len(required_tools - available_tools)
        if missing:
            return -float(missing)
        projected = self._project_routing(deployment)
        arrival_rates = self.simulator.current_arrival_rates()
        analytical = self.simulator.backend.evaluate(
            deployment, projected, arrival_rates
        )
        workflow = self.simulator.workflow.evaluate(
            deployment, projected, analytical, arrival_rates
        )
        cost = self.simulator._cost(deployment, analytical.link_load_mbps)
        attainment = (
            workflow.goodput_rps / workflow.total_arrival_rps
            if workflow.total_arrival_rps > 0
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
        return utility

    def _project_routing(self, deployment: DeploymentDecision) -> RoutingDecision:
        action = {
            "model": np.ones(self.layout.model_action_size, dtype=np.float32),
        }
        previous = self.current_deployment
        self.current_deployment = deployment
        try:
            projected = self.decode_routing(action)
        finally:
            self.current_deployment = previous
        return projected

    def _feature_vector(self) -> np.ndarray:
        features: list[float] = [
            self.simulator.slot / max(1, self.max_slots),
            float(self.phase),
        ]
        features.extend(
            float(self.current_deployment.llm_active.get(candidate_id, 0))
            for candidate_id in self.layout.candidates
        )
        features.extend(
            self.current_deployment.tool_replicas.get((tool_id, server), 0)
            / self.scenario.simulation.max_tool_replicas_per_server
            for tool_id in self.scenario.tools
            for server in self.layout.servers
        )
        features.extend(
            self.simulator.current_arrival_rates().get((app.id, ingress), rate)
            / self.arrival_scale
            for app in self.scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        )
        metrics = self.simulator.last_metrics
        current_rates = self.simulator.current_arrival_rates()
        reward_components = (
            self._slot_reward(metrics, current_rates)[1] if metrics else {}
        )
        features.extend(
            [
                reward_components.get("cost_normalized", 0.0),
                reward_components.get("latency_normalized", 0.0),
                metrics.slo_attainment if metrics else 0.0,
                metrics.quality if metrics else 0.0,
                self._slot_constraint_cost(metrics) if metrics else 0.0,
            ]
        )
        features.extend(
            min(2.0, metrics.llm_utilization.get(candidate_id, 0.0)) if metrics else 0.0
            for candidate_id in self.layout.candidates
        )
        features.extend(
            min(2.0, metrics.tool_utilization.get(f"{tool}@{server}", 0.0))
            if metrics
            else 0.0
            for tool in self.scenario.tools
            for server in self.layout.servers
        )
        features.extend(
            min(2.0, metrics.link_utilization.get(f"{link.source}->{link.target}", 0.0))
            if metrics
            else 0.0
            for link in self.scenario.links
        )
        return np.asarray(features, dtype=np.float32)

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
    if total <= 1e-12:
        uniform = 1.0 / len(positive)
        return {key: uniform for key in positive}
    return {key: value / total for key, value in positive.items()}


class RoutingOnlyEnv(AgentOrchestrationEnv):
    """Fixed capacity-planner deployment with PPO-controlled routing every slot."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.phase = self.ROUTING
        info = {"discount": 0.99, "phase": "routing"}
        return self._observation(), info

    def _step_routing(self, action: dict[str, Any]):
        observation, reward, terminated, truncated, info = super()._step_routing(action)
        if not terminated:
            self.phase = self.ROUTING
            observation = self._observation()
        return observation, reward, terminated, truncated, info


class DeploymentOnlyEnv(AgentOrchestrationEnv):
    """PPO deployment with greedy routing evaluated over each deployment period."""

    def _step_deployment(self, selected: np.ndarray):
        observation, reward, terminated, truncated, info = super()._step_deployment(selected)
        if self.phase != self.ROUTING:
            return observation, reward, terminated, truncated, info

        interval_reward = 0.0
        interval_constraint = float(info.get("constraint_cost", 0.0))
        discount = 1.0
        evaluated_slots = 0
        interval_metrics = []
        while (
            evaluated_slots < self.scenario.simulation.deployment_period_slots
            and self.simulator.slot < self.max_slots
        ):
            arrival_rates = self.simulator.current_arrival_rates()
            routing = GreedyPolicy(
                self.scenario, self._seed + self.simulator.slot
            ).routing(self.current_deployment, self.simulator.last_metrics)
            transition = self.simulator.step(self.current_deployment, routing)
            self.last_routing = routing
            interval_metrics.append(transition.metrics)
            slot_reward, _ = self._slot_reward(transition.metrics, arrival_rates)
            interval_reward += discount * slot_reward
            interval_constraint += self._slot_constraint_cost(transition.metrics)
            discount *= 0.99
            evaluated_slots += 1

        reward += interval_reward
        terminated = self.simulator.slot >= self.max_slots
        if not terminated:
            self.phase = self.DEPLOYMENT
        info = {
            "discount": discount,
            "phase": "deployment_interval",
            "evaluated_slots": evaluated_slots,
            "interval_metrics": interval_metrics,
            "constraint_cost": interval_constraint,
            "constraint_steps": max(1, evaluated_slots),
            "reward_components": {
                "utility": reward,
                "constraint_cost": interval_constraint,
            },
        }
        return self._observation(), reward, terminated, False, info
