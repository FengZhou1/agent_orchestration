from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import DeploymentDecision, NodeType, RoutingDecision, Scenario
from agent_orch.simulator import Simulator


@dataclass(frozen=True)
class StructuredActionLayout:
    models: tuple[str, ...]
    candidates: tuple[str, ...]
    servers: tuple[str, ...]
    model_groups: tuple[tuple[str, str], ...]
    llm_groups: tuple[tuple[str, str, str, str], ...]
    tool_groups: tuple[tuple[str, str, str, str], ...]
    deployment_items: tuple[tuple[Any, ...], ...]

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
        deployment_items: list[tuple[Any, ...]] = [
            ("llm", candidate_id) for candidate_id in candidates
        ]
        for tool_id in scenario.tools:
            for server in servers:
                for slot in range(scenario.simulation.max_tool_replicas_per_server):
                    deployment_items.append(("tool", tool_id, server, slot))
        return StructuredActionLayout(
            models,
            candidates,
            servers,
            model_groups,
            llm_groups,
            tuple(tool_groups_list),
            tuple(deployment_items),
        )

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
    ):
        super().__init__()
        self.scenario = scenario
        self.max_slots = max_slots
        self.potential_shaping = potential_shaping
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario)
        self._seed = seed
        self.phase = self.DEPLOYMENT
        self.deploy_cursor = 0
        self.current_deployment = self.planner.initial_deployment()
        self.pending_deployment = self.current_deployment.copy()
        self.tool_slot_selected: dict[tuple[str, str, int], bool] = {}
        self.last_routing = GreedyPolicy(scenario, seed).routing(self.current_deployment)

        feature_size = self._feature_vector().size
        self.observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(-10.0, 10.0, (feature_size,), dtype=np.float32),
                "action_type": gym.spaces.Discrete(2),
                "deploy_mask": gym.spaces.MultiBinary(2),
                "model_mask": gym.spaces.MultiBinary(self.layout.model_action_size),
                "llm_mask": gym.spaces.MultiBinary(self.layout.llm_action_size),
                "tool_mask": gym.spaces.MultiBinary(self.layout.tool_action_size),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "deploy": gym.spaces.Discrete(2),
                "model": gym.spaces.Box(
                    0.0, 1.0, (self.layout.model_action_size,), dtype=np.float32
                ),
                "llm": gym.spaces.Box(
                    0.0, 1.0, (self.layout.llm_action_size,), dtype=np.float32
                ),
                "tool": gym.spaces.Box(
                    0.0, 1.0, (self.layout.tool_action_size,), dtype=np.float32
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.simulator.reset(self._seed)
        self.current_deployment = self.planner.initial_deployment()
        self.pending_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(self.scenario, self._seed).routing(
            self.current_deployment
        )
        self.phase = self.DEPLOYMENT
        self.deploy_cursor = 0
        self._initialize_tool_slots()
        return self._observation(), {"discount": 1.0, "phase": "deployment"}

    def step(self, action: dict[str, Any]):
        if self.phase == self.DEPLOYMENT:
            return self._step_deployment(int(action["deploy"]))
        return self._step_routing(action)

    def _step_deployment(self, selected: int):
        before = self._potential(self.pending_deployment) if self.potential_shaping else 0.0
        mask = self._deploy_mask()
        invalid = not bool(mask[selected])
        if not invalid:
            self._apply_deployment_item(selected)
        self.deploy_cursor += 1
        if self.deploy_cursor >= len(self.layout.deployment_items):
            self.current_deployment = self.pending_deployment.copy()
            self.phase = self.ROUTING
        after = self._potential(self.pending_deployment) if self.potential_shaping else 0.0
        if self.phase == self.ROUTING:
            after = 0.0
        reward = after - before - (10.0 if invalid else 0.0)
        info = {
            "discount": 1.0,
            "phase": "deployment",
            "invalid_action": invalid,
            "potential_before": before,
            "potential_after": after,
        }
        return self._observation(), reward, False, False, info

    def _step_routing(self, action: dict[str, Any]):
        routing = self.decode_routing(action)
        transition = self.simulator.step(self.current_deployment, routing)
        self.last_routing = routing
        metrics = transition.metrics
        reward = self._slot_reward(metrics)
        terminated = self.simulator.slot >= self.max_slots
        if not terminated and self.simulator.slot % self.scenario.simulation.deployment_period_slots == 0:
            self.phase = self.DEPLOYMENT
            self.deploy_cursor = 0
            self.pending_deployment = self.current_deployment.copy()
            self._initialize_tool_slots()
        info = {
            "discount": 0.99,
            "phase": "routing",
            "metrics": metrics,
            "reward_components": transition.reward_components,
        }
        return self._observation(), reward, terminated, False, info

    @staticmethod
    def _slot_reward(metrics) -> float:
        return 0.25 * (
            -metrics.cost / 10.0
            - metrics.mean_latency_s / 10.0
            + metrics.slo_attainment
            + metrics.quality
        ) - 10.0 * metrics.violations

    def decode_routing(self, action: dict[str, Any]) -> RoutingDecision:
        model_raw = np.asarray(action["model"], dtype=float).reshape(
            len(self.layout.model_groups), len(self.layout.models)
        )
        llm_raw = np.asarray(action["llm"], dtype=float).reshape(
            len(self.layout.llm_groups), len(self.layout.candidates)
        )
        tool_raw = np.asarray(action["tool"], dtype=float).reshape(
            len(self.layout.tool_groups), len(self.layout.servers)
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
            candidates = {
                candidate_id: llm_raw[group_index, index]
                for index, candidate_id in enumerate(self.layout.candidates)
                if self.current_deployment.llm_active.get(candidate_id, 0)
                and self.scenario.candidates[candidate_id].model == model
            }
            conditional = _normalized_or_uniform(candidates)
            x = model_group_values[(app_id, ingress)].get(model, 0.0)
            for candidate_id, probability in conditional.items():
                llm_share[(app_id, ingress, node_id, candidate_id)] = x * probability

        for group_index, (app_id, source, target, source_server) in enumerate(
            self.layout.tool_groups
        ):
            tool_id = self.scenario.applications[app_id].nodes[target].tool or ""
            destinations = {
                server: tool_raw[group_index, index]
                for index, server in enumerate(self.layout.servers)
                if self.current_deployment.tool_replicas.get((tool_id, server), 0) > 0
            }
            normalized = _normalized_or_uniform(destinations)
            for server, probability in normalized.items():
                tool_route[(app_id, source, target, source_server, server)] = probability
        return RoutingDecision(model_share, llm_share, tool_route)

    def action_masks(self) -> dict[str, np.ndarray]:
        model_mask = np.zeros(
            (len(self.layout.model_groups), len(self.layout.models)), dtype=np.int8
        )
        llm_mask = np.zeros(
            (len(self.layout.llm_groups), len(self.layout.candidates)), dtype=np.int8
        )
        tool_mask = np.zeros(
            (len(self.layout.tool_groups), len(self.layout.servers)), dtype=np.int8
        )
        active_models = {
            self.scenario.candidates[candidate_id].model
            for candidate_id, active in self.current_deployment.llm_active.items()
            if active
        }
        for group_index in range(len(self.layout.model_groups)):
            for model_index, model in enumerate(self.layout.models):
                model_mask[group_index, model_index] = int(model in active_models)
        for group_index, (_, _, _, model) in enumerate(self.layout.llm_groups):
            for candidate_index, candidate_id in enumerate(self.layout.candidates):
                candidate = self.scenario.candidates[candidate_id]
                llm_mask[group_index, candidate_index] = int(
                    candidate.model == model
                    and self.current_deployment.llm_active.get(candidate_id, 0) == 1
                )
        for group_index, (app_id, _, target, _) in enumerate(self.layout.tool_groups):
            tool_id = self.scenario.applications[app_id].nodes[target].tool or ""
            for server_index, server in enumerate(self.layout.servers):
                tool_mask[group_index, server_index] = int(
                    self.current_deployment.tool_replicas.get((tool_id, server), 0) > 0
                )
        return {
            "deploy_mask": self._deploy_mask(),
            "model_mask": model_mask.reshape(-1),
            "llm_mask": llm_mask.reshape(-1),
            "tool_mask": tool_mask.reshape(-1),
        }

    def _deploy_mask(self) -> np.ndarray:
        if self.phase != self.DEPLOYMENT:
            return np.ones(2, dtype=np.int8)
        item = self.layout.deployment_items[self.deploy_cursor]
        mask = np.ones(2, dtype=np.int8)
        if item[0] == "llm":
            candidate_id = item[1]
            active_count = sum(self.pending_deployment.llm_active.values())
            if self.pending_deployment.llm_active.get(candidate_id, 0) and active_count <= 1:
                mask[0] = 0
            reduced = self.pending_deployment.copy()
            reduced.llm_active[candidate_id] = 0
            mask[1] = int(self.planner.feasible_activation(reduced, candidate_id))
        else:
            tool_id, server, slot = item[1:]
            if self.tool_slot_selected.get((tool_id, server, slot), False):
                active_tool_slots = sum(
                    int(value)
                    for (h, _, _), value in self.tool_slot_selected.items()
                    if h == tool_id
                )
                if active_tool_slots <= 1:
                    mask[0] = 0
            selected = dict(self.tool_slot_selected)
            selected[(tool_id, server, slot)] = True
            candidate = self.pending_deployment.copy()
            candidate.tool_replicas[(tool_id, server)] = sum(
                int(value)
                for (h, n, _), value in selected.items()
                if h == tool_id and n == server
            )
            mask[1] = int(self.planner.deployment_feasible(candidate))
        return mask

    def _apply_deployment_item(self, selected: int) -> None:
        item = self.layout.deployment_items[self.deploy_cursor]
        if item[0] == "llm":
            self.pending_deployment.llm_active[item[1]] = selected
            return
        tool_id, server, slot = item[1:]
        self.tool_slot_selected[(tool_id, server, slot)] = bool(selected)
        self.pending_deployment.tool_replicas[(tool_id, server)] = sum(
            int(value)
            for (h, n, _), value in self.tool_slot_selected.items()
            if h == tool_id and n == server
        )

    def _initialize_tool_slots(self) -> None:
        self.tool_slot_selected = {}
        for tool_id in self.scenario.tools:
            for server in self.scenario.servers:
                replicas = self.pending_deployment.tool_replicas.get((tool_id, server), 0)
                for slot in range(self.scenario.simulation.max_tool_replicas_per_server):
                    self.tool_slot_selected[(tool_id, server, slot)] = slot < replicas

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
            return -10.0 * missing
        projected = self._project_routing(deployment)
        arrival_rates = self.simulator.current_arrival_rates()
        analytical = self.simulator.backend.evaluate(
            deployment, projected, arrival_rates
        )
        workflow = self.simulator.workflow.evaluate(
            deployment, projected, analytical, arrival_rates
        )
        cost = self.simulator._cost(deployment, analytical.link_loads_mbit)
        attainment = (
            workflow.goodput_rps / workflow.total_arrival_rps
            if workflow.total_arrival_rps > 0
            else 0.0
        )
        return 0.25 * (
            -cost / 10.0
            - workflow.mean_latency_s / 10.0
            + attainment
            + workflow.quality
        ) - 10.0 * len(set(analytical.violations))

    def _project_routing(self, deployment: DeploymentDecision) -> RoutingDecision:
        action = {
            "model": np.ones(self.layout.model_action_size, dtype=np.float32),
            "llm": np.ones(self.layout.llm_action_size, dtype=np.float32),
            "tool": np.ones(self.layout.tool_action_size, dtype=np.float32),
        }
        previous = self.current_deployment
        self.current_deployment = deployment
        try:
            projected = self.decode_routing(action)
        finally:
            self.current_deployment = previous
        return projected

    def _feature_vector(self) -> np.ndarray:
        visible_deployment = (
            self.pending_deployment if self.phase == self.DEPLOYMENT else self.current_deployment
        )
        features: list[float] = [
            self.simulator.slot / max(1, self.max_slots),
            float(self.phase),
            self.deploy_cursor / max(1, len(self.layout.deployment_items)),
        ]
        features.extend(
            float(visible_deployment.llm_active.get(candidate_id, 0))
            for candidate_id in self.layout.candidates
        )
        features.extend(
            visible_deployment.tool_replicas.get((tool_id, server), 0)
            / self.scenario.simulation.max_tool_replicas_per_server
            for tool_id in self.scenario.tools
            for server in self.layout.servers
        )
        features.extend(
            self.simulator.current_arrival_rates().get((app.id, ingress), rate) / 10.0
            for app in self.scenario.applications.values()
            for ingress, rate in app.ingress_rates.items()
        )
        metrics = self.simulator.last_metrics
        features.extend(
            [
                metrics.cost / 10.0 if metrics else 0.0,
                metrics.mean_latency_s / 10.0 if metrics else 0.0,
                metrics.slo_attainment if metrics else 0.0,
                metrics.quality if metrics else 0.0,
                metrics.violations / 10.0 if metrics else 0.0,
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

    def _step_deployment(self, selected: int):
        observation, reward, terminated, truncated, info = super()._step_deployment(selected)
        if self.phase != self.ROUTING:
            return observation, reward, terminated, truncated, info

        interval_reward = 0.0
        discount = 1.0
        evaluated_slots = 0
        interval_metrics = []
        while (
            evaluated_slots < self.scenario.simulation.deployment_period_slots
            and self.simulator.slot < self.max_slots
        ):
            routing = GreedyPolicy(self.scenario, self._seed + self.simulator.slot).routing(
                self.current_deployment
            )
            transition = self.simulator.step(self.current_deployment, routing)
            self.last_routing = routing
            interval_metrics.append(transition.metrics)
            interval_reward += discount * self._slot_reward(transition.metrics)
            discount *= 0.99
            evaluated_slots += 1

        reward += interval_reward
        terminated = self.simulator.slot >= self.max_slots
        if not terminated:
            self.phase = self.DEPLOYMENT
            self.deploy_cursor = 0
            self.pending_deployment = self.current_deployment.copy()
            self._initialize_tool_slots()
        info = {
            "discount": discount,
            "phase": "deployment_interval",
            "evaluated_slots": evaluated_slots,
            "interval_metrics": interval_metrics,
        }
        return self._observation(), reward, terminated, False, info
