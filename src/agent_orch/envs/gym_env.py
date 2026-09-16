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
        targets = (("stop", ""),)
        targets += tuple(("add_candidate", item) for item in candidates)
        targets += tuple(("remove_candidate", item) for item in candidates)
        targets += tuple(("add_server", item) for item in servers)
        targets += tuple(("remove_server", item) for item in servers)
        return StructuredActionLayout(
            models, candidates, servers, tools, model_groups, targets
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

    def __init__(
        self,
        scenario: Scenario,
        max_slots: int = 600,
        potential_shaping: bool = True,
        seed: int = 0,
        arrival_trace: ArrivalTrace | None = None,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.scenario = scenario
        self.max_periods = max(1, int(max_slots))
        self.potential_shaping = potential_shaping
        self.layout = StructuredActionLayout.build(scenario)
        self.planner = CapacityPlanner(scenario)
        self.simulator = Simulator(scenario)
        self.simulator.set_arrival_trace(arrival_trace)
        self.physical_router = PhysicalRouter(scenario)
        self._seed = seed
        self.gamma = gamma
        self.deployment_gamma = 1.0
        self.phase = self.LLM_DEPLOYMENT
        self.cost_min, self.cost_max = self._fixed_cost_bounds()
        self.arrival_scale = max(
            1.0,
            sum(rate for app in scenario.applications.values() for rate in app.ingress_rates.values()),
        )
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.last_routing = GreedyPolicy(scenario, seed).routing(self.current_deployment)
        self._last_slot_components = self._empty_slot_components()
        self._last_constraint_vector = np.zeros(2, dtype=np.float32)
        self._planning_model_share = self._uniform_model_share()
        self._period_index = 0
        self._capacity_plan: CapacityPlan | None = None
        self._deployment_targets: list[tuple[str, str]] = []
        self._deployment_target_index = 0
        self._current_demand: tuple[str, str] | None = None
        self._touched_targets: set[tuple[str, str]] = set()
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
        action_kind, action_id = self.layout.deployment_targets[selected]
        target_kind, target_id = self._current_demand
        if action_kind == "stop":
            self._advance_deployment_target()
        elif target_kind == "llm" and action_kind in {"add_candidate", "remove_candidate"}:
            candidate = self.scenario.candidates[action_id]
            if candidate.model != target_id:
                raise ValueError("The selected candidate serves another model")
            self.current_deployment.llm_active[action_id] = int(action_kind == "add_candidate")
            self._touched_targets.add((action_kind, action_id))
            self._last_placement = selected
            self._deployment_actions_in_period += 1
        elif target_kind == "tool" and action_kind in {"add_server", "remove_server"}:
            key = (target_id, action_id)
            current = self.current_deployment.tool_replicas.get(key, 0)
            self.current_deployment.tool_replicas[key] = max(
                0, current + (1 if action_kind == "add_server" else -1)
            )
            self._touched_targets.add((action_kind, action_id))
            self._last_placement = selected
            self._deployment_actions_in_period += 1
        else:
            raise ValueError("The deployment action does not match the active target")
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
        if self.potential_shaping and action_kind != "stop":
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
        target_kind, target_id = self._current_demand
        for index, (action_kind, action_id) in enumerate(self.layout.deployment_targets):
            if action_kind == "stop":
                mask[index] = 1
                continue
            if (action_kind, action_id) in self._touched_targets:
                continue
            if target_kind == "llm" and action_kind in {"add_candidate", "remove_candidate"}:
                candidate = self.scenario.candidates[action_id]
                if candidate.model != target_id:
                    continue
                active = bool(self.current_deployment.llm_active.get(action_id, 0))
                if action_kind == "add_candidate":
                    mask[index] = int(not active and self.planner.feasible_activation(self.current_deployment, action_id))
                else:
                    active_count = sum(
                        self.current_deployment.llm_active.values()
                    )
                    mask[index] = int(active and active_count > 1)
            elif target_kind == "tool" and action_kind in {"add_server", "remove_server"}:
                key = (target_id, action_id)
                current = self.current_deployment.tool_replicas.get(key, 0)
                if action_kind == "add_server":
                    mask[index] = int(self.planner.feasible_tool_replica(self.current_deployment, target_id, action_id))
                else:
                    total = sum(
                        self.current_deployment.tool_replicas.get((target_id, server), 0)
                        for server in self.layout.servers
                    )
                    mask[index] = int(current > 0 and total > 1)
        return mask

    def _begin_deployment_cycle(self) -> None:
        self._base_deployment = self.current_deployment.copy()
        arrival_rates = self.simulator.current_arrival_rates()
        self._capacity_plan = self.planner.plan(
            arrival_rates, self._planning_model_share, self._period_index
        )
        model_targets = [("llm", model) for model in self.scenario.models]
        tool_targets = [("tool", tool) for tool in self.scenario.tools]
        self._deployment_targets = model_targets + tool_targets
        self._deployment_target_index = 0
        self._touched_targets = set()
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
        self._touched_targets = set()
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
        return self._normalized_utility(
            metrics.cost, metrics.mean_latency_s, metrics.slo_attainment,
            metrics.quality, metrics.app_latency_s, arrival_rates
        )

    def _normalized_utility(self, cost, mean_latency, attainment, quality, app_latency, arrival_rates):
        cost_normalized = float(np.clip((cost - self.cost_min) / max(self.cost_max - self.cost_min, 1.0e-12), 0.0, 1.0))
        total_rate = sum(max(0.0, rate) for rate in arrival_rates.values())
        if total_rate > 0.0 and app_latency:
            latency_normalized = sum(
                sum(max(0.0, arrival_rates.get((app.id, ingress), 0.0)) for ingress in app.ingress_rates)
                * min(1.0, app_latency.get(app.id, mean_latency) / self._app_latency_reference(app.id))
                for app in self.scenario.applications.values()
            ) / total_rate
        else:
            latency_normalized = min(1.0, mean_latency / max(float(np.mean([self._app_latency_reference(a) for a in self.scenario.applications])), 1.0e-12))
        components = {
            "cost_normalized": cost_normalized,
            "latency_normalized": float(latency_normalized),
            "goodput_normalized": float(np.clip(attainment, 0.0, 1.0)),
            "quality_normalized": float(np.clip(quality, 0.0, 1.0)),
        }
        weights = self.scenario.reward
        utility = (
            weights.goodput_weight * components["goodput_normalized"]
            + weights.quality_weight * components["quality_normalized"]
            - weights.cost_weight * cost_normalized
            - weights.latency_weight * latency_normalized
        )
        components["utility"] = float(utility)
        return float(utility), components

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

    def _fixed_cost_bounds(self):
        period_seconds = self.scenario.simulation.orchestration_period_s
        min_llm = min(
            self.scenario.llm_configs[c.config].running_cost_per_slot * period_seconds
            for c in self.scenario.candidates.values()
        )
        min_tools = sum(
            tool.running_cost_per_slot * period_seconds
            for tool in self.scenario.tools.values()
        )
        maximum = sum(
            self.scenario.llm_configs[c.config].running_cost_per_slot * period_seconds
            + self.scenario.llm_configs[c.config].load_cost
            for c in self.scenario.candidates.values()
        )
        maximum += sum(
            self.scenario.simulation.max_tool_replicas_per_server
            * (tool.running_cost_per_slot * period_seconds + tool.start_cost)
            for tool in self.scenario.tools.values() for _ in self.scenario.servers
        )
        return float(min_llm + min_tools), float(max(maximum, min_llm + min_tools + 1.0))

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
        return {"utility": 0.0, "cost_normalized": 0.0, "latency_normalized": 0.0, "goodput_normalized": 0.0, "quality_normalized": 0.0}

    def _phase_name(self) -> str:
        return {self.LLM_DEPLOYMENT: "llm_deployment", self.SERVICE_DEPLOYMENT: "service_deployment", self.COMPOSITION: "composition"}[self.phase]

    def _feature_vector(self) -> np.ndarray:
        features: list[float] = [self._period_index / max(1, self.max_periods)]
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
        features.extend([self._last_slot_components.get("cost_normalized", 0.0), self._last_slot_components.get("latency_normalized", 0.0), metrics.slo_attainment if metrics else 0.0, metrics.quality if metrics else 0.0, *self._last_constraint_vector.tolist()])
        features.extend(min(2.0, metrics.llm_utilization.get(cid, 0.0)) if metrics else 0.0 for cid in self.layout.candidates)
        features.extend(min(2.0, metrics.tool_utilization.get(f"{tool}@{server}", 0.0)) if metrics else 0.0 for tool in self.layout.tools for server in self.layout.servers)
        features.extend(min(2.0, metrics.link_utilization.get(f"{link.source}->{link.target}", 0.0)) if metrics else 0.0 for link in self.scenario.links)
        remaining = self.planner.normalized_remaining_resources(self.current_deployment)
        for server in self.layout.servers:
            features.extend(remaining[server])
        features.extend(float(self._current_demand == ("llm", model)) for model in self.layout.models)
        features.extend(float(self._current_demand == ("tool", tool)) for tool in self.layout.tools)
        features.extend([self._deployment_target_index / max(1, len(self._deployment_targets)), float(self.phase == self.COMPOSITION)])
        return np.clip(np.asarray(features, dtype=np.float32), -10.0, 10.0)

    def _observation(self):
        masks = self.action_masks()
        return {"features": self._feature_vector(), "action_type": self.phase, **masks}


class RoutingOnlyEnv(AgentOrchestrationEnv):
    """Fixed deployment with one composition/evaluation transition per period."""

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
        self._touched_targets = set()
        self._deployment_actions_in_period = 0
        self._period_initial_potential = self._deployment_potential()
        self._last_potential = self._period_initial_potential
        self.phase = self.COMPOSITION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, _ = super().reset(seed=seed, options=options)
        self.current_deployment = self.planner.initial_deployment()
        self._base_deployment = self.current_deployment.copy()
        self.phase = self.COMPOSITION
        return self._observation(), {"discount": self.gamma, "phase": "composition"}


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
