"""One physical slot, many deployment steps, then many model-share steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from agent_orch.baselines import GreedyPolicy
from agent_orch.capacity import CapacityPlanner
from agent_orch.objective import ObjectiveEvaluator
from agent_orch.routing import PhysicalRouter
from agent_orch.schema.models import NodeType, Scenario
from agent_orch.simulator import Simulator
from agent_orch.workload.slot_trajectory import SlotTrajectory

from .base import BaseOrchestrationEnv


class SlotSequentialJointEnv(BaseOrchestrationEnv):
    """Joint PPO on a replayed slot trajectory.

    Decision substeps do not advance physical time.  The final model-group
    action runs exactly one simulation and advances the slot.  All episodes use
    identical exogenous values; only the policy actions change.
    """

    def __init__(
        self,
        scenario: Scenario,
        trajectory: SlotTrajectory,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if scenario.id != trajectory.scenario_id or not trajectory.slots:
            raise ValueError("A nonempty trajectory for this scenario is required")
        self.base_scenario = scenario
        self.trajectory = trajectory
        self._model_group = 0
        self._working_share = np.zeros(
            (sum(len(app.ingress_rates) for app in scenario.applications.values()),
             len(scenario.models)), dtype=np.float64
        )
        self._previous_round: dict[int, dict[str, float]] = {}
        self._completed_once: set[int] = set()
        self._baseline: dict[int, dict[str, float]] | None = None
        self.training_phase = "joint"
        self._stage_deployment = None
        self.delta_reward = True
        kwargs.pop("max_slots", None)
        kwargs.pop("arrival_trace", None)
        kwargs.pop("deployment_periods", None)
        kwargs.pop("potential_shaping", None)
        super().__init__(
            scenario, *args, max_slots=len(trajectory), arrival_trace=None,
            deployment_periods=1, potential_shaping=False, **kwargs,
        )
        # A shared categorical head represents a binary LLM switch or the
        # replica count of one (tool, server) pool, depending on the target.
        deployment_width = max(2, scenario.simulation.max_tool_replicas_per_server + 1)
        self.action_space.spaces["deploy"] = gym.spaces.Discrete(deployment_width)
        self.observation_space.spaces["deploy_mask"] = gym.spaces.MultiBinary(
            deployment_width
        )
        self.observation_space.spaces["model_group"] = gym.spaces.Discrete(
            len(self.layout.model_groups)
        )
        self.observation_space.spaces["deployment_features"] = gym.spaces.Box(
            -10.0, 10.0, (self._deployment_features().size,), dtype=np.float32
        )
        self.observation_space.spaces["routing_features"] = gym.spaces.Box(
            -10.0, 10.0, (self._routing_features().size,), dtype=np.float32
        )

    def reward_state(self) -> dict[str, Any]:
        return {
            "trajectory_digest": self.trajectory.digest(),
            "previous_round": self._previous_round,
            "completed_once": sorted(self._completed_once),
        }

    def baseline_report(self) -> dict[int, dict[str, float]]:
        if self._baseline is None:
            self._baseline = self._fixed_baseline()
        return self._baseline

    def restore_reward_state(self, state: dict[str, Any]) -> None:
        if state["trajectory_digest"] != self.trajectory.digest():
            raise ValueError("Checkpoint reward baseline belongs to a different trajectory")
        self._previous_round = {
            int(slot): dict(values)
            for slot, values in state["previous_round"].items()
        }
        self._completed_once = set(int(slot) for slot in state["completed_once"])

    def _activate_slot(self) -> None:
        slot_scenario = self.trajectory.scenario_at(self._period_index, self.base_scenario)
        previous = self.simulator.previous_deployment
        last_metrics = self.simulator.last_metrics
        samples = self.simulator.workflow.max_mapping_samples
        self.simulator = Simulator(slot_scenario, max_mapping_samples=samples)
        self.simulator.slot = self._period_index
        self.simulator.previous_deployment = previous
        self.simulator.last_metrics = last_metrics
        self.scenario = slot_scenario
        self.planner = CapacityPlanner(slot_scenario)
        self.physical_router = PhysicalRouter(slot_scenario)
        self.objective = ObjectiveEvaluator(
            slot_scenario, self.objective.spec, self.objective.references
        )
        # The policy constructs the complete placement from an empty slate each
        # slot.  Simulator.previous_deployment still tracks actual switch-on cost.
        self.current_deployment = self.planner.empty_deployment()
        self._stage_deployment = (
            self._sample_stage_deployment()
            if self.training_phase == "composition" else None
        )
        self._model_group = 0
        self._working_share.fill(0.0)

    def _begin_deployment_cycle(self) -> None:
        self._activate_slot()
        super()._begin_deployment_cycle()
        # One binary decision per LLM candidate and one replica-count decision
        # per stateless-service pool. No substep advances physical time.
        self._deployment_targets = [
            ("llm", candidate_id, "") for candidate_id in self.planner.candidate_order()
        ] + [
            ("tool", tool, server)
            for tool in self.layout.tools
            for server in self.layout.servers
        ]
        self._deployment_target_index = 0
        self._current_demand = None
        self.phase = self.LLM_DEPLOYMENT
        self._advance_deployment_target()

    def _fixed_baseline(self) -> dict[int, dict[str, float]]:
        """Evaluate one deterministic greedy policy on every sampled slot."""

        result: dict[int, dict[str, float]] = {}
        previous = None
        last_metrics = None
        for slot in range(len(self.trajectory)):
            scenario = self.trajectory.scenario_at(slot, self.base_scenario)
            policy = GreedyPolicy(scenario, seed=0)
            deployment = policy.deployment()
            simulator = Simulator(
                scenario, max_mapping_samples=self.simulator.workflow.max_mapping_samples
            )
            simulator.slot = slot
            if previous is not None:
                simulator.previous_deployment = previous
            simulator.last_metrics = last_metrics
            routing = policy.routing(deployment, last_metrics)
            metrics = simulator.step(deployment, routing).metrics
            value = ObjectiveEvaluator(
                scenario, self.objective.spec, self.objective.references
            ).evaluate(metrics, simulator.current_arrival_rates())
            result[slot] = {
                **{key: float(value.components[key]) for key in _COMPONENTS},
                "utility": float(value.utility),
                "cost_raw": float(metrics.cost),
                "latency_raw": float(metrics.mean_latency_s),
                "goodput_raw": float(metrics.goodput_rps),
                "quality_raw": float(metrics.quality),
                "slo_attainment_raw": float(metrics.slo_attainment),
                **{
                    f"constraint_{name}": float(amount)
                    for name, amount in zip(self.constraint_names, value.constraints)
                },
            }
            previous = deployment.copy()
            last_metrics = metrics
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # The baseline is computed once and retained when PPO resets episodes.
        if self._baseline is None:
            self._baseline = self._fixed_baseline()
        self.scenario = self.base_scenario
        self.planner = CapacityPlanner(self.base_scenario)
        self.simulator = Simulator(
            self.base_scenario,
            max_mapping_samples=self.simulator.workflow.max_mapping_samples,
        )
        observation, info = super().reset(seed=seed, options=options)
        self.trace_offset = 0
        info["trajectory_digest"] = self.trajectory.digest()
        return observation, info

    def _draw_trace_offset(self, seed: int | None) -> int:
        return 0

    def step(self, action: dict[str, Any]):
        if self.phase != self.COMPOSITION:
            if self._stage_deployment is not None and self._current_demand is not None:
                kind, target, server = self._current_demand
                desired = (
                    int(self._stage_deployment.llm_active.get(target, 0))
                    if kind == "llm" else
                    int(self._stage_deployment.tool_replicas.get((target, server), 0))
                )
                if not self.action_masks()["deploy_mask"][desired]:
                    raise RuntimeError("Fixed routing-stage placement is not feasible")
                action = {**action, "deploy": desired}
            return super().step(action)
        raw = np.asarray(action["model"], dtype=float).reshape(self._working_share.shape)
        group = self._model_group
        row = np.maximum(0.0, raw[group])
        active = self.action_masks()["model_mask"].reshape(self._working_share.shape)[group]
        row = np.where(active, row, 0.0)
        total = float(row.sum())
        if total <= 0.0:
            row = active.astype(float) / max(1, int(active.sum()))
        else:
            row /= total
        self._working_share[group] = row
        if group + 1 < len(self.layout.model_groups):
            self._model_group += 1
            return self._observation(), 0.0, False, False, {
                "discount": 1.0,
                "phase": "composition",
                "period_complete": False,
                "model_group": group,
                "constraint_vector": [0.0] * self.constraint_count,
                "constraint_steps": 0,
                "reward_components": {},
            }
        slot = self._period_index
        observation, _, terminated, truncated, info = self._step_composition(
            {"deploy": 0, "model": self._working_share.reshape(-1).copy()}
        )
        current = {
            **{key: float(info["reward_components"][key]) for key in _COMPONENTS},
            "utility": float(info["utility"]),
            "cost_raw": float(info["metrics"].cost),
            "latency_raw": float(info["metrics"].mean_latency_s),
            "goodput_raw": float(info["metrics"].goodput_rps),
            "quality_raw": float(info["metrics"].quality),
            "slo_attainment_raw": float(info["metrics"].slo_attainment),
            **{
                f"constraint_{name}": float(amount)
                for name, amount in zip(self.constraint_names, info["constraint_vector"])
            },
        }
        assert self._baseline is not None
        previous = self._previous_round.get(slot, self._baseline[slot])
        delta = {key: current[key] - previous[key] for key in current}
        self._previous_round[slot] = current
        info["reward_components"] = {**current, **{f"delta_{k}": v for k, v in delta.items()}}
        info["learning_utility"] = delta["utility"]
        info["constraint_delta_vector"] = [
            delta[f"constraint_{name}"] for name in self.constraint_names
        ]
        info["reward_baseline"] = "previous_round" if slot in self._completed_once else "fixed_greedy"
        info["physical_slot"] = slot
        info["trajectory_digest"] = self.trajectory.digest()
        self._completed_once.add(slot)
        return observation, delta["utility"], terminated, truncated, info

    def _observation(self):
        observation = super()._observation()
        observation["model_group"] = self._model_group
        observation["deployment_features"] = self._deployment_features()
        observation["routing_features"] = self._routing_features()
        return observation

    def _sample_stage_deployment(self):
        """A feasible placement fixed per slot across routing warm-up episodes."""
        placement = self.planner.initial_deployment()
        rng = np.random.default_rng(
            self.trajectory.seed + 7919 * self._period_index
        )
        for candidate in rng.permutation(self.planner.candidate_order()):
            if not placement.llm_active.get(candidate, 0) and rng.random() < 0.25:
                if self.planner.feasible_activation(placement, candidate):
                    placement.llm_active[candidate] = 1
        pools = [(tool, server) for tool in self.layout.tools for server in self.layout.servers]
        for tool, server in rng.permutation(pools):
            if rng.random() < 0.20 and self.planner.feasible_tool_replica(placement, tool, server):
                placement.tool_replicas[(tool, server)] = placement.tool_replicas.get((tool, server), 0) + 1
        return placement

    def _deploy_mask(self) -> np.ndarray:
        width = max(2, self.scenario.simulation.max_tool_replicas_per_server + 1)
        mask = np.zeros(width, dtype=np.int8)
        if self._current_demand is None:
            return mask
        kind, target, server = self._current_demand
        mask[0] = 1
        if kind == "llm":
            mask[1] = int(self.planner.feasible_activation(self.current_deployment, target))
            return mask
        host = self.scenario.servers[server]
        used_cpu = sum(
            self.scenario.tools[tool].cpu_cores * replicas
            for (tool, host_id), replicas in self.current_deployment.tool_replicas.items()
            if host_id == server
        )
        used_memory = sum(
            self.scenario.tools[tool].memory_gb * replicas
            for (tool, host_id), replicas in self.current_deployment.tool_replicas.items()
            if host_id == server
        )
        current = int(self.current_deployment.tool_replicas.get((target, server), 0))
        tool = self.scenario.tools[target]
        for count in range(1, width):
            if count > self.scenario.simulation.max_tool_replicas_per_server:
                break
            cpu = used_cpu + (count - current) * tool.cpu_cores
            memory = used_memory + (count - current) * tool.memory_gb
            mask[count] = int(
                cpu <= host.cpu_cores + 1e-9
                and memory <= host.memory_gb + 1e-9
            )
        return mask

    def _step_deployment(self, selected: int):
        if self._current_demand is None:
            raise RuntimeError("No deployment target is active")
        mask = self._deploy_mask()
        if selected < 0 or selected >= len(mask) or not mask[selected]:
            raise ValueError("The sequential deployment action is masked or invalid")
        kind, target, server = self._current_demand
        if kind == "llm":
            self.current_deployment.llm_active[target] = selected
        else:
            self.current_deployment.tool_replicas[(target, server)] = selected
        self._last_placement = selected
        self._deployment_actions_in_period += 1
        self._advance_deployment_target()
        return self._observation(), 0.0, False, False, {
            "discount": 1.0,
            "phase": self._phase_name(),
            "deployment_complete": self.phase == self.COMPOSITION,
            "period_complete": False,
            "constraint_cost": 0.0,
            "constraint_vector": [0.0] * self.constraint_count,
            "constraint_steps": 0,
            "reward_components": {"utility": 0.0},
        }

    def _deployment_features(self) -> np.ndarray:
        previous = self.simulator.previous_deployment
        extras = [
            float(previous.llm_active.get(candidate, 0)) if previous else 0.0
            for candidate in self.layout.candidates
        ]
        extras.extend(
            previous.tool_replicas.get((tool, server), 0)
            / max(1, self.scenario.simulation.max_tool_replicas_per_server)
            if previous else 0.0
            for tool in self.layout.tools for server in self.layout.servers
        )
        return np.concatenate((self._feature_vector(), np.asarray(extras, dtype=np.float32)))

    def _routing_features(self) -> np.ndarray:
        app_id, ingress = self.layout.model_groups[self._model_group]
        app = self.scenario.applications[app_id]
        slo = app.slo
        llm_nodes = [node for node in app.nodes.values() if node.type is NodeType.LLM]
        rate = app.ingress_rates[ingress]
        total_rate = sum(sum(item.ingress_rates.values()) for item in self.scenario.applications.values())
        local = [rate / max(total_rate, 1e-9), float(slo.type == "lat"),
                 float(slo.type == "ddl"), float(slo.type == "cmp"),
                 (slo.ttft_s or 0.0) / 10.0, (slo.tbt_s or 0.0) / 0.05,
                 (slo.deadline_s or 0.0) / 20.0]
        local.extend(flow.probability for flow in app.pattern_flows)
        # A fixed-width summary of flow probabilities remains valid when apps
        # have different numbers of pattern flows.
        local = local[:7] + [sum(flow.probability ** 2 for flow in app.pattern_flows)]
        for model in self.layout.models:
            active = [cid for cid, enabled in self.current_deployment.llm_active.items()
                      if enabled and self.scenario.candidates[cid].model == model]
            local.extend((
                app.quality[model],
                sum(node.prompt_tokens[model] for node in llm_nodes) / 4096.0,
                sum(node.output_tokens[model] for node in llm_nodes) / 1024.0,
                len(active) / max(1, len(self.layout.candidates)),
            ))
        return np.clip(np.concatenate((self._feature_vector(), np.asarray(local, dtype=np.float32))), -10.0, 10.0)

    def _feature_vector(self) -> np.ndarray:
        base = super()._feature_vector()
        scenario = self.scenario
        original = self.base_scenario
        features: list[float] = []
        for app_id, app in scenario.applications.items():
            features.extend(flow.probability for flow in app.pattern_flows)
            for node_id, node in app.nodes.items():
                if node.prompt_tokens:
                    reference = original.applications[app_id].nodes[node_id]
                    for model in self.layout.models:
                        features.append(node.prompt_tokens[model] / reference.prompt_tokens[model])
                        features.append(node.output_tokens[model] / reference.output_tokens[model])
        features.extend(
            link.capacity_mbps / original.links[index].capacity_mbps
            for index, link in enumerate(scenario.links)
        )
        for server_id, server in scenario.servers.items():
            reference = original.servers[server_id]
            features.extend((
                server.cpu_cores / reference.cpu_cores,
                server.memory_gb / reference.memory_gb,
                server.gpu_count / max(1, reference.gpu_count),
            ))
        groups = len(self.layout.model_groups)
        width = len(self.layout.models)
        working = getattr(self, "_working_share", None)
        if working is None or working.shape != (groups, width):
            working = np.zeros((groups, width))
        features.append(self._model_group / max(1, groups))
        features.extend(working.reshape(-1).tolist())
        return np.clip(
            np.concatenate((base, np.asarray(features, dtype=np.float32))), -10.0, 10.0
        )


_COMPONENTS = (
    "cost_normalized", "latency_normalized",
    "goodput_normalized", "quality_normalized",
)
