from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping

from agent_orch.performance.analytical import evaluate_llm_instance
from agent_orch.performance.queueing import tool_response_time
from agent_orch.schema.models import DeploymentDecision, NodeType, Scenario


@dataclass(frozen=True)
class CapacityPlanningConfig:
    tool_target_utilization: float = 0.75
    llm_target_utilization: float = 0.80
    tool_delay_weight: float = 0.5
    tool_cost_weight: float = 0.5
    initial_prior_weight: float = 0.20


@dataclass(frozen=True)
class CapacityPlan:
    tool_required: dict[str, int]
    model_required_capacity: dict[str, float]
    candidate_capacity: dict[str, float]
    tool_arrival: dict[str, float]
    model_arrival: dict[str, float]
    planning_model_share: dict[tuple[str, str, str], float]
    service_order: tuple[tuple[str, str], ...]


class CapacityPlanner:
    """Agent-aware capacity planning and physical deployment feasibility."""

    def __init__(
        self,
        scenario: Scenario,
        config: CapacityPlanningConfig = CapacityPlanningConfig(),
    ):
        self.scenario = scenario
        self.config = config

    def plan(
        self,
        arrival_rates: Mapping[tuple[str, str], float],
        previous_model_share: Mapping[tuple[str, str, str], float] | None = None,
        period_index: int = 0,
    ) -> CapacityPlan:
        planning_share = self._planning_model_share(
            previous_model_share, period_index
        )
        tool_arrival = self._tool_arrivals(arrival_rates)
        model_arrival = self._model_arrivals(arrival_rates, planning_share)
        tool_required = {
            tool_id: self._required_tool_replicas(tool_id, rate)
            for tool_id, rate in tool_arrival.items()
        }
        candidate_capacity = self._candidate_capacities(
            arrival_rates, planning_share, model_arrival
        )
        model_required = {
            model: rate / self.config.llm_target_utilization
            for model, rate in model_arrival.items()
        }
        pressures: list[tuple[float, str, str]] = []
        for tool_id, rate in tool_arrival.items():
            reference_rate = self._reference_tool_rate(tool_id)
            pressures.append((rate / max(reference_rate, 1.0e-12), "tool", tool_id))
        for model, rate in model_arrival.items():
            capacities = [
                capacity
                for candidate_id, capacity in candidate_capacity.items()
                if self.scenario.candidates[candidate_id].model == model
            ]
            reference = sum(capacities) / len(capacities) if capacities else 0.0
            pressures.append((rate / max(reference, 1.0e-12), "llm", model))
        service_order = tuple(
            (kind, service)
            for _, kind, service in sorted(
                pressures, key=lambda item: (-item[0], item[1], item[2])
            )
        )
        return CapacityPlan(
            tool_required=tool_required,
            model_required_capacity=model_required,
            candidate_capacity=candidate_capacity,
            tool_arrival=tool_arrival,
            model_arrival=model_arrival,
            planning_model_share=planning_share,
            service_order=service_order,
        )

    def empty_deployment(self) -> DeploymentDecision:
        return DeploymentDecision(
            llm_active={candidate_id: 0 for candidate_id in self.scenario.candidates},
            tool_replicas={
                (tool_id, server_id): 0
                for tool_id in self.scenario.tools
                for server_id in self.scenario.servers
            },
        )

    def candidate_order(self) -> tuple[str, ...]:
        return tuple(
            candidate.id
            for candidate in sorted(
                self.scenario.candidates.values(),
                key=lambda item: (item.model, item.server, item.config, item.id),
            )
        )

    def feasible_activation(
        self, deployment: DeploymentDecision, candidate_id: str
    ) -> bool:
        candidate = self.scenario.candidates[candidate_id]
        config = self.scenario.llm_configs[candidate.config]
        server = self.scenario.servers[candidate.server]
        gpu_used = memory_used = 0.0
        for active_id, active in deployment.llm_active.items():
            if not active:
                continue
            other = self.scenario.candidates[active_id]
            if other.server != candidate.server:
                continue
            other_config = self.scenario.llm_configs[other.config]
            gpu_used += other_config.gpu_count * other_config.gpu_share
            memory_used += (
                other_config.gpu_count * other_config.reserved_memory_gb_per_gpu
            )
        gpu_used += config.gpu_count * config.gpu_share
        memory_used += config.gpu_count * config.reserved_memory_gb_per_gpu
        return (
            gpu_used <= server.gpu_count + 1e-9
            and memory_used <= server.gpu_count * server.gpu_memory_gb + 1e-9
        )

    def feasible_tool_replica(
        self, deployment: DeploymentDecision, tool_id: str, server_id: str
    ) -> bool:
        current = deployment.tool_replicas.get((tool_id, server_id), 0)
        if current >= self.scenario.simulation.max_tool_replicas_per_server:
            return False
        proposed = deployment.copy()
        proposed.tool_replicas[(tool_id, server_id)] = current + 1
        return self.deployment_feasible(proposed)

    def normalized_remaining_resources(
        self, deployment: DeploymentDecision
    ) -> dict[str, tuple[float, float, float, float]]:
        gpu: dict[str, float] = defaultdict(float)
        gpu_memory: dict[str, float] = defaultdict(float)
        cpu: dict[str, float] = defaultdict(float)
        memory: dict[str, float] = defaultdict(float)
        for candidate_id, active in deployment.llm_active.items():
            if not active:
                continue
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            gpu[candidate.server] += config.gpu_count * config.gpu_share
            gpu_memory[candidate.server] += (
                config.gpu_count * config.reserved_memory_gb_per_gpu
            )
        for (tool_id, server_id), replicas in deployment.tool_replicas.items():
            tool = self.scenario.tools[tool_id]
            cpu[server_id] += replicas * tool.cpu_cores
            memory[server_id] += replicas * tool.memory_gb
        result = {}
        for server_id, server in self.scenario.servers.items():
            result[server_id] = (
                max(0.0, 1.0 - cpu[server_id] / max(server.cpu_cores, 1.0e-12)),
                max(0.0, 1.0 - memory[server_id] / max(server.memory_gb, 1.0e-12)),
                max(0.0, 1.0 - gpu[server_id] / max(server.gpu_count, 1.0e-12)),
                max(
                    0.0,
                    1.0
                    - gpu_memory[server_id]
                    / max(server.gpu_count * server.gpu_memory_gb, 1.0e-12),
                ),
            )
        return result

    def deployment_feasible(self, deployment: DeploymentDecision) -> bool:
        for candidate_id, active in deployment.llm_active.items():
            if active:
                reduced = deployment.copy()
                reduced.llm_active[candidate_id] = 0
                if not self.feasible_activation(reduced, candidate_id):
                    return False
        cpu: dict[str, float] = defaultdict(float)
        memory: dict[str, float] = defaultdict(float)
        for (tool_id, server_id), replicas in deployment.tool_replicas.items():
            tool = self.scenario.tools[tool_id]
            cpu[server_id] += replicas * tool.cpu_cores
            memory[server_id] += replicas * tool.memory_gb
        return all(
            cpu[server_id] <= server.cpu_cores + 1e-9
            and memory[server_id] <= server.memory_gb + 1e-9
            for server_id, server in self.scenario.servers.items()
        )

    def resource_excess(self, deployment: DeploymentDecision) -> float:
        """Return normalized physical-resource excess for a complete deployment."""
        gpu: dict[str, float] = defaultdict(float)
        gpu_memory: dict[str, float] = defaultdict(float)
        cpu: dict[str, float] = defaultdict(float)
        memory: dict[str, float] = defaultdict(float)
        for candidate_id, active in deployment.llm_active.items():
            if not active:
                continue
            candidate = self.scenario.candidates[candidate_id]
            config = self.scenario.llm_configs[candidate.config]
            gpu[candidate.server] += config.gpu_count * config.gpu_share
            gpu_memory[candidate.server] += (
                config.gpu_count * config.reserved_memory_gb_per_gpu
            )
        for (tool_id, server_id), replicas in deployment.tool_replicas.items():
            tool = self.scenario.tools[tool_id]
            cpu[server_id] += replicas * tool.cpu_cores
            memory[server_id] += replicas * tool.memory_gb

        excess = 0.0
        for server_id, server in self.scenario.servers.items():
            capacities = (
                (gpu[server_id], float(server.gpu_count)),
                (
                    gpu_memory[server_id],
                    float(server.gpu_count) * server.gpu_memory_gb,
                ),
                (cpu[server_id], float(server.cpu_cores)),
                (memory[server_id], server.memory_gb),
            )
            for demand, capacity in capacities:
                excess += max(0.0, demand / max(capacity, 1e-12) - 1.0)
        return excess

    def initial_deployment(self) -> DeploymentDecision:
        deployment = self.empty_deployment()
        for model in self.scenario.models:
            candidates = [
                candidate
                for candidate in self.scenario.candidates.values()
                if candidate.model == model
            ]
            candidates.sort(
                key=lambda item: self.scenario.llm_configs[item.config].running_cost_per_slot
            )
            for candidate in candidates:
                if self.feasible_activation(deployment, candidate.id):
                    deployment.llm_active[candidate.id] = 1
                    break
        for tool_id in self.scenario.tools:
            best_server = max(
                self.scenario.servers,
                key=lambda server_id: self.scenario.tools[tool_id].service_rate[server_id],
            )
            deployment.tool_replicas[(tool_id, best_server)] = 1
        if not self.deployment_feasible(deployment):
            raise ValueError("The capacity planner could not construct a feasible deployment")
        return deployment

    def _planning_model_share(
        self,
        previous: Mapping[tuple[str, str, str], float] | None,
        period_index: int,
    ) -> dict[tuple[str, str, str], float]:
        prior_weight = self.config.initial_prior_weight / math.sqrt(period_index + 1.0)
        uniform = 1.0 / max(1, len(self.scenario.models))
        result: dict[tuple[str, str, str], float] = {}
        for app in self.scenario.applications.values():
            for ingress in app.ingress_rates:
                historical = {
                    model: max(
                        0.0,
                        float((previous or {}).get((app.id, ingress, model), uniform)),
                    )
                    for model in self.scenario.models
                }
                total = sum(historical.values())
                if total <= 0.0:
                    historical = {model: uniform for model in self.scenario.models}
                else:
                    historical = {
                        model: value / total for model, value in historical.items()
                    }
                for model in self.scenario.models:
                    result[(app.id, ingress, model)] = (
                        (1.0 - prior_weight) * historical[model]
                        + prior_weight * uniform
                    )
        return result

    def _tool_arrivals(
        self, arrival_rates: Mapping[tuple[str, str], float]
    ) -> dict[str, float]:
        arrivals = {tool_id: 0.0 for tool_id in self.scenario.tools}
        for app in self.scenario.applications.values():
            external_rate = sum(
                max(0.0, float(arrival_rates.get((app.id, ingress), base_rate)))
                for ingress, base_rate in app.ingress_rates.items()
            )
            for node in app.nodes.values():
                if node.type is NodeType.TOOL and node.tool is not None:
                    arrivals[node.tool] += external_rate * app.visit_probability(node.id)
        return arrivals

    def _model_arrivals(
        self,
        arrival_rates: Mapping[tuple[str, str], float],
        model_share: Mapping[tuple[str, str, str], float],
    ) -> dict[str, float]:
        arrivals = {model: 0.0 for model in self.scenario.models}
        for app in self.scenario.applications.values():
            llm_visits = sum(
                app.visit_probability(node.id)
                for node in app.nodes.values()
                if node.type is NodeType.LLM
            )
            for ingress, base_rate in app.ingress_rates.items():
                rate = max(
                    0.0, float(arrival_rates.get((app.id, ingress), base_rate))
                )
                for model in self.scenario.models:
                    arrivals[model] += (
                        rate
                        * model_share[(app.id, ingress, model)]
                        * llm_visits
                    )
        return arrivals

    def _required_tool_replicas(self, tool_id: str, arrival_rate: float) -> int:
        service_rate = self._reference_tool_rate(tool_id)
        initial = max(
            1,
            math.ceil(
                arrival_rate
                / max(
                    self.config.tool_target_utilization * service_rate, 1.0e-12
                )
            ),
        )
        maximum = (
            len(self.scenario.servers)
            * self.scenario.simulation.max_tool_replicas_per_server
        )
        replicas = min(initial, maximum)
        while replicas < maximum:
            current = self._tool_objective(tool_id, arrival_rate, replicas)
            candidate = self._tool_objective(tool_id, arrival_rate, replicas + 1)
            if candidate >= current:
                break
            replicas += 1
        return replicas

    def _tool_objective(
        self, tool_id: str, arrival_rate: float, replicas: int
    ) -> float:
        tool = self.scenario.tools[tool_id]
        service_rate = self._reference_tool_rate(tool_id)
        wait, process, _, _ = tool_response_time(
            arrival_rate,
            service_rate,
            replicas,
            tool.arrival_scv,
            self.scenario.simulation.overload_delay_s,
        )
        delay_reference = max(1.0 / service_rate, 1.0e-12)
        cost_reference = max(tool.running_cost_per_slot, 1.0e-12)
        return (
            self.config.tool_delay_weight * (wait + process) / delay_reference
            + self.config.tool_cost_weight
            * replicas
            * tool.running_cost_per_slot
            / cost_reference
        )

    def _reference_tool_rate(self, tool_id: str) -> float:
        rates = [
            rate
            for rate in self.scenario.tools[tool_id].service_rate.values()
            if rate > 0.0
        ]
        return sum(rates) / len(rates) if rates else 0.0

    def _candidate_capacities(
        self,
        arrival_rates: Mapping[tuple[str, str], float],
        model_share: Mapping[tuple[str, str, str], float],
        model_arrival: Mapping[str, float],
    ) -> dict[str, float]:
        capacities: dict[str, float] = {}
        for candidate_id, candidate in self.scenario.candidates.items():
            prompt, output, long_fraction, composition = self._model_workload(
                candidate.model, arrival_rates, model_share
            )

            config = self.scenario.llm_configs[candidate.config]
            instance, _ = evaluate_llm_instance(
                self.scenario.models[candidate.model],
                config,
                [(prompt, output)],
                [1.0],
                0.0,
                self.scenario.simulation.prefill_chunk_tokens,
                composition_mode="macro",
            )
            capacities[candidate_id] = instance.throughput_capacity_rps
        return capacities

    def _model_workload(
        self,
        model: str,
        arrival_rates: Mapping[tuple[str, str], float],
        model_share: Mapping[tuple[str, str, str], float],
    ) -> tuple[float, float, float, dict[str, float]]:
        weighted: list[tuple[float, float, float, str]] = []
        for app in self.scenario.applications.values():
            for ingress, base_rate in app.ingress_rates.items():
                rate = max(
                    0.0, float(arrival_rates.get((app.id, ingress), base_rate))
                )
                share = model_share[(app.id, ingress, model)]
                for node in app.nodes.values():
                    if node.type is not NodeType.LLM:
                        continue
                    weight = rate * share * app.visit_probability(node.id)
                    weighted.append(
                        (
                            weight,
                            node.prompt_tokens[model],
                            node.output_tokens[model],
                            app.family,
                        )
                    )
        total = sum(item[0] for item in weighted)
        if total <= 0.0:
            return 1.0, 1.0, 0.0, {}
        prompt = sum(weight * p for weight, p, _, _ in weighted) / total
        output = sum(weight * o for weight, _, o, _ in weighted) / total
        long_fraction = sum(
            weight for weight, p, o, _ in weighted if p + o >= 1024
        ) / total
        composition: dict[str, float] = defaultdict(float)
        for weight, _, _, family in weighted:
            composition[family] += weight / total
        return prompt, output, long_fraction, dict(composition)
