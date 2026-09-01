from __future__ import annotations

from collections import defaultdict

from agent_orch.schema.models import DeploymentDecision, Scenario


class CapacityPlanner:
    """Static feasibility, candidate ordering, and deployment masks only."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

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
        deployment = DeploymentDecision(
            llm_active={candidate_id: 0 for candidate_id in self.scenario.candidates},
            tool_replicas={
                (tool_id, server_id): 0
                for tool_id in self.scenario.tools
                for server_id in self.scenario.servers
            },
        )
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
