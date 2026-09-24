"""A sampled, immutable slot trajectory replayed in every training episode.

The draws below are provisional ranges, not dataset calibrations.  A slot holds
one mean arrival rate and one mean token count for each workload class; the
analytical simulator does not sample individual requests inside that slot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np

from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import NodeType, Scenario


@dataclass(frozen=True)
class SlotVariationSpec:
    arrival_range: tuple[float, float] = (0.6, 1.4)
    flow_concentration: float = 40.0
    prompt_range: tuple[float, float] = (0.8, 1.2)
    output_range: tuple[float, float] = (0.8, 1.2)
    bandwidth_range: tuple[float, float] = (0.7, 1.0)
    cpu_range: tuple[float, float] = (0.8, 1.0)
    memory_range: tuple[float, float] = (0.8, 1.0)
    # Physical GPU inventory and SLO contracts are held fixed.  A later,
    # calibrated availability trace may set a nonzero GPU outage probability.
    gpu_unavailable_probability: float = 0.0

    def validate(self) -> None:
        for name in (
            "arrival_range", "prompt_range", "output_range",
            "bandwidth_range", "cpu_range", "memory_range",
        ):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must be positive and ordered")
        if self.flow_concentration <= 0.0:
            raise ValueError("flow_concentration must be positive")
        if not 0.0 <= self.gpu_unavailable_probability < 1.0:
            raise ValueError("gpu_unavailable_probability must lie in [0, 1)")


@dataclass(frozen=True)
class SlotSnapshot:
    arrivals: dict[str, dict[str, float]]
    flow_probabilities: dict[str, dict[str, float]]
    prompt_tokens: dict[str, dict[str, dict[str, float]]]
    output_tokens: dict[str, dict[str, dict[str, float]]]
    link_capacity_mbps: tuple[float, ...]
    server_resources: dict[str, dict[str, float | int]]

    def apply(self, base: Scenario) -> Scenario:
        applications = {}
        for app_id, app in base.applications.items():
            nodes = {}
            for node_id, node in app.nodes.items():
                if node.type is NodeType.LLM:
                    node = replace(
                        node,
                        prompt_tokens=dict(self.prompt_tokens[app_id][node_id]),
                        output_tokens=dict(self.output_tokens[app_id][node_id]),
                    )
                nodes[node_id] = node
            flows = tuple(
                replace(flow, probability=self.flow_probabilities[app_id][flow.id])
                for flow in app.pattern_flows
            )
            applications[app_id] = replace(
                app, ingress_rates=dict(self.arrivals[app_id]),
                nodes=nodes, pattern_flows=flows,
            )
        servers = {
            server_id: replace(server, **self.server_resources[server_id])
            for server_id, server in base.servers.items()
        }
        links = tuple(
            replace(link, capacity_mbps=capacity)
            for link, capacity in zip(base.links, self.link_capacity_mbps, strict=True)
        )
        return replace(base, applications=applications, servers=servers, links=links)


@dataclass(frozen=True)
class SlotTrajectory:
    scenario_id: str
    seed: int
    variation: SlotVariationSpec
    slots: tuple[SlotSnapshot, ...]

    def __len__(self) -> int:
        return len(self.slots)

    def scenario_at(self, slot: int, base: Scenario) -> Scenario:
        if base.id != self.scenario_id:
            raise ValueError("The trajectory belongs to a different scenario")
        return self.slots[slot].apply(base)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SlotTrajectory":
        variation = SlotVariationSpec(**payload["variation"])
        variation.validate()
        slots = tuple(
            SlotSnapshot(
                arrivals=raw["arrivals"],
                flow_probabilities=raw["flow_probabilities"],
                prompt_tokens=raw["prompt_tokens"],
                output_tokens=raw["output_tokens"],
                link_capacity_mbps=tuple(raw["link_capacity_mbps"]),
                server_resources=raw["server_resources"],
            )
            for raw in payload["slots"]
        )
        return SlotTrajectory(payload["scenario_id"], int(payload["seed"]), variation, slots)

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def sample(
        scenario: Scenario,
        slots: int,
        seed: int,
        variation: SlotVariationSpec = SlotVariationSpec(),
    ) -> "SlotTrajectory":
        if slots <= 0:
            raise ValueError("slots must be positive")
        variation.validate()
        rng = np.random.default_rng(seed)
        snapshots: list[SlotSnapshot] = []
        for _ in range(slots):
            for attempt in range(100):
                snapshot = _sample_snapshot(scenario, variation, rng)
                # The fixed first-round reference must exist under the sampled
                # available resources.  Reject an infeasible draw deterministically.
                try:
                    CapacityPlanner(snapshot.apply(scenario)).initial_deployment()
                except ValueError:
                    continue
                snapshots.append(snapshot)
                break
            else:
                raise ValueError("Could not sample a feasible slot in 100 attempts")
        return SlotTrajectory(scenario.id, int(seed), variation, tuple(snapshots))


def _draw(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(*bounds))


def _sample_snapshot(
    base: Scenario, variation: SlotVariationSpec, rng: np.random.Generator
) -> SlotSnapshot:
    arrivals = {}
    flows = {}
    prompts = {}
    outputs = {}
    model_limits = {
        model: min(config.max_model_len for config in base.llm_configs.values() if config.model == model)
        for model in base.models
    }
    for app_id, app in base.applications.items():
        arrivals[app_id] = {
            ingress: max(0.0, rate * _draw(rng, variation.arrival_range))
            for ingress, rate in app.ingress_rates.items()
        }
        nonzero = [flow for flow in app.pattern_flows if flow.probability > 0.0]
        sampled = rng.dirichlet(
            np.asarray([flow.probability for flow in nonzero]) * variation.flow_concentration
        )
        flows[app_id] = {flow.id: 0.0 for flow in app.pattern_flows}
        flows[app_id].update({flow.id: float(p) for flow, p in zip(nonzero, sampled)})
        prompts[app_id] = {}
        outputs[app_id] = {}
        for node_id, node in app.nodes.items():
            if node.type is not NodeType.LLM:
                continue
            prompts[app_id][node_id] = {}
            outputs[app_id][node_id] = {}
            for model in base.models:
                prompt = node.prompt_tokens[model] * _draw(rng, variation.prompt_range)
                output = node.output_tokens[model] * _draw(rng, variation.output_range)
                limit = model_limits[model]
                if prompt + output > limit:
                    scale = limit / (prompt + output)
                    prompt *= scale
                    output *= scale
                prompts[app_id][node_id][model] = float(prompt)
                outputs[app_id][node_id][model] = float(output)
    servers = {}
    for server_id, server in base.servers.items():
        gpu_count = server.gpu_count
        if gpu_count and rng.random() < variation.gpu_unavailable_probability:
            gpu_count = max(0, gpu_count - 1)
        servers[server_id] = {
            "cpu_cores": max(1, int(round(server.cpu_cores * _draw(rng, variation.cpu_range)))),
            "memory_gb": float(server.memory_gb * _draw(rng, variation.memory_range)),
            "gpu_count": int(gpu_count),
        }
    links = tuple(
        float(link.capacity_mbps * _draw(rng, variation.bandwidth_range))
        for link in base.links
    )
    return SlotSnapshot(arrivals, flows, prompts, outputs, links, servers)
