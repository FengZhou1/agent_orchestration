"""A sampled, immutable slot trajectory replayed in every training episode.

The draws below are provisional ranges, not dataset calibrations.  A slot holds
one mean arrival rate and one mean token count for each workload class; the
analytical simulator does not sample individual requests inside that slot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from agent_orch.capacity import CapacityPlanner
from agent_orch.schema.models import NodeType, Scenario


@dataclass(frozen=True)
class SlotVariationSpec:
    arrival_range: tuple[float, float] = (0.6, 1.4)
    bandwidth_range: tuple[float, float] = (0.7, 1.0)
    cpu_range: tuple[float, float] = (0.8, 1.0)
    memory_range: tuple[float, float] = (0.8, 1.0)
    # Physical GPU inventory and SLO contracts are held fixed.  A later,
    # calibrated availability trace may set a nonzero GPU outage probability.
    gpu_unavailable_probability: float = 0.0

    def validate(self) -> None:
        for name in (
            "arrival_range", "bandwidth_range", "cpu_range", "memory_range",
        ):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must be positive and ordered")
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
        trajectory = SlotTrajectory(payload["scenario_id"], int(payload["seed"]), variation, slots)
        trajectory.validate_fixed_workload()
        return trajectory

    def validate_fixed_workload(self) -> None:
        """Reject traces that change application token lengths or pattern probabilities."""
        if not self.slots:
            raise ValueError("A nonempty slot trajectory is required")
        first = self.slots[0]
        for snapshot in self.slots[1:]:
            if (snapshot.flow_probabilities != first.flow_probabilities
                    or snapshot.prompt_tokens != first.prompt_tokens
                    or snapshot.output_tokens != first.output_tokens):
                raise ValueError("Token lengths and pattern probabilities must stay fixed across slots")

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate_against(self, base: Scenario) -> None:
        """Fixed values must match the base, not merely match the first slot."""
        self.validate_fixed_workload()
        if self.scenario_id != base.id:
            raise ValueError("Trajectory belongs to a different scenario")
        expected_flows = {a.id: {f.id: f.probability for f in a.pattern_flows}
                          for a in base.applications.values()}
        expected_prompt = {a.id: {n.id: n.prompt_tokens for n in a.nodes.values() if n.type is NodeType.LLM}
                           for a in base.applications.values()}
        expected_output = {a.id: {n.id: n.output_tokens for n in a.nodes.values() if n.type is NodeType.LLM}
                           for a in base.applications.values()}
        for snapshot in self.slots:
            if (snapshot.flow_probabilities != expected_flows or
                    snapshot.prompt_tokens != expected_prompt or snapshot.output_tokens != expected_output):
                raise ValueError("Trajectory token lengths or probabilities differ from the base scenario")
            if (set(snapshot.arrivals) != set(base.applications) or
                    set(snapshot.server_resources) != set(base.servers) or
                    len(snapshot.link_capacity_mbps) != len(base.links)):
                raise ValueError("Trajectory application/server/link layout differs from base")
            for a in base.applications.values():
                rates = snapshot.arrivals[a.id]
                if set(rates) != set(a.ingress_rates) or any(
                    not math.isfinite(r) or r < 0 or ((r > 0) != (a.ingress_rates[n] > 0))
                    for n, r in rates.items()):
                    raise ValueError("Trajectory changed the fixed positive-ingress support")
            for n, s in base.servers.items():
                resources = snapshot.server_resources[n]
                if (set(resources) != {"cpu_cores", "memory_gb", "gpu_count"} or
                    not math.isfinite(resources["cpu_cores"]) or resources["cpu_cores"] < 1 or
                    not math.isfinite(resources["memory_gb"]) or resources["memory_gb"] <= 0 or
                    not 0 <= resources["gpu_count"] <= s.gpu_count or
                    (self.variation.gpu_unavailable_probability == 0 and resources["gpu_count"] != s.gpu_count)):
                    raise ValueError("Trajectory contains invalid resources or changed GPU inventory")
            if any(not math.isfinite(c) or c <= 0 for c in snapshot.link_capacity_mbps):
                raise ValueError("Trajectory has invalid bandwidth")

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
    for app_id, app in base.applications.items():
        arrivals[app_id] = {
            ingress: max(0.0, rate * _draw(rng, variation.arrival_range))
            for ingress, rate in app.ingress_rates.items()
        }
        flows[app_id] = {flow.id: flow.probability for flow in app.pattern_flows}
        prompts[app_id] = {}
        outputs[app_id] = {}
        for node_id, node in app.nodes.items():
            if node.type is not NodeType.LLM:
                continue
            prompts[app_id][node_id] = dict(node.prompt_tokens)
            outputs[app_id][node_id] = dict(node.output_tokens)
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
