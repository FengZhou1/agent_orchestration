"""Fixed, resource-feasible deployment contexts for model-selection warm-up.

Constructive coverage, not policy supervision: no reward, SLO score, teacher,
or PPO output enters generation. Every context is bound to its physical slot.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Callable

import yaml

from agent_orch.capacity import CapacityPlanner
from agent_orch.performance.analytical import AnalyticalBackend, reference_llm_capacity
from agent_orch.performance.network import NetworkBackend
from agent_orch.schema.models import DeploymentDecision, NodeType, Scenario
from agent_orch.workload.slot_trajectory import SlotTrajectory


VERSION = "fixed_warmup_v1"
DEFAULT_CONFIG = Path("configs/warmup_deployments.yaml")


def _canonical(value):
    if is_dataclass(value):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _canonical(v) for k, v in value.items()}
        return [[_canonical(k), _canonical(v)] for k, v in sorted(value.items(), key=lambda p: repr(p[0]))]
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def scenario_digest(scenario: Scenario) -> str:
    """Bind applications, arrivals, token, SLO and resources, not just a catalog."""
    return _digest(scenario)


def physical_signature(scenario: Scenario, deployment: DeploymentDecision) -> str:
    """Replica-index permutations are not different physical deployments."""
    groups = Counter((scenario.candidates[c].model, scenario.candidates[c].config,
                      scenario.candidates[c].server)
                     for c, active in deployment.llm_active.items() if active)
    tools = [(h, n, z) for (h, n), z in sorted(deployment.tool_replicas.items()) if z]
    return _digest([sorted((*k, v) for k, v in groups.items()), tools])


@dataclass(frozen=True)
class WarmupSpec:
    gpu_fractions: tuple[float, ...] = (0.35, 0.55, 0.75, 0.90)
    model_biases: tuple[str, ...] = ("balanced", "light", "large")
    placements: tuple[str, ...] = ("dispersed", "ingress_near", "concentrated", "tool_near")
    tool_utilization_targets: tuple[float, ...] = (0.45, 0.70, 0.90)
    extra_tool_probabilities: tuple[float, ...] = (0.0, 0.20, 0.45)
    attempts_per_slot: int = 48
    repair_gpu_fraction: float = 0.95

    def validate(self):
        if (not self.gpu_fractions or any(not 0 < x <= 1 for x in self.gpu_fractions)
                or not 0 < self.repair_gpu_fraction <= 1
                or not self.tool_utilization_targets
                or any(not 0 < x < 1 for x in self.tool_utilization_targets)
                or not self.extra_tool_probabilities
                or any(not 0 <= x <= 1 for x in self.extra_tool_probabilities)
                or self.attempts_per_slot < 1):
            raise ValueError("Invalid warm-up resource/attempt settings")
        if not self.model_biases or set(self.model_biases) - {"balanced", "light", "large"}:
            raise ValueError("Invalid model allocation profiles")
        if not self.placements or set(self.placements) - {"dispersed", "ingress_near", "concentrated", "tool_near"}:
            raise ValueError("Invalid placement profiles")

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        spec = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        spec.validate()
        return spec


def _encode(deployment):
    return {"llm_active": dict(deployment.llm_active),
            "tool_replicas": {h: {n: z for (tool, n), z in deployment.tool_replicas.items() if tool == h}
                              for h in sorted({h for h, _ in deployment.tool_replicas})}}


def _decode(raw):
    return DeploymentDecision(dict(raw["llm_active"]),
        {(h, n): z for h, hosts in raw["tool_replicas"].items() for n, z in hosts.items()})


def _tool_rates(scenario):
    result = {h: 0.0 for h in scenario.tools}
    for app in scenario.applications.values():
        rate = sum(app.ingress_rates.values())
        for node in app.nodes.values():
            if node.type is NodeType.TOOL:
                result[node.tool] += rate * app.visit_probability(node.id)
    return result


def _capacities(scenario):
    """Actual probability/arrival-weighted token mixture for each model/config."""
    capacities, total_calls = {}, 0.0
    for model in scenario.models:
        classes, weights = [], []
        for app in scenario.applications.values():
            rate = sum(app.ingress_rates.values())
            for node in app.nodes.values():
                weight = rate * app.visit_probability(node.id)
                if node.type is NodeType.LLM and weight > 0:
                    classes.append((node.prompt_tokens[model], node.output_tokens[model]))
                    weights.append(weight)
        total_calls = sum(weights)
        # All diagnostic mixtures use one global share per model. This scaling
        # leaves the conditional class mixture unchanged; no mean-token shortcut.
        for config in scenario.llm_configs.values():
            if config.model == model:
                capacities[config.id] = reference_llm_capacity(scenario.models[model], config,
                    classes, weights, scenario.simulation.prefill_chunk_tokens)[0]
    return capacities, total_calls


def _active_models(scenario, deployment):
    return sorted({scenario.candidates[c].model for c, v in deployment.llm_active.items() if v})


def validate_placement(scenario, deployment):
    planner = CapacityPlanner(scenario)
    if (set(deployment.llm_active) != set(scenario.candidates)
            or set(deployment.tool_replicas) != set(planner.empty_deployment().tool_replicas)
            or any(type(v) is not int or v not in (0, 1) for v in deployment.llm_active.values())
            or any(type(v) is not int or v < 0 for v in deployment.tool_replicas.values())):
        raise ValueError("Invalid or incomplete warm-up deployment vector")
    if not planner.deployment_feasible(deployment):
        raise ValueError("Warm-up deployment exceeds physical resources")
    if not _active_models(scenario, deployment):
        raise ValueError("Warm-up deployment has no online model")
    required = {h for h, rate in _tool_rates(scenario).items() if rate > 0}
    for h in required:
        if not any(z > 0 and scenario.tools[h].service_rate[n] > 0
                   for (tool, n), z in deployment.tool_replicas.items() if tool == h):
            raise ValueError(f"Missing positive-workload tool {h}")
    # Warm-up contexts require paths for every positive routing support pair.
    network = NetworkBackend(scenario.links)
    hosts = {scenario.candidates[c].server for c, v in deployment.llm_active.items() if v}
    hosts |= {n for (h, n), z in deployment.tool_replicas.items() if z and h in required}
    hosts |= {n for a in scenario.applications.values() for n, r in a.ingress_rates.items() if r > 0}
    for u in hosts:
        for v in hosts:
            network.path(u, v)


@dataclass(frozen=True)
class FixedWarmupDeployments:
    scenario_hash: str
    trajectory_digest: str
    seed: int
    spec: dict[str, Any]
    entries: tuple[dict[str, Any], ...]

    def to_dict(self):
        return {"version": VERSION, **asdict(self)}

    def digest(self):
        return _digest(self.to_dict())

    def deployment_at(self, slot):
        return _decode(self.entries[slot]["deployment"])

    def validate(self, scenario, trajectory):
        trajectory.validate_against(scenario)
        if self.scenario_hash != scenario_digest(scenario) or self.trajectory_digest != trajectory.digest():
            raise ValueError("Warm-up contexts belong to a different scenario or trajectory")
        WarmupSpec(**self.spec).validate()
        if len(self.entries) != len(trajectory):
            raise ValueError("Warm-up context count differs from physical slots")
        for slot, entry in enumerate(self.entries):
            current = trajectory.scenario_at(slot, scenario)
            deployment = self.deployment_at(slot)
            if entry["slot"] != slot or entry["signature"] != physical_signature(current, deployment):
                raise ValueError("Invalid warm-up slot or physical signature")
            validate_placement(current, deployment)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps({**self.to_dict(), "digest": self.digest()},
            indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path, scenario, trajectory):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("version") != VERSION:
            raise ValueError("Unknown warm-up deployment protocol")
        digest = payload.pop("digest")
        payload["entries"] = tuple(payload["entries"])
        result = cls(**payload)
        if digest != result.digest():
            raise ValueError("Warm-up deployment digest mismatch")
        result.validate(scenario, trajectory)
        return result


def _construct(scenario, models, capacities, profile, rng):
    planner = CapacityPlanner(scenario)
    deployment = planner.empty_deployment()
    network = NetworkBackend(scenario.links)
    ingress = Counter()
    for app in scenario.applications.values():
        for n, rate in app.ingress_rates.items():
            ingress[n] += rate
    distances = {n: sum(r * sum(network.links[e].propagation_ms for e in network.path(g, n))
                        for g, r in ingress.items() if r > 0) for n in scenario.servers}
    anchor = rng.choice(list(scenario.servers))
    jitter = {n: rng.random() for n in scenario.servers}

    def host_order(kind):
        llm_hosts = {scenario.candidates[c].server for c, v in deployment.llm_active.items() if v}
        tool_hosts = Counter()
        for (h, n), z in deployment.tool_replicas.items():
            tool_hosts[n] += z
        remaining = planner.normalized_remaining_resources(deployment)
        def rank(n):
            placement = profile["placement"]
            used = (1 - remaining[n][2]) if kind == "llm" else (1 - remaining[n][0])
            if placement == "dispersed": return (used, jitter[n])
            if placement == "ingress_near": return (distances[n], used, jitter[n])
            if placement == "tool_near": return (-(tool_hosts[n] if kind == "llm" else int(n in llm_hosts)), used, jitter[n])
            return (sum(network.links[e].propagation_ms for e in network.path(anchor, n)), -used, jitter[n])
        return {n: i for i, n in enumerate(sorted(scenario.servers, key=rank))}

    def gpu_used():
        return sum(scenario.llm_configs[scenario.candidates[c].config].gpu_count
                   * scenario.llm_configs[scenario.candidates[c].config].gpu_share
                   for c, v in deployment.llm_active.items() if v)

    available_gpu = sum(s.gpu_count for s in scenario.servers.values())
    budget = available_gpu * profile["gpu_fraction"]
    config_ids = sorted({c.config for c in scenario.candidates.values() if c.model in models})
    preferred = config_ids[profile["config_rotation"] % len(config_ids)]

    def add_llm(model, limit):
        order = host_order("llm")
        choices = [c for c in scenario.candidates.values() if c.model == model
                   and not deployment.llm_active[c.id] and capacities[c.config] > 0
                   and gpu_used() + scenario.llm_configs[c.config].gpu_count
                       * scenario.llm_configs[c.config].gpu_share <= limit + 1e-9
                   and planner.feasible_activation(deployment, c.id)]
        choices.sort(key=lambda c: (order[c.server], c.config != preferred,
                                   jitter[c.server], c.replica_index, c.id))
        if not choices: return False
        deployment.llm_active[choices[0].id] = 1
        return True

    # Install scarce models first. Coverage can exceed a soft occupancy target,
    # but never the actual fleet. This is recorded rather than called target met.
    ordered = sorted(models, key=lambda m: (sum(c.model == m and capacities[c.config] > 0
                                               for c in scenario.candidates.values()), -scenario.models[m].parameter_count))
    for model in ordered:
        if not add_llm(model, available_gpu):
            raise ValueError(f"Cannot pack requested model subset: {models}")

    rates = _tool_rates(scenario)
    required = [h for h, r in rates.items() if r > 0]
    tool_order = sorted(required, key=lambda h: (-scenario.tools[h].cpu_cores, -rates[h], h))
    for h in tool_order:
        tool = scenario.tools[h]
        target = rates[h] / profile["tool_utilization"]
        desired_redundancy = 1 + int(rng.random() < profile["extra_tool_probability"])
        placed = 0
        while (sum(z * tool.service_rate[n] for (k, n), z in deployment.tool_replicas.items() if k == h)
               < target or placed < desired_redundancy):
            order = host_order("tool")
            choices = [n for n in scenario.servers if tool.service_rate[n] > 0
                       and planner.feasible_tool_replica(deployment, h, n)]
            if not choices:
                if placed == 0: raise ValueError(f"Cannot cover tool {h}")
                break  # capacity/redundancy shortfall is measured below
            n = min(choices, key=lambda n: (deployment.tool_replicas[(h, n)], order[n]))
            deployment.tool_replicas[(h, n)] += 1
            placed += 1

    weights = _bias_weights(scenario, models, profile["model_bias"])
    while True:
        capacity = _model_capacities(scenario, deployment, capacities)
        priority = sorted(models, key=lambda m: (capacity[m] / weights[m], rng.random()))
        if not any(add_llm(m, budget) for m in priority): break
    return deployment


def _bias_weights(scenario, models, bias):
    if bias == "balanced": return {m: 1.0 for m in models}
    ordered = sorted(models, key=lambda m: scenario.models[m].parameter_count,
                     reverse=bias == "large")
    return {m: 2.0 ** (-i) for i, m in enumerate(ordered)}


def _model_capacities(scenario, deployment, capacities):
    totals = Counter({m: 0.0 for m in _active_models(scenario, deployment)})
    for c, v in deployment.llm_active.items():
        if v:
            candidate = scenario.candidates[c]
            totals[candidate.model] += capacities[candidate.config]
    return dict(totals)


def capacity_probes(scenario, deployment, capacities, total_calls):
    """Three declared mixtures, not optimization or proof of SLO feasibility.

    Global model shares preserve each model's exact conditional token mixture.
    Capacity ratios screen inference load; tool/link checks use the current
    capacity-proportional execution probabilities. No workflow scores computed.
    """
    model_capacity = _model_capacities(scenario, deployment, capacities)
    models = sorted(model_capacity)
    probes = []
    backend = AnalyticalBackend(scenario)
    for bias in ("balanced", "light", "large"):
        weights = _bias_weights(scenario, models, bias)
        shares = {m: weights[m] / sum(weights.values()) for m in models}
        # Construct the manuscript's capacity-proportional routing directly from
        # the same reference capacities; all supports have been path-validated.
        from agent_orch.schema.models import RoutingDecision
        llm_share, tool_route = {}, {}
        for app in scenario.applications.values():
            for g in app.ingress_rates:
                for node in app.nodes.values():
                    if node.type is NodeType.LLM:
                        for c, active in deployment.llm_active.items():
                            if active:
                                candidate = scenario.candidates[c]
                                llm_share[(app.id, g, node.id, c)] = (
                                    shares[candidate.model] * capacities[candidate.config] / model_capacity[candidate.model])
            for i, j in sorted({e for f in app.pattern_flows for e in f.edges
                                if app.nodes[e[1]].type is NodeType.TOOL}):
                h = app.nodes[j].tool
                pool = {n: z * scenario.tools[h].service_rate[n]
                        for (k, n), z in deployment.tool_replicas.items() if k == h and z > 0}
                for u in scenario.servers:
                    for v, cap in pool.items(): tool_route[(app.id, i, j, u, v)] = cap / sum(pool.values())
        routing = RoutingDecision({(a.id, g, m): share for a in scenario.applications.values()
                                   for g in a.ingress_rates for m, share in shares.items()}, llm_share, tool_route)
        distributions = backend.node_distributions(deployment, routing)
        tool_rates = backend.tool_arrivals(routing, distributions)
        tool_util = {f"{h}|{n}": rate / (deployment.tool_replicas[(h, n)] * scenario.tools[h].service_rate[n])
                     for (h, n), rate in tool_rates.items() if rate > 0}
        network_util = backend.network.utilization(backend.link_loads(routing, distributions))
        llm_util = {m: total_calls * shares[m] / cap for m, cap in model_capacity.items()}
        maximum = max([*llm_util.values(), *tool_util.values(), *network_util.values()], default=0)
        probes.append({"mixture": bias, "model_shares": shares, "llm_capacity_ratios": llm_util,
            "max_tool_utilization": max(tool_util.values(), default=0),
            "max_network_utilization": max(network_util.values(), default=0),
            "max_capacity_ratio": maximum, "capacity_screen_pass": maximum < 1})
    return probes


def _coverage_profile(slot, size_count, subset, spec):
    block, within = divmod(slot, size_count)
    # Different periods avoid tying model cardinality to a single bias or
    # tool tier. This is a coverage schedule, not a full factorial experiment.
    return {"models": list(subset),
        "gpu_fraction": spec.gpu_fractions[block % len(spec.gpu_fractions)],
        "model_bias": spec.model_biases[(block + within) % len(spec.model_biases)],
        "placement": spec.placements[slot % len(spec.placements)],
        "tool_utilization": spec.tool_utilization_targets[
            (block // len(spec.model_biases) + within) % len(spec.tool_utilization_targets)],
        "extra_tool_probability": spec.extra_tool_probabilities[
            (block // len(spec.gpu_fractions) + within) % len(spec.extra_tool_probabilities)],
        "config_rotation": slot}


def generate_warmup_deployments(scenario: Scenario, trajectory: SlotTrajectory,
        seed: int = 20261014, spec: WarmupSpec = WarmupSpec(),
        on_slot: Callable[[dict], None] | None = None) -> FixedWarmupDeployments:
    spec.validate()
    trajectory.validate_against(scenario)
    capacities, _ = _capacities(scenario)
    models = sorted({c.model for c in scenario.candidates.values() if capacities[c.config] > 0},
                    key=lambda m: scenario.models[m].parameter_count)
    if not models: raise ValueError("No candidate has positive inference capacity")
    # Cycle subset cardinality as well as identities; do not put all pairs first.
    subsets_by_size = {n: list(itertools.combinations(models, n))
                       for n in range(min(2, len(models)), len(models) + 1)}
    sizes = list(subsets_by_size)
    entries, seen = [], set()
    for slot in range(len(trajectory)):
        current = trajectory.scenario_at(slot, scenario)
        # Per-application rates fluctuate independently: even fixed token
        # lengths yield a different conditional token mixture in each slot.
        capacities, calls = _capacities(current)
        n = sizes[slot % len(sizes)]
        subsets = subsets_by_size[n]
        subset = subsets[(slot // len(sizes)) % len(subsets)]
        profile = _coverage_profile(slot, len(sizes), subset, spec)
        errors = []
        for attempt in range(spec.attempts_per_slot):
            try:
                deployment = _construct(current, subset, capacities, profile,
                                        random.Random(f"{seed}:{slot}:{attempt}"))
                validate_placement(current, deployment)
            except ValueError as error:
                errors.append(str(error))
                continue
            signature = physical_signature(current, deployment)
            if signature in seen and attempt + 1 < spec.attempts_per_slot: continue
            break
        else:
            raise ValueError(f"Slot {slot}: no valid context after {spec.attempts_per_slot} attempts; {errors[-3:]}")
        probes = capacity_probes(current, deployment, capacities, calls)
        repairs = []
        # If none of the fixed mixtures passes the capacity screen, add replicas
        # on the most overloaded model of the balanced probe. Never pick a best
        # model-action target, change load, or reject by SLO attainment.
        if not any(p["capacity_screen_pass"] for p in probes):
            planner = CapacityPlanner(current)
            fleet = sum(s.gpu_count for s in current.servers.values())
            while True:
                totals = _model_capacities(current, deployment, capacities)
                used = sum(current.llm_configs[current.candidates[c].config].gpu_count
                           * current.llm_configs[current.candidates[c].config].gpu_share
                           for c, v in deployment.llm_active.items() if v)
                ranked = sorted(subset, key=lambda m: calls / len(subset) / totals[m], reverse=True)
                choices = [c for m in ranked for c in current.candidates.values()
                           if c.model == m and not deployment.llm_active[c.id] and capacities[c.config] > 0
                           and planner.feasible_activation(deployment, c.id)
                           and used + current.llm_configs[c.config].gpu_count * current.llm_configs[c.config].gpu_share
                               <= fleet * spec.repair_gpu_fraction + 1e-9]
                if not choices: break
                c = choices[0]
                deployment.llm_active[c.id] = 1
                repairs.append(c.id)
                # Cheap capacity-only stopping condition. Final tool/link probes
                # still run and any unresolved risk remains explicitly recorded.
                totals = _model_capacities(current, deployment, capacities)
                if all(calls / len(subset) / totals[m] < 1 for m in subset): break
            if repairs: probes = capacity_probes(current, deployment, capacities, calls)
        validate_placement(current, deployment)
        signature = physical_signature(current, deployment)
        duplicate = signature in seen
        seen.add(signature)
        remaining = CapacityPlanner(current).normalized_remaining_resources(deployment)
        gpu_used = sum(current.llm_configs[current.candidates[c].config].gpu_count
                       * current.llm_configs[current.candidates[c].config].gpu_share
                       for c, v in deployment.llm_active.items() if v)
        entry = {"slot": slot, "profile": profile, "attempts": attempt + 1,
            "construction_failures": dict(Counter(errors)), "repairs_added": repairs,
            "signature": signature, "duplicate_physical_layout": duplicate,
            "deployment": _encode(deployment), "active_models": _active_models(current, deployment),
            "llm_instances": sum(deployment.llm_active.values()),
            "tool_replicas": sum(deployment.tool_replicas.values()),
            "gpu_used": gpu_used, "available_gpu": sum(s.gpu_count for s in current.servers.values()),
            "gpu_fraction_actual": gpu_used / max(1, sum(s.gpu_count for s in current.servers.values())),
            "llm_hosts": len({current.candidates[c].server for c, v in deployment.llm_active.items() if v}),
            "tool_hosts": len({n for (h, n), z in deployment.tool_replicas.items() if z}),
            "cpu_fraction_max": max(1 - v[0] for v in remaining.values()),
            "memory_fraction_max": max(1 - v[1] for v in remaining.values()),
            "model_capacities_rps": _model_capacities(current, deployment, capacities),
            "probes": probes, "capacity_risk": not any(p["capacity_screen_pass"] for p in probes)}
        entries.append(entry)
        if on_slot: on_slot(entry)
    result = FixedWarmupDeployments(scenario_digest(scenario), trajectory.digest(), int(seed),
                                   asdict(spec), tuple(entries))
    result.validate(scenario, trajectory)
    return result


def coverage_report(bank: FixedWarmupDeployments):
    entries = bank.entries
    return {"version": VERSION, "digest": bank.digest(), "slots": len(entries),
        "unique_physical_deployments": len({e["signature"] for e in entries}),
        "model_count_histogram": dict(Counter(str(len(e["active_models"])) for e in entries)),
        "model_subsets": dict(Counter("+".join(e["active_models"]) for e in entries)),
        "model_exposure_slots": dict(Counter(m for e in entries for m in e["active_models"])),
        "profiles_by_model_count": {str(n): {
            "model_biases": sorted({e["profile"]["model_bias"] for e in entries if len(e["active_models"]) == n}),
            "gpu_targets": sorted({e["profile"]["gpu_fraction"] for e in entries if len(e["active_models"]) == n}),
            "tool_utilization_targets": sorted({e["profile"]["tool_utilization"] for e in entries if len(e["active_models"]) == n}),
            "extra_tool_probabilities": sorted({e["profile"]["extra_tool_probability"] for e in entries if len(e["active_models"]) == n}),
            "placements": sorted({e["profile"]["placement"] for e in entries if len(e["active_models"]) == n})}
            for n in sorted({len(e["active_models"]) for e in entries})},
        "capacity_risk_slots": [e["slot"] for e in entries if e["capacity_risk"]],
        "repair_slots": [e["slot"] for e in entries if e["repairs_added"]],
        "duplicate_slots": [e["slot"] for e in entries if e["duplicate_physical_layout"]],
        "note": "Capacity screens use three declared mixtures, not optimal choices or SLO guarantees.",
        "rows": [{k: e[k] for k in ("slot", "active_models", "llm_instances", "tool_replicas",
            "gpu_used", "available_gpu", "gpu_fraction_actual", "llm_hosts", "tool_hosts",
            "cpu_fraction_max", "memory_fraction_max", "model_capacities_rps", "capacity_risk")}
            for e in entries]}


def reuse_layouts_fixed_exogenous(scenario: Scenario, trajectory: SlotTrajectory,
        source: FixedWarmupDeployments, source_trajectory: SlotTrajectory,
        on_slot: Callable[[dict], None] | None = None) -> FixedWarmupDeployments:
    """Rebind, never repair/reselect, saved layouts for the fixed-base control.

    Source provenance is checked against its original trajectory. Capacity and
    resource diagnostics are recomputed under the new demand, not copied.
    """
    source.validate(scenario, source_trajectory)
    trajectory.validate_against(scenario)
    if not trajectory.variation.is_fixed_base or len(trajectory) != len(source.entries):
        raise ValueError("Layout reuse requires fixed-base conditions and the original slot count")
    current = trajectory.scenario_at(0, scenario)
    capacities, calls = _capacities(current)
    source_digest = source.digest()
    entries, seen = [], set()
    for slot, original in enumerate(source.entries):
        deployment = source.deployment_at(slot)
        validate_placement(current, deployment)  # Fail instead of silently repairing.
        entry = copy.deepcopy(original)
        remaining = CapacityPlanner(current).normalized_remaining_resources(deployment)
        signature = physical_signature(current, deployment)
        probes = capacity_probes(current, deployment, capacities, calls)
        entry.update({
            "attempts": 0, "construction_failures": {}, "repairs_added": [],
            "signature": signature, "duplicate_physical_layout": signature in seen,
            "model_capacities_rps": _model_capacities(current, deployment, capacities),
            "probes": probes, "capacity_risk": not any(p["capacity_screen_pass"] for p in probes),
            "available_gpu": sum(s.gpu_count for s in current.servers.values()),
            "cpu_fraction_max": max(1-v[0] for v in remaining.values()),
            "memory_fraction_max": max(1-v[1] for v in remaining.values()),
            "source_context": {"bank_digest": source_digest, "trajectory_digest": source_trajectory.digest(),
                               "slot": slot, "signature": original["signature"],
                               "attempts": original["attempts"], "repairs_added": original["repairs_added"]},
        })
        entry["gpu_fraction_actual"] = entry["gpu_used"] / max(1, entry["available_gpu"])
        if signature != original["signature"] or entry["deployment"] != original["deployment"]:
            raise AssertionError("Rebinding changed a physical deployment")
        seen.add(signature)
        entries.append(entry)
        if on_slot: on_slot(entry)
    result = FixedWarmupDeployments(scenario_digest(scenario), trajectory.digest(), source.seed,
                                   copy.deepcopy(source.spec), tuple(entries))
    result.validate(scenario, trajectory)
    return result
