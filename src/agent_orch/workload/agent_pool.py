"""Fixed synthetic application pools, explicit provenance and nested scenarios."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import platform

import networkx as nx
import numpy as np

from agent_orch.schema.models import (
    ApplicationSpec, NodeType, PatternFlow, SLOSpec, SLOType, ToolSpec, WorkflowNode,
)
from agent_orch.schema.loader import ScenarioLoader
from .agent_graph import GraphExpander, ExpandedGraph, validate_graph, pattern_graph, replay_control
from .agent_lengths import (
    INPUT, OUTPUT, TOOL, LengthDistribution, describe, generate_lengths, representative_prompts,
)


GENERATOR_VERSION = "probabilistic-agent-pool-v1"


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def keyed_seed(seed, domain, index=0):
    return int.from_bytes(hashlib.sha256(f"{seed}:{domain}:{index}".encode()).digest()[:4], "big")


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20261013
    applications: int = 200
    tool_types: int = 200
    pure_fraction: float = .05
    llm_bounds: tuple[int, int] = (2, 15)
    tool_bounds: tuple[int, int] = (1, 30)
    llm_log_mu: float = math.log(4)
    llm_log_sigma: float = .8
    tool_log_mu: float = math.log(5)
    tool_log_sigma: float = .9
    max_depth: int = 30
    max_parallel: int = 4
    max_patterns: int = 32
    operation_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("serial", "parallel", "exclusive", "repeat"), 1.))
    root_stop_range: tuple[float, float] = (.05, .30)
    short_stop_range: tuple[float, float] = (.60, .85)
    later_stop_range: tuple[float, float] = (.15, .55)
    structural_attempts: int = 200
    length_attempts: int = 200
    shared_fraction: float = .5
    private_request_probability: float = .5
    shared_zipf_exponent: float = 1.1
    tool_memory_gb: tuple[float, float] = (.5, 3.)
    tool_service_rate_rps: tuple[float, float] = (2., 8.)
    tool_running_cost: float = .05
    tool_start_cost: float = .1
    tool_cpu_cost_per_s: float | None = .02
    tool_memory_cost_per_gb_s: float | None = .001
    tool_start_cost_ratio_s: float | None = .1
    nontext_return_mb: tuple[float, float] = (2., 10.)
    text_bytes_per_token: float = 4.
    application_rate_rps: tuple[float, float] = (.005, .02)
    ingress_probability: float = .3
    slo_type_weights: tuple[float, float, float] = (1., 1., 1.)
    ttft_s_range: tuple[float, float] = (2., 20.)
    tbt_s_range: tuple[float, float] = (.02, .2)
    deadline_s_range: tuple[float, float] = (20., 180.)
    quality_range: tuple[float, float] = (.4, .95)
    quality_noise: float = .1
    quality_base: float = .58
    quality_size_gain: float = .24
    calibration_passes: int = 20

    def validate(self):
        prices = (self.tool_cpu_cost_per_s, self.tool_memory_cost_per_gb_s,
                  self.tool_start_cost_ratio_s)
        if any(v is not None for v in prices) and any(v is None or v < 0 for v in prices):
            raise ValueError("Resource-based tool pricing requires all three nonnegative coefficients")
        if self.applications < 1 or self.tool_types < 1:
            raise ValueError("Application and tool counts must be positive")
        if not (0 <= self.pure_fraction <= .1 and 0 <= self.shared_fraction <= 1
                and 0 <= self.private_request_probability <= 1 and 0 < self.ingress_probability <= 1):
            raise ValueError("Invalid mixture/support fractions")
        if not (2 <= self.llm_bounds[0] <= self.llm_bounds[1] <= 15
                and 1 <= self.tool_bounds[0] <= self.tool_bounds[1] <= 30
                and 2 <= self.max_depth <= 30 and 2 <= self.max_parallel <= 4
                and self.max_patterns >= 3):
            raise ValueError("Invalid structural ranges")
        if min(self.structural_attempts, self.length_attempts) < 1 or self.calibration_passes < 0:
            raise ValueError("Invalid generation attempt budget")
        if (self.llm_log_sigma <= 0 or self.tool_log_sigma <= 0 or self.quality_noise < 0
                or self.shared_zipf_exponent < 0 or self.quality_range[1] > 1
                or min(self.tool_running_cost, self.tool_start_cost) < 0):
            raise ValueError("Invalid distribution, quality or price parameters")
        for name in ("tool_memory_gb", "tool_service_rate_rps", "application_rate_rps",
                     "ttft_s_range", "tbt_s_range", "deadline_s_range", "quality_range"):
            lo, hi = getattr(self, name)
            if not (0 < lo <= hi):
                raise ValueError(f"Invalid {name}")
        if not (0 <= self.nontext_return_mb[0] <= self.nontext_return_mb[1]
                and self.text_bytes_per_token > 0 and sum(self.slo_type_weights) > 0
                and min(self.slo_type_weights) >= 0):
            raise ValueError("Invalid payload or SLO mixture")
        if set(self.operation_weights) != {"serial", "parallel", "exclusive", "repeat"} or min(self.operation_weights.values()) < 0:
            raise ValueError("Invalid expansion-operation weights")
        for bounds in (self.root_stop_range, self.short_stop_range, self.later_stop_range):
            if not 0 < bounds[0] <= bounds[1] < 1:
                raise ValueError("Response probabilities must be positive and below one")


@dataclass
class GeneratedApplication:
    spec: ApplicationSpec
    graph: ExpandedGraph
    lengths: dict
    record: dict


def sample_count(rng, bounds, mu, sigma):
    for _ in range(10000):
        n = int(round(rng.lognormal(mu, sigma)))
        if bounds[0] <= n <= bounds[1]:
            return n
    raise ValueError("Count distribution has negligible mass inside bounds")


def synthetic_attributes(template, graph, lengths, cfg, index):
    rng = np.random.default_rng(keyed_seed(cfg.seed, "ingress", index))
    models = sorted(template.models, key=lambda m: template.models[m].parameter_count)
    servers = sorted(template.servers)
    active = [n for n in servers if rng.random() < cfg.ingress_probability]
    if not active:
        active = [str(rng.choice(servers))]
    weights = rng.dirichlet(np.ones(len(active)))
    rate = float(rng.uniform(*cfg.application_rate_rps))
    arrivals = {n: rate * float(w) for n, w in zip(active, weights)}
    rng = np.random.default_rng(keyed_seed(cfg.seed, "slo", index))
    type_weights = np.asarray(cfg.slo_type_weights) / sum(cfg.slo_type_weights)
    slo_type = SLOType(str(rng.choice(["lat", "ddl", "cmp"], p=type_weights)))
    deadline = float(rng.uniform(*cfg.deadline_s_range))
    slo = SLOSpec(slo_type,
        ttft_s=float(rng.uniform(*cfg.ttft_s_range)) if slo_type is not SLOType.DEADLINE else None,
        tbt_s=float(rng.uniform(*cfg.tbt_s_range)) if slo_type is not SLOType.DEADLINE else None,
        deadline_s=deadline if slo_type is not SLOType.LATENCY else None)
    quality = {m: float(np.clip(cfg.quality_base + cfg.quality_size_gain * rank / max(1, len(models) - 1)
                  + np.random.default_rng(keyed_seed(cfg.seed, "quality:" + m, index)).normal(0, cfg.quality_noise), *cfg.quality_range))
               for rank, m in enumerate(models)}
    prompts = representative_prompts(graph, lengths)
    full = graph.graph()
    depth = {}
    for n in nx.lexicographical_topological_sort(full):
        depth[n] = 1 + max((depth[p] for p in full.predecessors(n)), default=0)
    total_depth = max(depth.values())
    nodes = {}
    for n, kind in graph.node_types.items():
        nodes[n] = WorkflowNode(n, NodeType(kind),
            tool="unbound" if kind == "tool" else None,
            prompt_tokens={m: prompts[n] for m in models} if kind == "llm" else {},
            output_tokens={m: float(lengths["outputs"][n]) for m in models} if kind == "llm" else {},
            stage_deadline_s=(deadline * depth[n] / total_depth
                             if kind == "llm" and slo_type is SLOType.COMPOUND else None))
    flows = tuple(PatternFlow(f"p{i}", v.probability, v.exits[0], (),
                             tuple(sorted(v.nodes)), tuple(sorted(v.edges)))
                  for i, v in enumerate(graph.patterns))
    payloads = {}
    nontext = {n: float(np.random.default_rng(keyed_seed(cfg.seed, "payload:" + n, index)).uniform(
                        *cfg.nontext_return_mb)) for n in lengths["tool_text"]}
    for n, kind in graph.node_types.items():
        payloads[n] = ((lengths["outputs"][n] if kind == "llm" else lengths["tool_text"][n])
                       * cfg.text_bytes_per_token / 1e6 + nontext.get(n, 0.))
    root = "l0"
    mean_final_output = sum(v.probability * lengths["outputs"][v.exits[0]] for v in graph.patterns)
    spec = ApplicationSpec(f"app-{index:04d}", arrivals, nodes, flows, slo, quality,
        {m: prompts[root] * cfg.text_bytes_per_token / 1e6 for m in models},
        {m: mean_final_output * cfg.text_bytes_per_token / 1e6 for m in models},
        {(m, u, v): payloads[u] for m in models for u, v in full.edges},
        family="probabilistic_expansion", template_id="grammar-v1", length_class="synthetic",
        exit_data_by_flow={flow.id: {m: lengths["outputs"][flow.final_node]
                           * cfg.text_bytes_per_token / 1e6 for m in models} for flow in flows})
    return spec, {"eligible_ingress": servers, "zero_ingress": sorted(set(servers) - set(active)),
                  "tool_nontext_mb": nontext, "node_payload_mb": payloads}


def bind_tools(applications, template, cfg, uniform_profile=False):
    rng = np.random.default_rng(keyed_seed(cfg.seed, "bindings"))
    slots = [(i, n) for i, app in enumerate(applications) for n, node in app.spec.nodes.items()
             if node.type is NodeType.TOOL]
    if cfg.tool_types > len(slots):
        raise ValueError(f"{cfg.tool_types} active tool types need at least as many calls; got {len(slots)}")
    shared_count = int(round(cfg.tool_types * cfg.shared_fraction))
    if shared_count and len({i for i, _ in slots}) < 2:
        raise ValueError("Shared tools need at least two non-pure applications")
    remaining = list(slots)
    rng.shuffle(remaining)
    binding = {}
    owners = {}
    shared = [f"shared-{i:04d}" for i in range(shared_count)]
    private = [f"private-{i:04d}" for i in range(cfg.tool_types - shared_count)]
    # Every catalogue type is actually used. Shared eligibility is not a claim
    # that every shared type must occur in two applications in every subset.
    for tool in shared:
        first = remaining.pop()
        binding[first] = tool
    if not shared and len(private) < len({i for i, _ in slots}):
        raise ValueError("An all-private catalogue needs one type per non-pure application")
    reserved = []
    if not shared:
        for owner in sorted({i for i, _ in slots}):
            position = next(j for j, item in enumerate(remaining) if item[0] == owner)
            reserved.append(remaining.pop(position))
    private_by_app = defaultdict(list)
    for tool in private:
        slot = reserved.pop(0) if reserved else remaining.pop()
        owners[tool] = applications[slot[0]].spec.id
        private_by_app[slot[0]].append(tool)
        binding[slot] = tool
    popularity = np.arange(1, len(shared) + 1, dtype=float) ** (-cfg.shared_zipf_exponent)
    if len(popularity):
        popularity /= popularity.sum()
    for slot in remaining:
        choices = private_by_app[slot[0]]
        if choices and (not shared or rng.random() < cfg.private_request_probability):
            binding[slot] = str(rng.choice(choices))
        elif shared:
            binding[slot] = str(rng.choice(shared, p=popularity))
        else:
            # No shared catalogue: allocate one existing type only to its owner.
            raise ValueError("An application has calls but no eligible private tool")
    for i, app in enumerate(applications):
        app.spec = replace(app.spec, nodes={n: replace(node, tool=binding[(i, n)])
                           if node.type is NodeType.TOOL else node for n, node in app.spec.nodes.items()})
    tools = {}
    for name in shared + private:
        trng = np.random.default_rng(keyed_seed(cfg.seed, "tool-profile:" + name))
        rate = float(sum(cfg.tool_service_rate_rps) / 2 if uniform_profile
                     else trng.uniform(*cfg.tool_service_rate_rps))
        memory = float(sum(cfg.tool_memory_gb) / 2 if uniform_profile
                       else trng.uniform(*cfg.tool_memory_gb))
        running = (cfg.tool_cpu_cost_per_s + memory * cfg.tool_memory_cost_per_gb_s
                   if cfg.tool_cpu_cost_per_s is not None else cfg.tool_running_cost)
        start = (running * cfg.tool_start_cost_ratio_s
                 if cfg.tool_start_cost_ratio_s is not None else cfg.tool_start_cost)
        tools[name] = ToolSpec(name, 1, memory, {n: rate for n in template.servers},
                              1., running, start)
    return tools, {"private_owners": owners, "shared_tools": shared,
                   "profile_mode": "uniform" if uniform_profile else "heterogeneous"}


def identity_graph(app):
    """Identifier-free attributed graph including weighted pattern membership."""
    graph = nx.DiGraph()
    for n, node in app.spec.nodes.items():
        attributes = asdict(node)
        attributes.pop("id")
        if node.type is NodeType.TOOL:
            attributes["return_text_tokens"] = app.lengths["tool_text"][n]
        graph.add_node("call:" + n, signature=digest(attributes))
    for u, v in app.graph.graph().edges:
        graph.add_edge("call:" + u, "call:" + v, signature=digest({"dependency": True,
            "payloads": {m: app.spec.edge_data_mb[(m, u, v)] for m in app.spec.quality}}))
    for i, flow in enumerate(app.spec.pattern_flows):
        pattern = "pattern:" + str(i)
        graph.add_node(pattern, signature=digest({"probability": flow.probability}))
        for n in flow.nodes:
            call = app.lengths["patterns"][flow.id]["calls"].get(n)
            features = None if call is None else {k: v for k, v in call.items() if k != "input_messages"}
            graph.add_edge(pattern, "call:" + n, signature=digest(
                {"member": True, "final": n == flow.final_node, "call_lengths": features}))
    return graph


def config_global_identity(app):
    return digest({"quality": app.spec.quality, "slo": asdict(app.spec.slo),
                   "ingress": app.spec.ingress_rates, "entry": app.spec.entry_data_mb,
                   "exit": app.spec.exit_data_mb,
                   "exit_by_flow_multiset": sorted(digest(payload) for payload in app.spec.exit_data_by_flow.values())})


def same_application(left, right):
    return (config_global_identity(left) == config_global_identity(right)
            and nx.is_isomorphic(identity_graph(left), identity_graph(right),
                node_match=lambda a, b: a["signature"] == b["signature"],
                edge_match=lambda a, b: a["signature"] == b["signature"]))


def pool_statistics(applications, tools):
    values = {name: [] for name in ("execution_llms", "input", "output", "tool_text")}
    weights = {name: [] for name in values}
    arrived = {name: [] for name in values}
    topology_buckets = defaultdict(list)
    topology_count = 0
    structures = []
    operations = Counter()
    use = defaultdict(set)
    for app in applications:
        s = validate_graph(app.graph)
        structures.append(s)
        operations.update(app.graph.operation_counts)
        full = app.graph.graph()
        key = nx.weisfeiler_lehman_graph_hash(full, node_attr="type")
        if not any(nx.is_isomorphic(full, old, node_match=lambda a, b: a["type"] == b["type"])
                   for old in topology_buckets[key]):
            topology_buckets[key].append(full)
            topology_count += 1
        rate = sum(app.spec.ingress_rates.values())
        for n, node in app.spec.nodes.items():
            if node.type is NodeType.TOOL:
                use[node.tool].add(app.spec.id)
        for flow in app.spec.pattern_flows:
            record = app.lengths["patterns"][flow.id]
            nllm = len(record["calls"])
            values["execution_llms"].append(nllm)
            weights["execution_llms"].append(flow.probability)
            arrived["execution_llms"].append(rate * flow.probability)
            for call in record["calls"].values():
                for name, fieldname in (("input", "input_tokens"), ("output", "output_tokens")):
                    values[name].append(call[fieldname])
                    weights[name].append(flow.probability)
                    arrived[name].append(rate * flow.probability)
            for n in flow.nodes:
                if n in app.lengths["tool_text"]:
                    values["tool_text"].append(app.lengths["tool_text"][n])
                    weights["tool_text"].append(flow.probability)
                    arrived["tool_text"].append(rate * flow.probability)
    return {"applications": len(applications), "unique_topologies": topology_count,
            "actual_tool_types": len(use), "catalogue_tool_types": len(tools),
            "actually_shared_types": sum(len(owners) > 1 for owners in use.values()),
            "non_reused_types": sum(len(owners) == 1 for owners in use.values()),
            "full_graph": {name: describe([s[name] for s in structures])
                           for name in ("llm_nodes", "tool_nodes", "depth", "width", "patterns")},
            "application_equal_pattern_weighted": {name: describe(values[name], weights[name]) for name in values},
            "arrival_weighted": {name: describe(values[name], arrived[name]) for name in values},
            "operation_counts": dict(sorted(operations.items())),
            "length_rejections": sum(app.lengths["length_attempts"] - 1 for app in applications),
            "structural_rejections": sum(app.record["structural_attempts"] - 1 for app in applications)}


@dataclass
class ApplicationPool:
    config: GeneratorConfig
    applications: list[GeneratedApplication]
    tools: dict
    provenance: dict

    def scenario(self, template, count=None):
        count = len(self.applications) if count is None else count
        if not 1 <= count <= len(self.applications):
            raise ValueError("Subset count is outside the fixed pool")
        apps = {a.spec.id: a.spec for a in self.applications[:count]}
        used = {node.tool for app in apps.values() for node in app.nodes.values() if node.type is NodeType.TOOL}
        variant_key = digest({"config": asdict(self.config), "bindings": self.provenance["bindings"]})[:12]
        scene = replace(template, id=f"agent-pool-{self.config.seed}-a{count}-h{len(used)}-{variant_key}",
            applications=apps, tools={t: self.tools[t] for t in sorted(used)},
            metadata={"infrastructure_template_metadata": template.metadata,
                "parameter_receipt": template.metadata.get("parameter_receipt"),
                "generator_config": asdict(self.config),
                "raw_pattern_details_required": True,
                "role": "synthetic-generator-validation", "quality_status": "synthetic proxy",
                "slo_status": "declared synthetic ranges",
                "generator_version": GENERATOR_VERSION, "pool_seed": self.config.seed,
                "application_ids": list(apps), "actual_tool_types": len(used),
                "workload_fixed_across_slots": True, "parameter_status": "declared synthetic; not calibrated"})
        ScenarioLoader.validate(scene)
        return scene

    def to_dict(self):
        return plain({"version": GENERATOR_VERSION, "config": asdict(self.config),
            "provenance": self.provenance, "tools": [asdict(t) for t in self.tools.values()],
            "applications": [{"spec": application_mapping(a.spec), "control": a.graph.control,
                              "operations": a.graph.operation_counts,
                              "lengths": a.lengths, "record": a.record} for a in self.applications],
            "statistics": pool_statistics(self.applications, self.tools)})

    def validate(self, template):
        self.scenario(template)
        identities = defaultdict(list)
        for app in self.applications:
            validate_graph(app.graph, self.config.max_depth)
            expected = replay_control(app.graph.control, app.graph.node_types, self.config.max_patterns).variants
            def key(v):
                return (v.nodes, v.edges, frozenset(v.entries), frozenset(v.exits))
            expected_probabilities = defaultdict(float)
            for variant in expected:
                expected_probabilities[key(variant)] += variant.probability
            actual_probabilities = defaultdict(float)
            for variant in app.graph.patterns:
                actual_probabilities[key(variant)] += variant.probability
            if set(expected_probabilities) != set(actual_probabilities) or any(
                abs(actual_probabilities[k] - expected_probabilities[k]) > 1e-10 for k in expected_probabilities):
                raise ValueError("Patterns violate serial/AND/OR control semantics")
            if any(not 48 <= length <= 5000 for length in app.lengths["tool_text"].values()):
                raise ValueError("Tool text length bound violated")
            representatives = representative_prompts(app.graph, app.lengths)
            for n, mean in representatives.items():
                if any(abs(value - mean) > 1e-8 for value in app.spec.nodes[n].prompt_tokens.values()):
                    raise ValueError("Node representative input disagrees with pattern lengths")
            for flow in app.spec.pattern_flows:
                record = app.lengths["patterns"][flow.id]
                if record["total_tokens"] > 128000:
                    raise ValueError("Application budget violated")
                if sum(call["input_tokens"] + call["output_tokens"] for call in record["calls"].values()) != record["total_tokens"]:
                    raise ValueError("Application budget record is inconsistent")
                for call in record["calls"].values():
                    if not (256 <= call["input_tokens"] <= 16384 and 16 <= call["output_tokens"] <= 2048
                            and call["input_tokens"] + call["output_tokens"] <= 32768
                            and sum(call["input_messages"].values()) == call["input_tokens"]
                            and call["new_input_tokens"] >= 0):
                        raise ValueError("Invalid call length or duplicated history")
            key = (config_global_identity(app), nx.weisfeiler_lehman_graph_hash(identity_graph(app), node_attr="signature", edge_attr="signature"))
            if any(same_application(app, other) for other in identities[key]):
                raise ValueError("Duplicate application configuration, ignoring identifiers")
            identities[key].append(app)
        used = {n.tool for a in self.applications for n in a.spec.nodes.values() if n.type is NodeType.TOOL}
        if used != set(self.tools):
            raise ValueError("Tool catalogue contains unused types")
        for tool, owner in self.provenance["bindings"]["private_owners"].items():
            if any(a.spec.id != owner and any(n.tool == tool for n in a.spec.nodes.values()) for a in self.applications):
                raise ValueError("Private tool escaped its application")
        return pool_statistics(self.applications, self.tools)


def application_mapping(app):
    result = asdict(app)
    result["nodes"] = [asdict(n) for n in app.nodes.values()]
    result["edge_data_mb"] = {f"{m}:{u}->{v}": x for (m, u, v), x in app.edge_data_mb.items()}
    return plain(result)


def scenario_mapping(scene):
    return plain({"id": scene.id, "metadata": scene.metadata,
        "simulation": asdict(scene.simulation), "reward": asdict(scene.reward),
        **{name: [asdict(x) for x in getattr(scene, name).values()]
           for name in ("servers", "models", "llm_configs", "candidates", "tools")},
        "links": [asdict(x) for x in scene.links],
        "applications": [application_mapping(a) for a in scene.applications.values()]})


def generate_pool(template, config=GeneratorConfig(), progress=None):
    config.validate()
    if not template.models or not template.servers:
        raise ValueError("Infrastructure template needs models and servers")
    context_limit = min(32768, min(c.max_model_len for c in template.llm_configs.values()))
    if context_limit < 8192 + OUTPUT.lower:
        raise ValueError("Template context is too small for the requested input p90")
    pick = np.random.default_rng(keyed_seed(config.seed, "pure-indices"))
    pure_ids = set(pick.permutation(config.applications)[:round(config.applications * config.pure_fraction)])
    skeletons = []
    for i in range(config.applications):
        rng = np.random.default_rng(keyed_seed(config.seed, "structure", i))
        lcount = 1 if i in pure_ids else sample_count(rng, config.llm_bounds, config.llm_log_mu, config.llm_log_sigma)
        tcount = 0 if i in pure_ids else sample_count(rng, config.tool_bounds, config.tool_log_mu, config.tool_log_sigma)
        for attempt in range(config.structural_attempts):
            graph = GraphExpander(rng, config.max_patterns, config.max_parallel,
                config.operation_weights, config.root_stop_range, config.short_stop_range,
                config.later_stop_range).expand(lcount, tcount)
            try:
                stats = validate_graph(graph, config.max_depth)
            except ValueError:
                continue
            skeletons.append((graph, {**stats, "structural_attempts": attempt + 1,
                                     "source_index": i, "seed": keyed_seed(config.seed, "structure", i)}))
            break
        else:
            raise ValueError(f"Application {i} has no feasible structure within attempt budget")
    distributions = [INPUT, OUTPUT, TOOL]
    history = []
    apps = []
    def materialize(proposals):
        generated = []
        for i, (graph, record) in enumerate(skeletons):
            lengths = generate_lengths(graph, keyed_seed(config.seed, "lengths", i),
                *proposals, call_budget=context_limit, max_attempts=config.length_attempts)
            spec, payloads = synthetic_attributes(template, graph, lengths, config, i)
            generated.append(GeneratedApplication(spec, graph, lengths, {**record, **payloads}))
        return generated
    for iteration in range(config.calibration_passes + 1):
        apps = materialize(distributions)
        statistics = pool_statistics(apps, {})["application_equal_pattern_weighted"]
        history.append({"pass": iteration, "statistics": statistics,
                        "proposal_parameters": [asdict(d) for d in distributions]})
        if progress:
            progress(iteration, statistics)
        if iteration == config.calibration_passes:
            break
        # Offline distribution calibration only. No policy, deployment or performance feedback.
        targets = [(2048., 8192.), (200., 1200.), (256., 1500.)]
        for index, name in enumerate(("input", "output", "tool_text")):
            measured = statistics[name]
            median, tail = targets[index]
            scale = math.log(median / measured["p50"])
            spread = math.log(tail / median) / max(.1, math.log(measured["p90"] / measured["p50"]))
            old = distributions[index]
            distributions[index] = replace(old, mu=old.mu + .5 * scale,
                sigma=float(np.clip(old.sigma * spread ** .25, .25, 4.)))
    # Refine the coupled INPUT quantiles with a finite-difference two-parameter
    # update. Output/tool draws and structures use common random numbers.
    refinements = []
    if config.calibration_passes:
        x = np.asarray([distributions[0].mu, math.log(distributions[0].sigma)])
        def evaluate_input(point):
            proposals = [replace(distributions[0], mu=float(point[0]), sigma=math.exp(float(point[1]))),
                         *distributions[1:]]
            candidate = materialize(proposals)
            summary = pool_statistics(candidate, {})["application_equal_pattern_weighted"]
            residual = np.log([summary["input"]["p50"] / 2048., summary["input"]["p90"] / 8192.])
            return residual, candidate, summary, proposals
        residual, apps, summary, proposals = evaluate_input(x)
        for step in range(8):
            refinements.append({"step": step, "residual": residual.tolist(), "statistics": summary})
            if np.max(np.abs(residual)) < .035:
                break
            jacobian = np.column_stack([
                (evaluate_input(x + np.eye(2)[j] * .08)[0] - residual) / .08 for j in range(2)])
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            delta = np.clip(delta, [-.5, -.3], [.5, .3])
            refinements[-1]["jacobian"] = jacobian.tolist()
            refinements[-1]["delta"] = delta.tolist()
            improved = False
            for fraction in (1., .5, .25):
                candidate_x = np.clip(x + fraction * delta, [1., math.log(.25)], [11., math.log(4.)])
                r, a, s, p = evaluate_input(candidate_x)
                if np.linalg.norm(r) < np.linalg.norm(residual):
                    x, residual, apps, summary, proposals = candidate_x, r, a, s, p
                    improved = True
                    break
            if not improved:
                # Quantiles are discrete and can have a poorly conditioned
                # local Jacobian. Try a bounded coarse grid rather than claiming convergence.
                candidates = [(evaluate_input(x + [a, b]), x + [a, b])
                              for a in (-.5, 0., .5, 1.) for b in (-.3, 0., .3)
                              if a or b]
                result, point = min(candidates, key=lambda item: np.linalg.norm(item[0][0]))
                if np.linalg.norm(result[0]) >= np.linalg.norm(residual):
                    break
                residual, apps, summary, proposals = result
                x = point
        distributions = proposals
        if progress:
            progress("input-refined", summary)
    tools, bindings = bind_tools(apps, template, config)
    # Duplicate identity is checked after binding because service identity is
    # part of an application. A collision gets fresh lengths, never just an ID.
    accepted = defaultdict(list)
    duplicate_resamples = 0
    for i, app in enumerate(apps):
        for attempt in range(config.length_attempts):
            key = (config_global_identity(app), nx.weisfeiler_lehman_graph_hash(
                identity_graph(app), node_attr="signature", edge_attr="signature"))
            if not any(same_application(app, other) for other in accepted[key]):
                accepted[key].append(app)
                break
            duplicate_resamples += 1
            lengths = generate_lengths(app.graph, keyed_seed(config.seed, "deduplicate", i * 1000 + attempt),
                *distributions, call_budget=context_limit, max_attempts=config.length_attempts)
            spec, payload = synthetic_attributes(template, app.graph, lengths, config, i)
            app.spec = replace(spec, nodes={n: replace(node, tool=app.spec.nodes[n].tool)
                                for n, node in spec.nodes.items()})
            app.lengths = lengths
            app.record.update(payload)
            app.record["duplicate_length_resamples"] = attempt + 1
        else:
            raise ValueError("Unable to produce a distinct application without renaming")
    # Fixed stratified interleaving by full-graph size, not performance/SLO success.
    rng = np.random.default_rng(keyed_seed(config.seed, "pool-order"))
    strata = defaultdict(list)
    for app in apps:
        size = app.record["llm_nodes"]
        strata[-1 if app.record["pure"] else min(3, (size - 1) // 4)].append(app)
    for group in strata.values():
        rng.shuffle(group)
    ordered = []
    counts = {k: len(v) for k, v in strata.items()}
    delivered = Counter()
    while any(strata.values()):
        step = len(ordered) + 1
        key = max((k for k in strata if strata[k]),
                  key=lambda k: (step * counts[k] / len(apps) - delivered[k], -k))
        ordered.append(strata[key].pop())
        delivered[key] += 1
    pool = ApplicationPool(config, ordered, tools,
        {"bindings": bindings, "length_calibration": history,
         "input_quantile_refinement": refinements,
         "duplicate_resamples": duplicate_resamples,
         "length_distributions": [asdict(d) for d in distributions],
         "reference_method": "declared synthetic grammar; no LLM or PPO filtering",
         "inference_perf_reference_commit": "af6a7c2cba7b34f2567d1bbc3f3cfc514cb8f96b",
         "software": {"python": platform.python_version(), "numpy": np.__version__,
                      "networkx": nx.__version__},
         "units": {"rates": "request/s", "latency": "s", "payload": "MB (decimal)",
                   "token": "synthetic integer length, no real tokenizer"}})
    pool.validate(template)
    return pool


def rebind_pool(pool, template, tool_types, shared_fraction=None, uniform_profile=True):
    """Tool-scale variant: identical structures, probabilities, lengths and SLOs."""
    import copy
    cfg = replace(pool.config, tool_types=tool_types,
                  shared_fraction=pool.config.shared_fraction if shared_fraction is None else shared_fraction)
    cfg.validate()
    applications = copy.deepcopy(pool.applications)
    tools, bindings = bind_tools(applications, template, cfg, uniform_profile=uniform_profile)
    result = ApplicationPool(cfg, applications, tools, {**copy.deepcopy(pool.provenance),
        "bindings": bindings, "variant": "tool-binding-only; uniform resource profiles recommended",
        "source_pool_digest": digest(pool.to_dict())})
    result.validate(template)
    return result


def pool_from_mapping(payload, template):
    if payload["version"] != GENERATOR_VERSION:
        raise ValueError("Unsupported application-pool version")
    raw = scenario_mapping(template)
    raw["applications"] = [record["spec"] for record in payload["applications"]]
    raw["tools"] = payload["tools"]
    scenario = ScenarioLoader.from_mapping(raw)
    if len(scenario.applications) != len(payload["applications"]):
        raise ValueError("Duplicate application identifiers in pool file")
    apps = []
    from .agent_graph import Variant
    for record in payload["applications"]:
        spec = scenario.applications[record["spec"]["id"]]
        variants = [Variant(frozenset(flow.nodes), frozenset(flow.edges), flow.sources,
                            (flow.final_node,), flow.probability) for flow in spec.pattern_flows]
        graph = ExpandedGraph({n: node.type.value for n, node in spec.nodes.items()},
                              variants, record["control"], record["operations"])
        apps.append(GeneratedApplication(spec, graph, record["lengths"], record["record"]))
    pool = ApplicationPool(GeneratorConfig(**payload["config"]), apps, scenario.tools, payload["provenance"])
    pool.validate(template)
    return pool
