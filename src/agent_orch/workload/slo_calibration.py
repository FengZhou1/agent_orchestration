"""Frozen synthetic SLOs from an isolated fixed-hardware reference.

Quantiles are over the declared pattern distribution of analytical workflow
latencies. They are NOT empirical request-tail or queue-wait quantiles. No PPO,
resource-relative load adjustment, or policy-score-based threshold tuning occurs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from agent_orch.action_decoder import ActionDecoder
from agent_orch.performance import AnalyticalBackend
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.schema.models import (
    CandidateInstance, DeploymentDecision, LinkSpec, NodeType, ServerSpec, SLOSpec, SLOType,
)
from .agent_pool import application_mapping, digest

VERSION = "fixed-reference-slo-v1"


@dataclass(frozen=True)
class CalibrationSpec:
    model: str = "qwen3-8b"
    gpu_type: str = "L20"
    tensor_parallel: int = 1
    quantile: float = .95
    probe_arrival_rps: float = 1e-6
    maximum_reference_utilization: float = .01
    link_capacity_mbps: float = 100.
    propagation_ms: float = 1.
    ttft_multiplier: float = 1.5
    tbt_multiplier: float = 1.25
    deadline_multiplier: float = 1.5
    tight_scale: float = .8

    def validate(self):
        if self.tensor_parallel < 1 or not 0 < self.quantile <= 1:
            raise ValueError("Invalid reference configuration or quantile")
        values = (self.probe_arrival_rps, self.maximum_reference_utilization,
                  self.link_capacity_mbps, self.propagation_ms, self.ttft_multiplier,
                  self.tbt_multiplier, self.deadline_multiplier, self.tight_scale)
        if any(not math.isfinite(v) or v <= 0 for v in values) or self.tight_scale >= 1:
            raise ValueError("Calibration constants must be finite and positive; tight_scale < 1")


def weighted_quantile(values, weights, quantile):
    if len(values) != len(weights) or not values or not 0 < quantile <= 1:
        raise ValueError("A nonempty aligned distribution and quantile in (0,1] are required")
    if any(not math.isfinite(v) or v < 0 for v in (*values, *weights)) or sum(weights) <= 0:
        raise ValueError("Finite nonnegative values and positive total probability are required")
    ordered = sorted((v, w) for v, w in zip(values, weights) if w > 0)
    target, accumulated = quantile * sum(weights), 0.
    for value, weight in ordered:
        accumulated += weight
        if accumulated >= target:
            return float(value)
    return float(ordered[-1][0])


def _tool_profile(tool):
    rates = set(tool.service_rate.values())
    if len(rates) != 1:
        raise ValueError("The reference needs the registered homogeneous per-type service rates")
    return {"cpu_cores": tool.cpu_cores, "memory_gb": tool.memory_gb,
            "arrival_scv": tool.arrival_scv, "service_rate_rps": next(iter(rates))}


def workload_signature(app, tools):
    """Arrival/location/quality/SLO are not used to regenerate a task contract."""
    raw = application_mapping(app)
    for node in raw["nodes"]:
        node.pop("stage_deadline_s", None)
    used = sorted({n.tool for n in app.nodes.values() if n.type is NodeType.TOOL})
    return digest({"nodes": raw["nodes"], "patterns": raw["pattern_flows"],
                   "entry_data_mb": raw["entry_data_mb"], "exit_data_mb": raw["exit_data_mb"],
                   "exit_data_by_flow": raw["exit_data_by_flow"], "edge_data_mb": raw["edge_data_mb"],
                   "tools": {key: _tool_profile(tools[key]) for key in used}})


def request_signature(app):
    """A task contract survives changes of placement, service binding/capacity.

    The calibrated reference's full tool profiles remain separately recorded.
    Graph/type/probability/token/payload changes are different request workloads.
    """
    raw = application_mapping(app)
    for node in raw["nodes"]:
        node.pop("stage_deadline_s", None)
        node.pop("tool", None)
    return digest({k: raw[k] for k in ("nodes", "pattern_flows", "entry_data_mb",
                                     "exit_data_mb", "exit_data_by_flow", "edge_data_mb")})


def reference_environment(source, app, spec):
    """Gateway + one LLM host + one host/replica per used tool type.

    This is an isolated component reference, NOT a proposed deployment of all
    applications in the training network. Source objects and rates stay intact.
    """
    spec.validate()
    choices = [c for c in source.llm_configs.values() if c.model == spec.model and
               c.gpu_type == spec.gpu_type and c.gpu_count == spec.tensor_parallel]
    if len(choices) != 1:
        raise ValueError("Exactly one fixed reference model/GPU/configuration is required")
    config = choices[0]
    servers = {
        "reference-gateway": ServerSpec("reference-gateway", 8, 16., 0, 0., "none"),
        "reference-llm": ServerSpec("reference-llm", 8, 16., config.gpu_count,
                                    config.reserved_memory_gb_per_gpu, config.gpu_type),
    }
    pairs = [("reference-gateway", "reference-llm")]
    used = sorted({n.tool for n in app.nodes.values() if n.type is NodeType.TOOL})
    locations = {}
    for index, key in enumerate(used):
        host = f"reference-tool-{index:03d}"
        tool = source.tools[key]
        servers[host] = ServerSpec(host, max(8, tool.cpu_cores), max(16., tool.memory_gb), 0, 0., "none")
        pairs.append((host, "reference-llm"))
        locations[key] = host
    links = tuple(LinkSpec(u, v, spec.link_capacity_mbps, spec.propagation_ms)
                  for a, b in pairs for u, v in ((a, b), (b, a)))
    tools = {key: replace(source.tools[key], service_rate={n: _tool_profile(source.tools[key])[
        "service_rate_rps"] for n in servers}) for key in used}
    candidate = CandidateInstance("reference-instance", spec.model, config.id, "reference-llm")
    # Non-binding contracts inside this temporary reference avoid any dependence
    # on the old placeholder thresholds, including an unset old threshold.
    isolated_app = replace(app, ingress_rates={"reference-gateway": spec.probe_arrival_rps},
        slo=SLOSpec(app.slo.type, math.inf, math.inf, math.inf),
        nodes={key: replace(node, stage_deadline_s=None) for key, node in app.nodes.items()})
    scene = replace(source, id=f"reference-{app.id}", servers=servers, links=links,
                    llm_configs={config.id: config}, candidates={candidate.id: candidate},
                    tools=tools, applications={app.id: isolated_app}, metadata={"role": "isolated-SLO-reference"})
    ScenarioLoader.validate(scene)
    deployment = DeploymentDecision({candidate.id: 1}, {(h, n): int(locations[h] == n)
                                     for h in tools for n in servers})
    ActionDecoder(scene).validate_deployment(deployment)
    shares = {(app.id, "reference-gateway", m): float(m == spec.model) for m in scene.models}
    routing = CapacityProportionalRouter(scene).route(deployment, shares)
    ActionDecoder(scene).validate_routing(deployment, routing)
    return scene, deployment, routing


def _variant(app, reference, stages, spec, scale):
    values = {"ttft_s": reference["ttft_s"] * (spec.ttft_multiplier * scale),
              "tbt_s": reference["tbt_s"] * (spec.tbt_multiplier * scale),
              "deadline_s": reference["e2e_s"] * (spec.deadline_multiplier * scale)}
    kind = app.slo.type
    slo = {"type": kind.value}
    if kind is not SLOType.DEADLINE:
        slo.update(ttft_s=values["ttft_s"], tbt_s=values["tbt_s"])
    if kind is not SLOType.LATENCY:
        slo["deadline_s"] = values["deadline_s"]
    # Stage times are absolute completion times from request entry, conditional
    # on visiting the node as an intermediate stage (not its service time).
    deadlines = ({node: min(values["deadline_s"], value * (spec.deadline_multiplier * scale))
                  for node, value in stages.items()} if kind is SLOType.COMPOUND else {})
    return {"slo": slo, "stage_deadlines_s": deadlines}


def calibrated_application(app, values):
    slo_raw = values["slo"]
    slo = SLOSpec(**{**slo_raw, "type": SLOType(slo_raw["type"])})
    nodes = {n: replace(node, stage_deadline_s=values["stage_deadlines_s"].get(n))
             for n, node in app.nodes.items()}
    return replace(app, slo=slo, nodes=nodes)


def calibrate_application(source, app, spec):
    scene, deployment, routing = reference_environment(source, app, spec)
    backend = AnalyticalBackend(scene)
    analytical = backend.evaluate(deployment, routing)
    utilization = max([0., *analytical.llm_utilization.values(), *analytical.tool_utilization.values(),
                       *backend.network.utilization(analytical.link_load_mbps).values()])
    if analytical.violations or not math.isfinite(utilization) or utilization > spec.maximum_reference_utilization:
        raise ValueError(f"{app.id}: fixed reference is not stable/low-load; do not silently discard it")
    result = WorkflowEvaluator(scene, backend, max_mapping_samples=1).evaluate(deployment, routing, analytical)
    records = []
    for flow in app.pattern_flows:
        key = f"{app.id}:reference-gateway:{spec.model}:{flow.id}"
        row = result.flow_metrics[key]
        if not math.isclose(row["completion_probability"], 1., abs_tol=1e-10):
            raise ValueError(f"{app.id}/{flow.id}: a reference pattern is incomplete")
        records.append({"pattern": flow.id, "probability": flow.probability,
                        **{k: v for k, v in row.items() if k in ("ttft_s", "tbt_s", "e2e_s") or k.startswith("stage:")}})
    weights = [r["probability"] for r in records]
    reference = {k: weighted_quantile([r[k] for r in records], weights, spec.quantile)
                 for k in ("ttft_s", "tbt_s", "e2e_s")}
    stages = {}
    names = {k for r in records for k in r if k.startswith("stage:")}
    for name in names:
        rows = [r for r in records if name in r]
        stages[name.removeprefix("stage:").removesuffix("_s")] = weighted_quantile(
            [r[name] for r in rows], [r["probability"] for r in rows], spec.quantile)
    variants = {name: _variant(app, reference, stages, spec, scale)
                for name, scale in (("main", 1.), ("tight", spec.tight_scale))}
    attainment = {}
    for name, variant in variants.items():
        updated = replace(scene, applications={app.id: calibrated_application(scene.applications[app.id], variant)})
        value = WorkflowEvaluator(updated, backend, max_mapping_samples=1).evaluate(deployment, routing, analytical)
        attainment[name] = value.goodput_rps / value.total_arrival_rps
    return {"workload_digest": workload_signature(app, source.tools), "request_digest": request_signature(app),
            "reference_tool_profiles": {key: _tool_profile(source.tools[key]) for key in sorted({
                n.tool for n in app.nodes.values() if n.type is NodeType.TOOL})}, "type": app.slo.type.value,
            "reference_quantile_s": reference, "reference_stage_quantile_s": stages,
            "variants": variants, "reference_joint_attainment": attainment,
            "maximum_reference_utilization": utilization}, records


def sign_contracts(payload):
    return {**payload, "contract_digest": digest(payload)}


def validate_contracts(payload):
    if payload.get("version") != VERSION or payload.get("contract_digest") != digest(
            {k: v for k, v in payload.items() if k != "contract_digest"}):
        raise ValueError("SLO contract version/digest is invalid")
    for record in payload["applications"].values():
        for variant in record["variants"].values():
            values = [v for k, v in variant["slo"].items() if k != "type"]
            if any(not math.isfinite(v) or v < 0 for v in (*values, *variant["stage_deadlines_s"].values())):
                raise ValueError("SLO contract has invalid numeric thresholds")


def apply_contracts(scene, payload, variant="main"):
    validate_contracts(payload)
    applications = {}
    for key, app in scene.applications.items():
        record = payload["applications"].get(key)
        if record is None or record["request_digest"] != request_signature(app):
            raise ValueError(f"{key}: SLO contracts do not match graph/token/payload request workload; calibrate explicitly")
        if record["type"] != app.slo.type.value or variant not in record["variants"]:
            raise ValueError("SLO type/variant differs from its frozen contract")
        applications[key] = calibrated_application(app, record["variants"][variant])
    old = scene.metadata.get("slo_calibration", {})
    old_suffix = f"-slo-{old.get('variant')}-{str(old.get('contract_digest', ''))[:10]}"
    base_id = scene.id.removesuffix(old_suffix)
    return replace(scene, id=f"{base_id}-slo-{variant}-{payload['contract_digest'][:10]}",
                   applications=applications,
                   metadata={**scene.metadata, "slo_status": "fixed synthetic reference-calibrated contract",
                             "slo_calibration": {"contract_digest": payload["contract_digest"],
                                                 "variant": variant, "version": VERSION,
                                                 "inheritance": "fixed logical request contract; no resource-dependent recalibration"}})


def matches_contracts(scene, payload):
    metadata = scene.metadata.get("slo_calibration", {})
    if metadata.get("contract_digest") != payload.get("contract_digest"):
        return False
    try:
        expected = apply_contracts(scene, payload, metadata.get("variant", "main"))
    except (ValueError, KeyError):
        return False
    return expected.applications == scene.applications
