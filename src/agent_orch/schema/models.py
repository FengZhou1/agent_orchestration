from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    LLM = "llm"
    TOOL = "tool"


class SLOType(str, Enum):
    LATENCY = "lat"
    DEADLINE = "ddl"
    COMPOUND = "cmp"


@dataclass(frozen=True)
class ServerSpec:
    id: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    gpu_type: str = "generic"


@dataclass(frozen=True)
class LinkSpec:
    source: str
    target: str
    capacity_mbps: float
    propagation_ms: float
    cost_per_mb: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    parameter_count: float
    layers: int
    hidden_size: int
    weight_bytes: float
    kv_bytes_per_token: float


@dataclass(frozen=True)
class LLMConfigSpec:
    id: str
    model: str
    gpu_count: int
    gpu_share: float
    reserved_memory_gb_per_gpu: float
    effective_flops: float
    effective_bandwidth_bytes_s: float
    kv_token_capacity: float
    running_cost_per_slot: float
    load_cost: float = 0.0
    gpu_type: str = "generic"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 128
    chunked_prefill: bool = True
    prefix_cache: bool = False


@dataclass(frozen=True)
class ToolSpec:
    id: str
    cpu_cores: int
    memory_gb: float
    service_rate: dict[str, float]
    arrival_scv: float = 1.0
    running_cost_per_slot: float = 0.0
    start_cost: float = 0.0


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    type: NodeType
    tool: str | None = None
    prompt_tokens: dict[str, float] = field(default_factory=dict)
    output_tokens: dict[str, float] = field(default_factory=dict)
    stage_deadline_s: float | None = None


@dataclass(frozen=True)
class PatternFlow:
    id: str
    probability: float
    final_node: str
    chains: tuple[tuple[str, ...], ...]

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(node for chain in self.chains for node in chain))

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        edges: list[tuple[str, str]] = []
        for chain in self.chains:
            edges.extend(zip(chain[:-1], chain[1:]))
        return tuple(dict.fromkeys(edges))

    @property
    def sources(self) -> tuple[str, ...]:
        incoming = {v for _, v in self.edges}
        return tuple(node for node in self.nodes if node not in incoming)


@dataclass(frozen=True)
class SLOSpec:
    type: SLOType
    ttft_s: float | None = None
    tbt_s: float | None = None
    deadline_s: float | None = None


@dataclass(frozen=True)
class ApplicationSpec:
    id: str
    ingress_rates: dict[str, float]
    nodes: dict[str, WorkflowNode]
    pattern_flows: tuple[PatternFlow, ...]
    slo: SLOSpec
    quality: dict[str, float]
    entry_data_mb: dict[str, float]
    exit_data_mb: dict[str, float]
    edge_data_mb: dict[tuple[str, str, str], float]
    family: str = "unspecified"
    template_id: str = "unspecified"
    length_class: str = "unspecified"

    def visit_probability(self, node_id: str) -> float:
        return sum(
            flow.probability
            for flow in self.pattern_flows
            if node_id in flow.nodes
        )


@dataclass(frozen=True)
class CandidateInstance:
    id: str
    model: str
    config: str
    server: str


@dataclass(frozen=True)
class SimulationSpec:
    slot_seconds: float = 1.0
    prefill_chunk_tokens: int = 512
    overload_delay_s: float = 60.0
    orchestration_period_s: float = 60.0
    max_tool_replicas_per_server: int = 4


@dataclass(frozen=True)
class RewardSpec:
    cost_weight: float = 0.25
    latency_weight: float = 0.25
    goodput_weight: float = 0.25
    quality_weight: float = 0.25


@dataclass(frozen=True)
class Scenario:
    id: str
    servers: dict[str, ServerSpec]
    links: tuple[LinkSpec, ...]
    models: dict[str, ModelSpec]
    llm_configs: dict[str, LLMConfigSpec]
    tools: dict[str, ToolSpec]
    candidates: dict[str, CandidateInstance]
    applications: dict[str, ApplicationSpec]
    metadata: dict[str, Any] = field(default_factory=dict)
    simulation: SimulationSpec = field(default_factory=SimulationSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)


@dataclass
class DeploymentDecision:
    llm_active: dict[str, int]
    tool_replicas: dict[tuple[str, str], int]

    def copy(self) -> "DeploymentDecision":
        return DeploymentDecision(dict(self.llm_active), dict(self.tool_replicas))


@dataclass
class RoutingDecision:
    model_share: dict[tuple[str, str, str], float]
    llm_share: dict[tuple[str, str, str, str], float]
    tool_route: dict[tuple[str, str, str, str, str], float]


@dataclass
class LLMClassPerformance:
    service_s: float
    prefill_s: float
    decode_s: float
    ttft_s: float
    tbt_s: float
    response_s: float


@dataclass
class SlotMetrics:
    slot: int
    cost: float
    mean_latency_s: float
    goodput_rps: float
    quality: float
    total_arrival_rps: float
    slo_attainment: float
    violations: int
    app_latency_s: dict[str, float] = field(default_factory=dict)
    llm_utilization: dict[str, float] = field(default_factory=dict)
    tool_utilization: dict[str, float] = field(default_factory=dict)
    link_utilization: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    observation: dict[str, Any]
    reward_components: dict[str, float]
    metrics: SlotMetrics
    terminated: bool = False
    truncated: bool = False
