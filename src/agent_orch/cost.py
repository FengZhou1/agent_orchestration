"""Cost rates and one-off switching charges; neither uses slot duration."""
from collections import Counter
from agent_orch.schema.models import Scenario, DeploymentDecision


def operating_cost_rate(scenario: Scenario, deployment: DeploymentDecision) -> float:
    # Historical catalogue field name is retained for read compatibility;
    # values are price per second, not an amount charged per physical slot.
    return float(sum(scenario.llm_configs[scenario.candidates[c].config].running_cost_per_slot
                     for c, active in deployment.llm_active.items() if active)
                 + sum(z * scenario.tools[h].running_cost_per_slot
                       for (h, _), z in deployment.tool_replicas.items()))


def switching_charge(scenario: Scenario, current: DeploymentDecision,
                     previous: DeploymentDecision) -> float:
    def counts(deployment):
        return Counter((scenario.candidates[c].config, scenario.candidates[c].server)
                       for c, active in deployment.llm_active.items() if active)
    now, before = counts(current), counts(previous)
    # Replica indices distinguish identical candidates, not different service
    # configurations. Relabelling an unchanged replica count is not a reload.
    return float(sum(max(0, count - before[pool]) * scenario.llm_configs[pool[0]].load_cost
                     for pool, count in now.items())
                 + sum(max(0, z - previous.tool_replicas.get(pool, 0))
                       * scenario.tools[pool[0]].start_cost
                       for pool, z in current.tool_replicas.items()))
