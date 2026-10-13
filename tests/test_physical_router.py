from agent_orch.routing import CapacityProportionalRouter
from agent_orch.schema.models import SlotMetrics


def _metrics(llm_utilization=None, tool_utilization=None):
    return SlotMetrics(
        slot=0,
        cost=0.0,
        mean_latency_s=0.0,
        goodput_rps=0.0,
        quality=0.0,
        total_arrival_rps=0.0,
        slo_attainment=0.0,
        violations=0,
        llm_utilization=llm_utilization or {},
        tool_utilization=tool_utilization or {},
    )


def test_physical_router_preserves_model_and_service_probabilities(scenario):
    from agent_orch.capacity import CapacityPlanner

    deployment = CapacityPlanner(scenario).initial_deployment()
    deployment.llm_active["small-n0"] = 1
    deployment.llm_active["small-n2"] = 1
    model_share = {}
    for app in scenario.applications.values():
        for ingress in app.ingress_rates:
            model_share[(app.id, ingress, "small")] = 0.6
            model_share[(app.id, ingress, "large")] = 0.4
    routing = CapacityProportionalRouter(scenario).route(deployment, model_share, _metrics())

    for app in scenario.applications.values():
        for ingress in app.ingress_rates:
            for node in app.nodes.values():
                if node.type.value != "llm":
                    continue
                small_sum = sum(
                    routing.llm_share.get((app.id, ingress, node.id, candidate), 0.0)
                    for candidate in ("small-n0", "small-n2")
                )
                assert abs(small_sum - 0.6) < 1.0e-7


def test_capacity_router_ignores_historical_instance_utilization(scenario):
    from agent_orch.capacity import CapacityPlanner

    deployment = CapacityPlanner(scenario).initial_deployment()
    deployment.llm_active["small-n0"] = 1
    deployment.llm_active["small-n2"] = 1
    app = next(iter(scenario.applications.values()))
    ingress = next(iter(app.ingress_rates))
    model_share = {
        (candidate_app.id, candidate_ingress, model): float(model == "small")
        for candidate_app in scenario.applications.values()
        for candidate_ingress in candidate_app.ingress_rates
        for model in scenario.models
    }
    router = CapacityProportionalRouter(scenario)
    routing = router.route(
        deployment,
        model_share,
        _metrics({"small-n0": 0.9, "small-n2": 0.0}),
    )
    llm_node = next(node for node in app.nodes.values() if node.type.value == "llm")
    loaded = routing.llm_share[(app.id, ingress, llm_node.id, "small-n0")]
    idle = routing.llm_share[(app.id, ingress, llm_node.id, "small-n2")]
    total_capacity = router.reference_capacities["small-n0"] + router.reference_capacities["small-n2"]
    assert abs(loaded - router.reference_capacities["small-n0"] / total_capacity) < 1e-7
    assert abs(loaded + idle - 1.0) < 1e-7
    assert routing == router.route(deployment, model_share, _metrics())
